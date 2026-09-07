# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
from libc.math cimport fabs
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy, memset


cpdef cnp.ndarray[cnp.float32_t, ndim=1] inner_stage(
    float[::1] center,
    float[:, ::1] Z,
    float[::1] y,
    cnp.int64_t[::1] indices,
    float eta,
    float lam,
    unsigned char[::1] accumulate,
):
    """Run one non-projected ASSG-c stage."""
    cdef Py_ssize_t m_inner = indices.shape[0]
    cdef Py_ssize_t r = center.shape[0]
    cdef Py_ssize_t k, j
    cdef cnp.int64_t i
    cdef float s = 1.0
    cdef float eta_lam = eta * lam
    cdef float shrink = 1.0 - eta_lam
    cdef float d, coef, inv
    cdef int count = 0
    cdef float* w = <float*> malloc(r * sizeof(float))
    cdef float* v_sum = <float*> malloc(r * sizeof(float))
    cdef cnp.ndarray[cnp.float32_t, ndim=1] out = np.empty(r, dtype=np.float32)

    if w == NULL or v_sum == NULL:
        if w != NULL:
            free(w)
        if v_sum != NULL:
            free(v_sum)
        raise MemoryError()

    try:
        for j in range(r):
            w[j] = center[j]
        memset(v_sum, 0, r * sizeof(float))

        for k in range(m_inner):
            i = indices[k]
            s = s * shrink
            if s < 1e-9:
                for j in range(r):
                    w[j] *= s
                s = 1.0

            d = 0.0
            for j in range(r):
                d += w[j] * Z[i, j]

            if y[i] * s * d < 1.0:
                coef = eta * y[i] / s
                for j in range(r):
                    w[j] += coef * Z[i, j]

            if accumulate[k] != 0:
                for j in range(r):
                    v_sum[j] += s * w[j]
                count += 1

        for j in range(r):
            w[j] *= s

        inv = 1.0 / count
        for j in range(r):
            out[j] = v_sum[j] * inv
        return out
    finally:
        free(w)
        free(v_sum)


cpdef cnp.ndarray[cnp.float32_t, ndim=1] run_all_stages_fixed_decay(
    float[:, ::1] Z,
    float[::1] y,
    cnp.int64_t[:, ::1] indices,
    unsigned char[:, ::1] accumulate_masks,
    float eta0,
    float lam,
):
    """Run all fixed-decay ASSG-c stages in one compiled loop."""
    cdef Py_ssize_t n_stages = indices.shape[0]
    cdef Py_ssize_t m_inner = indices.shape[1]
    cdef Py_ssize_t r = Z.shape[1]
    cdef Py_ssize_t stage, k, j
    cdef cnp.int64_t i
    cdef float eta = eta0
    cdef float eta_lam, shrink, s, d, coef, inv
    cdef int count
    cdef float* center = <float*> malloc(r * sizeof(float))
    cdef float* w = <float*> malloc(r * sizeof(float))
    cdef float* v_sum = <float*> malloc(r * sizeof(float))
    cdef cnp.ndarray[cnp.float32_t, ndim=1] out = np.empty(r, dtype=np.float32)

    if center == NULL or w == NULL or v_sum == NULL:
        if center != NULL:
            free(center)
        if w != NULL:
            free(w)
        if v_sum != NULL:
            free(v_sum)
        raise MemoryError()

    try:
        memset(center, 0, r * sizeof(float))

        for stage in range(n_stages):
            memcpy(w, center, r * sizeof(float))
            memset(v_sum, 0, r * sizeof(float))
            count = 0
            s = 1.0
            eta_lam = eta * lam
            shrink = 1.0 - eta_lam

            for k in range(m_inner):
                i = indices[stage, k]
                s = s * shrink
                if s < 1e-9:
                    for j in range(r):
                        w[j] *= s
                    s = 1.0

                d = 0.0
                for j in range(r):
                    d += w[j] * Z[i, j]

                if y[i] * s * d < 1.0:
                    coef = eta * y[i] / s
                    for j in range(r):
                        w[j] += coef * Z[i, j]

                if accumulate_masks[stage, k] != 0:
                    for j in range(r):
                        v_sum[j] += s * w[j]
                    count += 1

            inv = 1.0 / count
            for j in range(r):
                center[j] = v_sum[j] * inv

            eta *= 0.5

        for j in range(r):
            out[j] = center[j]
        return out
    finally:
        free(center)
        free(w)
        free(v_sum)


cpdef cnp.ndarray[cnp.float32_t, ndim=1] inner_stage_regularized(
    float[::1] center,
    float[:, ::1] Z,
    float[::1] y,
    cnp.int64_t[::1] indices,
    float eta,
    float lam,
    float beta,
    unsigned char[::1] accumulate,
):
    """Run one ASSG-r stage using the paper update with identity projection."""
    cdef Py_ssize_t m_inner = indices.shape[0]
    cdef Py_ssize_t r = center.shape[0]
    cdef Py_ssize_t k, j
    cdef cnp.int64_t i
    cdef float tau, d, inv, w_coef, center_coef, hinge_coef
    cdef int count = 0
    cdef float* w = <float*> malloc(r * sizeof(float))
    cdef float* u_sum = <float*> malloc(r * sizeof(float))
    cdef cnp.ndarray[cnp.float32_t, ndim=1] out = np.empty(r, dtype=np.float32)
    cdef bint active

    if w == NULL or u_sum == NULL:
        if w != NULL:
            free(w)
        if u_sum != NULL:
            free(u_sum)
        raise MemoryError()

    try:
        for j in range(r):
            w[j] = center[j]
        memset(u_sum, 0, r * sizeof(float))

        for k in range(m_inner):
            i = indices[k]
            tau = k + 1.0

            d = 0.0
            for j in range(r):
                d += w[j] * Z[i, j]
            active = y[i] * d < 1.0

            w_coef = 1.0 - (2.0 / tau) - ((2.0 * beta * lam) / tau)
            center_coef = 2.0 / tau
            if active:
                hinge_coef = 2.0 * beta * y[i] / tau
                for j in range(r):
                    w[j] = w_coef * w[j] + center_coef * center[j] + hinge_coef * Z[i, j]
            else:
                for j in range(r):
                    w[j] = w_coef * w[j] + center_coef * center[j]

            if accumulate[k] != 0:
                for j in range(r):
                    u_sum[j] += w[j]
                count += 1

        inv = 1.0 / count
        for j in range(r):
            out[j] = u_sum[j] * inv
        return out
    finally:
        free(w)
        free(u_sum)


cpdef cnp.ndarray[cnp.float32_t, ndim=1] run_all_stages_regularized_fixed_decay(
    float[:, ::1] Z,
    float[::1] y,
    cnp.int64_t[:, ::1] indices,
    unsigned char[:, ::1] accumulate_masks,
    float eta0,
    float lam,
    float beta0,
    float beta_decay,
):
    """Run all fixed-decay ASSG-r stages in one compiled loop."""
    cdef Py_ssize_t n_stages = indices.shape[0]
    cdef Py_ssize_t m_inner = indices.shape[1]
    cdef Py_ssize_t r = Z.shape[1]
    cdef Py_ssize_t stage, k, j
    cdef cnp.int64_t i
    cdef float beta = beta0
    cdef float tau, d, inv, w_coef, center_coef, hinge_coef
    cdef int count
    cdef bint active
    cdef float* center = <float*> malloc(r * sizeof(float))
    cdef float* w = <float*> malloc(r * sizeof(float))
    cdef float* u_sum = <float*> malloc(r * sizeof(float))
    cdef cnp.ndarray[cnp.float32_t, ndim=1] out = np.empty(r, dtype=np.float32)

    if center == NULL or w == NULL or u_sum == NULL:
        if center != NULL:
            free(center)
        if w != NULL:
            free(w)
        if u_sum != NULL:
            free(u_sum)
        raise MemoryError()

    try:
        memset(center, 0, r * sizeof(float))

        for stage in range(n_stages):
            for j in range(r):
                w[j] = center[j]
            memset(u_sum, 0, r * sizeof(float))
            count = 0

            for k in range(m_inner):
                i = indices[stage, k]
                tau = k + 1.0

                d = 0.0
                for j in range(r):
                    d += w[j] * Z[i, j]
                active = y[i] * d < 1.0

                w_coef = 1.0 - (2.0 / tau) - ((2.0 * beta * lam) / tau)
                center_coef = 2.0 / tau
                if active:
                    hinge_coef = 2.0 * beta * y[i] / tau
                    for j in range(r):
                        w[j] = w_coef * w[j] + center_coef * center[j] + hinge_coef * Z[i, j]
                else:
                    for j in range(r):
                        w[j] = w_coef * w[j] + center_coef * center[j]

                if accumulate_masks[stage, k] != 0:
                    for j in range(r):
                        u_sum[j] += w[j]
                    count += 1

            inv = 1.0 / count
            for j in range(r):
                center[j] = u_sum[j] * inv

            beta /= beta_decay

        for j in range(r):
            out[j] = center[j]
        return out
    finally:
        free(center)
        free(w)
        free(u_sum)


cpdef cnp.ndarray[cnp.float32_t, ndim=2] run_restarted_regularized_path(
    float[:, ::1] Z,
    float[::1] y,
    cnp.int64_t[:, ::1] indices,
    unsigned char[:, ::1] accumulate_masks,
    cnp.int64_t[::1] stage_m_inner,
    float[::1] stage_beta,
    float lam,
):
    """Run the complete restarted RASSG-r path in one compiled loop."""
    cdef Py_ssize_t n_stages = indices.shape[0]
    cdef Py_ssize_t r = Z.shape[1]
    cdef Py_ssize_t stage, k, j, m_inner
    cdef cnp.int64_t i
    cdef float beta, tau, d, inv, w_coef, center_coef, hinge_coef
    cdef int count
    cdef bint active
    cdef float* center = <float*> malloc(r * sizeof(float))
    cdef float* w = <float*> malloc(r * sizeof(float))
    cdef float* u_sum = <float*> malloc(r * sizeof(float))
    cdef cnp.ndarray[cnp.float32_t, ndim=2] out = np.empty((n_stages, r), dtype=np.float32)

    if center == NULL or w == NULL or u_sum == NULL:
        if center != NULL:
            free(center)
        if w != NULL:
            free(w)
        if u_sum != NULL:
            free(u_sum)
        raise MemoryError()

    try:
        memset(center, 0, r * sizeof(float))

        for stage in range(n_stages):
            m_inner = stage_m_inner[stage]
            beta = stage_beta[stage]
            for j in range(r):
                w[j] = center[j]
            memset(u_sum, 0, r * sizeof(float))
            count = 0

            for k in range(m_inner):
                i = indices[stage, k]
                tau = k + 1.0

                d = 0.0
                for j in range(r):
                    d += w[j] * Z[i, j]
                active = y[i] * d < 1.0

                w_coef = 1.0 - (2.0 / tau) - ((2.0 * beta * lam) / tau)
                center_coef = 2.0 / tau
                if active:
                    hinge_coef = 2.0 * beta * y[i] / tau
                    for j in range(r):
                        w[j] = w_coef * w[j] + center_coef * center[j] + hinge_coef * Z[i, j]
                else:
                    for j in range(r):
                        w[j] = w_coef * w[j] + center_coef * center[j]

                if accumulate_masks[stage, k] != 0:
                    for j in range(r):
                        u_sum[j] += w[j]
                    count += 1

            inv = 1.0 / count
            for j in range(r):
                center[j] = u_sum[j] * inv
                out[stage, j] = center[j]

        return out
    finally:
        free(center)
        free(w)
        free(u_sum)
