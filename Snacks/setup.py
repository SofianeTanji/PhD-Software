from __future__ import annotations

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


extensions = [
    Extension(
        "snacks._cython_solver",
        ["src/snacks/_cython_solver.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "nonecheck": False,
            "cdivision": True,
        },
    )
)
