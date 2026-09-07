from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

# Register BuRd colormap for sequential plots
_clrs = [
    "#2166AC",
    "#4393C3",
    "#92C5DE",
    "#D1E5F0",
    "#F7F7F7",
    "#FDDBC7",
    "#F4A582",
    "#D6604D",
    "#B2182B",
]
_cmap_burd = LinearSegmentedColormap.from_list("BuRd", _clrs)
_cmap_burd.set_bad("#FFEE99")
plt.colormaps.register(_cmap_burd)
