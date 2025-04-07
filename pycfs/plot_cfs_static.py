import os

import numpy as np
import pandas as pd
from scipy.ndimage import zoom
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LogNorm, BoundaryNorm
from matplotlib.colors import from_levels_and_colors
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize

from pycfs.configuration import CfsConfig
from pycfs.geo import convert_sub_faults_geo2ned

plt.rcParams.update(
    {
        #"font.size": 10,
        "font.family": "Arial",
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)


def plot_cfs_static(
        config: CfsConfig, ind_obs, sub_length_strike_km, sub_length_dip_km
):
    sub_stress = pd.read_csv(
        str(os.path.join(config.path_output,
                         'results',
                         'static',
                         "cfs_static_plane%d.csv" % ind_obs)),
        index_col=False,
        header=None,
    ).to_numpy()
    min_stress = np.min(sub_stress)
    max_stress = np.max(sub_stress)
    max_stress_abs = max(abs(min_stress), abs(max_stress))
    print(min_stress / 1e6, "MPa", max_stress / 1e6, "MPa")
    tick_range = [-max_stress_abs/1e6, max_stress_abs/1e6]
    ind = int(np.argwhere(np.array(config.obs_inds)==ind_obs)[0][0])
    sub_stress: np.ndarray = sub_stress.reshape(
        config.obs_shapes[ind][0], config.obs_shapes[ind][1])
    # sub_stress = zoom(sub_stress, [4, 4], order=3)
    cmap = matplotlib.colormaps["seismic"]
    norm = Normalize(vmin=tick_range[0], vmax=tick_range[1])

    length = 20/2.54
    height = length/0.9*(config.obs_shapes[ind][1]/config.obs_shapes[ind][0])

    fig, ax = plt.subplots(figsize=(length,height))
    X, Y = np.meshgrid(
        np.arange(sub_stress.shape[0]),
        np.arange(sub_stress.shape[1]),
    )
    C = sub_stress / 1e6

    m_plane = ax.pcolormesh(
        X.T,
        Y.T,
        C,
        cmap=cmap,
        norm=norm,
        shading="auto",
    )
    ax.invert_yaxis()
    ax.set_aspect(1)
    cax = fig.add_axes((0.9, 0.2, 0.025, 0.6))
    m = cm.ScalarMappable(cmap=cmap)
    m.set_clim(tick_range[0], tick_range[1])
    cbar = fig.colorbar(m, cax=cax)
    cbar.set_label('Static Coulomb Failure Stress Change (MPa)')

    # ax.set_axis_off()
    # ax.grid(False)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlabel("Along Strike (km)")
    ax.set_ylabel("Along Dip (km)")

    xticks = [x for x in range(round(xlim[1]))]
    xtick_labels = ["%d" % (x * sub_length_strike_km) for x in range(round(xlim[1]))]
    ax.set_xticks(xticks[::5])
    ax.set_xticklabels(xtick_labels[::5])

    yticks = [x for x in range(round(ylim[0]))]
    ytick_labels = ["%d" % (y * sub_length_dip_km) for y in range(round(ylim[0]))]
    ax.set_yticks(yticks[::5])
    ax.set_yticklabels(ytick_labels[::5])

    #ax.text(xlim[0] + 1, ylim[1] + 1, "static", ha="left", va="top", weight="bold")
    fig.subplots_adjust(left=0.05, right=0.85, bottom=0, top=1)
    plt.show()
    return sub_stress


if __name__ == "__main__":
    pass
