import glob
import os
import warnings

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

from pycfs.geo import convert_sub_faults_geo2ned

plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "Arial",
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)


def reshape_sub_faults(sub_faults, num_strike, num_dip):
    mu_strike = sub_faults[num_dip] - sub_faults[0]
    mu_dip = sub_faults[1] - sub_faults[0]
    sub_faults = sub_faults - mu_strike / 2 - mu_dip / 2
    X: np.ndarray = sub_faults[:, 0]
    Y: np.ndarray = sub_faults[:, 1]
    Z: np.ndarray = sub_faults[:, 2]

    X = X.reshape(num_strike, num_dip)
    Y = Y.reshape(num_strike, num_dip)
    Z = Z.reshape(num_strike, num_dip)

    X = np.concatenate([X, np.array([X[:, -1] + mu_dip[0]]).T], axis=1)
    Y = np.concatenate([Y, np.array([Y[:, -1] + mu_dip[1]]).T], axis=1)
    Z = np.concatenate([Z, np.array([Z[:, -1] + mu_dip[2]]).T], axis=1)

    X = np.concatenate([X, np.array([X[-1, :] + mu_strike[0]])], axis=0)
    Y = np.concatenate([Y, np.array([Y[-1, :] + mu_strike[1]])], axis=0)
    Z = np.concatenate([Z, np.array([Z[-1, :] + mu_strike[2]])], axis=0)

    zoom_x = 1
    zoom_y = 1
    order = 4
    mode = "constant"
    X = zoom(X, [zoom_x, zoom_y], order=order, mode=mode)
    Y = zoom(Y, [zoom_x, zoom_y], order=order, mode=mode)
    Z = zoom(Z, [zoom_x, zoom_y], order=order, mode=mode)

    return X, Y, Z, zoom_x, zoom_y


def plot_dynamic_coulomb_stress_one_time_point(
        n_t,
        srate,
        elev,
        azim,
        path_input,
        path_output,
        obs_inds,
        obs_shapes,
        show=False,
):
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(22 / 2.54, 11 / 2.54),
        subplot_kw={"projection": "3d"},
    )
    ax.view_init(elev=elev, azim=azim)
    max_stress_abs = -np.inf
    stress_list = []
    ref = None
    for ind_obs in range(len(obs_inds)):
        print(ind_obs)
        obs_plane = pd.read_csv(
            str(os.path.join(path_input, 'obs_plane%d.csv' % obs_inds[ind_obs])),
            index_col=False, header=None
        ).to_numpy()
        sub_faults = obs_plane[:, :3]
        sub_faults[:, 2] = sub_faults[:, 2] * 1e3
        if ref is None:
            ref = sub_faults[0].tolist()
        sub_faults = convert_sub_faults_geo2ned(
            sub_faults=sub_faults, source_point=ref
        )

        # sub_faults[:, 2] = sub_faults[:, 2] + source1[2]

        X, Y, Z, zoom_x, zoom_y = reshape_sub_faults(
            sub_faults=sub_faults,
            num_strike=obs_shapes[ind_obs][0],
            num_dip=obs_shapes[ind_obs][1]
        )
        X = X / 1e3
        Y = Y / 1e3
        Z = Z / 1e3

        sub_stress = pd.read_csv(
            str(os.path.join(path_output,
                             'results',
                             'dynamic',
                             "cfs_dynamic_plane%d.csv" % obs_inds[ind_obs])),
            index_col=False, header=None
        ).to_numpy()
        print(np.min(sub_stress) / 1e6, "MPa", np.max(sub_stress) / 1e6, "MPa")
        sub_stress = sub_stress[:, n_t].flatten()
        sub_stress: np.ndarray = sub_stress.reshape(
            obs_shapes[ind_obs][0], obs_shapes[ind_obs][1])
        sub_stress = np.concatenate(
            [
                sub_stress,
                np.array([np.ones_like(sub_stress[:, -1])]).T,
            ],
            axis=1,
        )
        sub_stress = np.concatenate(
            [
                sub_stress,
                np.array([np.ones_like(sub_stress[-1, :])]),
            ],
            axis=0,
        )

        sub_stress = zoom(sub_stress, [zoom_x, zoom_y], order=3)
        stress_list.append([X, Y, Z, sub_stress])
        max_stress_abs_ind = np.max(abs(sub_stress))
        if max_stress_abs_ind > max_stress_abs:
            max_stress_abs = np.max(abs(sub_stress))

    tick_range = [-max_stress_abs / 1e6, max_stress_abs / 1e6]
    cmap = matplotlib.colormaps["seismic"]
    norm = Normalize(vmin=tick_range[0], vmax=tick_range[1])

    for ind_obs in range(len(obs_inds)):
        # print(
        #     "sub_stress_%d %d min:%f max:%f"
        #     % (
        #         obs_inds[ind_obs],
        #         n_t,
        #         float(np.min(stress_list[ind_obs][-1]) / 1e6),
        #         float(np.max(stress_list[ind_obs][-1]) / 1e6),
        #     )
        # )
        m_plane = ax.plot_surface(
            stress_list[ind_obs][1],
            stress_list[ind_obs][0],
            -stress_list[ind_obs][2],
            rstride=1,
            cstride=1,
            facecolors=cmap(norm(stress_list[ind_obs][-1] / 1e6)),
            antialiased=False,
            shade=False,
        )

    cax = inset_axes(
        ax,
        width="5%",
        height="75%",
        loc="upper right",
        bbox_to_anchor=(0.2, -0.05, 1, 1),
        bbox_transform=ax.transAxes,
    )
    m = cm.ScalarMappable(cmap=cmap)
    m.set_clim(tick_range[0], tick_range[1])
    cbar = fig.colorbar(m, cax=cax)
    # cbar.set_ticks(np.linspace(tick_range[0], tick_range[1], 7))
    # tick_labels = ['%.2f' % tick for tick in levels]
    # cbar.set_ticklabels(tick_labels)
    cbar.set_label("Coulomb Failure Stress Change (MPa)")

    # ax.set_xlim(-115000, 173000)
    # ax.set_ylim(-114000, 97000)
    ax.set_box_aspect([1, 1, 0.2])
    # ax.set_axis_off()
    # ax.grid(False)
    ax.set_xlabel("W-E (km)")
    ax.set_ylabel("S-N (km)")
    ax.set_zlabel("D-U (km)")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    title = "Time: %.0f" % float(n_t / srate)
    fig.suptitle(title)
    # plt.savefig(os.path.join(path_output, "Time_%.1f.pdf" % float(n_t / srate)))
    # plt.savefig(os.path.join(path_output, "Time_%.1f.png" % float(n_t / srate)))
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    plot_dynamic_coulomb_stress_one_time_point(
        n_t=35,
        srate=1,
        elev=30,
        azim=-70,
        path_input='/home/zjc/e/layercfs/t_py/input',
        path_output='/home/zjc/e/layercfs/t_mat/output',
        obs_inds=[6, 7, 8],
        obs_shapes=[[12,16], [12,16], [20,16]],
        show=True,
    )

    # plot_dynamic_coulomb_stress_one_time_point(
    #     n_t=35,
    #     srate=1,
    #     elev=30,
    #     azim=-70,
    #     path_input='/home/zjc/e/layercfs/t_mat/input',
    #     path_output='/home/zjc/e/layercfs/t_mat/output',
    #     obs_inds=[9],
    #     obs_shapes=[[46,16]],
    #     show=True,
    # )