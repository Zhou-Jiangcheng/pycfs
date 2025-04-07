import datetime
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from .configuration import CfsConfig
from .focal_mechanism import plane2nd
from .create_edgrn_bulk import (
    pre_process_edgrn2,
    create_grnlib_edgrn2_parallel_single_node,
)
from .create_edcmp_bulk import (
    pre_process_edcmp2,
    compute_static_stress_edcmp2_sequential,
    compute_static_stress_edcmp2_parallel_single_node,
)
from .read_edcmp import seek_edcmp2
from .utils import read_source_array


def cal_coulomb_failure_stress(
        norm_stress_drop,
        shear_stress_drop,
        mu_f_eff=0.4,
):
    """
    :param norm_stress_drop:
    :param shear_stress_drop:
    :param mu_f_eff: effective coefficient of friction
    :return:
    """
    coulomb_stress = shear_stress_drop + mu_f_eff * norm_stress_drop
    return coulomb_stress


def cal_coulomb_failure_stress_poroelasticity(
        norm_stress_drop,
        shear_stress_drop,
        mean_stress_drop,
        mu_f_pore=0.6,
        B_pore=0.75,
):
    """

    :param norm_stress_drop:
    :param shear_stress_drop:
    :param mean_stress_drop:
    :param mu_f_pore: coefficient of friction
    :param B_pore: Skempton's coefficient
    :return:
    """
    coulomb_stress = shear_stress_drop + mu_f_pore * (
            norm_stress_drop + B_pore * mean_stress_drop
    )
    return coulomb_stress


def cal_cfs_static_single_point(
        focal_mechanism, stress, mu_f_eff=0.4, mu_f_pore=0.6, B_pore=0.75
):
    n, d = plane2nd(*focal_mechanism)
    n = np.array([n.flatten()]).T
    d = np.array([d.flatten()]).T
    sigma_tensor = np.array(
        [
            [stress[0], stress[3], stress[5]],
            [stress[3], stress[1], stress[4]],
            [stress[5], stress[4], stress[2]],
        ]
    )
    sigma_vector = np.dot(sigma_tensor, n)
    sigma = np.dot(sigma_vector.T, n)[0][0]
    # tau = np.linalg.norm(sigma_vector - sigma * n)
    tau = np.dot(sigma_vector.T, d)[0][0]
    mean_stress = np.sum(stress[:3]) / 3
    cfs = cal_coulomb_failure_stress(
        norm_stress_drop=sigma, shear_stress_drop=tau, mu_f_eff=mu_f_eff
    )
    cfs_pore = cal_coulomb_failure_stress_poroelasticity(
        norm_stress_drop=sigma,
        shear_stress_drop=tau,
        mean_stress_drop=mean_stress,
        mu_f_pore=mu_f_pore,
        B_pore=B_pore,
    )
    return sigma_vector, sigma, tau, mean_stress, cfs, cfs_pore


def create_static_lib(config: CfsConfig):
    s = datetime.datetime.now()
    print("Preprocessing")
    pre_process_edgrn2(
        processes_num=config.processes_num,
        path_green=config.path_green_staic,
        path_bin=config.path_bin_edgrn,
        grn_source_depth_range=config.grn_source_depth_range,
        grn_source_delta_depth=config.grn_source_delta_depth,
        grn_dist_range=config.grn_dist_range,
        grn_delta_dist=config.grn_delta_dist,
        obs_depth_list=config.obs_depth_list,
        wavenumber_sampling_rate=config.wavenumber_sampling_rate,
        path_nd=config.path_nd,
        earth_model_layer_num=config.earth_model_layer_num,
    )
    create_grnlib_edgrn2_parallel_single_node(
        path_green=config.path_green_staic, check_finished=config.check_finished
    )
    e = datetime.datetime.now()
    print("run time:", e - s)


def compute_static_cfs_sequential(config: CfsConfig):
    s = datetime.datetime.now()

    source_array = read_source_array(
        source_inds=config.source_inds,
        path_input=config.path_input,
        shift2corner=True,
        source_shapes=config.source_shapes
    )
    source_array_new = np.zeros((len(source_array), 9))
    source_array_new[:, 0] = source_array[:, 8]
    source_array_new[:, 1:9] = source_array[:, 0:8]
    pre_process_edcmp2(
        processes_num=config.processes_num,
        path_green=config.path_green_staic,
        path_bin=config.path_bin_edcmp,
        obs_depth_list=config.obs_depth_list,
        obs_x_range=config.obs_x_range,
        obs_y_range=config.obs_y_range,
        obs_delta_x=config.obs_delta_x,
        obs_delta_y=config.obs_delta_y,
        source_array=source_array_new,
        source_ref=config.source_ref,
        obs_ref=config.obs_ref,
        layered=config.layered,
        lam=config.lam,
        mu=config.mu,
    )
    if config.processes_num == 1:
        compute_static_stress_edcmp2_sequential(
            path_green=config.path_green_staic, check_finished=False)
    elif config.processes_num>1:
        compute_static_stress_edcmp2_parallel_single_node(
            path_green=config.path_green_staic, check_finished=False
        )

    path_output_results = os.path.join(config.path_output, "results", "static")
    os.makedirs(path_output_results, exist_ok=True)

    for ind_obs in config.obs_inds:
        obs_plane = pd.read_csv(
            str(os.path.join(config.path_input, "obs_plane%d.csv" % ind_obs)),
            index_col=False,
            header=None,
        ).to_numpy()
        N = len(obs_plane)
        stress_tensor_array = seek_edcmp2(
            str(os.path.join(config.path_output, "grn_s")),
            "stress",
            obs_plane[:, :3],
            geo_coordinate=True,
        )

        stress_vector_array = np.zeros((N, 3))
        norm_stress_array = np.zeros(N)
        shear_stress_array = np.zeros(N)
        mean_stress_array = np.zeros(N)
        cfs_array = np.zeros(N)
        cfs_pore_array = np.zeros(N)

        for i in tqdm(
                range(N),
                desc="Computing static Coulomb Failure Stress change at No.%d plane"
                     % ind_obs,
        ):
            (sigma_vector, sigma, tau, mean_stress, cfs, cfs_pore) = (
                cal_cfs_static_single_point(
                    obs_plane[i, 3:],
                    stress_tensor_array[i, :],
                    config.mu_f_eff,
                    config.mu_f_pore,
                    config.B_pore,
                )
            )
            stress_vector_array[i] = sigma_vector.flatten()  # ned
            norm_stress_array[i] = sigma
            shear_stress_array[i] = tau
            mean_stress_array[i] = mean_stress
            cfs_array[i] = cfs
            cfs_pore_array[i] = cfs_pore

            df_cfs_plane = pd.DataFrame(cfs_array)
            df_cfs_plane.to_csv(
                str(
                    os.path.join(
                        path_output_results, "cfs_static_plane%d.csv" % ind_obs
                    )
                ),
                header=False,
                index=False,
            )
            df_cfs_pore_plane = pd.DataFrame(cfs_pore_array)
            df_cfs_pore_plane.to_csv(
                str(
                    os.path.join(
                        path_output_results, "cfs_pore_static_plane%d.csv" % ind_obs
                    )
                ),
                header=False,
                index=False,
            )
            df_normal_stress_plane = pd.DataFrame(norm_stress_array)
            df_normal_stress_plane.to_csv(
                str(
                    os.path.join(
                        path_output_results,
                        "normal_stress_static_plane%d.csv" % ind_obs,
                    )
                ),
                header=False,
                index=False,
            )
            df_shear_stress_plane = pd.DataFrame(shear_stress_array)
            df_shear_stress_plane.to_csv(
                str(
                    os.path.join(
                        path_output_results, "shear_stress_static_plane%d.csv" % ind_obs
                    )
                ),
                header=False,
                index=False,
            )
    e = datetime.datetime.now()
    print("runtime:", e - s)


def run_all_static(config: CfsConfig):
    create_static_lib(config)
    compute_static_cfs_sequential(config)


if __name__ == "__main__":
    pass
