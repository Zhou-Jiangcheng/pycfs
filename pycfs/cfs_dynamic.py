import datetime
import os
import json

import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

from .configuration import CfsConfig
from .create_qssp_bulk import (
    pre_process_qssp2020,
    create_grnlib_qssp2020_parallel_single_node,
    convert_pd2bin_qssp2020_all,
    remove_dat_files,
)
from .read_qssp import seek_qssp2020
from .focal_mechanism import plane2nd
from .obspy_geo import gps2dist_azimuth
from .signal_process import resample
from .cfs_static import (
    cal_coulomb_failure_stress,
    cal_coulomb_failure_stress_poroelasticity,
)
from .utils import read_source_array

d2km = 111.19492664455874


def cal_stress_vector_ned_dynamic(stress_enz, n):
    sigma11, sigma12, sigma13, sigma22, sigma23, sigma33 = stress_enz.T
    stress_tensor_ned = np.array(
        [
            [sigma22, sigma12, -sigma23],
            [sigma12, sigma11, -sigma13],
            [-sigma23, -sigma13, sigma33],
        ]
    ).T  # Shape will be (n, 3, 3)
    sigma_vector = np.einsum("ijk,k->ij", stress_tensor_ned, n.flatten())
    return sigma_vector


def cal_cfs_dynamic_single_point(
    path_green,
    source_array,
    obs_array_single_point,
    srate_stf,
    mu_f_eff=0.4,
    mu_f_pore=0.6,
    B_pore=0.75,
    green_info=None,
    rm_zero_offset=False,
    slowness_cut=True,
):
    """
    :param path_green: Root directory of Green's function library created
                       by create_grnlib_qssp2020_*.
    :param source_array: 2D numpy array, each line contains
                        [lat(deg), lon(deg), depth(km), strike(deg), dip(deg), rake(deg),
                        length_strike(km), length_dip(km), slip(m), m0(Nm),
                        stf_array(dimensionless)]
                        The stf at the end will be normalized by m0.
    :param obs_array_single_point: 1D numpy array,
                        [lat(deg), lon(deg), depth(km),
                        strike(deg), dip(deg), rake(deg)]
    :param srate_stf: Sampling rate of stf in Hz.
    :param mu_f_eff: Effective coefficient of friction.
    :param mu_f_pore: Coefficient of friction.
    :param B_pore: Skempton's coefficient.
    :param green_info:
    :param rm_zero_offset: Should only be used if green_info['T_rise'] is 0
    :param slowness_cut: Should only be used if green_info['T_rise'] is 0

    Note: The sampling interval for all return values is the same as green'info ['sampling_interval ']
    :return: (
        stress_enz,
        sigma_vector,
        sigma,
        tau,
        mean_stress,
        cfs,
        cfs_pore
    )
    """
    if green_info is None:
        with open(os.path.join(path_green, "green_lib_info.json"), "r") as fr:
            green_info = json.load(fr)
    sampling_num = (
        round(green_info["time_window"] / green_info["sampling_interval"]) + 1
    )
    max_slowness = green_info["max_slowness"]
    model_name = green_info["path_nd_without_Q"]
    srate_cfs = 1 / green_info["sampling_interval"]
    # srate_cfs = srate_stf
    # min_slowness = 1 / 10
    average_length = 10

    # lat lon dep len_strike len_dip slip m0 strike dip rake stf
    stress_enz = np.zeros((sampling_num, 6))
    sub_faults_source = source_array[:, :3]
    sub_fms = source_array[:, 3:6]
    sub_m0s = source_array[:, 9]
    sub_stfs = source_array[:, 10:]
    for i in range(sub_faults_source.shape[0]):
        sub_stf = resample(
            sub_stfs[i],
            srate_old=srate_stf,
            srate_new=srate_cfs,
            zero_phase=True,
        )
        sub_stf = sub_stf / (np.sum(sub_stf) / srate_cfs) * sub_m0s[i]
        dist_in_m, az_in_deg, baz_in_deg = gps2dist_azimuth(
            lat1=sub_faults_source[i][0],
            lon1=sub_faults_source[i][1],
            lat2=obs_array_single_point[0],
            lon2=obs_array_single_point[1],
        )
        focal_mechanism = sub_fms[i]
        (
            stress_enz_1source,
            tpts_table,
            first_p,
            first_s,
            grn_dep,
            grn_receiver,
            green_dist,
        ) = seek_qssp2020(
            path_green,
            sub_faults_source[i,2],
            obs_array_single_point[2],
            az_in_deg,
            dist_in_m / 1e3,
            focal_mechanism,
            srate_cfs,
            before_p=None,
            pad_zeros=True,
            shift=False,
            rotate=True,
            only_seismograms=False,
            output_type="stress",
            model_name=model_name,
            green_info=green_info,
        )
        stress_enz_1source = stress_enz_1source.T
        if rm_zero_offset:
            # stf_T_rise = create_stf(4*green_info['sampling_interval'], srate_cfs)
            # stress_enz_1source = signal.convolve(
            #     stress_enz_1source, stf_T_rise[:, None], mode="same"
            # )
            taps = round(2 ** (np.floor(np.log2(len(stress_enz_1source) // 8))) + 1)
            lowpass = signal.firwin(
                numtaps=taps,
                cutoff=min(green_info["max_frequency"]*1.25, srate_cfs/2)/ srate_cfs,
                window="hamming"
            )
            stress_enz_1source = signal.filtfilt(
                lowpass, 1.0, stress_enz_1source, axis=0
            )

            start_idx = 0
            end_idx = round(np.floor(tpts_table["p_onset"] * srate_cfs))
            if end_idx !=0:
                value_before_p = np.mean(stress_enz_1source[start_idx:end_idx, :], axis=0)
                stress_enz_1source = stress_enz_1source - value_before_p[None, :]
                stress_enz_1source[:end_idx, :] = 0


        conv_result = signal.convolve(stress_enz_1source, sub_stf[:, None], mode="full")
        stress_enz_1source = conv_result[:sampling_num, :] / srate_cfs

        if slowness_cut:
            # Compute the index after which the constant averaging is applied.
            t_max_slowness = green_dist * max_slowness
            ind_const = round((t_max_slowness + 1) * srate_cfs + len(sub_stf))
            # If there are enough samples, replace values from ind_const onward with the average
            # computed over a window of length average_length for each column.
            if ind_const + average_length < sampling_num:
                avg_val = np.mean(
                    stress_enz_1source[ind_const : ind_const + average_length, :],
                    axis=0,
                )
                stress_enz_1source[ind_const:, :] = avg_val[None, :]

        stress_enz = stress_enz + stress_enz_1source

    n_obs, d_obs = plane2nd(*obs_array_single_point[3:])
    n = np.array([n_obs.flatten()]).T
    d = np.array([d_obs.flatten()]).T
    sigma_vector = cal_stress_vector_ned_dynamic(stress_enz, n)  # ned
    sigma = np.dot(sigma_vector, np.array([n]).T).flatten()
    tau = np.dot(sigma_vector, np.array([d]).T).flatten()
    mean_stress = (stress_enz[:, 0] + stress_enz[:, 3] + stress_enz[:, 5]) / 3
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

    return (
        stress_enz,
        sigma_vector,
        sigma,
        tau,
        mean_stress,
        cfs,
        cfs_pore,
    )


def create_dynamic_lib(config: CfsConfig):
    s = datetime.datetime.now()
    pre_process_qssp2020(
        processes_num=config.processes_num,
        path_green=config.path_green_dynamic,
        path_bin=config.path_bin_qssp,
        event_depth_list=config.event_depth_list,
        receiver_depth_list=config.receiver_depth_list,
        spec_time_window=config.spec_time_window,
        sampling_interval=config.sampling_interval_cfs,
        max_frequency=config.max_frequency,
        max_slowness=config.max_slowness,
        anti_alias=config.anti_alias,
        turning_point_filter=config.turning_point_filter,
        turning_point_d1=config.turning_point_d1,
        turning_point_d2=config.turning_point_d2,
        free_surface_filter=config.free_surface_filter,
        gravity_fc=config.gravity_fc,
        gravity_harmonic=config.gravity_harmonic,
        cal_sph=config.cal_sph,
        cal_tor=config.cal_tor,
        min_harmonic=config.min_harmonic,
        max_harmonic=config.max_harmonic,
        source_radius=config.source_radius,
        source_duration=config.source_duration,
        output_observables=config.output_observables,
        time_window=config.time_window,
        time_reduction=config.time_reduction,
        dist_range=config.dist_range,
        delta_dist=config.delta_dist,
        path_nd=config.path_nd,
        earth_model_layer_num=config.earth_model_layer_num,
        physical_dispersion=config.physical_dispersion,
        check_finished_tpts_table=config.check_finished,
    )
    create_grnlib_qssp2020_parallel_single_node(
        path_green=config.path_green_dynamic,
        cal_spec=config.compute_spec,
        check_finished=config.check_finished,
    )
    if not config.check_finished:
        convert_pd2bin_qssp2020_all(config.path_green_dynamic)
        remove_dat_files(config.path_green_dynamic)
    e = datetime.datetime.now()
    print("runtime:", e - s)


result_name_list = [
    "stress_enz",
    "sigma_vector",
    "sigma",
    "tau",
    "mean_stress",
    "cfs",
    "cfs_pore",
]


def read_stress_results(obs_plane, ind_obs, path_output, sampling_num):
    print("Outputting results for No.%d obs_plane to csv files" % ind_obs)
    path_results_each = os.path.join(path_output, "grn_d", "results_each")
    path_output_results = os.path.join(path_output, "results", "dynamic")
    os.makedirs(path_output_results, exist_ok=True)
    cfs_plane = np.zeros((len(obs_plane), sampling_num))
    cfs_pore_plane = np.zeros((len(obs_plane), sampling_num))
    normal_stress_plane = np.zeros((len(obs_plane), sampling_num))
    shear_stress_plane = np.zeros((len(obs_plane), sampling_num))
    for i in range(len(obs_plane)):
        lat, lon, dep = list(obs_plane[i, :3])
        fname = "%.4f_%.4f_%.4f" % (lat, lon, dep)
        # mdict = loadmat(os.path.join(path_results_each, fname + '.mat'))
        # cfs = mdict['cfs']
        # cfs_pore = mdict['cfs_pore']
        # sigma = mdict['normal_stress']
        # tau = mdict['shear_stress']

        cfs = pd.read_csv(
            str(os.path.join(path_results_each, fname + "_cfs.csv")),
            index_col=False,
            header=None,
        ).to_numpy()
        cfs_pore = pd.read_csv(
            str(os.path.join(path_results_each, fname + "_cfs_pore.csv")),
            index_col=False,
            header=None,
        ).to_numpy()
        sigma = pd.read_csv(
            str(os.path.join(path_results_each, fname + "_sigma.csv")),
            index_col=False,
            header=None,
        ).to_numpy()
        tau = pd.read_csv(
            str(os.path.join(path_results_each, fname + "_tau.csv")),
            index_col=False,
            header=None,
        ).to_numpy()

        cfs_plane[i] = cfs.flatten()
        cfs_pore_plane[i] = cfs_pore.flatten()
        normal_stress_plane[i] = sigma.flatten()
        shear_stress_plane[i] = tau.flatten()

        df_cfs_plane = pd.DataFrame(cfs_plane)
        df_cfs_plane.to_csv(
            str(os.path.join(path_output_results, "cfs_dynamic_plane%d.csv" % ind_obs)),
            header=False,
            index=False,
        )
        df_cfs_pore_plane = pd.DataFrame(cfs_plane)
        df_cfs_pore_plane.to_csv(
            str(
                os.path.join(
                    path_output_results, "cfs_pore_dynamic_plane%d.csv" % ind_obs
                )
            ),
            header=False,
            index=False,
        )
        df_normal_stress_plane = pd.DataFrame(normal_stress_plane)
        df_normal_stress_plane.to_csv(
            str(
                os.path.join(
                    path_output_results, "normal_stress_dynamic_plane%d.csv" % ind_obs
                )
            ),
            header=False,
            index=False,
        )
        df_shear_stress_plane = pd.DataFrame(shear_stress_plane)
        df_shear_stress_plane.to_csv(
            str(
                os.path.join(
                    path_output_results, "shear_stress_dynamic_plane%d.csv" % ind_obs
                )
            ),
            header=False,
            index=False,
        )


def compute_dynamic_cfs_sequential(config: CfsConfig):
    s = datetime.datetime.now()
    source_array = read_source_array(
        source_inds=config.source_inds,
        path_input=config.path_input,
    )

    with open(
        os.path.join(config.path_green_dynamic, "green_lib_info.json"), "r"
    ) as fr:
        green_info = json.load(fr)
    path_results_each = os.path.join(config.path_output, "grn_d", "results_each")
    os.makedirs(path_results_each, exist_ok=True)
    for ind_obs in config.obs_inds:
        obs_plane = pd.read_csv(
            str(os.path.join(config.path_input, "obs_plane%d.csv" % ind_obs)),
            index_col=False,
            header=None,
        ).to_numpy()
        for i in tqdm(
            range(len(obs_plane)),
            desc="Computing dynamic Coulomb Failure Stress change at No.%d plane"
            % ind_obs,
        ):
            results = cal_cfs_dynamic_single_point(
                path_green=config.path_green_dynamic,
                source_array=source_array,
                obs_array_single_point=obs_plane[i, :],
                srate_stf=1 / config.sampling_interval_stf,
                mu_f_eff=config.mu_f_eff,
                mu_f_pore=config.mu_f_pore,
                B_pore=config.B_pore,
                green_info=green_info,
                rm_zero_offset=config.rm_zero_offset,
                slowness_cut=config.slowness_cut,
            )
            file_name = "%.4f_%.4f_%.4f" % (
                float(obs_plane[i, 0]),
                float(obs_plane[i, 1]),
                float(obs_plane[i, 2]),
            )
            for j in range(len(result_name_list)):
                df_each = pd.DataFrame(results[j])
                df_each.to_csv(
                    str(
                        os.path.join(
                            path_results_each,
                            file_name + "_%s.csv" % result_name_list[j],
                        )
                    ),
                    index=False,
                    header=False,
                )
        read_stress_results(
            obs_plane=obs_plane,
            ind_obs=ind_obs,
            path_output=config.path_output,
            sampling_num=config.sampling_num,
        )
    e = datetime.datetime.now()
    print("run time:", e - s)


def run_all_dynamic(config: CfsConfig):
    create_dynamic_lib(config)
    compute_dynamic_cfs_sequential(config)


if __name__ == "__main__":
    pass
