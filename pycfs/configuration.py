import os
import platform

if platform.system() == "Windows":
    platform_exec = "exe"
else:
    platform_exec = "bin"
import configparser
import ast

import numpy as np
import pandas as pd

from .utils import read_nd
from .geo import d2m


def bool2int(input_bool):
    if input_bool:
        return 1
    else:
        return 0


class CfsConfig(object):
    def __init__(self):
        # set by user
        self.path_input = ""
        self.path_output = ""
        self.earth_model_layer_num = None
        self.processes_num = None
        self.compute_spec = None
        self.check_finished = None
        self.source_depth_range = None
        self.delta_source_depth = None
        self.obs_depth_range = None
        self.delta_obs_depth = None
        self.dist_range = None
        self.delta_dist = None
        self.source_inds = None
        self.obs_inds = None
        self.source_shapes = None
        self.obs_shapes = None
        self.sampling_interval_stf = None
        self.sampling_interval_cfs = None
        self.sampling_num = None
        self.rm_zero_offset = None
        self.slowness_cut = None
        self.max_frequency = None
        self.source_duration = None

        # path
        self.path_nd = ""
        self.path_green_staic = ""
        self.path_green_dynamic = ""
        self.path_output_results = ""
        self.path_bin_edgrn = ""
        self.path_bin_edcmp = ""
        self.path_bin_qssp = ""

        # edgrn2
        self.grn_source_depth_range = None
        self.grn_source_delta_depth = None
        self.grn_dist_range = None
        self.grn_delta_dist = None
        self.obs_depth_list = None
        # the following variables will be set by func self.set_default()
        self.wavenumber_sampling_rate = None

        # edcmp2
        # the following variables will be set by func self.set_default()
        self.obs_x_range = None
        self.obs_y_range = None
        self.obs_delta_x = None
        self.obs_delta_y = None
        self.source_ref = None
        self.obs_ref = None
        self.layered = None
        self.lam = None
        self.mu = None
        self.mu_f_eff = None
        self.mu_f_pore = None
        self.B_pore = None

        # qssp2020
        self.event_depth_list = None
        self.receiver_depth_list = None
        self.time_window = None
        # the following variables will be set by func self.set_default()
        self.time_reduction = None
        self.spec_time_window = None
        self.max_slowness = None
        self.anti_alias = None
        self.source_radius = None
        self.turning_point_filter = None
        self.turning_point_d1 = None
        self.turning_point_d2 = None
        self.free_surface_filter = None
        self.gravity_fc = None
        self.gravity_harmonic = None
        self.cal_sph = None
        self.cal_tor = None
        self.min_harmonic = None
        self.max_harmonic = None
        self.output_observables = None
        self.physical_dispersion = None

        self.default_config = None

    def read_config(self, path_conf):
        config = configparser.ConfigParser()
        config.read(path_conf)
        # [path]
        self.path_input = config["path"]["path_input"]
        self.path_output = config["path"]["path_output"]

        # other path
        self.path_nd = os.path.join(self.path_input, "model.nd")
        self.path_green_staic = os.path.join(self.path_output, "grn_s")
        os.makedirs(self.path_green_staic, exist_ok=True)
        self.path_green_dynamic = os.path.join(self.path_output, "grn_d")
        os.makedirs(self.path_green_dynamic, exist_ok=True)
        self.path_output_results = os.path.join(self.path_output, "results")
        os.makedirs(self.path_output_results, exist_ok=True)
        self.path_bin_edgrn = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "exec",
            "edgrn2.%s" % platform_exec,
        )
        self.path_bin_edcmp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "exec",
            "edcmp2.%s" % platform_exec,
        )
        self.path_bin_qssp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "exec",
            "qssp2020.%s" % platform_exec,
        )

        # [input_addition]
        self.source_inds = ast.literal_eval(
            config["input_addition"]["source_inds"].strip()
        )
        self.obs_inds = ast.literal_eval(config["input_addition"]["obs_inds"].strip())
        self.source_shapes = ast.literal_eval(
            config["input_addition"]["source_shapes"].strip()
        )
        self.obs_shapes = ast.literal_eval(
            config["input_addition"]["obs_shapes"].strip()
        )
        self.earth_model_layer_num = int(
            config["input_addition"]["earth_model_layer_num"]
        )

        # [parallel]
        self.processes_num = int(config["parallel"]["processes_num"])
        self.compute_spec = config["parallel"].getboolean("compute_spec")
        self.check_finished = config["parallel"].getboolean("check_finished")

        # [friction]
        self.mu_f_eff = float(config["friction"]["mu_f_eff"])
        self.mu_f_pore = float(config["friction"]["mu_f_pore"])
        self.B_pore = float(config["friction"]["B_pore"])

        # [region]
        self.source_depth_range = ast.literal_eval(
            config["region"]["source_depth_range"].strip()
        )
        self.delta_source_depth = float(config["region"]["delta_source_depth"])
        self.obs_depth_range = ast.literal_eval(
            config["region"]["obs_depth_range"].strip()
        )
        self.delta_obs_depth = float(config["region"]["delta_obs_depth"])
        self.dist_range = ast.literal_eval(config["region"]["dist_range"].strip())
        self.delta_dist = float(config["region"]["delta_dist"])

        # [time_window]
        self.sampling_interval_stf = float(
            config["time_window"]["sampling_interval_stf"]
        )
        self.sampling_interval_cfs = float(
            config["time_window"]["sampling_interval_cfs"]
        )
        self.sampling_num = int(config["time_window"]["sampling_num"])
        self.time_window = (self.sampling_num - 1) * self.sampling_interval_cfs
        try:
            self.source_duration = float(config["time_window"]["source_duration"])
        except:
            self.source_duration = 0
        try:
            self.max_frequency = float(config["time_window"]["max_frequency"])
        except:
            self.max_frequency = self.sampling_interval_cfs / 2
        try:
            self.rm_zero_offset = config['time_window'].getboolean('rm_zero_offset')
        except:
            self.rm_zero_offset = False
        try:
            self.slowness_cut = config['time_window'].getboolean('slowness_cut')
        except:
            self.slowness_cut = False

        # depth_list from range and delta_dep
        event_depth_list = np.linspace(
            self.source_depth_range[0],
            self.source_depth_range[1],
            round(
                (self.source_depth_range[1] - self.source_depth_range[0])
                / self.delta_source_depth
                + 1
            ),
        ).tolist()
        obs_depth_list = np.linspace(
            self.obs_depth_range[0],
            self.obs_depth_range[1],
            round(
                (self.obs_depth_range[1] - self.obs_depth_range[0])
                / self.delta_obs_depth
                + 1
            ),
        ).tolist()
        # edgrn2
        self.grn_source_depth_range = self.source_depth_range
        self.grn_source_delta_depth = self.delta_source_depth
        self.grn_dist_range = self.dist_range
        self.grn_delta_dist = self.delta_dist
        self.obs_depth_list = obs_depth_list

        # qssp2020
        self.event_depth_list = event_depth_list
        self.receiver_depth_list = obs_depth_list

        if config["default_config"].getboolean("default_config"):
            self.default_config = True
            self.set_default()
        else:
            self.default_config = False

            # edgrn2
            self.wavenumber_sampling_rate = int(
                config["static"]["wavenumber_sampling_rate"]
            )

            # edcmp2
            # the following variables will be set by func self.set_default()
            self.obs_x_range = ast.literal_eval(config["static"]["obs_x_range"].strip())
            self.obs_y_range = ast.literal_eval(config["static"]["obs_y_range"].strip())
            self.obs_delta_x = float(config["static"]["obs_delta_x"])
            self.obs_delta_y = float(config["static"]["obs_delta_y"])
            self.source_ref = ast.literal_eval(config["static"]["source_ref"].strip())
            self.obs_ref = ast.literal_eval(config["static"]["obs_ref"].strip())
            self.layered = config["static"].getboolean("layered")
            self.lam = float(config["static"]["lam"])
            self.mu = float(config["static"]["mu"])

            # qssp2
            self.time_reduction = float(config["dynamic"]["time_reduction"])
            self.spec_time_window = (
                                            int(config["dynamic"]["spec_sampling_num"]) - 1
                                    ) * self.sampling_interval_cfs
            self.max_slowness = float(config["dynamic"]["max_slowness"])
            self.anti_alias = float(config["dynamic"]["anti_alias"])
            self.source_radius = float(config["dynamic"]["source_radius"])
            self.turning_point_filter = bool2int(
                config["dynamic"].getboolean("turning_point_filter")
            )
            self.turning_point_d1 = float(config["dynamic"]["turning_point_d1"])
            self.turning_point_d2 = float(config["dynamic"]["turning_point_d1"])
            self.free_surface_filter = bool2int(
                config["dynamic"].getboolean("free_surface_filter")
            )
            self.gravity_fc = float(config["dynamic"]["gravity_fc"])
            self.gravity_harmonic = int(config["dynamic"]["gravity_harmonic"])
            self.cal_sph = bool2int(config["dynamic"].getboolean("cal_sph"))
            self.cal_tor = bool2int(config["dynamic"].getboolean("cal_tor"))
            self.min_harmonic = int(config["dynamic"]["min_harmonic"])
            self.max_harmonic = int(config["dynamic"]["max_harmonic"])
            self.output_observables = ast.literal_eval(
                config["dynamic"]["output_observables"].strip()
            )
            self.physical_dispersion = bool2int(
                config["dynamic"].getboolean("physical_dispersion")
            )

    def set_default(self):
        # calculate reference point
        # source_csvs = [file for file in os.listdir(os.path.join(self.path_input))
        #                if file.startswith('source') and file.endswith('.csv')]
        # obs_csvs = [file for file in os.listdir(os.path.join(self.path_input))
        #             if file.startswith('source') and file.endswith('.csv')]
        source_points = None
        for ind_src in range(len(self.source_inds)):
            source_plane = pd.read_csv(
                str(
                    os.path.join(
                        self.path_input,
                        "source_plane%d.csv" % self.source_inds[ind_src],
                    )
                ),
                index_col=False,
                header=None,
            ).to_numpy()
            if ind_src == 0:
                source_points = source_plane[:, :3]
            else:
                source_points = np.concatenate(
                    [source_points, source_plane[:, :3]], axis=0
                )

        obs_points = None
        for ind_obs in range(len(self.obs_inds)):
            obs_plane = pd.read_csv(
                str(
                    os.path.join(
                        self.path_input, "obs_plane%d.csv" % self.obs_inds[ind_obs]
                    )
                ),
                index_col=False,
                header=None,
            ).to_numpy()
            if ind_obs == 0:
                obs_points = obs_plane[:, :3]
            else:
                obs_points = np.concatenate([obs_points, obs_plane[:, :3]], axis=0)

        points = np.concatenate([source_points[:, :2], obs_points[:, :2]])
        points[:, 1] = points[:, 1] + 180
        ref_point = np.mean(points, axis=0)
        ref_point[1] = ref_point[1] - 180

        # calculate obs_x_range, obs_y_range
        obs_x_min = float(np.min(obs_points[:, 0]) - self.delta_dist)
        obs_x_max = float(np.max(obs_points[:, 0]) + self.delta_dist)
        obs_y_min = float(np.min(obs_points[:, 1]) - self.delta_dist)
        obs_y_max = float(np.max(obs_points[:, 1]) + self.delta_dist)

        # calculate max_slowness
        nd_model = read_nd(self.path_nd)
        vs_min = np.min(nd_model[nd_model[:, 2] != 0][:, 2])  # km/s
        max_slowness = 1 / vs_min + 0.01  # s/km

        # edgrn2
        self.wavenumber_sampling_rate = 12

        # edcmp2
        self.obs_x_range = [obs_x_min, obs_x_max]
        self.obs_y_range = [obs_y_min, obs_y_max]
        self.obs_delta_x = self.delta_dist / (d2m / 1e3)
        self.obs_delta_y = self.delta_dist / (d2m / 1e3)
        self.source_ref = ref_point.tolist()
        self.obs_ref = ref_point.tolist()
        self.layered = True
        if not self.layered:
            self.lam = 30516224000
            self.mu = 33701888000

        # qssp2020
        self.time_reduction = 0
        self.spec_time_window = self.time_window
        self.max_slowness = max_slowness
        self.anti_alias = 0
        self.source_radius = 0
        self.turning_point_filter = 0
        self.turning_point_d1 = 0
        self.turning_point_d2 = 0
        self.free_surface_filter = 1
        self.gravity_fc = 0
        self.gravity_harmonic = 0
        self.cal_sph = 1
        self.cal_tor = 1
        self.min_harmonic = 6000
        self.max_harmonic = 25000
        self.output_observables = [0 for _ in range(11)]
        self.output_observables[5] = 1
        self.physical_dispersion = 0


if __name__ == "__main__":
    pass
