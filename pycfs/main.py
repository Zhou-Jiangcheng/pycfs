import argparse

from .configuration import CfsConfig
from .cfs_dynamic import (
    create_dynamic_lib,
    compute_dynamic_cfs_sequential,
    run_all_dynamic,
)
from .cfs_static import create_static_lib, compute_static_cfs_sequential, run_all_static


def main():
    # Create an argument parser with a description for the CLI tool
    parser = argparse.ArgumentParser(description="pycfs command line tool")

    # Add the --config argument expecting a string for the configuration file path
    # (required for library creation)
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file"
    )

    # static
    parser.add_argument(
        "--create-static-lib", action="store_true", help="Create static stress library"
    )
    parser.add_argument(
        "--compute-static-cfs",
        action="store_true",
        help="Compute static Coulomb Failure Stress change",
    )
    parser.add_argument(
        "--run-static",
        action="store_true",
        help="Create static stress library and "
        "Compute static Coulomb Failure Stress change",
    )

    # dynamic
    parser.add_argument(
        "--create-dynamic-lib",
        action="store_true",
        help="Create dynamic stress library",
    )
    parser.add_argument(
        "--compute-dynamic-cfs",
        action="store_true",
        help="Compute dynamic Coulomb Failure Stress change",
    )
    parser.add_argument(
        "--run-dynamic",
        action="store_true",
        help="Create dynamic stress library and "
        "Compute dynamic Coulomb Failure Stress change",
    )

    # run all
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Create static and dynamic stress library and "
        "Compute static and dynamic Coulomb Failure Stress change",
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Load the configuration using the provided config file path
    print(f"Using configuration file: {args.config}")
    config = CfsConfig()
    config.read_config(args.config)

    if args.create_static_lib:
        create_static_lib(config)
    if args.compute_static_cfs:
        compute_static_cfs_sequential(config)
    if args.run_static:
        run_all_static(config)

    if args.create_dynamic_lib:
        create_dynamic_lib(config)
    if args.compute_dynamic_cfs:
        compute_dynamic_cfs_sequential(config)
    if args.run_dynamic:
        run_all_dynamic(config)

    if args.run_all:
        run_all_static(config)
        run_all_dynamic(config)


if __name__ == "__main__":
    main()
