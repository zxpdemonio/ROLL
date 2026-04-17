import argparse
import os
import socket

from dacite import from_dict
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.rlvr.rlvr_vlm_pipeline import RLVRConfig, RLVRVLMPipeline


def _set_default_env(name: str, value: str) -> None:
    if not os.getenv(name):
        os.environ[name] = value


def _resolve_local_hostname() -> str:
    configured_hostname = os.getenv("MOONCAKE_LOCAL_HOSTNAME")
    if configured_hostname:
        return configured_hostname

    host_name = socket.gethostname()
    try:
        return socket.gethostbyname(host_name)
    except OSError:
        return host_name


def _configure_mooncake_rdma_defaults() -> None:
    master_addr = os.getenv("MASTER_ADDR")
    if not master_addr:
        raise ValueError("MASTER_ADDR must be set before using the rollout transfer benchmark launcher")

    _set_default_env("MOONCAKE_PROTOCOL", "rdma")
    _set_default_env("MOONCAKE_MASTER", f"{master_addr}:50051")
    _set_default_env("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE")
    _set_default_env("MOONCAKE_LOCAL_HOSTNAME", _resolve_local_hostname())


def _log_effective_transfer_env(cfg: RLVRConfig) -> None:
    print("=== Rollout transfer benchmark settings ===")
    print(f"backend: {cfg.rollout_transfer_backend}")
    print(f"protocol: {cfg.rollout_transfer_protocol}")
    print(f"num_gpus_per_node: {cfg.num_gpus_per_node}")
    print(f"actor_infer.device_mapping: {cfg.actor_infer.device_mapping}")
    print(f"actor_train.device_mapping: {cfg.actor_train.device_mapping}")
    print(f"reference.device_mapping: {cfg.reference.device_mapping}")
    if cfg.rollout_transfer_backend == "mooncake":
        print(f"MOONCAKE_PROTOCOL: {os.getenv('MOONCAKE_PROTOCOL')}")
        print(f"MOONCAKE_MASTER: {os.getenv('MOONCAKE_MASTER')}")
        print(f"MOONCAKE_LOCAL_HOSTNAME: {os.getenv('MOONCAKE_LOCAL_HOSTNAME')}")
        print(f"MOONCAKE_DEVICE: {os.getenv('MOONCAKE_DEVICE', '')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        help="The path of the main configuration file",
        default="examples/qwen3-vl-4B-rlvr_megatron",
    )
    parser.add_argument(
        "--config_name",
        help="The name of the main configuration file (without extension).",
        default="rlvr_dual_node_transfer_benchmark",
    )
    args, overrides = parser.parse_known_args()

    initialize(config_path=args.config_path, job_name="app")
    cfg = compose(config_name=args.config_name, overrides=overrides)
    benchmark_config: RLVRConfig = from_dict(
        data_class=RLVRConfig,
        data=OmegaConf.to_container(cfg, resolve=True),
    )

    if benchmark_config.rollout_transfer_backend == "mooncake":
        _configure_mooncake_rdma_defaults()

    print(OmegaConf.to_yaml(cfg, resolve=True))
    _log_effective_transfer_env(benchmark_config)

    init()
    pipeline = RLVRVLMPipeline(pipeline_config=benchmark_config)
    pipeline.run()


if __name__ == "__main__":
    main()
