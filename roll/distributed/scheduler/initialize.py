import os
import subprocess
import sys
import time

import ray

from roll.distributed.scheduler.driver_utils import (
    get_driver_rank,
    get_driver_master_addr,
    get_driver_node_name,
    get_driver_master_port,
    get_driver_world_size,
    get_driver_dashboard_port,
    get_ray_status,
    is_ray_cluster_running,
    wait_for_nodes,
)
from roll.distributed.scheduler.log_monitor import LogMonitorListener
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.logging import get_logger
from roll.platforms import current_platform

logger = get_logger()


def start_ray_cluster():
    rank = get_driver_rank()
    world_size = get_driver_world_size()
    master_addr = get_driver_master_addr()
    master_port = get_driver_master_port()
    node_name = get_driver_node_name()
    dashboard_port = get_driver_dashboard_port()

    if os.getenv("RAY_ADDRESS"):
        logger.info("RAY_ADDRESS is set, skip starting a local Ray cluster")
        return False

    if is_ray_cluster_running():
        logger.info("Ray cluster already initialized")
        return False

    if rank == 0:
        cmd = f"ray start --head --port={master_port} --node-name={node_name} --dashboard-port={dashboard_port}"
    else:
        # fix: 处理大规模下可能会出现的head/worker node创建顺序不一致问题
        time.sleep(5)
        cmd = f"ray start --address={master_addr}:{master_port} --node-name={node_name} --dashboard-port={dashboard_port}"

    logger.info(f"Starting ray cluster: {cmd}")
    ret = subprocess.run(cmd, shell=True, capture_output=True)
    if ret.returncode != 0:
        logger.error(f"Failed to start ray cluster: {cmd}")
        logger.error(f"ret.stdout: {ret.stdout}")
        logger.error(f"ret.stderr: {ret.stderr}")
        sys.exit(1)
    return True


def init():
    rank = get_driver_rank()
    world_size = get_driver_world_size()
    master_addr = get_driver_master_addr()
    master_port = get_driver_master_port()

    manual_start = start_ray_cluster()

    runtime_env_env_vars = current_platform.get_custom_env_vars()
    for env_var in [
        "NVTE_CUDA_INCLUDE_DIR",
        "PYTHONPATH",
        "MOONCAKE_PROTOCOL",
        "MOONCAKE_MASTER",
        "MOONCAKE_TE_META_DATA_SERVER",
        "MOONCAKE_LOCAL_HOSTNAME",
        "MOONCAKE_DEVICE",
        "MOONCAKE_GLOBAL_SEGMENT_SIZE",
        "MOONCAKE_LOCAL_BUFFER_SIZE",
        "ROLL_MOONCAKE_STRICT",
        "ROLL_MOONCAKE_ZEROCOPY_BUFFER_SIZE",
        "ROLL_MOONCAKE_SCATTER_MAX_BYTES",
        "ROLL_MOONCAKE_DISABLE_FAST_PATH",
        "ROLL_MOONCAKE_PROFILE_RANGES",
        "VLLM_USE_V1",
        "CUDA_VISIBLE_DEVICES",
    ]:
        env_value = os.getenv(env_var)
        if env_value:
            runtime_env_env_vars[env_var] = env_value

    runtime_env = {
        "env_vars": runtime_env_env_vars,
    }

    if not ray.is_initialized():
        ray.init(
            address=f"{master_addr}:{master_port}" if manual_start else None,
            namespace=RAY_NAMESPACE,
            ignore_reinit_error=True,
            log_to_driver=not manual_start,
            runtime_env=runtime_env,
        )
        logger.info("Ray cluster initialized")

    if manual_start:
        wait_for_nodes(expected=world_size)
        listener = LogMonitorListener()
        listener.start()

    logger.info(f"Current ray cluster resources: {ray.available_resources()}")

    if manual_start and rank > 0:
        sys.exit(0)
