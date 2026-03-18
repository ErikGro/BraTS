from brats.utils.algorithm_config import AlgorithmData
from pathlib import Path

import subprocess
from typing import Dict, List, Optional, Union
from brats.core.docker import (
    _log_algorithm_info,
    _sanity_check_output,
    _get_additional_files_path,
    _get_volume_mappings_mlcube,
    _get_parameters_arg,
    _handle_device_requests,
    _get_volume_mappings_docker_only,
    _ensure_image as _ensure_docker_image,
)
from brats.constants import PARAMETERS_DIR
from loguru import logger
import time
from spython.main import Client
import docker
import tempfile
import os
import urllib.request
import json

try:
    docker_client = docker.from_env()
except docker.errors.DockerException as e:
    logger.debug(
        "Could not connect to the Docker daemon. Docker functionality is disabled, so the Singularity container's working directory may not be set correctly."
    )
    docker_client = None


def _build_command_args(
    algorithm: AlgorithmData,
) -> List[str]:
    """Build the command arguments for the singularity container.

    Args:
        algorithm (AlgorithmData): The algorithm data

    Returns:
        List[str]: The command arguments
    """

    command_args = ["--data_path=/mlcube_io0", "--output_path=/mlcube_io2"]
    if algorithm.additional_files is not None:
        for i, param in enumerate(algorithm.additional_files.param_name):
            additional_files_arg = f"--{param}=/mlcube_io1"
            if algorithm.additional_files.param_path:
                additional_files_arg += f"/{algorithm.additional_files.param_path[i]}"
            command_args.append(additional_files_arg)

    # Add parameters file arg if required
    params_arg = _get_parameters_arg(algorithm=algorithm)
    if params_arg:
        command_args.append(params_arg.strip())

    return command_args


def _ensure_image(image: str) -> str:
    """
    Ensure the Singularity image is present on the system. If not, pull it as a Sandbox.
    This function checks if the specified Singularity image exists locally in the temporary directory.
    If the image is not found, it pulls the image from Docker Hub, creates a Singularity Sandbox at the target location.

    Args:
        image (str): The Docker image to pull and convert into a Singularity Sandbox.

    Returns:
        str: The path to the Singularity image Sandbox.
    """
    persistent_dir = os.path.join(tempfile.gettempdir(), "brats_singularity_images")
    os.makedirs(persistent_dir, exist_ok=True)
    logger.debug(f"Persistent folder: {persistent_dir}")
    temp_folder = Path(persistent_dir)
    image_path = temp_folder.joinpath(image.replace(":", "_"))
    if not image_path.exists():
        image_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(
            f"Pulling Singularity image {image} and creating a Sandbox at {image_path}"
        )
        subprocess.run(
            [
                "singularity",
                "build",
                "--sandbox",
                "--fakeroot",
                str(image_path),
                f"docker://{image}",
            ],
            check=True,
        )

    return str(image_path)


def _convert_volume_mappings_to_singularity_format(
    volume_mappings: Dict[Union[Path, str], Dict[str, str]],
) -> List[str]:
    """Convert volume mappings from Docker format to Singularity format.

    Args:
        volume_mappings (Dict[Path | str, Dict[str, str]]): The volume mappings in Docker format

    Returns:
        List[str]: The volume mappings in Singularity format
    """
    singularity_bindings = []
    for host_path, val in volume_mappings.items():
        container_path = val["bind"]
        singularity_bindings.append(f"{str(host_path)}:{container_path}")
    return singularity_bindings

def _get_docker_working_dir(image: str) -> Optional[Path]:
    """
    Fetch the WorkingDir directly from the Docker Hub API.
    This bypasses the need for a local Docker Daemon, making it HPC-friendly.
    """
    logger.debug(f"Fetching OCI config for {image} from Docker Hub API...")
    try:
        # Parse image string (e.g., "brainles/brats25_inpainting_ying_weng:latest")
        if ":" in image:
            repo, tag = image.split(":")
        else:
            repo, tag = image, "latest"
            
        # Add implicit 'library/' for official images if missing
        if "/" not in repo:
            repo = f"library/{repo}"

        # 1. Get anonymous bearer token
        token_url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
        req = urllib.request.Request(token_url)
        with urllib.request.urlopen(req) as resp:
            token = json.loads(resp.read().decode())["token"]

        # 2. Setup headers to accept both single and multi-arch manifests
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.index.v1+json"
        }
        
        # 3. Fetch manifest
        manifest_url = f"https://registry-1.docker.io/v2/{repo}/manifests/{tag}"
        req = urllib.request.Request(manifest_url, headers=headers)
        with urllib.request.urlopen(req) as resp:
            manifest = json.loads(resp.read().decode())

        # If it's a multi-arch index, grab the first manifest's digest
        if "manifests" in manifest:
            digest = manifest["manifests"][0]["digest"]
            req = urllib.request.Request(f"https://registry-1.docker.io/v2/{repo}/manifests/{digest}", headers=headers)
            with urllib.request.urlopen(req) as resp:
                manifest = json.loads(resp.read().decode())

        # 4. Fetch the actual config blob
        config_digest = manifest["config"]["digest"]
        req = urllib.request.Request(f"https://registry-1.docker.io/v2/{repo}/blobs/{config_digest}", headers=headers)
        with urllib.request.urlopen(req) as resp:
            config_blob = json.loads(resp.read().decode())
            
        # Extract WorkingDir
        workdir = config_blob.get("config", {}).get("WorkingDir")
        
        if workdir:
            logger.debug(f"Found WorkingDir via API: {workdir}")
            return Path(workdir)
        else:
            return None

    except Exception as e:
        logger.warning(f"Failed to fetch WORKDIR from registry: {e}. Falling back to default.")
        return None


def run_container(
    algorithm: AlgorithmData,
    data_path: Path,
    output_path: Path,
    cuda_devices: str,
    force_cpu: bool,
    internal_external_name_map: Optional[Dict[str, str]] = None,
    overlay_size: int = 1024,
):
    """Run a Singularity container for the provided algorithm.

    Args:
        algorithm (AlgorithmData): The data of the algorithm to run
        data_path (Path | str): The path to the input data
        output_path (Path | str): The path to save the output
        cuda_devices (str): The CUDA devices to use
        force_cpu (bool): Whether to force CPU execution
        internal_external_name_map (Dict[str, str]): Dictionary mapping internal name (in standardized format) to external subject name provided by user (only used for batch inference)
        overlay_size (int): The size of the overlay image in MB. Defaults to 1024.
    """
    if overlay_size <= 0:
        raise ValueError("Overlay size must be greater than 0.")
    _log_algorithm_info(algorithm=algorithm)
    # ensure image is present, if not pull it
    image = _ensure_image(image=algorithm.run_args.docker_image)

    additional_files_path = _get_additional_files_path(algorithm)
    logger.debug(f"Additional files path: {additional_files_path}")
    # ensure output folder exists
    output_path.mkdir(parents=True, exist_ok=True)

    command_args = _build_command_args(algorithm=algorithm)
    command_args_str = " ".join(command_args)
    logger.debug(f"Command args: {command_args_str}")
    if algorithm.meta.year <= 2024:
        volume_mappings = _get_volume_mappings_mlcube(
            data_path=data_path,
            additional_files_path=additional_files_path,
            output_path=output_path,
            parameters_path=PARAMETERS_DIR,
        )
        args = ["infer", *command_args]
    else:
        volume_mappings = _get_volume_mappings_docker_only(
            data_path=data_path,
            output_path=output_path,
        )
        args = None

    logger.debug(f"Volume mappings: {volume_mappings}")

    # device setup
    device_requests = _handle_device_requests(
        algorithm=algorithm, cuda_devices=cuda_devices, force_cpu=force_cpu
    )
    logger.debug(f"GPU Device requests: {device_requests}")

    # Run the container
    logger.info("Starting inference")
    start_time = time.time()

    singularity_bindings = _convert_volume_mappings_to_singularity_format(
        volume_mappings
    )

    #options = []
    options = ["--no-home"]
    if len(device_requests) > 0 and not force_cpu:
        logger.info(f"Using CUDA devices: {cuda_devices}")
        options.append("--nv")  # Singularity uses --nv to enable GPU support

    # TODO: The --fakeroot option may be required for certain algorithms that need root privileges inside the Singularity container.
    docker_working_dir = _get_docker_working_dir(algorithm.run_args.docker_image)
    if docker_working_dir is not None:
        options.append("--pwd")
        options.append(str(docker_working_dir))
    else:
        logger.warning(
            "Docker working directory not found. Using default working directory."
        )
    overlay_path = Path(image).parent / (Path(image).name + "_overlay.img")
    options.append("--overlay")
    options.append(str(overlay_path))

    overlay_created = False
    if not overlay_path.exists():
        subprocess.run(
            [
                "singularity",
                "overlay",
                "create",
                "--size",
                str(overlay_size),
                str(overlay_path),
            ],
            check=True,
        )
        overlay_created = True


    # 1. Manually construct the singularity command
    cmd = ["singularity", "run"]
    cmd.extend(options)
    
    for bind in singularity_bindings:
        cmd.extend(["--bind", bind])
        
    cmd.append(image)
    if args:
        cmd.extend(args)
        
    logger.debug(f"Executing Singularity command: {' '.join(cmd)}")
    
    container_output = []

    try:
        # 2. Use Popen, redirecting stderr to stdout (STDOUT) so we catch everything
        with subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1 # Line-buffered
        ) as proc:
            for line in proc.stdout:
                # 3. Use logger or a flushed print. 
                # (Ideally, pass a tqdm.write callback here, but for now, we force a flush)
                print(line, end="", flush=True) 
                container_output.append(line)
        
        if proc.returncode != 0:
            logger.error(f"Container exited with return code {proc.returncode}")
            r
        _sanity_check_output(
            data_path=data_path,
            output_path=output_path,
            container_output="\n".join(container_output),
            internal_external_name_map=internal_external_name_map,
        )
    finally:
        try:
            if overlay_created:
                overlay_path.unlink()
        except FileNotFoundError:
            pass
    logger.info(f"Finished inference in {time.time() - start_time:.2f} seconds")
