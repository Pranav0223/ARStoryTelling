'''
conda activate triposr
i am using triposr env so activate that and run the subprocess and get the output you can check the rigging/rigging_inference.py code as a referenece.
'''

import argparse
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

SERVICE_ROOT = Path(__file__).resolve().parent
TRIPOSR = SERVICE_ROOT / "TripoSR"
CONDA_ENV = "triposr"


def _conda_base() -> str:
    try:
        r = subprocess.run(["conda", "info", "--base"], capture_output=True, text=True, timeout=10)
        base = r.stdout.strip()
        if base:
            return base
    except Exception as e:
        logger.warning("Could not detect conda base: %s", e)
    return os.path.expanduser("~/miniconda3")


def _subprocess_env() -> dict:
    conda_env_bin = str(Path(_conda_base()) / "envs" / CONDA_ENV / "bin")

    env = os.environ.copy()
    for key in ["VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT", "PYTHONPATH"]:
        env.pop(key, None)
    for key in [k for k in env if k.startswith("UV_")]:
        env.pop(key)

    env["PATH"] = f"{conda_env_bin}:{env.get('PATH', '/usr/bin:/bin')}"
    logger.info("Subprocess python: %s/python", conda_env_bin)
    return env


def convert_obj_to_glb(obj_path: Path, glb_path: Path):
    import trimesh
    logger.info("convert_obj_to_glb | obj=%s glb=%s", obj_path, glb_path)
    mesh = trimesh.load(obj_path, force="mesh")
    mesh.export(glb_path)
    logger.info("Converted OBJ to GLB: %s", glb_path)


def generate_glb(input_image: str, asset_name: str | None = None) -> str:
    """
    Generate a GLB 3D model from a 2D image using TripoSR.

    Runs TripoSR's run.py via subprocess using the triposr conda env.
    If TripoSR outputs a GLB directly it is copied to the final dir; if it
    outputs an OBJ, trimesh converts it to GLB. The intermediate job dir is
    always removed in a finally block.

    Args:
        input_image: Path to the input 2D image.
        asset_name:  Optional name for the output asset; defaults to image stem.

    Returns:
        Absolute path to the generated .glb file.

    Raises:
        FileNotFoundError: If the input image or any expected output is missing.
        RuntimeError:      If the TripoSR subprocess fails.
    """
    input_path = Path(os.path.abspath(input_image))
    logger.info("generate_glb | input=%s asset_name=%s", input_path, asset_name)

    if not input_path.exists():
        logger.error("Input image not found: %s", input_path)
        raise FileNotFoundError(f"Input image not found: {input_path}")

    if asset_name is None:
        asset_name = input_path.stem

    job_id = f"{asset_name}_{int(time.time())}"

    out_dir = SERVICE_ROOT / "outputs" / "jobs" / job_id
    final_dir = SERVICE_ROOT / "outputs" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Job dir: %s", out_dir)

    try:
        cmd = [
            "python",
            str(TRIPOSR / "run.py"),
            str(input_path),
            "--output-dir",
            str(out_dir),
        ]
        logger.info("Running TripoSR: %s", " ".join(cmd))
        logger.info("cwd: %s", TRIPOSR)

        result = subprocess.run(
            cmd,
            cwd=str(TRIPOSR),
            env=_subprocess_env(),
            capture_output=True,
            text=True,
        )

        for line in result.stdout.splitlines():
            logger.info("[subprocess] %s", line)
        for line in result.stderr.splitlines():
            logger.info("[subprocess stderr] %s", line)

        if result.returncode != 0:
            logger.error("TripoSR exited with code %s", result.returncode)
            raise RuntimeError(f"TripoSR failed (exit {result.returncode}). Check logs above.")

        final_glb = final_dir / f"{job_id}.glb"

        glbs = list(out_dir.rglob("*.glb"))
        if glbs:
            shutil.copy(glbs[0], final_glb)
            logger.info("GLB ready: %s", final_glb)
            return str(final_glb)

        objs = list(out_dir.rglob("*.obj"))
        if objs:
            convert_obj_to_glb(objs[0], final_glb)
            logger.info("GLB ready: %s", final_glb)
            return str(final_glb)

        logger.error("No GLB or OBJ found in %s", out_dir)
        raise FileNotFoundError(f"No GLB or OBJ found in {out_dir}")

    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
        logger.debug("Cleaned up job dir: %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Generate GLB from any 2D image using TripoSR")
    parser.add_argument("--input", "-i", required=True, help="Path to input image")
    parser.add_argument("--name", "-n", default=None, help="Asset/model name")
    args = parser.parse_args()

    generate_glb(args.input, args.name)
