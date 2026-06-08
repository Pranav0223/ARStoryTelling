import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

EXTRACTION_DIR = Path(__file__).resolve().parent


def extract_story_graphs(image_path: str) -> dict:
    """
    Run the SHRI JSON extraction pipeline on a scripture image.

    Invokes pipeline.py via subprocess with EXTRACTION_DIR as cwd so all
    local imports (ocr, rule_extractor, beat_extractor, etc.) resolve correctly.
    Output JSON files are written to a temp directory and read back.

    Args:
        image_path: Absolute (or resolvable) path to the input .jpg/.png image.

    Returns:
        {
            "graph1": dict,   — 3D model generation (characters + image prompts)
            "graph2": dict,   — animation generation (actions + durations)
            "graph3": dict,   — scene composition (positions + timelines)
            "story_id": str
        }

    Raises:
        FileNotFoundError: If the input image does not exist.
        RuntimeError:      If the extraction subprocess fails or produces no output.
    """
    image_path = os.path.abspath(image_path)
    logger.info("extract_story_graphs | image=%s", image_path)

    if not os.path.exists(image_path):
        logger.error("Input image not found: %s", image_path)
        raise FileNotFoundError(f"Input image not found: {image_path}")

    output_dir = tempfile.mkdtemp(prefix="extraction_out_")
    try:
        cmd = [sys.executable, "pipeline.py", image_path, "--output", output_dir]
        logger.info("Extraction cmd: %s", " ".join(cmd))
        logger.info("cwd: %s", EXTRACTION_DIR)

        result = subprocess.run(
            cmd,
            cwd=str(EXTRACTION_DIR),
            capture_output=True,
            text=True,
            timeout=300,
        )

        for line in result.stdout.splitlines():
            logger.info("[extraction] %s", line)
        for line in result.stderr.splitlines():
            logger.warning("[extraction stderr] %s", line)

        if result.returncode != 0:
            logger.error("Extraction exited with code %s", result.returncode)
            raise RuntimeError(
                f"JSON extraction failed (exit {result.returncode}). Check logs above."
            )

        g1_path = Path(output_dir) / "graph1_3d_generation.json"
        g2_path = Path(output_dir) / "graph2_animation_generation.json"
        g3_path = Path(output_dir) / "graph3_scene_composition.json"

        if not g1_path.exists():
            logger.error("graph1_3d_generation.json not found in %s", output_dir)
            raise RuntimeError(
                "Extraction subprocess succeeded but graph1_3d_generation.json was not produced."
            )

        graph1 = json.loads(g1_path.read_text(encoding="utf-8"))
        graph2 = json.loads(g2_path.read_text(encoding="utf-8")) if g2_path.exists() else {}
        graph3 = json.loads(g3_path.read_text(encoding="utf-8")) if g3_path.exists() else {}

        story_id = graph1.get("story", "unknown")
        logger.info(
            "Extraction complete | story=%s  chars=%d  anims=%d",
            story_id,
            len(graph1.get("characters", [])),
            len(graph2.get("animations", [])),
        )

        return {
            "graph1": graph1,
            "graph2": graph2,
            "graph3": graph3,
            "story_id": story_id,
        }

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
        logger.debug("Cleaned up extraction output dir: %s", output_dir)
