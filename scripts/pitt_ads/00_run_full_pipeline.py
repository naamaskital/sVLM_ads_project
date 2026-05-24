"""
Run the full Pitt Ads preprocessing pipeline.

Run from the project root:
    python3 scripts/pitt_ads/00_run_full_pipeline.py
"""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PIPELINE_STEPS = [
    ("Build Pitt topics CSV", "scripts/pitt_ads/03_build_pitt_topics_csv.py"),
    ("Build Pitt moderation dataset", "scripts/pitt_ads/04_build_pitt_moderation_dataset.py"),
    ("Build Pitt binary topic dataset", "scripts/pitt_ads/05_build_pitt_binary_topic_dataset.py"),
    ("Build Pitt coarse dataset", "scripts/pitt_ads/05_build_pitt_coarse_dataset.py"),
    ("Create balanced coarse dataset", "scripts/pitt_ads/06_create_balanced_coarse_dataset.py"),
    ("Split coarse dataset", "scripts/pitt_ads/07_split_coarse_dataset.py"),
    ("Check splits and create coarse viewer", "scripts/pitt_ads/08_check_splits_and_make_viewer.py"),
    ("Create binary topic viewer", "scripts/pitt_ads/09_create_binary_topic_viewer.py"),
]


def run_step(step_name, script_path):
    """
    Run one pipeline step from the project root.
    If the step fails, stop the pipeline.
    """
    print("\n" + "=" * 80)
    print(f"Step: {step_name}")
    print("=" * 80)

    command = ["python3", script_path]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
    )

    if result.returncode != 0:
        print("\nPipeline failed.")
        print(f"Failed step: {step_name}")
        print(f"Command: {' '.join(command)}")
        sys.exit(result.returncode)


def run_pipeline(skip_viewers):
    """
    Run the Pitt Ads pipeline in the correct order.
    """
    print("Starting Pitt Ads pipeline")
    print(f"Project root: {PROJECT_ROOT}")

    for step_name, script_path in PIPELINE_STEPS:
        if skip_viewers and "viewer" in step_name.lower():
            print(f"\nSkipping viewer step: {step_name}")
            continue

        run_step(step_name, script_path)

    print("\nPipeline finished successfully.")


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--skip-viewers",
        action="store_true",
        help="Run the data pipeline without generating HTML viewers.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    run_pipeline(skip_viewers=args.skip_viewers)


if __name__ == "__main__":
    main()
