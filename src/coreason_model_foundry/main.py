# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_model_foundry

import argparse
from pathlib import Path

from coreason_model_foundry.service import ModelFoundryService


def main() -> None:
    parser = argparse.ArgumentParser(description="Coreason Model Foundry - The Refinery")
    parser.add_argument("--manifest", type=str, required=True, help="Path to the Training Manifest YAML")
    args = parser.parse_args()

    # Use the Facade to run the service synchronously
    with ModelFoundryService() as service:
        service.orchestrate_training(Path(args.manifest))


if __name__ == "__main__":
    main()
