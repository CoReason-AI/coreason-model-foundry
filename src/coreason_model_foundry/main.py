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

from coreason_identity.models import UserContext
from coreason_identity.types import SecretStr
from coreason_model_foundry.service import ModelFoundryService


def main() -> None:
    parser = argparse.ArgumentParser(description="Coreason Model Foundry - The Refinery")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train Command
    train_parser = subparsers.add_parser("train", help="Train a model using a manifest")
    train_parser.add_argument("--manifest", type=str, required=True, help="Path to the Training Manifest YAML")

    # Evaluate Command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--model", type=str, required=True, help="Path to the model directory")

    # Publish Command
    pub_parser = subparsers.add_parser("publish", help="Publish a model artifact")
    pub_parser.add_argument("--model", type=str, required=True, help="Path to the model directory")
    pub_parser.add_argument("--registry", type=str, required=True, help="Target registry URI")
    pub_parser.add_argument("--tag", type=str, required=True, help="Version tag")

    args = parser.parse_args()

    system_context = UserContext(
        user_id=SecretStr("cli-user"),
        roles=["system"],
        metadata={"source": "cli"},
    )

    # Use the Facade to run the service synchronously
    with ModelFoundryService() as service:
        if args.command == "train":
            service.orchestrate_training(Path(args.manifest), context=system_context)
        elif args.command == "evaluate":
            service.evaluate_model(Path(args.model), context=system_context)
        elif args.command == "publish":
            service.publish_model(Path(args.model), args.registry, args.tag, context=system_context)


if __name__ == "__main__":
    main()
