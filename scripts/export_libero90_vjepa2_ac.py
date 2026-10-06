"""Combine the frozen V-JEPA encoder and a LIBERO-90 tuned AC predictor.

The evaluation adapter reads only ``encoder`` and ``predictor`` from the
checkpoint, so this export omits training optimizer/target-encoder tensors.
The full resumable training checkpoint remains untouched.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def normalize(name: str) -> str:
    return name.replace("module.", "").replace("backbone.", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    base = torch.load(args.base, map_location="cpu", weights_only=True, mmap=True)
    tuned = torch.load(args.trained, map_location="cpu", weights_only=False, mmap=True)
    if int(tuned["step"]) <= 0 or not tuned["optimizer"]["state"]:
        raise ValueError("Training checkpoint lacks a completed step or optimizer state")

    base_predictor = {normalize(k): v for k, v in base["predictor"].items()}
    tuned_predictor = {normalize(k): v for k, v in tuned["predictor"].items()}
    if set(base_predictor) != set(tuned_predictor):
        raise ValueError("Predictor state keys differ from the original checkpoint")
    for name, value in tuned_predictor.items():
        if value.shape != base_predictor[name].shape:
            raise ValueError(f"Shape mismatch for predictor tensor {name}")

    payload = {
        "encoder": base["encoder"],
        "predictor": tuned_predictor,
        "source_checkpoint": str(args.base),
        "fine_tune_checkpoint": str(args.trained),
        "fine_tune_step": int(tuned["step"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    print(f"Exported step {tuned['step']} to {args.output}", flush=True)


if __name__ == "__main__":
    main()
