"""Download and normalize the VoxelNeXt Argoverse 2 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

FILE_ID = "1YP2UOz-yO-cWfYQkIqILEu6bodvCBVrR"
DEFAULT_SOURCE = Path.home() / "scratch" / "voxelnext_av2_source.pth"
DEFAULT_OUTPUT = Path("tasks/perception_task/models/voxelnext_av2.pth")
PREFIXES = ("module.", "model.", "net.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def download_checkpoint(source: Path) -> None:
    if source.exists():
        print(f"Source already present, skipping download: {source}")
        return

    try:
        import gdown
    except ImportError as error:
        raise SystemExit(
            "gdown is required to download the checkpoint; install it as a "
            "developer tool with `.venv/bin/python -m pip install gdown`."
        ) from error

    source.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Google Drive file {FILE_ID} to {source}")
    result = gdown.download(id=FILE_ID, output=str(source), quiet=False)
    if result is None or not source.is_file():
        raise RuntimeError("Google Drive download did not produce the source file")


def load_checkpoint(source: Path) -> Mapping[str, Any]:
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Expected a mapping checkpoint, got {type(checkpoint).__name__}"
        )
    return checkpoint


def unwrap_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    top_level_keys = list(checkpoint)
    print(f"Top-level keys ({len(top_level_keys)}): {top_level_keys}")

    wrapper_keys = ("model_state", "state_dict")
    wrappers = [key for key in wrapper_keys if key in checkpoint]
    if len(wrappers) > 1:
        raise ValueError(f"Multiple recognized wrapper keys found: {wrappers}")
    if wrappers:
        wrapper_key = wrappers[0]
        wrapped = checkpoint[wrapper_key]
        if not isinstance(wrapped, Mapping):
            raise TypeError(f"Wrapper {wrapper_key!r} does not contain a mapping")
        state = dict(wrapped)
        print(f"Detected format: wrapped state dict under {wrapper_key!r}")
    else:
        state = dict(checkpoint)
        print("Detected format: direct state dict with checkpoint metadata entry")

    if not state or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise ValueError("Checkpoint state contains non-tensor entries")

    source_tensor_count = len(state)
    source_parameter_count = sum(value.numel() for value in state.values())
    print(f"Loaded tensor count: {source_tensor_count}")
    print(f"Loaded total parameter count: {source_parameter_count}")

    matching_prefixes = [
        prefix for prefix in PREFIXES if all(key.startswith(prefix) for key in state)
    ]
    if len(matching_prefixes) > 1:
        raise ValueError(f"Multiple shared prefixes detected: {matching_prefixes}")
    if matching_prefixes:
        prefix = matching_prefixes[0]
        state = {key.removeprefix(prefix): value for key, value in state.items()}
        print(f"Stripped shared prefix: {prefix!r}")
    else:
        print("Shared prefix stripping: no-op (no shared module./model./net. prefix)")

    if (
        len(state) != source_tensor_count
        or sum(value.numel() for value in state.values()) != source_parameter_count
    ):
        raise AssertionError("Unwrapping changed tensor or parameter counts")

    if "global_step" not in state:
        raise ValueError(
            "Expected exactly one global_step tensor, but it was not found"
        )
    state_without_step = {
        key: value for key, value in state.items() if key != "global_step"
    }
    removed_keys = set(state) - set(state_without_step)
    if removed_keys != {"global_step"}:
        raise AssertionError(f"Unexpected keys removed: {removed_keys}")

    output_tensor_count = len(state_without_step)
    output_parameter_count = sum(value.numel() for value in state_without_step.values())
    if output_tensor_count != 487 or output_parameter_count != 7_894_088:
        raise AssertionError(
            "Unexpected normalized state: "
            f"{output_tensor_count} tensors, {output_parameter_count} parameters"
        )
    print("Removed exactly one checkpoint entry: 'global_step'")
    print(f"Output tensor count: {output_tensor_count}")
    print(f"Output total parameter count: {output_parameter_count}")
    return state_without_step


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    download_checkpoint(args.source)
    state = unwrap_checkpoint(load_checkpoint(args.source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    print(f"Saved output: {args.output}")
    print(f"Output size: {args.output.stat().st_size} bytes")
    print(f"SHA256: {sha256(args.output)}")


if __name__ == "__main__":
    main()
