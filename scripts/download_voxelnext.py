"""Download the VoxelNeXt Argoverse 2 checkpoint."""


from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

FILE_ID = "1YP2UOz-yO-cWfYQkIqILEu6bodvCBVrR"
DEFAULT_SOURCE = Path.home() / "scratch" / "voxelnext_av2_source.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    download_checkpoint(args.source)
    print(f"Source size: {args.source.stat().st_size} bytes")
    print(f"SHA256: {sha256(args.source)}")


if __name__ == "__main__":
    main()
