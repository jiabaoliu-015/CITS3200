import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tasks.perception_task.inference.pointpillars import (
    run_pointpillars_inference,
)

def fake_torch(cuda_available: bool):
    module = ModuleType("torch")
    module.cuda = SimpleNamespace(
        is_available=lambda: cuda_available
    )
    return module

def test_requires_cuda(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        fake_torch(False),
    )

    with pytest.raises(
        RuntimeError,
        match="PointPillars inference requires a CUDA-capable GPU",
    ):
        run_pointpillars_inference(
            scene=None,
            openpcdet_root=Path("fake_openpcdet"),
            checkpoint_path=Path("fake_checkpoint.pth"),
        )

def test_missing_openpcdet_root(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        fake_torch(True),
    )

    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()

    with pytest.raises(
        FileNotFoundError,
        match="OpenPCDet repository not found",
    ):
        run_pointpillars_inference(
            scene=None,
            openpcdet_root=tmp_path / "missing_openpcdet",
            checkpoint_path=checkpoint,
        )
def test_missing_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        fake_torch(True),
    )

    openpcdet_root = tmp_path / "OpenPCDet"
    openpcdet_root.mkdir()

    with pytest.raises(
        FileNotFoundError,
        match="PointPillars checkpoint not found",
    ):
        run_pointpillars_inference(
            scene=None,
            openpcdet_root=openpcdet_root,
            checkpoint_path=tmp_path / "missing.pth",
        )