from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from py123d.api import SceneAPI

@contextmanager
def _working_directory(path: Path):
    old_directory = Path.cwd()
    os.chdir(path)

    try:
        yield
    finally:
        os.chdir(old_directory)


def _filter_points(
    points: np.ndarray,
    point_cloud_range: list[float],
) -> np.ndarray:
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range

    mask = (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
    )

    return np.ascontiguousarray(
        points[mask],
        dtype=np.float32,
    )


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in (
        "model_state",
        "state_dict",
        "model",
    ):
        value = checkpoint.get(key)

        if isinstance(value, dict):
            return value

    return checkpoint


def _normalise_key(key: str) -> str:
    prefixes = (
        "module.",
        "model.",
        "net.",
    )

    changed = True

    while changed:
        changed = False

        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True

    return key


def run_voxelnext_inference(
    scene: SceneAPI,
    voxelnext_root: Path,
    checkpoint_path: Path,
    frame_index: int = 0,
    score_threshold: float = 0.25,
    point_cloud_range: list[float] | None = None,
    max_test_voxels: int | None = None,
) -> dict:
    """
    Run VoxelNeXt inference on one py123D LiDAR frame.

    VoxelNeXt/OpenPCDet must already be installed and available
    at voxelnext_root.

    Args:
        scene:
            py123D scene containing LiDAR data.
        voxelnext_root:
            Path to the cloned VoxelNeXt repository.
        checkpoint_path:
            Path to the VoxelNeXt .pth checkpoint.
        frame_index:
            LiDAR frame to evaluate.
        score_threshold:
            Minimum prediction confidence returned.
        point_cloud_range:
            Optional reduced inference range.
        max_test_voxels:
            Optional reduced maximum voxel count.

    Returns:
        Dictionary containing predicted boxes, scores,
        labels, and class names.
    """

    import torch

    from tasks.perception_task.adapters.py123d_voxelnext import (
        load_voxelnext_frame,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "VoxelNeXt inference requires a CUDA-capable GPU"
        )

    voxelnext_root = Path(voxelnext_root)
    checkpoint_path = Path(checkpoint_path)

    if not voxelnext_root.exists():
        raise FileNotFoundError(
            f"VoxelNeXt repository not found: {voxelnext_root}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"VoxelNeXt checkpoint not found: {checkpoint_path}"
        )

    if str(voxelnext_root) not in sys.path:
        sys.path.insert(0, str(voxelnext_root))

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import DatasetTemplate
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils

    config_path = (
        voxelnext_root
        / "tools"
        / "cfgs"
        / "argo2_models"
        / "cbgs_voxel01_voxelnext.yaml"
    )

    tools_directory = voxelnext_root / "tools"

    with _working_directory(tools_directory):
        cfg_from_yaml_file(
            str(config_path),
            cfg,
        )

    if point_cloud_range is not None:
        cfg.DATA_CONFIG.POINT_CLOUD_RANGE = point_cloud_range

        cfg.MODEL.DENSE_HEAD.POST_PROCESSING.POST_CENTER_LIMIT_RANGE = (
            point_cloud_range
        )

    if max_test_voxels is not None:
        for processor in cfg.DATA_CONFIG.DATA_PROCESSOR:
            if processor.NAME == "transform_points_to_voxels":
                processor.MAX_NUMBER_OF_VOXELS["test"] = (
                    max_test_voxels
                )

    points = load_voxelnext_frame(
        scene,
        frame_index=frame_index,
    )

    if point_cloud_range is not None:
        points = _filter_points(
            points,
            point_cloud_range,
        )

    class Py123DVoxelNeXtDataset(DatasetTemplate):
        def __init__(self, points):
            super().__init__(
                dataset_cfg=cfg.DATA_CONFIG,
                class_names=cfg.CLASS_NAMES,
                training=False,
                root_path=Path("."),
                logger=common_utils.create_logger(),
            )

            self.points = points

        def __len__(self):
            return 1

        def __getitem__(self, index):
            input_dict = {
                "points": self.points.copy(),
                "frame_id": index,
            }

            return self.prepare_data(
                data_dict=input_dict,
            )

    dataset = Py123DVoxelNeXtDataset(
        points=points,
    )

    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset,
    )

    try:
        raw_checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        raw_checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    checkpoint_state = _extract_state_dict(
        raw_checkpoint
    )

    model_state = model.state_dict()

    compatible_state = {}

    for original_key, value in checkpoint_state.items():
        key = _normalise_key(original_key)

        if key not in model_state:
            continue

        if model_state[key].shape != value.shape:
            continue

        compatible_state[key] = value

    match_ratio = len(compatible_state) / max(
        len(model_state),
        1,
    )

    if match_ratio < 0.90:
        raise RuntimeError(
            "VoxelNeXt checkpoint does not match the model "
            f"({len(compatible_state)}/{len(model_state)} parameters)"
        )

    model.load_state_dict(
        compatible_state,
        strict=False,
    )

    model.cuda()
    model.eval()

    data_dict = dataset[0]

    batch_dict = dataset.collate_batch(
        [data_dict]
    )

    load_data_to_gpu(
        batch_dict
    )

    with torch.no_grad():
        prediction_dicts, _ = model(
            batch_dict
        )

    prediction = prediction_dicts[0]

    boxes = (
        prediction["pred_boxes"]
        .detach()
        .cpu()
        .numpy()
    )

    scores = (
        prediction["pred_scores"]
        .detach()
        .cpu()
        .numpy()
    )

    labels = (
        prediction["pred_labels"]
        .detach()
        .cpu()
        .numpy()
    )

    keep = scores >= score_threshold

    return {
        "boxes": boxes[keep],
        "scores": scores[keep],
        "labels": labels[keep],
        "class_names": list(cfg.CLASS_NAMES),
        "num_input_points": len(points),
    }