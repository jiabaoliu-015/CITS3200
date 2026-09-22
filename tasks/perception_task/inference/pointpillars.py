from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

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

def run_pointpillars_inference(
    scene: SceneAPI,
    openpcdet_root: Path,
    checkpoint_path: Path,
    frame_index: int = 0,
    score_threshold: float = 0.25,
) -> dict:
    """
    Run OpenPCDet PointPillars inference on one py123D LiDAR frame.
    """

    import torch

    from tasks.perception_task.adapters.py123d_pointpillars import (
        load_pointpillars_frame,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PointPillars inference requires a CUDA-capable GPU"
        )

    openpcdet_root = Path(openpcdet_root)
    checkpoint_path = Path(checkpoint_path)

    if not openpcdet_root.exists():
        raise FileNotFoundError(
            f"OpenPCDet repository not found: {openpcdet_root}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"PointPillars checkpoint not found: {checkpoint_path}"
        )

    if str(openpcdet_root) not in sys.path:
        sys.path.insert(0, str(openpcdet_root))

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import DatasetTemplate
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils

    config_path = (
        openpcdet_root
        / "tools"
        / "cfgs"
        / "nuscenes_models"
        / "cbgs_pp_multihead.yaml"
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"PointPillars config not found: {config_path}"
        )

    with _working_directory(openpcdet_root / "tools"):
        cfg_from_yaml_file(
            str(config_path),
            cfg,
        )

    points = load_pointpillars_frame(
        scene,
        frame_index=frame_index,
    )

    logger = common_utils.create_logger()

    class Py123DPointPillarsDataset(DatasetTemplate):
        def __init__(self, points):
            super().__init__(
                dataset_cfg=cfg.DATA_CONFIG,
                class_names=cfg.CLASS_NAMES,
                training=False,
                root_path=Path("."),
                logger=logger,
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

    dataset = Py123DPointPillarsDataset(
        points=points,
    )

    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset,
    )

    model.load_params_from_file(
        filename=str(checkpoint_path),
        logger=logger,
        to_cpu=True,
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