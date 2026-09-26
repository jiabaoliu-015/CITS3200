from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from py123d.api import SceneAPI
    from py123d.datatypes import Lidar


def lidar_to_second_points(lidar: Lidar) -> np.ndarray:
    """
    Convert a py123D LiDAR frame into the point format expected by OpenPCDet SECOND on nuScenes.

    Output columns:
        [x, y, z, intensity, time_lag]

    For a single current LiDAR sweep, time_lag is zero.
    """

    if lidar.intensity is None:
        raise ValueError("LiDAR frame does not contain intensity values")

    xyz = lidar.xyz.astype(np.float32)

    intensity = (
        lidar.intensity
        .astype(np.float32)
        .reshape(-1, 1)
    )

    time_lag = np.zeros(
        (xyz.shape[0], 1),
        dtype=np.float32,
    )

    return np.concatenate(
        [
            xyz,
            intensity,
            time_lag,
        ],
        axis=1,
    ).astype(np.float32)


def load_second_frame(
    scene: SceneAPI,
    frame_index: int = 0,
) -> np.ndarray:
    """
    Load one LiDAR frame from a py123D scene and convert it into the format expected by SECOND.
    """

    from py123d.datatypes import LidarID

    lidar = scene.get_lidar_at_iteration(
        iteration=frame_index,
        lidar_id=LidarID.LIDAR_TOP,
    )

    if lidar is None:
        lidar = scene.get_lidar_at_iteration(
            iteration=frame_index,
            lidar_id=LidarID.LIDAR_MERGED,
        )

    if lidar is None:
        raise ValueError(
            f"No LiDAR data found for frame {frame_index}"
        )

    return lidar_to_second_points(lidar)