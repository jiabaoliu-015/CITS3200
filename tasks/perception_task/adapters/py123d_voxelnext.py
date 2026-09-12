import numpy as np
from py123d.api import SceneAPI
from py123d.datatypes import Lidar, LidarID


def lidar_to_voxelnext_points(lidar: Lidar) -> np.ndarray:
    """
    Convert a py123D LiDAR frame into the [x, y, z, intensity]
    float32 format expected by VoxelNeXt.
    """

    if lidar.intensity is None:
        raise ValueError("LiDAR frame does not contain intensity values")

    xyz = lidar.xyz.astype(np.float32)
    intensity = lidar.intensity.astype(np.float32).reshape(-1, 1)

    return np.concatenate(
        [xyz, intensity],
        axis=1,
    ).astype(np.float32)


def load_voxelnext_frame(
    scene: SceneAPI,
    frame_index: int = 0,
) -> np.ndarray:
    """
    Load one LiDAR frame from a py123D scene and convert it
    into the format expected by VoxelNeXt.
    """

    lidar = scene.get_lidar_at_iteration(
        iteration=frame_index,
        lidar_id=LidarID.LIDAR_MERGED,
    )

    if lidar is None:
        lidar = scene.get_lidar_at_iteration(
            iteration=frame_index,
            lidar_id=LidarID.LIDAR_TOP,
        )

    if lidar is None:
        raise ValueError(
            f"No LiDAR data found for frame {frame_index}"
        )

    return lidar_to_voxelnext_points(lidar)