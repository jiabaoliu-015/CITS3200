import numpy as np
import pytest

from tasks.perception_task.adapters.py123d_pointpillars import (
    lidar_to_pointpillars_points,
)

class FakeLidar:
    def __init__(self, xyz, intensity):
        self.xyz = xyz
        self.intensity = intensity

def test_lidar_to_pointpillars_points():
    lidar = FakeLidar(
        xyz=np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        ),
        intensity=np.array([10, 20], dtype=np.uint8),
    )

    points = lidar_to_pointpillars_points(lidar)

    assert points.shape == (2, 5)
    assert points.dtype == np.float32

    np.testing.assert_array_equal(
        points[:, :3],
        lidar.xyz,
    )

    np.testing.assert_array_equal(
        points[:, 3],
        np.array([10.0, 20.0], dtype=np.float32),
    )

    np.testing.assert_array_equal(
        points[:, 4],
        np.zeros(2, dtype=np.float32),
    )

def test_missing_intensity_raises_error():
    lidar = FakeLidar(
        xyz=np.zeros((2, 3), dtype=np.float32),
        intensity=None,
    )

    with pytest.raises(
        ValueError,
        match="LiDAR frame does not contain intensity values",
    ):
        lidar_to_pointpillars_points(lidar)