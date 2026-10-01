from types import SimpleNamespace

import numpy as np

from tasks.perception_task.adapters.py123d_voxelnext import (
    lidar_to_voxelnext_points,
)


def test_lidar_to_voxelnext_points():
    lidar = SimpleNamespace(
        xyz=np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        ),
        intensity=np.array(
            [10, 20],
            dtype=np.uint8,
        ),
    )

    result = lidar_to_voxelnext_points(lidar)

    expected = np.array(
        [
            [1.0, 2.0, 3.0, 10.0],
            [4.0, 5.0, 6.0, 20.0],
        ],
        dtype=np.float32,
    )

    assert result.shape == (2, 4)
    assert result.dtype == np.float32

    np.testing.assert_array_equal(result, expected)