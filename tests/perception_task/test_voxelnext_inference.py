import numpy as np

from tasks.perception_task.inference.voxelnext import (
    _extract_state_dict,
    _filter_points,
    _normalise_key,
)


def test_filter_points():
    points = np.array(
        [
            [0.0, 0.0, 0.0, 10.0],
            [100.0, 0.0, 0.0, 20.0],
            [-10.0, 5.0, 1.0, 30.0],
        ],
        dtype=np.float32,
    )

    result = _filter_points(
        points,
        [-60.0, -60.0, -5.0, 60.0, 60.0, 5.0],
    )

    expected = np.array(
        [
            [0.0, 0.0, 0.0, 10.0],
            [-10.0, 5.0, 1.0, 30.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_normalise_key():
    assert _normalise_key(
        "module.model.backbone.weight"
    ) == "backbone.weight"

    assert _normalise_key(
        "backbone.weight"
    ) == "backbone.weight"


def test_extract_state_dict():
    state = {
        "layer.weight": "test-weight",
    }

    checkpoint = {
        "model_state": state,
    }

    result = _extract_state_dict(
        checkpoint
    )

    assert result == state