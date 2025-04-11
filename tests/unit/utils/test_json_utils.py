import pytest
import json
import numpy as np

# テスト対象のEncoderをインポート
try:
    from src.utils.json_utils import NumpyEncoder
except ImportError:
    pytest.skip("Skipping json_utils tests due to missing src modules", allow_module_level=True)
    # Dummy class for static analysis
    class NumpyEncoder(json.JSONEncoder): pass

# --- Test NumpyEncoder ---

@pytest.mark.parametrize("input_data, expected_json_str", [
    # Basic Python types (should work as default)
    ({"a": 1, "b": "text", "c": [1.0, 2.5], "d": True, "e": None}, '{"a": 1, "b": "text", "c": [1.0, 2.5], "d": true, "e": null}'),
    # Numpy integer types
    ({"np_int32": np.int32(10)}, '{"np_int32": 10}'),
    ({"np_int64": np.int64(-50)}, '{"np_int64": -50}'),
    # Numpy float types
    ({"np_float32": np.float32(0.125)}, '{"np_float32": 0.125}'),
    ({"np_float64": np.float64(1.23e-4)}, '{"np_float64": 0.000123}'),
    # Numpy boolean
    ({"np_bool": np.bool_(True)}, '{"np_bool": true}'),
    ({"np_bool_false": np.bool_(False)}, '{"np_bool_false": false}'),
    # Numpy array (1D)
    ({"np_array_1d": np.array([1, 2, 3])}, '{"np_array_1d": [1, 2, 3]}'),
    # Numpy array (2D)
    ({"np_array_2d": np.array([[1.1, 2.2], [3.3, 4.4]])}, '{"np_array_2d": [[1.1, 2.2], [3.3, 4.4]]}'),
    # Mixed data
    ({"mixed": [np.int64(5), 6, np.float32(7.7)], "nested": {"arr": np.array([True, False])}},
     '{"mixed": [5, 6, 7.7], "nested": {"arr": [true, false]}}'),
    # Numpy NaN and Inf (should become null by default in standard JSON)
    # Note: JSON standard doesn't support NaN/Inf. Some libs allow it, but default is null.
    # Check NumpyEncoder's specific handling if non-standard output is desired.
    ({"np_nan": np.nan}, '{"np_nan": null}'),
    ({"np_inf": np.inf}, '{"np_inf": null}'),
    ({"np_neg_inf": -np.inf}, '{"np_neg_inf": null}'),
    ({"array_with_nan": np.array([1.0, np.nan, 3.0])}, '{"array_with_nan": [1.0, null, 3.0]}'),
])
def test_numpy_encoder(input_data, expected_json_str):
    """Tests that NumpyEncoder correctly serializes various numpy types."""
    # Use separators to ensure consistent spacing for comparison
    result_json_str = json.dumps(input_data, cls=NumpyEncoder, separators=(',', ':'))

    # Load the expected string back into a dict for potential comparison if needed,
    # but comparing the JSON string directly is often sufficient and stricter.
    # expected_dict = json.loads(expected_json_str)
    # result_dict = json.loads(result_json_str)
    # assert result_dict == expected_dict

    assert result_json_str == expected_json_str

def test_numpy_encoder_unhandled_type():
    """Tests that NumpyEncoder raises TypeError for unhandled types (like complex numbers)."""
    unhandled_data = {"complex": np.complex128(1 + 2j)}

    with pytest.raises(TypeError):
        json.dumps(unhandled_data, cls=NumpyEncoder) 