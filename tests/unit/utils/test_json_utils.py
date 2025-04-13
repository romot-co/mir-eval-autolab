# -*- coding: utf-8 -*-
import pytest
import json
import numpy as np
from pathlib import Path

# テスト対象のEncoderをインポート
from src.utils.json_utils import NumpyEncoder

# --- Test NumpyEncoder ---


@pytest.mark.parametrize(
    "input_data, expected_json_str",
    [
        # Basic Python types (should work as default)
        (
            {"a": 1, "b": "text", "c": [1.0, 2.5], "d": True, "e": None},
            '{"a":1,"b":"text","c":[1.0,2.5],"d":true,"e":null}',
        ),
        # Numpy integer types
        ({"np_int32": np.int32(10)}, '{"np_int32":10}'),
        ({"np_int64": np.int64(-50)}, '{"np_int64":-50}'),
        # Numpy float types
        ({"np_float32": np.float32(0.125)}, '{"np_float32":0.125}'),
        ({"np_float64": np.float64(1.23e-4)}, '{"np_float64":0.000123}'),
        # Numpy boolean
        ({"np_bool": np.bool_(True)}, '{"np_bool":true}'),
        ({"np_bool_false": np.bool_(False)}, '{"np_bool_false":false}'),
        # Numpy array (1D)
        ({"np_array_1d": np.array([1, 2, 3])}, '{"np_array_1d":[1,2,3]}'),
        # Numpy array (2D)
        (
            {"np_array_2d": np.array([[1.1, 2.2], [3.3, 4.4]])},
            '{"np_array_2d":[[1.1,2.2],[3.3,4.4]]}',
        ),
        # Path object
        ({"path": Path("/some/test/path")}, '{"path":"/some/test/path"}'),
    ],
)
def test_numpy_encoder(input_data, expected_json_str):
    """Tests that NumpyEncoder correctly serializes various numpy types."""
    # Use separators to ensure consistent spacing for comparison
    result_json_str = json.dumps(input_data, cls=NumpyEncoder, separators=(",", ":"))
    assert result_json_str == expected_json_str


def test_numpy_encoder_float32_precision():
    """Tests that NumpyEncoder handles float32 precision correctly."""
    data = {
        "mixed": [np.int64(5), 6, np.float32(7.7)],
        "nested": {"arr": np.array([True, False])},
    }
    result_json_str = json.dumps(data, cls=NumpyEncoder, separators=(",", ":"))
    parsed = json.loads(result_json_str)

    # float32は精度が低いため、正確に7.7にならないことを確認
    assert parsed["mixed"][2] != 7.7
    assert round(parsed["mixed"][2], 1) == 7.7
    assert parsed["nested"]["arr"] == [True, False]


def test_numpy_encoder_special_floats():
    """Tests that NumpyEncoder handles NaN and Infinity values."""
    # NaN
    data = {"np_nan": np.nan}
    result_json_str = json.dumps(data, cls=NumpyEncoder, separators=(",", ":"))
    assert result_json_str == '{"np_nan":NaN}'

    # Positive Infinity
    data = {"np_inf": np.inf}
    result_json_str = json.dumps(data, cls=NumpyEncoder, separators=(",", ":"))
    assert result_json_str == '{"np_inf":Infinity}'

    # Negative Infinity
    data = {"np_neg_inf": -np.inf}
    result_json_str = json.dumps(data, cls=NumpyEncoder, separators=(",", ":"))
    assert result_json_str == '{"np_neg_inf":-Infinity}'

    # Array with NaN
    data = {"array_with_nan": np.array([1.0, np.nan, 3.0])}
    result_json_str = json.dumps(data, cls=NumpyEncoder, separators=(",", ":"))
    assert result_json_str == '{"array_with_nan":[1.0,NaN,3.0]}'


def test_numpy_complex_encoding():
    """Tests that NumpyEncoder correctly serializes complex numbers."""
    # 複素数型
    complex_data = {"complex": np.complex128(1 + 2j)}
    result = json.dumps(complex_data, cls=NumpyEncoder)
    parsed = json.loads(result)

    # 実装では辞書として返される
    assert "complex" in parsed
    assert "real" in parsed["complex"]
    assert "imag" in parsed["complex"]
    assert parsed["complex"]["real"] == 1.0
    assert parsed["complex"]["imag"] == 2.0

    # 複素数配列
    complex_array_data = {"complex_array": np.array([1 + 2j, 3 + 4j])}
    result = json.dumps(complex_array_data, cls=NumpyEncoder)
    parsed = json.loads(result)

    assert "complex_array" in parsed
    assert len(parsed["complex_array"]) == 2
    assert parsed["complex_array"][0]["real"] == 1.0
    assert parsed["complex_array"][0]["imag"] == 2.0
    assert parsed["complex_array"][1]["real"] == 3.0
    assert parsed["complex_array"][1]["imag"] == 4.0


def test_custom_to_json_object():
    """Tests that NumpyEncoder handles objects with to_json method."""

    class CustomObject:
        def to_json(self):
            return {"custom": "data"}

    data = {"obj": CustomObject()}
    result = json.dumps(data, cls=NumpyEncoder)
    parsed = json.loads(result)

    assert parsed["obj"]["custom"] == "data"


def test_numpy_encoder_unhandled_type():
    """Tests that NumpyEncoder raises TypeError for unhandled types."""

    class UnhandledType:
        pass

    unhandled_data = {"obj": UnhandledType()}

    with pytest.raises(TypeError):
        json.dumps(unhandled_data, cls=NumpyEncoder)
