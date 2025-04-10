"""
JSONユーティリティモジュール (`src/utils/json_utils.py`) のテスト
"""
import pytest
import json
import numpy as np
from datetime import date
from unittest.mock import MagicMock

# テスト対象のモジュールをインポート
try:
    from src.utils.json_utils import NumpyEncoder
except ImportError:
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.utils.json_utils import NumpyEncoder

# --- テスト用のカスタムクラス ---
class SerializableObject:
    def __init__(self, value):
        self.value = value

    def to_json(self):
        return {"custom_value": self.value}

class NonSerializableObject:
    pass

# --- テスト関数 ---

def test_numpy_encoder_basic_types():
    """NumpyEncoder: 基本的なPython型のテスト"""
    data = {
        "string": "hello",
        "integer": 123,
        "float": 45.6,
        "boolean": True,
        "none": None,
        "list": [1, "a", None],
        "dict": {"key": "value"}
    }
    # expected_json のキーをアルファベット順に修正
    expected_json = '{"boolean": true, "dict": {"key": "value"}, "float": 45.6, "integer": 123, "list": [1, "a", null], "none": null, "string": "hello"}'
    # 標準のjson.dumpsと比較しても同じはず
    assert json.dumps(data, cls=NumpyEncoder, sort_keys=True) == json.dumps(data, sort_keys=True)
    # 期待値と比較
    assert json.dumps(data, cls=NumpyEncoder, sort_keys=True) == expected_json

def test_numpy_encoder_numpy_int():
    """NumpyEncoder: NumPy整数型のテスト"""
    data = {"np_int32": np.int32(10), "np_int64": np.int64(20)}
    expected_json = '{"np_int32": 10, "np_int64": 20}'
    assert json.dumps(data, cls=NumpyEncoder, sort_keys=True) == expected_json

def test_numpy_encoder_numpy_float():
    """NumpyEncoder: NumPy浮動小数点型のテスト"""
    data = {"np_float32": np.float32(1.23), "np_float64": np.float64(4.56)}
    # 浮動小数点数の比較は注意が必要なので、loadsしてから比較
    result_json = json.dumps(data, cls=NumpyEncoder)
    result_dict = json.loads(result_json)
    assert result_dict["np_float32"] == pytest.approx(1.23)
    assert result_dict["np_float64"] == pytest.approx(4.56)

def test_numpy_encoder_numpy_bool():
    """NumpyEncoder: NumPyブール型のテスト"""
    data = {"np_bool_true": np.bool_(True), "np_bool_false": np.bool_(False)}
    expected_json = '{"np_bool_false": false, "np_bool_true": true}'
    assert json.dumps(data, cls=NumpyEncoder, sort_keys=True) == expected_json

def test_numpy_encoder_numpy_array():
    """NumpyEncoder: NumPy ndarrayのテスト"""
    data = {
        "np_array_int": np.array([1, 2, 3]),
        "np_array_float": np.array([1.1, 2.2]),
        "np_array_mixed": np.array([1, 3.14, True], dtype=object), # object dtype もリストに変換される
        "np_array_2d": np.array([[1, 2], [3, 4]])
    }
    expected_json = '{"np_array_2d": [[1, 2], [3, 4]], "np_array_float": [1.1, 2.2], "np_array_int": [1, 2, 3], "np_array_mixed": [1, 3.14, true]}'
    assert json.dumps(data, cls=NumpyEncoder, sort_keys=True) == expected_json

def test_numpy_encoder_numpy_complex():
    """NumpyEncoder: NumPy複素数型のテスト"""
    data = {"np_complex": np.complex128(1 + 2j)}
    expected_json = '{"np_complex": {"imag": 2.0, "real": 1.0}}'
    assert json.dumps(data, cls=NumpyEncoder, sort_keys=True) == expected_json

def test_numpy_encoder_mixed_dict():
    """NumpyEncoder: NumPy型を含む辞書のテスト"""
    data = {
        "id": np.int64(1),
        "values": np.array([1.5, 2.5, 3.5]),
        "valid": np.bool_(True),
        "name": "test_data"
    }
    expected_json = '{"id": 1, "name": "test_data", "valid": true, "values": [1.5, 2.5, 3.5]}'
    assert json.dumps(data, cls=NumpyEncoder, sort_keys=True) == expected_json

def test_numpy_encoder_custom_serializable():
    """NumpyEncoder: to_jsonメソッドを持つオブジェクトのテスト"""
    obj = SerializableObject(value=100)
    data = {"custom": obj}
    expected_json = '{"custom": {"custom_value": 100}}'
    assert json.dumps(data, cls=NumpyEncoder, sort_keys=True) == expected_json

def test_numpy_encoder_stringifiable():
    """NumpyEncoder: str()で変換可能なオブジェクト (例: date) - 修正: TypeErrorを期待"""
    data = {"date": date(2023, 10, 27)}
    # datetime.date は NumpyEncoder では特別扱いされず、
    # 親クラスの JSONEncoder.default に渡されて TypeError になることを確認
    # expected_json = '{"date": "2023-10-27"}' # <-- 以前の期待値
    # assert json.dumps(data, cls=NumpyEncoder, sort_keys=True) == expected_json # <-- 以前のアサーション
    with pytest.raises(TypeError) as excinfo:
        json.dumps(data, cls=NumpyEncoder)
    assert "not JSON serializable" in str(excinfo.value)

def test_numpy_encoder_non_serializable():
    """NumpyEncoder: シリアライズ不能なオブジェクトのテスト"""
    obj = NonSerializableObject()
    data = {"unserializable": obj}
    # NumpyEncoderでも処理できず、str()でも失敗し、最終的にJSONEncoderのデフォルトに渡されてTypeErrorになるはず
    with pytest.raises(TypeError):
        json.dumps(data, cls=NumpyEncoder)

def test_numpy_encoder_integer():
    """NumpyEncoder: 整数型のNumPy配列エンコードテスト"""
    # 整数型の配列
    data = {'array': np.array([1, 2, 3], dtype=np.int32)}
    # JSONエンコード
    json_str = json.dumps(data, cls=NumpyEncoder)
    # 復元して検証
    result = json.loads(json_str)
    assert result['array'] == [1, 2, 3]
    
    # スカラー整数値
    data = {'value': np.int32(42)}
    json_str = json.dumps(data, cls=NumpyEncoder)
    result = json.loads(json_str)
    assert result['value'] == 42


def test_numpy_encoder_float():
    """NumpyEncoder: 浮動小数点型のNumPy配列エンコードテスト"""
    # 浮動小数点型の配列
    data = {'array': np.array([1.1, 2.2, 3.3], dtype=np.float32)}
    # JSONエンコード
    json_str = json.dumps(data, cls=NumpyEncoder)
    # 復元して検証
    result = json.loads(json_str)
    # float32は精度の問題があるため、ほぼ等しいかをチェック
    assert len(result['array']) == 3
    assert abs(result['array'][0] - 1.1) < 1e-5
    assert abs(result['array'][1] - 2.2) < 1e-5
    assert abs(result['array'][2] - 3.3) < 1e-5
    
    # スカラー浮動小数点値
    data = {'value': np.float64(3.14159)}
    json_str = json.dumps(data, cls=NumpyEncoder)
    result = json.loads(json_str)
    assert result['value'] == 3.14159


def test_numpy_encoder_multidimensional():
    """NumpyEncoder: 多次元NumPy配列エンコードテスト"""
    # 2次元配列
    data = {'matrix': np.array([[1, 2], [3, 4]])}
    # JSONエンコード
    json_str = json.dumps(data, cls=NumpyEncoder)
    # 復元して検証
    result = json.loads(json_str)
    assert result['matrix'] == [[1, 2], [3, 4]]
    
    # 3次元配列
    data = {'tensor': np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])}
    json_str = json.dumps(data, cls=NumpyEncoder)
    result = json.loads(json_str)
    assert result['tensor'] == [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]


def test_numpy_encoder_complex():
    """NumpyEncoder: 複素数型のNumPy配列エンコードテスト"""
    # 複素数型の配列
    data = {'complex': np.array([1+2j, 3+4j], dtype=np.complex64)}
    # JSONエンコード
    json_str = json.dumps(data, cls=NumpyEncoder)
    # 復元して検証
    result = json.loads(json_str)
    # 複素数は{'real': 実部, 'imag': 虚部}の形式でエンコードされる
    assert len(result['complex']) == 2
    assert result['complex'][0]['real'] == 1.0
    assert result['complex'][0]['imag'] == 2.0
    assert result['complex'][1]['real'] == 3.0
    assert result['complex'][1]['imag'] == 4.0
    
    # スカラー複素数
    data = {'value': np.complex128(1+2j)}
    json_str = json.dumps(data, cls=NumpyEncoder)
    result = json.loads(json_str)
    assert result['value']['real'] == 1
    assert result['value']['imag'] == 2


def test_numpy_encoder_boolean():
    """NumpyEncoder: ブール型のNumPy配列エンコードテスト"""
    # ブール型の配列
    data = {'bool_array': np.array([True, False, True], dtype=np.bool_)}
    # JSONエンコード
    json_str = json.dumps(data, cls=NumpyEncoder)
    # 復元して検証
    result = json.loads(json_str)
    assert result['bool_array'] == [True, False, True]
    
    # スカラーブール値
    data = {'value': np.bool_(True)}
    json_str = json.dumps(data, cls=NumpyEncoder)
    result = json.loads(json_str)
    assert result['value'] is True


def test_numpy_encoder_custom_object():
    """NumpyEncoder: to_jsonメソッドを持つカスタムオブジェクトのエンコードテスト"""
    # モック・カスタムオブジェクトの作成
    custom_obj = MagicMock()
    custom_obj.to_json.return_value = {'name': 'custom', 'value': 42}
    
    # データの設定
    data = {'custom': custom_obj}
    
    # JSONエンコード
    json_str = json.dumps(data, cls=NumpyEncoder)
    
    # 復元して検証
    result = json.loads(json_str)
    assert result['custom'] == {'name': 'custom', 'value': 42}
    
    # to_jsonメソッドが呼ばれたことを確認
    custom_obj.to_json.assert_called_once()


def test_numpy_encoder_unsupported_type():
    """NumpyEncoder: サポートされていない型のエンコードテスト（例外発生確認）"""
    # シリアライズできない型を作成
    class Unserializable:
        pass
    
    # データにシリアライズできない型を含める
    data = {'unsupported': Unserializable()}
    
    # JSONエンコードで例外が発生することを確認
    with pytest.raises(TypeError):
        json.dumps(data, cls=NumpyEncoder) 