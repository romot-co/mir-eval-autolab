"""
JSONシリアライズ関連のユーティリティ関数

このモジュールは、NumPy配列などをJSONに変換するための
ユーティリティクラスを提供します。
"""

import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    NumPy配列やndarray型などをJSON形式に変換するためのエンコーダー
    
    Examples
    --------
    >>> import json
    >>> import numpy as np
    >>> from src.utils.json_utils import NumpyEncoder
    >>> data = {'array': np.array([1, 2, 3])}
    >>> json.dumps(data, cls=NumpyEncoder)
    '{"array": [1, 2, 3]}'
    """
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif hasattr(obj, 'to_json'):
            # カスタムオブジェクトのシリアライズをサポート
            return obj.to_json()
        try:
            # DatetimeやTimedeltaなど、NumPy以外のカスタム型も文字列に変換できる場合
            return str(obj)
        except:
            # その他は親クラスのdefaultメソッドに委譲
            return super(NumpyEncoder, self).default(obj) 