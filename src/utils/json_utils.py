"""
JSONシリアライズ関連のユーティリティ関数

このモジュールは、NumPy配列などをJSONに変換するための
ユーティリティクラスを提供します。
"""

import json
import numpy as np
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """
    NumPy配列やndarray型、PathオブジェクトなどをJSON形式に変換するためのエンコーダー
    
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
            # 複素数の配列を処理
            if np.issubdtype(obj.dtype, np.complexfloating):
                return [{'real': item.real, 'imag': item.imag} for item in obj.tolist()]
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif hasattr(obj, 'to_json'):
            # カスタムオブジェクトのシリアライズをサポート
            return obj.to_json()
        elif isinstance(obj, Path):
            return str(obj)
        # try:
        #     # DatetimeやTimedeltaなど、NumPy以外のカスタム型も文字列に変換できる場合
        #     # この try-except があると、意図せず str() にフォールバックしてしまい、
        #     # シリアライズ不能な型で TypeError が発生しない原因になる
        #     return str(obj)
        # except:
            # その他は親クラスのdefaultメソッドに委譲
        return super(NumpyEncoder, self).default(obj) 