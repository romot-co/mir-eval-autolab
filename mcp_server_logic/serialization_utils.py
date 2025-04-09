import json
import numpy as np
from pathlib import Path

# JSON Encoder for Numpy types and Path objects
class JsonNumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)): # Convert ndarray to list
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.void)): # Handle void type (e.g., convert to None)
            return None
        elif isinstance(obj, Path): # Convert Path object to string
            return str(obj)
        return super(JsonNumpyEncoder, self).default(obj)

# Potentially add a decoder function here if needed later 