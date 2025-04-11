import pytest
import re
import time # time モジュールをインポート
from unittest.mock import patch, MagicMock

# Assume functions are defined in src.utils.misc_utils
# Provide dummies if the file doesn't exist
try:
    from src.utils.misc_utils import generate_id, get_timestamp
    # Check how datetime is imported in the actual misc_utils.py
    # We need to know the exact object to patch for datetime.now()
    # Let's try to import datetime from there to see if it works
    try:
        # Try importing 'dt' alias first
        from src.utils.misc_utils import dt as datetime_alias
        DATETIME_PATCH_TARGET = 'src.utils.misc_utils.dt'
    except ImportError:
        try:
            # Maybe imported as 'from datetime import datetime, timezone'
            from src.utils.misc_utils import datetime as datetime_direct
            DATETIME_PATCH_TARGET = 'src.utils.misc_utils.datetime'
        except ImportError:
            try:
                # Maybe imported as 'import datetime'
                import src.utils.misc_utils # Import the module itself
                if hasattr(src.utils.misc_utils, 'datetime'):
                     DATETIME_PATCH_TARGET = 'src.utils.misc_utils.datetime'
                else:
                     # Fallback or raise error if patching target cannot be determined
                     DATETIME_PATCH_TARGET = 'datetime' # Hope it's imported globally? Risky.
                     print("Warning: Could not determine how datetime is imported in misc_utils. Patching global 'datetime'.")
            except ImportError:
                 # If src.utils.misc_utils itself doesn't exist
                 raise ImportError("Could not import misc_utils or determine datetime import style.")

except ImportError:
    # Dummy implementation if src.utils.misc_utils doesn't exist
    import uuid
    import datetime as dt # Use alias in dummy

    DATETIME_PATCH_TARGET = 'tests.unit.utils.test_misc_utils.dt' # Patch the dummy's alias

    print(f"Using dummy implementations for misc_utils. Patch target: {DATETIME_PATCH_TARGET}")

    def generate_id(prefix="", length=8):
        # Dummy implementation
        random_part = uuid.uuid4().hex[:length]
        return f"{prefix}{random_part}" if prefix else random_part

    def get_timestamp():
        # Dummy implementation matching the actual one
        return time.time()


# --- generate_id Tests ---

def test_generate_id_default():
    """デフォルト設定でIDが生成されることを確認"""
    generated_id = generate_id()
    assert isinstance(generated_id, str)
    # Default length depends on implementation, check for non-zero length
    assert len(generated_id) > 0
    # Usually hex characters
    assert re.match(r'^[a-f0-9]+$', generated_id)

def test_generate_id_with_prefix():
    """プレフィックス付きでIDが生成されることを確認"""
    prefix = "job_"
    generated_id = generate_id(prefix=prefix)
    assert isinstance(generated_id, str)
    assert generated_id.startswith(prefix)
    id_part = generated_id[len(prefix):]
    assert len(id_part) > 0
    assert re.match(r'^[a-f0-9]+$', id_part)

def test_generate_id_custom_length():
    """カスタム長でIDが生成されることを確認"""
    length = 12
    generated_id = generate_id(length=length)
    assert isinstance(generated_id, str)
    assert len(generated_id) == length
    assert re.match(r'^[a-f0-9]+$', generated_id)

def test_generate_id_uniqueness():
    """複数回生成したIDが一意であることを（確率的に）確認"""
    ids = {generate_id() for _ in range(100)} # Generate 100 IDs
    assert len(ids) == 100 # All generated IDs should be unique

def test_generate_id_prefix_and_length():
    """プレフィックスとカスタム長を同時に指定できることを確認"""
    prefix = "sess_"
    length = 6 # Shorter length for testing
    generated_id = generate_id(prefix=prefix, length=length)
    assert generated_id.startswith(prefix)
    id_part = generated_id[len(prefix):]
    assert len(id_part) == length
    assert re.match(r'^[a-f0-9]+$', id_part)


# --- get_timestamp Tests ---

# Define a fixed timestamp for mocking
MOCK_TIME = 1672531200.123456 # Example: 2023-01-01 00:00:00.123456 UTC

@patch('src.utils.misc_utils.time') # Patch the time module used in misc_utils
def test_get_timestamp_returns_float(mock_time):
    """get_timestampがtime.time()のfloat値を返すことを確認"""
    # Configure the mock for time.time()
    mock_time.time.return_value = MOCK_TIME

    timestamp_val = get_timestamp()

    # Check that time.time() was called
    mock_time.time.assert_called_once()

    # Check the return value type and value
    assert isinstance(timestamp_val, float)
    assert timestamp_val == MOCK_TIME 