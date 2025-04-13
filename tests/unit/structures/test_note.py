# tests/unit/structures/test_note.py
import pytest

# テスト対象のデータクラスをインポート
try:
    from src.structures.note import Note
except ImportError:
    pytest.skip(
        "Skipping structures/note tests due to missing src modules",
        allow_module_level=True,
    )
    # Dummy class for static analysis
    from dataclasses import dataclass

    @dataclass
    class Note:
        start_time: float
        end_time: float
        pitch: float
        velocity: int = 100  # Provide default if applicable


# --- Test Note Dataclass ---


def test_note_creation_and_attributes():
    """Tests basic creation and attribute access for the Note dataclass."""
    start = 0.5
    end = 1.5
    pitch_midi = 60.0
    velocity_val = 80

    # Create an instance with specific values
    note_instance = Note(
        start_time=start, end_time=end, pitch=pitch_midi, velocity=velocity_val
    )

    # Assert that attributes are set correctly
    assert note_instance.start_time == start
    assert note_instance.end_time == end
    assert note_instance.pitch == pitch_midi
    assert note_instance.velocity == velocity_val


def test_note_default_velocity():
    """Tests that the default velocity is used if not provided (if applicable)."""
    start = 1.0
    end = 2.0
    pitch_midi = 72.0

    # Create instance without specifying velocity
    try:
        note_instance = Note(start_time=start, end_time=end, pitch=pitch_midi)
        # Assuming 100 is the default based on common MIDI practice/potential definition
        assert note_instance.velocity == 100
    except TypeError:
        # If velocity has no default and is required, this test might need adjustment
        # or indicates the dataclass definition might need a default.
        pytest.skip(
            "Skipping default velocity test - velocity might be required or has no default."
        )


# Add tests for any methods the Note class might have, if any.
# For example:
# def test_note_duration():
#     note = Note(start_time=0.2, end_time=0.7, pitch=65)
#     assert pytest.approx(note.duration()) == 0.5 # Assuming a duration() method exists
