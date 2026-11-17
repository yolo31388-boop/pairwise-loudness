import pytest

from audio.loudness import (
    gated_lufs,
    true_peak,
    limiter_gain,
    channel_gain,
    replaygain_tag,
    resampled_lufs
)


def test_lufs_empty():
    assert gated_lufs([], 48000) == -70.0

def test_lufs_silence():
    assert gated_lufs([0.0] * 100, 48000) == -70.0

def test_lufs_louder():
    assert gated_lufs([1.0] * 100, 48000) > gated_lufs([0.5] * 100, 48000)

def test_lufs_gating_ignores_silence():
    assert gated_lufs([1.0] * 4800 + [0.0] * 4800, 48000) == pytest.approx(gated_lufs([1.0] * 4800, 48000))

def test_true_peak_ge_sample():
    assert true_peak([0.5]) >= 0.5

def test_true_peak_zero():
    assert true_peak([0.0]) == 0.0

def test_limiter_under_ceiling():
    assert limiter_gain(0.0, -1.0) == 0.0

def test_limiter_over_ceiling():
    assert limiter_gain(-6.0, -1.0) < 0.0

def test_channel_stereo():
    assert channel_gain('stereo', -23.0) == {'L': -23.0, 'R': -23.0}

def test_channel_5_1_center():
    assert channel_gain('5.1', -23.0)['C'] == -23.0

def test_replaygain_negative():
    assert replaygain_tag(-6.0) == '-6.00 dB'

def test_replaygain_positive():
    assert replaygain_tag(3.0) == '3.00 dB'

def test_resample_same_rate():
    assert resampled_lufs(-6.0, 48000, 48000) == -6.0

def test_resample_other_rate():
    assert resampled_lufs(-6.0, 44100, 48000) == -6.0

def test_true_peak_oversample():
    assert true_peak([1.0, -1.0], 4) >= 1.0
