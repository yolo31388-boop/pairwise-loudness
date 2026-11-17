"""Loudness normalization core, following ITU-R BS.1770-4.

Implements the six pieces needed by the middleware:

* :func:`gated_lufs`      - K-weighted integrated loudness with the BS.1770
                            absolute (-70 LKFS) and relative (-10 LU) gates,
                            so silence/silence-padded sections do not pull the
                            integrated level up.
* :func:`true_peak`       - true-peak detection on an oversampled (default 4x)
                            reconstruction, catching inter-sample peaks that a
                            sample-point scan misses (mastering clipping).
* :func:`limiter_gain`    - ceiling-clamped limiter gain with an adaptive,
                            program-dependent release (slow release for deep,
                            sustained reduction, fast release on transients) to
                            stop pumping on dense drum passages.
* :func:`channel_gain`    - per-channel-layout gains (stereo vs 5.1 keep their
                            own gains; the 5.1 center/dialogue channel is no
                            longer flattened by a stereo-shared gain).
* :func:`resampled_lufs`  - loudness re-integration bookkeeping after sample
                            rate conversion.
* :func:`replaygain_tag`  - standard ReplayGain 2.0 ``REPLAYGAIN_TRACK_GAIN``
                            text ("-6.00 dB"), which current players read.

Pure standard library, deterministic, no numpy/scipy dependency.
"""

from __future__ import annotations

import cmath
import math
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants (ITU-R BS.1770-4)
# ---------------------------------------------------------------------------

#: Integrated loudness reported when no block passes the gates (silence).
SILENCE_LUFS = -70.0

#: Absolute gate threshold, LKFS (BS.1770-4 section 7, step 2).
ABSOLUTE_GATE_LKFS = -70.0

#: Relative gate offset below the absolute-gated mean (BS.1770-4 step 5).
RELATIVE_GATE_OFFSET_LU = -10.0

#: Block / hop lengths in seconds (400 ms blocks, 75 % overlap).
BLOCK_SECONDS = 0.4
HOP_SECONDS = 0.1

#: Block mean-square energy equivalent to the -70 LKFS absolute gate.
#: LKFS = -0.691 + 10*log10(z)  =>  z = 10**((LKFS + 0.691) / 10)
_ABSOLUTE_GATE_ENERGY = 10.0 ** ((ABSOLUTE_GATE_LKFS + 0.691) / 10.0)

#: BS.1770 channel weights (5.1 layout); the LFE channel is not measured.
_CHANNEL_WEIGHTS = {
    "L": 1.0,
    "R": 1.0,
    "C": 1.0,
    "LFE": 0.0,
    "Ls": 1.41,  # sqrt(2), the 1.5 dB surround weight
    "Rs": 1.41,
}

#: Reference sample rate the published K-weighting coefficients target.
_REF_RATE = 48000

# K-weighting biquad coefficients at 48 kHz, normalized so a0 == 1.
# Stage 1: high-shelf ("pre-filter"), +4 dB above 1681.76 Hz.
_SHELF_48K = (
    (1.53512485958697, -2.69169618940638, 1.19839281085285),
    (1.0, -1.69065929318241, 0.73248077421585),
)
# Stage 2: second-order high-pass, ~38 Hz, Q = 0.5.
_HIGHPASS_48K = (
    (1.0, -2.0, 1.0),
    (1.0, -1.99004745483398, 0.99007225036621),
)


# ---------------------------------------------------------------------------
# K-weighting filters (BS.1770-4 stage 1 + stage 2), via biquads
# ---------------------------------------------------------------------------

def _quadratic_roots(b: float, c: float) -> Tuple[complex, complex]:
    """Roots of z^2 + b*z + c = 0."""
    disc = cmath.sqrt(b * b - 4.0 * c)
    return (-b + disc) / 2.0, (-b - disc) / 2.0


def _biquad_at_rate(
    coeffs_ref: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    sample_rate: int,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Map reference 48 kHz biquad coefficients to ``sample_rate``.

    The published BS.1770 coefficients are only defined at 48 kHz, so the
    analog prototype they discretize is recovered through the poles and
    zeros: invert the bilinear transform on the reference poles/zeros to get
    the continuous-time locations, then apply the bilinear transform again
    with the new rate.  Frequencies therefore stay anchored to the same
    analog values (the ~38 Hz high-pass and the ~1.2 kHz shelf) instead of
    drifting with sample rate.  The overall gain is matched at Nyquist,
    which is well defined for both the shelf and the high-pass (whose DC
    gain is zero).
    """
    if sample_rate == _REF_RATE:
        return coeffs_ref

    (b0, b1, b2), (_, a1, a2) = coeffs_ref
    ref_zeros = _quadratic_roots(b1 / b0, b2 / b0)
    ref_poles = _quadratic_roots(a1, a2)

    k_ref = 2.0 * _REF_RATE
    k_new = 2.0 * sample_rate

    def to_s(z: complex) -> complex:  # inverse bilinear: s = k(z-1)/(z+1)
        return k_ref * (z - 1.0) / (z + 1.0)

    def to_z(s: complex) -> complex:  # forward bilinear: z = (k+s)/(k-s)
        return (k_new + s) / (k_new - s)

    zeros = (to_z(to_s(ref_zeros[0])), to_z(to_s(ref_zeros[1])))
    poles = (to_z(to_s(ref_poles[0])), to_z(to_s(ref_poles[1])))

    # Rebuild real-coefficient polynomials from conjugate pairs.
    den0 = 1.0
    den1 = -(poles[0] + poles[1])
    den2 = poles[0] * poles[1]
    num1 = -(zeros[0] + zeros[1])
    num2 = zeros[0] * zeros[1]

    # Scale the numerator so the Nyquist gain equals the reference Nyquist
    # gain (the denominator is already normalized to a0 == 1).
    ref_hf = abs((b0 - b1 + b2) / (1.0 - a1 + a2))
    new_hf_unnorm = abs((1.0 - num1 + num2) / (den0 - den1 + den2))
    scale = ref_hf / new_hf_unnorm
    numerator = (
        scale,
        (scale * num1).real,
        (scale * num2).real,
    )
    denominator = (1.0, den1.real, den2.real)
    return numerator, denominator


def _apply_biquad(samples: Sequence[float], coeffs) -> List[float]:
    """Run one biquad section, direct form II transposed (a0 == 1)."""
    (b0, b1, b2), (_, a1, a2) = coeffs
    z1 = 0.0
    z2 = 0.0
    out: List[float] = []
    for x in samples:
        y = b0 * x + z1
        z1 = b1 * x - a1 * y + z2
        z2 = b2 * x - a2 * y
        out.append(y)
    return out


def _k_weight(samples: Sequence[float], sample_rate: int) -> List[float]:
    """Apply the BS.1770 K-weighting (high-shelf then high-pass)."""
    shelf = _biquad_at_rate(_SHELF_48K, sample_rate)
    highpass = _biquad_at_rate(_HIGHPASS_48K, sample_rate)
    return _apply_biquad(_apply_biquad(samples, shelf), highpass)


# ---------------------------------------------------------------------------
# Gated integrated loudness, ITU-R BS.1770-4
# ---------------------------------------------------------------------------

#: Samples at or below this absolute amplitude are treated as digital silence.
#: Programme material is orders of magnitude louder; only true zeros and
#: floating-point noise land here.
_SILENCE_FLOOR = 1e-9


def _material_end(samples: Sequence[float]) -> int:
    """Index just past the last sample above the digital-silence floor.

    The boundary is detected on the *raw* input, before K-weighting, so the
    measurement filters' high-pass ring-down stays inside the material instead
    of decaying into appended silence and producing boundary blocks whose
    diluted energy still clears the -70 LKFS absolute gate.  The loudness
    gates themselves remain block-level, exactly as BS.1770 defines.
    """
    end = len(samples)
    while end > 0 and abs(samples[end - 1]) <= _SILENCE_FLOOR:
        end -= 1
    return end


def _block_energies(weighted: Sequence[float], sample_rate: int) -> List[float]:
    """Mean-square energy of each 400 ms block on the 100 ms hop grid.

    Blocks are rectangular windows.  A final block shorter than 400 ms is
    normalized by its own sample count rather than zero-padded and divided by
    the full block length: padding would dilute the block and make the
    measured level depend on how much silence happens to follow the material.
    Every block is still run through the absolute and relative gates.
    """
    block = max(1, int(round(BLOCK_SECONDS * sample_rate)))
    hop = max(1, int(round(HOP_SECONDS * sample_rate)))
    n = len(weighted)
    energies: List[float] = []
    for start in range(0, n, hop):
        end = min(start + block, n)
        acc = 0.0
        for i in range(start, end):
            v = weighted[i]
            acc += v * v
        energies.append(acc / (end - start))
    return energies


def _gated_level(energies: Sequence[float]) -> float:
    """Apply the BS.1770-4 absolute then relative gates to block energies."""
    # Absolute gate: blocks louder than -70 LKFS.
    absolute = [e for e in energies if e > _ABSOLUTE_GATE_ENERGY]
    if not absolute:
        return SILENCE_LUFS

    # Mean over the absolute-gated blocks sets the relative threshold.
    mean_abs = sum(absolute) / len(absolute)
    lufs_abs = -0.691 + 10.0 * math.log10(mean_abs)
    relative_threshold_energy = 10.0 ** (
        (lufs_abs + RELATIVE_GATE_OFFSET_LU + 0.691) / 10.0
    )

    # Relative gate, then re-integrate over the surviving blocks.
    gated = [e for e in absolute if e > relative_threshold_energy]
    if not gated:
        return SILENCE_LUFS
    return -0.691 + 10.0 * math.log10(sum(gated) / len(gated))


def gated_lufs(
    samples: Sequence[float],
    sample_rate: int,
    material_length: Optional[int] = None,
) -> float:
    """Integrated loudness in LUFS with BS.1770-4 gating.

    The programme material is K-weighted (high-shelf followed by the ~38 Hz
    high-pass), split into overlapping 400 ms blocks on a 100 ms hop grid,
    and the blocks are first filtered by the -70 LKFS absolute gate and then
    by the relative gate (10 LU below the absolute-gated mean).

    Digital silence is excluded from the measurement: trailing silence is
    detected on the raw signal (or bounded explicitly by
    ``material_length``), the K-weighting filters run over the material only
    so their ring-down cannot leak into silent sections, and any silent block
    fails the absolute gate anyway.  Appending silence therefore leaves the
    integrated level unchanged instead of pulling a quiet scene up to match
    louder ones.

    Returns :data:`SILENCE_LUFS` (-70.0) for empty or fully gated input.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not samples:
        return SILENCE_LUFS

    if material_length is None:
        material = _material_end(samples)
    else:
        material = min(max(0, material_length), len(samples))
    if material == 0:
        return SILENCE_LUFS

    # Filter the material only; samples beyond the boundary stay exactly
    # silent instead of carrying the filters' ring-down.
    weighted = _k_weight(samples[:material], sample_rate)
    return _gated_level(_block_energies(weighted, sample_rate))


# ---------------------------------------------------------------------------
# True peak via oversampled reconstruction
# ---------------------------------------------------------------------------

def _windowed_sinc_table(oversample: int, half_taps: int) -> List[List[float]]:
    """Polyphase windowed-sinc interpolation kernels.

    Building the upsampler as zero-stuffing followed by a low-pass, the
    prototype impulse response is ``sinc(pi*t/OS)/OS`` (the 1/OS scaling is
    what makes the interpolator unity-gain instead of an identity network
    that leaves the spectral images in), windowed by a length-``2*half_taps``
    Blackman window.  Each row of the table is one fractional phase of the
    polyphase decomposition.
    """
    table: List[List[float]] = []
    for phase in range(oversample):
        kernel: List[float] = []
        for m in range(-half_taps, half_taps):
            t = m - phase / oversample
            if t == 0.0:
                h = 1.0 / oversample
            else:
                h = (
                    math.sin(math.pi * t / oversample)
                    / (math.pi * t)
                )
            # Blackman window over the 2*half_taps tap span: m/half_taps
            # runs from -1 (exclusive of +1) across the kernel.
            window = (
                0.42
                + 0.5 * math.cos(math.pi * m / half_taps)
                + 0.08 * math.cos(2.0 * math.pi * m / half_taps)
            )
            kernel.append(h * window)
        table.append(kernel)
    return table


def true_peak(samples: Sequence[float], oversample: int = 4) -> float:
    """True-peak level in linear amplitude (always >= the sample-point peak).

    The signal is upsampled ``oversample`` times (default 4x, as recommended
    by BS.1770 for true-peak metering) with a windowed-sinc reconstruction
    filter, so peaks that form *between* the original samples — invisible to a
    scan of sample values and the cause of mastering clipping after D/A
    reconstruction — are detected.  Original samples are included in the
    maximum, so the result can never be lower than the sample-point peak.
    ``oversample=1`` falls back to a plain sample-point peak scan.
    """
    if oversample < 1:
        raise ValueError("oversample must be >= 1")
    if not samples:
        return 0.0

    peak = 0.0
    for x in samples:
        ax = abs(x)
        if ax > peak:
            peak = ax
    if oversample == 1:
        return peak

    half_taps = 16
    table = _windowed_sinc_table(oversample, half_taps)
    n = len(samples)
    for i in range(n):
        for phase in range(1, oversample):
            kernel = table[phase]
            acc = 0.0
            for j, m in enumerate(range(-half_taps, half_taps)):
                idx = i + m
                if 0 <= idx < n:
                    acc += samples[idx] * kernel[j]
            a = abs(acc)
            if a > peak:
                peak = a
    return peak


# ---------------------------------------------------------------------------
# Limiter with adaptive (program-dependent) release
# ---------------------------------------------------------------------------

def _adaptive_release_ms(
    reduction_db: float,
    transient: bool,
    fast_ms: float = 50.0,
    slow_ms: float = 400.0,
) -> float:
    """Program-dependent release time in milliseconds.

    Deep/sustained gain reduction releases slowly, which keeps dense drum
    passages level without the gain bouncing back between hits (pumping).
    Shallow reduction, or an explicit transient, releases fast so transients
    stay open.  The time constant is interpolated by reduction depth between
    ``fast_ms`` and ``slow_ms`` and halved on a transient.
    """
    depth = min(1.0, max(0.0, reduction_db / 12.0))
    base = fast_ms + (slow_ms - fast_ms) * depth
    if transient:
        base *= 0.5
    return max(fast_ms * 0.5, base)


def _smoothing_coeff(release_ms: float, sample_rate: int) -> float:
    """One-pole smoothing coefficient for a given release time."""
    if release_ms <= 0.0 or sample_rate <= 0:
        return 1.0
    return math.exp(-1.0 / (release_ms * 0.001 * sample_rate))


def limiter_gain(
    ceiling_db: float,
    peak_db: float,
    current_gain_db: float = 0.0,
    sample_rate: Optional[int] = None,
    transient: bool = False,
) -> float:
    """Limiter gain in dB (always <= 0) needed to stay under ``ceiling_db``.

    Arguments are ordered ceiling first, then the signal peak::

        gain = min(0, ceiling_db - peak_db)

    so a peak at or below the ceiling gets 0 dB (no limiting) and an
    over-ceiling peak gets a negative reduction.  When ``sample_rate`` is
    given, the gain is smoothed sample-to-sample with the adaptive,
    program-dependent release from :func:`_adaptive_release_ms` rather than a
    fixed release constant: attack is instant (an over-ceiling peak is pulled
    down immediately, never allowed through), and only the release back
    toward 0 dB is smoothed.  Without a sample rate the stateless
    instantaneous gain is returned.
    """
    target = min(0.0, ceiling_db - peak_db)
    if sample_rate is None:
        return target
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    release_ms = _adaptive_release_ms(max(0.0, -target), transient)
    coeff = _smoothing_coeff(release_ms, sample_rate)
    # Instant attack: whenever the required reduction is deeper than the
    # current gain, jump straight to the target.  Only the release (target
    # above the current gain) is smoothed — that asymmetry, with the adaptive
    # time constant, is what removes pumping on dense drum passages.
    if target <= current_gain_db:
        return target
    return current_gain_db + coeff * (target - current_gain_db)


# ---------------------------------------------------------------------------
# Per-channel-layout normalization
# ---------------------------------------------------------------------------

_LAYOUT_CHANNELS: Dict[str, Tuple[str, ...]] = {
    "mono": ("C",),
    "stereo": ("L", "R"),
    "5.1": ("L", "R", "C", "LFE", "Ls", "Rs"),
}


def channel_gain(layout: str, target_gain_db: float) -> Dict[str, float]:
    """Per-channel gain map for a speaker layout.

    Stereo and 5.1 no longer share one gain: each layout is normalized on its
    own BS.1770 weighted loudness (surrounds carry the sqrt(2) / 1.5 dB
    weight, the band-limited LFE effects channel is excluded from the
    measurement), and every channel of a layout receives that layout's
    coherent gain — so the 5.1 center dialogue is no longer flattened by a
    gain derived from stereo content.
    """
    try:
        channels = _LAYOUT_CHANNELS[layout]
    except KeyError as exc:
        raise ValueError(f"unknown channel layout: {layout!r}") from exc

    return {ch: float(target_gain_db) for ch in channels}


def layout_loudness(
    channels: Dict[str, Sequence[float]], sample_rate: int
) -> float:
    """BS.1770 integrated loudness across a whole multi-channel layout.

    Each channel's material is K-weighted separately; per-block energy is
    multiplied by the channel weight (surrounds sqrt(2), LFE excluded) and
    summed, then the combined block energies go through exactly the same
    absolute/relative gating as the mono :func:`gated_lufs` path.  The
    material boundary and block normalization rules match the mono path, so
    trailing silence cannot move the result.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not channels:
        return SILENCE_LUFS

    # Boundary on the raw per-channel data, before filtering (see gated_lufs).
    material = 0
    for name, data in channels.items():
        if _CHANNEL_WEIGHTS.get(name, 0.0) > 0.0 and data:
            material = max(material, _material_end(data))
    if material == 0:
        return SILENCE_LUFS

    weighted = {
        name: _k_weight(data[:material], sample_rate)
        for name, data in channels.items()
        if _CHANNEL_WEIGHTS.get(name, 0.0) > 0.0 and data
    }

    block = max(1, int(round(BLOCK_SECONDS * sample_rate)))
    hop = max(1, int(round(HOP_SECONDS * sample_rate)))
    energies: List[float] = []
    for start in range(0, material, hop):
        end = min(start + block, material)
        acc = 0.0
        for name, w in weighted.items():
            weight = _CHANNEL_WEIGHTS[name]
            sub = 0.0
            stop = min(end, len(w))
            for i in range(start, stop):
                v = w[i]
                sub += v * v
            acc += weight * sub / (stop - start)
        energies.append(acc)

    return _gated_level(energies)


# ---------------------------------------------------------------------------
# Re-integration after sample rate conversion
# ---------------------------------------------------------------------------

def resampled_lufs(
    lufs_before: float,
    source_rate: int,
    target_rate: int,
    samples_after: Optional[Sequence[float]] = None,
) -> float:
    """Loudness after sample-rate conversion.

    Rate conversion itself is level-preserving in the ideal case (a proper
    resampler neither creates nor removes energy), so without the converted
    samples the previously integrated LUFS is returned unchanged rather than
    reusing stale rate-dependent numbers.  In practice anti-alias filtering
    and converter ripple make the integrated level drift, so when the
    converted samples are supplied the loudness is **re-integrated from
    scratch** — K-weighting coefficients remapped to the new rate and the
    full gating re-run via :func:`gated_lufs` — instead of trusting the old
    measurement.
    """
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if samples_after is not None:
        return gated_lufs(samples_after, target_rate)
    return float(lufs_before)


# ---------------------------------------------------------------------------
# Standard ReplayGain 2.0 tag
# ---------------------------------------------------------------------------

def replaygain_tag(track_gain_db: float) -> str:
    """Format a ReplayGain 2.0 ``REPLAYGAIN_TRACK_GAIN`` tag value.

    The standard form is a dB value with two decimals and an explicit unit,
    e.g. ``"-6.00 dB"`` / ``"3.00 dB"``.  No ``"+"`` prefix and no legacy
    field names: this is the ReplayGain 2.0 spelling current players
    actually read, replacing the old-version tags they ignored.
    """
    return f"{track_gain_db:.2f} dB"
