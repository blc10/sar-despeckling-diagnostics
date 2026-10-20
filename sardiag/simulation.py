"""System-response-aware speckle simulation.

The observation model is Z = |h * (sqrt(B) s + D)|^2 with circular complex Gaussian speckle s, reflectivity B and
deterministic point targets D. The reference is the expected observation X = E[Z] = |h|^2 * B + |h * D|^2, so that Z/X is
unit-mean speckle. The transfer function of h is a band-limited generalized Hamming window along each axis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

PI = math.pi
SIZE = 256
TEXTURE_LEVELS = (1.0, 2.0, 4.0, 8.0, 16.0)
HOMOGENEITY_CV = 0.02
MIN_MASK_PIXELS = 200
EXTRA_DATES = 11
SEEDS = {"eval": 20260929, "dv": 20260930, "cal": 20261003, "stack": 20261004}
SET_SIZES = {"eval": 200, "dv": 80, "cal": 10}
CLASSES = ("structural", "texture")

#: Sentinel-1 IW subswaths: oversampling (sampling rate / processed bandwidth) and window parameter per axis.
SENTINEL1_IW = {
    "IW1": {"os_az": 1.488, "os_rg": 1.139, "a_az": 0.70, "a_rg": 0.75},
    "IW2": {"os_az": 1.554, "os_rg": 1.332, "a_az": 0.75, "a_rg": 0.75},
    "IW3": {"os_az": 1.549, "os_rg": 1.504, "a_az": 0.75, "a_rg": 0.75},
}
#: Umbra spotlight: uniform window, median oversampling of 15 SICD products.
UMBRA_SPOTLIGHT = {"os_az": 1.252, "os_rg": 1.255, "a_az": 1.0, "a_rg": 1.0}
#: SARLO-80 resampled to 80 cm: white speckle.
SARLO80 = {"os_az": 1.0, "os_rg": 1.0, "a_az": 1.0, "a_rg": 1.0}


def response(sensor: str, subswath: str) -> dict:
    """System response of a sensor; for Sentinel-1 the subswath drawn for the phantom is used."""
    if sensor == "s1":
        return SENTINEL1_IW[subswath]
    if sensor == "umbra":
        return UMBRA_SPOTLIGHT
    if sensor == "sarlo":
        return SARLO80
    raise ValueError(f"unknown sensor '{sensor}'")


# ------------------------------------------------------------------------------------------------ system response
def window_1d(n: int, oversampling: float, a: float) -> np.ndarray:
    """Band-limited generalized Hamming weighting on the FFT frequency grid (cycles per sample)."""
    f = np.fft.fftfreq(n)
    half = 0.5 / oversampling
    inside = np.abs(f) <= half
    theta = 2.0 * PI * (f + half) / (2.0 * half)
    return np.where(inside, a - (1.0 - a) * np.cos(theta), 0.0)


def transfer(h: int, w: int, resp: dict) -> np.ndarray:
    """Separable transfer function normalized to preserve the total power."""
    H = np.outer(window_1d(h, resp["os_az"], resp["a_az"]), window_1d(w, resp["os_rg"], resp["a_rg"]))
    return H / math.sqrt(float(np.mean(H ** 2)) + 1e-12)


def impulse_response_1d(oversampling: float, a: float, n: int = 4096, upsample: int = 64) -> np.ndarray:
    """One-dimensional impulse response |h|^2 on a fine grid, normalized to a unit peak."""
    spec = window_1d(n, oversampling, a)
    padded = np.zeros(n * upsample, dtype=np.complex128)
    half = n // 2
    padded[:half] = spec[:half]
    padded[-half:] = spec[-half:]
    h = np.fft.fftshift(np.fft.ifft(padded))
    power = np.abs(h) ** 2
    return power / power.max()


def pslr_irw(oversampling: float, a: float) -> dict:
    """Peak sidelobe ratio in dB and -3 dB impulse response width in samples."""
    upsample = 64
    p = impulse_response_1d(oversampling, a, upsample=upsample)
    c = int(np.argmax(p))
    right = c + 1
    while right + 1 < len(p) and p[right + 1] < p[right]:
        right += 1
    left = c - 1
    while left - 1 > 0 and p[left - 1] < p[left]:
        left -= 1
    side = max(p[:left].max(), p[right + 1:].max())
    above = np.where(p >= 0.5)[0]
    irw = (above.max() - above.min() + 1) / upsample
    return {"pslr_db": float(10 * np.log10(side)), "irw_samples": float(irw)}


def intensity_correlation(oversampling: float, a: float, lag: int, n: int = 4096) -> float:
    """Intensity correlation of speckle at a lag, from the Siegert relation |rho_z(k)|^2."""
    w2 = window_1d(n, oversampling, a) ** 2
    f = np.fft.fftfreq(n)
    return float(abs(np.sum(w2 * np.exp(2j * np.pi * f * lag)) / np.sum(w2)) ** 2)


def lag_correlation(intensity: np.ndarray, lag_az: int, lag_rg: int) -> float:
    """Empirical intensity correlation of an image at a given lag."""
    a = intensity[: intensity.shape[0] - lag_az, : intensity.shape[1] - lag_rg].ravel()
    b = intensity[lag_az:, lag_rg:].ravel()
    return float(np.corrcoef(a, b)[0, 1])


# ------------------------------------------------------------------------------------------------ phantoms
def _line_mask(size: int, rng: np.random.Generator, width: float, length: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    cy, cx = rng.uniform(0.15 * size, 0.85 * size, 2)
    ang = rng.uniform(0.0, PI)
    dy, dx = math.sin(ang), math.cos(ang)
    t = (yy - cy) * dy + (xx - cx) * dx
    d = np.abs(-(yy - cy) * dx + (xx - cx) * dy)
    inside = np.abs(t) <= length / 2.0
    return np.clip(width / 2.0 + 0.5 - d, 0.0, 1.0) * inside


def make_phantom(rng: np.random.Generator, size: int, klass: str) -> dict:
    """Piecewise-constant tiles, a homogeneous square, Gamma texture, linear structures and point targets.

    klass is 'structural' (a local textured patch) or 'texture' (texture everywhere outside the homogeneous square).
    """
    refl = np.zeros((size, size), dtype=np.float64)
    n_tiles = int(rng.integers(3, 6))
    ys = np.linspace(0, size, n_tiles + 1).astype(int)
    xs = np.linspace(0, size, n_tiles + 1).astype(int)
    for i in range(n_tiles):
        for j in range(n_tiles):
            refl[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = float(rng.uniform(0.3, 3.0))

    hy0, hx0 = int(rng.integers(8, size // 3)), int(rng.integers(8, size // 3))
    hy1, hx1 = hy0 + size // 3, hx0 + size // 3
    refl[hy0:hy1, hx0:hx1] = float(rng.uniform(0.8, 1.6))
    homog_box = np.zeros((size, size), dtype=bool)
    homog_box[hy0:hy1, hx0:hx1] = True
    homog = np.zeros((size, size), dtype=bool)
    homog[hy0 + 3:hy1 - 3, hx0 + 3:hx1 - 3] = True

    def gamma_texture(nu: float, shape) -> np.ndarray:
        g = rng.gamma(shape=nu, scale=1.0 / nu, size=shape)
        g = gaussian_filter(g, sigma=2.0, mode="reflect")
        return np.clip(g / float(g.mean()), 1e-3, None)

    if klass == "structural":
        nu = float(rng.choice(TEXTURE_LEVELS[1:]))
        ty0, tx0 = int(rng.integers(0, size // 2)), int(rng.integers(0, size // 2))
        patch = np.zeros((size, size), dtype=bool)
        patch[ty0:ty0 + size // 2, tx0:tx0 + size // 2] = True
        patch &= ~homog_box
        tex = gamma_texture(nu, (size, size))
        refl = np.where(patch, refl * tex, refl)
    else:
        nu = float(rng.choice(TEXTURE_LEVELS))
        tex = gamma_texture(nu, (size, size))
        refl = np.where(homog_box, refl, refl * tex)

    lines = []
    for _ in range(int(rng.integers(2, 5))):
        bright = bool(rng.random() < 0.5)
        width = float(rng.uniform(1.0, 3.0))
        length = float(rng.uniform(60.0, 200.0))
        factor = float(rng.uniform(3.0, 8.0) if bright else rng.uniform(0.05, 0.25))
        m = _line_mask(size, rng, width, length)
        m[homog_box] = 0.0
        refl = refl * (1.0 + (factor - 1.0) * m)
        lines.append({"bright": bright, "width": width, "length": length, "factor": factor})

    points = []
    for _ in range(int(rng.integers(4, 9))):
        py, px = int(rng.integers(6, size - 6)), int(rng.integers(6, size - 6))
        if homog_box[py, px]:
            continue
        power = float(10.0 ** rng.uniform(1.0, 2.0))
        points.append((py, px, power, float(rng.uniform(-PI, PI))))

    return {"background": refl.astype(np.float32), "homog_mask": homog, "points": points,
            "lines": lines, "texture_nu": nu, "klass": klass}


def homogeneous_mask(phantom: dict, size: int = 7) -> np.ndarray:
    """Pixels of the declared homogeneous square that remain flat after all structures have been written."""
    background = np.asarray(phantom["background"], dtype=np.float64)
    declared = np.asarray(phantom["homog_mask"], dtype=bool)
    if not declared.any():
        return declared
    mean = uniform_filter(background, size=size, mode="reflect")
    mean_sq = uniform_filter(background * background, size=size, mode="reflect")
    var = np.maximum(mean_sq - mean * mean, 1e-8)
    cv = np.sqrt(np.maximum(var, 0.0)) / np.maximum(mean, 1e-8)
    return declared & (cv < HOMOGENEITY_CV)


# ------------------------------------------------------------------------------------------------ observations
@dataclass
class Observation:
    clean: np.ndarray        # reference X, float32
    z: np.ndarray            # complex observation, complex64
    coords: list             # point-target coordinates
    point_component: np.ndarray
    scale: float
    H: np.ndarray

    @property
    def intensity(self) -> np.ndarray:
        return (np.abs(self.z) ** 2).astype(np.float32)


def simulate(phantom: dict, resp: dict, rng: np.random.Generator) -> Observation:
    """Single-look complex observation and expected-value reference, both scaled to unit mean observed power."""
    background = phantom["background"].astype(np.float64)
    h, w = background.shape
    H = transfer(h, w, resp)
    s = (rng.standard_normal((h, w)) + 1j * rng.standard_normal((h, w))) / math.sqrt(2.0)
    z_bg = np.fft.ifft2(np.fft.fft2(np.sqrt(background) * s) * H)
    delta = np.zeros((h, w), dtype=np.complex128)
    for (py, px, power, phase) in phantom["points"]:
        delta[py, px] = math.sqrt(power) * np.exp(1j * phase)
    z_pt = np.fft.ifft2(np.fft.fft2(delta) * H)
    z = z_bg + z_pt
    psf = np.abs(np.fft.ifft2(H)) ** 2
    base = np.real(np.fft.ifft2(np.fft.fft2(background) * np.fft.fft2(psf)))
    clean = np.maximum(base, 1e-6) + np.abs(z_pt) ** 2
    scale = math.sqrt(max(float(np.mean(np.abs(z) ** 2)), 1e-8))
    return Observation(clean=(clean / scale ** 2).astype(np.float32), z=(z / scale).astype(np.complex64),
                       coords=[(py, px) for (py, px, _, _) in phantom["points"]], point_component=z_pt, scale=scale, H=H)


def extra_dates(phantom: dict, obs: Observation, class_id: int, index: int, n: int = EXTRA_DATES) -> np.ndarray:
    """Additional dates of the same scene with independent speckle, at the power scale of the target date."""
    background = phantom["background"].astype(np.float64)
    h, w = background.shape
    root = np.sqrt(background)
    out = np.empty((n, h, w), dtype=np.float32)
    for k in range(n):
        rk = np.random.default_rng([SEEDS["stack"], class_id, index, k])
        s = (rk.standard_normal((h, w)) + 1j * rk.standard_normal((h, w))) / math.sqrt(2.0)
        zk = ((np.fft.ifft2(np.fft.fft2(root * s) * obs.H) + obs.point_component) / obs.scale).astype(np.complex64)
        out[k] = (np.abs(zk) ** 2).astype(np.float32)
    return out


def phantom_set(which: str, sensor: str, n: int | None = None, with_stack: bool = False) -> Iterator[dict]:
    """Phantom stream of the study: 'eval' (200 + 200), 'dv' (80 structural) or 'cal' (10 + 10).

    Phantoms and speckle draws are identical for every sensor; only the system response changes. Phantoms with fewer
    than 200 homogeneous pixels or without point targets are skipped, as in the study.
    """
    if which == "dv":
        plans = [(0, "structural", np.random.default_rng(SEEDS["dv"]), n or SET_SIZES["dv"])]
    else:
        plans = [(ci, k, np.random.default_rng(SEEDS[which] + ci), n or SET_SIZES[which]) for ci, k in enumerate(CLASSES)]
    for ci, klass, rng, count in plans:
        for index in range(count):
            subswath = str(rng.choice(sorted(SENTINEL1_IW)))
            phantom = make_phantom(rng, SIZE, klass)
            obs = simulate(phantom, response(sensor, subswath), rng)
            mask = homogeneous_mask(phantom)
            if mask.sum() < MIN_MASK_PIXELS or not obs.coords:
                continue
            item = {"pid": f"{klass}_{index:04d}" if which != "dv" else f"dv_{index:04d}", "klass": klass, "index": index,
                    "subswath": subswath, "phantom": phantom, "clean": obs.clean, "noisy": obs.intensity, "z": obs.z,
                    "coords": obs.coords, "mask": mask}
            if with_stack:
                item["stack"] = extra_dates(phantom, obs, ci, index)
            yield item


def median_bias(n: int) -> float:
    """Expected value of the median of n independent single-look intensities divided by the reflectivity (n odd)."""
    if n % 2 == 0:
        raise ValueError("n must be odd")
    return float(sum(1.0 / j for j in range((n + 1) // 2, n + 1)))
