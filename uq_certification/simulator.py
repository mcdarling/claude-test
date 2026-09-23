"""Toy synthetic-data generator for a UAS (drone) detection sensor.

Stands in for the paper's closed-loop synthetic image pipeline. Every sample
is produced from explicit *scene parameters* (the knobs a synthetic-data
pipeline controls) and turned into a vector of noisy *sensor features* (what
the detector actually sees). Keeping the two separate lets us:

* train the detector only on sensor features, as in a real system;
* attribute uncertainty back to scene parameters ("feature attribution");
* generate new data "similar" to uncertain samples by perturbing their scene
  parameters (the paper's feature-space similarity).

Scene parameters
----------------
drone     1 if a UAS is present, else 0 (the label)
bird      1 if a bird (confuser) is present
size_m    object size in metres
range_km  distance to the object
lighting  0 = night ... 1 = full daylight
weather   0 = clear ... 1 = heavy fog/rain
terrain   index into TERRAINS (sets background clutter and RF noise)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TERRAINS = ("urban", "forest", "maritime", "desert")
CLUTTER = np.array([0.6, 0.9, 0.3, 0.2])  # visual clutter per terrain
RF_NOISE = np.array([0.8, 0.2, 0.3, 0.1])  # RF/acoustic noise per terrain

FEATURE_NAMES = (
    "optical_blob",
    "thermal_blob",
    "rotor_periodicity",
    "motion_regularity",
    "shape_rigidity",
    "clutter_energy",
    "ambient_light",
    "visibility",
)
CONTINUOUS_PARAMS = ("size_m", "range_km", "lighting", "weather")
PARAM_BOUNDS = {
    "size_m": (0.2, 1.5),
    "range_km": (0.1, 2.0),
    "lighting": (0.0, 1.0),
    "weather": (0.0, 1.0),
}

# Operational-domain grid used for structured testing and conformal groups.
LIGHTING_BINS = (("night", 0.0, 0.2), ("dusk", 0.2, 0.45), ("day", 0.45, 1.0001))
WEATHER_BINS = (("clear", 0.0, 0.33), ("moderate", 0.33, 0.66), ("severe", 0.66, 1.0001))


@dataclass
class Scenes:
    """A batch of scene parameters (one entry per sample)."""

    drone: np.ndarray
    bird: np.ndarray
    size_m: np.ndarray
    range_km: np.ndarray
    lighting: np.ndarray
    weather: np.ndarray
    terrain: np.ndarray

    def __len__(self) -> int:
        return len(self.drone)

    def subset(self, idx) -> "Scenes":
        return Scenes(**{k: v[idx] for k, v in vars(self).items()})

    @staticmethod
    def concat(parts: list["Scenes"]) -> "Scenes":
        return Scenes(**{k: np.concatenate([getattr(p, k) for p in parts]) for k in vars(parts[0])})

    def param_matrix(self) -> np.ndarray:
        """Scene parameters as a numeric matrix, used for attribution."""
        return np.column_stack(
            [self.size_m, self.range_km, self.lighting, self.weather, self.terrain, self.bird]
        )


PARAM_MATRIX_NAMES = ("size_m", "range_km", "lighting", "weather", "terrain", "bird")


def sample_scenes(n: int, rng: np.random.Generator, biased: bool = False) -> Scenes:
    """Sample scene parameters.

    ``biased=False`` covers the whole operational domain uniformly (used for
    testing, calibration and candidate pools). ``biased=True`` mimics an
    initial real-world collection campaign: mostly clear daylight in urban
    and desert settings, with little night, fog or forest data.
    """
    drone = rng.integers(0, 2, n)
    if biased:
        lighting = rng.beta(5.0, 1.5, n)
        weather = rng.beta(1.0, 5.0, n)
        terrain = rng.choice(len(TERRAINS), n, p=[0.4, 0.05, 0.15, 0.4])
    else:
        lighting = rng.uniform(0, 1, n)
        weather = rng.uniform(0, 1, n)
        terrain = rng.integers(0, len(TERRAINS), n)
    return Scenes(
        drone=drone,
        bird=_sample_birds(drone, rng),
        size_m=rng.uniform(*PARAM_BOUNDS["size_m"], n),
        range_km=rng.uniform(*PARAM_BOUNDS["range_km"], n),
        lighting=lighting,
        weather=weather,
        terrain=terrain,
    )


def _sample_birds(drone: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Half of the empty scenes contain a bird (the classic confuser);
    # birds occasionally share the frame with a drone.
    p_bird = np.where(drone == 1, 0.2, 0.5)
    return (rng.uniform(size=len(drone)) < p_bird).astype(int)


def perturb_scenes(seeds: Scenes, n_per_seed: int, rng: np.random.Generator, scale: float = 0.08) -> Scenes:
    """Generate new scenes *similar* to ``seeds`` (closed-loop acquisition).

    Continuous parameters are jittered by ``scale`` times their range and the
    terrain is kept. The drone/bird content is resampled so the detector sees
    both classes under the difficult condition.
    """
    idx = np.repeat(np.arange(len(seeds)), n_per_seed)
    base = seeds.subset(idx)
    jittered = {}
    for name in CONTINUOUS_PARAMS:
        lo, hi = PARAM_BOUNDS[name]
        values = getattr(base, name) + rng.normal(0, scale * (hi - lo), len(idx))
        jittered[name] = np.clip(values, lo, hi)
    drone = rng.integers(0, 2, len(idx))
    return Scenes(drone=drone, bird=_sample_birds(drone, rng), terrain=base.terrain, **jittered)


def render_features(scenes: Scenes, rng: np.random.Generator) -> np.ndarray:
    """Turn scene parameters into noisy sensor features (the model's input)."""
    n = len(scenes)
    drone, bird = scenes.drone, scenes.bird
    has_object = np.maximum(drone, bird)
    # Birds are smaller than most drones.
    size = np.where(drone == 1, scenes.size_m, 0.6 * scenes.size_m)
    apparent = size / (0.2 + scenes.range_km)
    clutter = CLUTTER[scenes.terrain]
    rf_noise = RF_NOISE[scenes.terrain]

    optical_vis = scenes.lighting * (1 - 0.8 * scenes.weather)
    optical_snr = apparent * optical_vis / (0.3 + clutter)
    # Thermal imaging works at night but degrades in fog/rain.
    thermal_vis = (1 - 0.85 * scenes.weather) * (0.6 + 0.4 * (1 - scenes.lighting))
    heat = np.where(drone == 1, 1.0, 0.7)
    thermal_snr = apparent * thermal_vis * heat

    def noise(scale):
        return rng.normal(0, 1, n) * scale

    optical_blob = has_object * optical_snr + noise(0.25 + 0.3 * clutter)
    thermal_blob = has_object * thermal_snr + noise(0.25)
    # Rotor signature (RF/acoustic): drone-only, falls off with range.
    rotor = drone * 0.9 / (0.4 + scenes.range_km) * (1 - 0.5 * scenes.weather) + noise(0.15 + 0.4 * rf_noise)
    # Drones fly smoothly, birds flap. Hard to see when the image is poor.
    true_regularity = np.where(drone == 1, 0.8, np.where(bird == 1, 0.2, 0.5))
    motion = true_regularity + noise(0.15 + 0.6 / (1 + 4 * optical_snr))
    shape_signal = np.where(drone == 1, 1.0, np.where(bird == 1, -0.6, 0.0))
    shape = shape_signal * np.minimum(1.0, 1.5 * optical_snr) + noise(0.35)
    # Context the sensor suite can measure about its own conditions.
    clutter_energy = clutter + noise(0.08)
    ambient_light = scenes.lighting + noise(0.05)
    visibility = 1 - scenes.weather + noise(0.05)

    return np.column_stack(
        [optical_blob, thermal_blob, rotor, motion, shape, clutter_energy, ambient_light, visibility]
    )


def generate(n: int, rng: np.random.Generator, biased: bool = False) -> tuple[Scenes, np.ndarray]:
    scenes = sample_scenes(n, rng, biased=biased)
    return scenes, render_features(scenes, rng)


def condition_cell(scenes: Scenes, with_terrain: bool = True) -> np.ndarray:
    """Integer id of each sample's operational-domain cell (lighting x weather x terrain)."""
    light = _bin(scenes.lighting, LIGHTING_BINS)
    weather = _bin(scenes.weather, WEATHER_BINS)
    cell = light * len(WEATHER_BINS) + weather
    if with_terrain:
        cell = cell * len(TERRAINS) + scenes.terrain
    return cell


def cell_label(cell: int) -> str:
    terrain = cell % len(TERRAINS)
    rest = cell // len(TERRAINS)
    light, weather = divmod(rest, len(WEATHER_BINS))
    return f"{LIGHTING_BINS[light][0]}/{WEATHER_BINS[weather][0]}/{TERRAINS[terrain]}"


N_CELLS = len(LIGHTING_BINS) * len(WEATHER_BINS) * len(TERRAINS)


def _bin(values: np.ndarray, bins) -> np.ndarray:
    edges = [b[2] for b in bins[:-1]]
    return np.digitize(values, edges)
