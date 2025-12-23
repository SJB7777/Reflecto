from dataclasses import dataclass, field

import numpy as np

@dataclass
class Instrument:
    wavelength: float = 1.54
    I0: float = 1.0
    Ibkg: float = 1e-10
    res: float = 0.0001

@dataclass
class Layer:
    thickness: float
    roughness: float
    sld: float
    f: complex = field(default=complex(1, 0), repr=False)

    @property
    def density(self) -> float:
        return self.sld / self.f.real if self.f.real != 0 else 0.0

@dataclass
class Sample:
    layers: list[Layer]
    Ambient: Layer
    Substrate: Layer

@dataclass
class Experiment:
    q: np.ndarray
    R: np.ndarray
    instrument: Instrument
    sample: Sample

@dataclass
class FitParam:
    value: float
    min: float
    max: float
    active: bool = True

