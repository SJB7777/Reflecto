from typing import Final, overload, Tuple

import numpy as np
import scipy.constants as const
from numpy.typing import NDArray

r_e: Final[float] = 2.8179403227e-5

@overload
def tth2q(tth: float, wavelen: float = 1.54) -> float: ...

@overload
def tth2q(tth: NDArray[np.float64], wavelen: float = 1.54) -> NDArray[np.float64]: ...

def tth2q(tth, wavelen=1.54):
    """
    Convert 2θ (in degrees) and wavelength (in Å) to scattering vector q (in 1/Å).
    Formula: q = (4 * pi / lambda) * sin(theta)
    """
    th_rad = np.deg2rad(0.5 * tth)
    result = (4 * np.pi / wavelen) * np.sin(th_rad)

    if isinstance(tth, (int, float)):
        return float(result)
    return result

def sld_to_density(sld: float | np.ndarray) -> float | np.ndarray:
    """
    SLD (10^-6 A^-2)를 Electron Density (electrons/cm^3)로 변환합니다.
    Formula: rho_e = (SLD * 1e-6) / r_e * 1e24
    """
    return (sld * 1e-6 / r_e) * 1e24

def calc_theoretical_sld(density: float, total_z: int, molar_mass: float) -> Tuple[float, float]:
    """
    물질의 밀도와 화학 정보를 받아 이론적인 SLD와 전자 밀도를 계산합니다.
    
    Args:
        density (float): 밀도 (g/cm^3)
        total_z (int): 분자당 총 원자 번호 (총 전자 수)
        molar_mass (float): 분자량 (g/mol)
        
    Returns:
        Tuple[float, float]: (SLD [x10^-6 A^-2], Electron Density [e/A^3])
    """
    # 1. 전자 밀도 (electrons/A^3)
    # rho_e = (density * N_A * Total_Z) / Molar_Mass * (1e-24 cm^3/A^3)
    rho_e = (density * const.Avogadro * total_z) / molar_mass * 1e-24

    # 2. SLD (A^-2) -> XRR 툴 호환을 위해 1e6을 곱해 반환
    # SLD = r_e * rho_e
    sld_val = r_e * rho_e * 1e6
    
    return sld_val, rho_e