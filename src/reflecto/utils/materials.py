import numpy as np
from .physics import r_e 

MATERIAL_DB = {
    "Si":   {"rho": 2.33,  "Z": 14, "M": 28.0855},
    "SiO2": {"rho": 2.20,  "Z": 30, "M": 60.0843},
    "Au":   {"rho": 19.32, "Z": 79, "M": 196.967},
    "Cu":   {"rho": 8.96,  "Z": 29, "M": 63.546},
    "Fe":   {"rho": 7.87,  "Z": 26, "M": 55.845},
    "Air":  {"rho": 0.00,  "Z": 0,  "M": 1.0},
    "H2O":  {"rho": 1.00,  "Z": 10, "M": 18.015},
}

def get_sld_from_material(name: str, density: float = None) -> float:
    """
    물질 이름이나 밀도를 받아 SLD (10^-6 A^-2)를 반환합니다.
    
    Args:
        name: 물질 화학식 (예: 'Si', 'Au')
        density: (선택) 밀도 값 (g/cm^3). 입력 시 DB 값 대신 사용됨.
    """
    if name not in MATERIAL_DB:
        raise ValueError(f"Unknown material: {name}. Available: {list(MATERIAL_DB.keys())}")
    
    mat = MATERIAL_DB[name]
    rho = density if density is not None else mat['rho']
    
    # 아보가드로 수 (mol^-1)
    N_A = 6.02214076e23
    
    # 전자 밀도 (electrons / cm^3)
    # rho_e = (rho * N_A * Z) / M
    rho_e = (rho * N_A * mat['Z']) / mat['M']
    
    # SLD = r_e * rho_e
    # r_e는 usually Angstrom 단위 (approx 2.817e-5 A)
    # rho_e는 cm^-3 이므로 Angstrom^-3으로 변환 필요 (1e-24 곱함)
    
    sld_val = r_e * rho_e * 1e-24
    
    # XRR에서는 보통 10^-6 A^-2 단위를 쓰므로 1e6을 곱해서 반환
    return sld_val * 1e6

def calculate_sld(density: float, Z: int, M: float) -> float:
    """수동으로 물성치를 알 때 SLD 계산"""
    N_A = 6.02214076e23
    rho_e = (density * N_A * Z) / M
    sld = r_e * rho_e * 1e-24
    return sld * 1e6