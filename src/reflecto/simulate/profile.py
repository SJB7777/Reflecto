import numpy as np
from scipy.special import erf

from ..simulate.simulate import ParamSet

def calc_profile(layers: list, z_step: float = 0.5, padding: float = 30.0):
    """
    ParamSet 리스트를 받아 깊이(z)에 따른 SLD 프로파일을 계산합니다.
    
    Args:
        layers: [ParamSet(Top), ParamSet(Middle), ..., ParamSet(Bottom)] 순서
    """
    # 1. 전체 깊이 계산 (속성 접근 방식 변경: ['d'] -> .thickness)
    total_thickness = sum(L.thickness for L in layers)
    z_max = total_thickness + padding
    z = np.arange(-padding, z_max, z_step)
    
    # 2. 초기값: 맨 위층(보통 Air)의 SLD로 시작
    current_sld = layers[0].sld
    sld_profile = np.full_like(z, current_sld, dtype=float)
    
    current_interface_z = 0.0
    
    # 3. 계면 순회 (Layer i -> Layer i+1)
    for i in range(len(layers) - 1):
        top_layer = layers[i]
        bot_layer = layers[i+1]
        
        # 계면 거칠기 (보통 해당 계면을 형성하는 아래층의 roughness 사용)
        sigma = bot_layer.roughness
        if sigma < 1e-3: sigma = 1e-3
        
        sld_diff = bot_layer.sld - top_layer.sld
        
        # Error Function Smoothing
        transition = 0.5 * (1 + erf((z - current_interface_z) / (np.sqrt(2) * sigma)))
        sld_profile += sld_diff * transition
        
        # 다음 계면 위치: Top Layer의 두께만큼 이동 (단, Air는 두께 0 취급)
        # (주의: 첫번째 층이 Air라면 두께가 0이어야 z=0에서 시작됨)
        d = top_layer.thickness
        current_interface_z += d
        
    return z, sld_profile
