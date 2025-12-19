from pathlib import Path

import reflecto 

from reflecto.utils.physics import calc_theoretical_sld, tth2q 
from reflecto.simulate.simulate import ParamSet
from reflecto.simulate.profile import calc_profile
from reflecto.visualize import plot_profile
from reflecto.utils.consts_genx import SUBSTRATE_SI, AIR
from reflecto.utils.math_utils import i0_normalize

def main():
    print("🚀 Reflecto Test Program Started...")

    # --- [Step 0] In2O3 이론값 계산 (Physics 모듈 활용) ---
    # In2O3 화학 정보 정의
    # In (Z=49, M=114.818), O (Z=8, M=15.999)
    in2o3_total_z = (49 * 2) + (8 * 3)          # 98 + 24 = 122
    in2o3_molar_mass = (114.818 * 2) + (15.999 * 3) # 277.633 g/mol
    target_density = 7.18                       # g/cm^3

    # physics.py의 함수 호출
    theo_sld, theo_rho_e = calc_theoretical_sld(target_density, in2o3_total_z, in2o3_molar_mass)
    
    print("[Physics] Theoretical In2O3 Properties:")
    print(f" - Density : {target_density} g/cm³")
    print(f" - SLD     : {theo_sld:.2f} (x10⁻⁶ Å⁻²)") 
    print(f" - e-Dens  : {theo_rho_e:.3f} e/Å³")
    print("-" * 30)

    print("[1] Loading Data...")
    file = Path(r"D:\03_Resources\Data\XRR_AI\XRR_data\In2O3_Ar_30nm") / "#1_xrr.dat"
    if not file.exists():
        raise FileNotFoundError(f"'{file}' not found.")
    
    df = reflecto.read_dat(file)
    R_raw = df["intensity"].to_numpy()
    tth_deg = df["tth"].to_numpy()
    
    # 전처리
    R = i0_normalize(R_raw)
    qs = tth2q(tth_deg) # physics.py의 함수 사용

    print("[2] Running Analysis (AI + GenX)...")
    result = reflecto.analyze(qs, R, fit=True, verbose=True)

    # 3. 결과 리포트
    print("\n" + "="*40)
    print(" ANALYSIS REPORT")
    print("="*40)
    
    fit = result['fit_params']
    fit_sld = fit['set_f_sld']

    print(f"Final Fit       : d={fit['set_f_d']:.2f}, sig={fit['set_f_sig']:.2f}, sld={fit_sld:.2f}")
    
    # 이론값 검증
    diff = abs(fit_sld - theo_sld)
    print(f"Theoretical     : {theo_sld:.2f}")
    print(f"Difference      : {diff:.2f}")
    
    if diff > 5.0:
        print(">> Warning: High discrepancy. Possible low density or porosity.")
    else:
        print(">> Pass: Fits well with theoretical In2O3 properties.")

    print(f"FOM (Error)     : {result['fom']:.2e}")
    print("="*40)

    # 4. 시각화
    print("[3] Plotting Result...")
    
    # 레이어 구성
    ambient = ParamSet.from_genx_layer(AIR)
    final_film = ParamSet(fit['set_f_d'], fit['set_f_sig'], fit_sld)
    final_sio2 = ParamSet(fit['set_s_d'], fit['set_s_sig'], fit['set_s_sld'])
    final_substrate = ParamSet.from_genx_layer(SUBSTRATE_SI)
    layers = [ambient, final_film, final_sio2, final_substrate]

    reflecto.plot_analysis_result(qs, R, result)

    z, sld_profile = calc_profile(layers)
    plot_profile(z, sld_profile, title="Final Fit Profile (In2O3 Model)")

if __name__ == "__main__":
    main()