from pathlib import Path

import reflecto 
from reflecto.utils.physics_utils import calc_theoretical_sld, tth2q, sld_to_material_properties
from reflecto.simulate.simulate import ParamSet
from reflecto.simulate.profile import calc_profile
from reflecto.visualize import plot_profile
from reflecto.utils.consts_genx import SUBSTRATE_SI, AIR
from reflecto.utils.math_utils import i0_normalize

def main():
    # --- [Option] 설정 ---
    show_plots = True  # 그래프를 보고 싶지 않으면 False로 설정하세요.
    # -------------------

    print("🚀 Reflecto Test Program Started...")

    # [Step 0] In2O3 물성 정의 및 이론값 계산
    in2o3_total_z = (49 * 2) + (8 * 3) 
    in2o3_molar_mass = (114.818 * 2) + (15.999 * 3)
    target_density = 7.18 

    theo_sld, theo_rho_e = calc_theoretical_sld(target_density, in2o3_total_z, in2o3_molar_mass)

    # [Step 1] 데이터 로드
    data_dir = Path("test_data") / "In2O3_Ar_30nm"
    file_path = data_dir / "#2_xrr.dat"
    
    if not file_path.exists():
        print(f"❌ Error: File not found at {file_path}")
        return

    df = reflecto.read_dat(file_path)
    R = i0_normalize(df["intensity"].to_numpy())
    qs = tth2q(df["tth"].to_numpy())

    # [Step 2] AI 예측 + GenX 피팅 실행
    print("[2] Running Analysis (AI + GenX)...")
    result = reflecto.analyze(qs, R, fit=True, verbose=True)

    # =========================================================================
    # [Step 3] 결과 리포트 (AI Guess & Final Fit)
    # =========================================================================
    print("\n" + "="*50)
    print(" 📊 ANALYSIS REPORT")
    print("="*50)
    
    # 1. AI Initial Guess (요청하신 기능)
    pred = result['prediction']
    print(f"[AI Initial Guess]")
    print(f"  - Thickness : {pred['thickness']:.2f} Å")
    print(f"  - Roughness : {pred['roughness']:.2f} Å")
    print(f"  - SLD       : {pred['sld']:.2f} (x10⁻⁶ Å⁻²)")
    print("-" * 30)

    # 2. Final Fit Parameters
    fit = result['fit_params']
    fit_sld, fit_d, fit_sig = fit['set_f_sld'], fit['set_f_d'], fit['set_f_sig']

    print(f"[Final GenX Fit]")
    print(f"  - Thickness : {fit_d:.2f} Å")
    print(f"  - Roughness : {fit_sig:.2f} Å")
    print(f"  - SLD       : {fit_sld:.2f} (x10⁻⁶ Å⁻²)")
    
    # [검증] 밀도 환산
    calc_rho_e, calc_rho_mass = sld_to_material_properties(fit_sld, in2o3_total_z, in2o3_molar_mass)
    dens_ratio = (calc_rho_mass / target_density) * 100
    
    print("-" * 30)
    print(f"Material Quality: {dens_ratio:.1f}% of Bulk Density")
    print(f"FOM (Fit Error) : {result['fom']:.2e}")
    print("="*50)

    # =========================================================================
    # [Step 4] 시각화 (옵션에 따라 실행)
    # =========================================================================
    if show_plots:
        print("[3] Plotting Result...")
        
        # 1. XRR Curve Fitting Plot (AI Guess와 Final Fit이 모두 그려집니다)
        reflecto.plot_analysis_result(qs, R, result)

        # 2. Electron Density Profile Plot
        ambient = ParamSet.from_genx_layer(AIR)
        final_film = ParamSet(fit_d, fit_sig, fit_sld)
        final_sio2 = ParamSet(fit['set_s_d'], fit['set_s_sig'], fit['set_s_sld'])
        final_substrate = ParamSet.from_genx_layer(SUBSTRATE_SI)
        
        z, sld_profile = calc_profile([ambient, final_film, final_sio2, final_substrate])
        plot_profile(z, sld_profile, title=f"EDP (In2O3, {dens_ratio:.0f}%)")
    else:
        print("[3] Plotting skipped by user option.")

if __name__ == "__main__":
    main()