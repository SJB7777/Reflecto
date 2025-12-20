from pathlib import Path

import reflecto 
from reflecto.utils.physics import calc_theoretical_sld, tth2q, sld_to_material_properties
from reflecto.simulate.simulate import ParamSet
from reflecto.simulate.profile import calc_profile
from reflecto.visualize import plot_profile
from reflecto.utils.consts_genx import SUBSTRATE_SI, AIR
from reflecto.utils.math_utils import i0_normalize

def main():
    print("🚀 Reflecto Test Program Started...")

    # =========================================================================
    # [Step 0] In2O3 물성 정의 및 이론값 계산
    # =========================================================================
    # Chemical Info: In2O3
    # In (Z=49, M=114.818), O (Z=8, M=15.999)
    in2o3_total_z = (49 * 2) + (8 * 3)          # Total Z = 122
    in2o3_molar_mass = (114.818 * 2) + (15.999 * 3) # Molar Mass = 277.633 g/mol
    target_density = 7.18                       # Theoretical Density (g/cm^3)

    # Calculate Theoretical SLD
    theo_sld, theo_rho_e = calc_theoretical_sld(target_density, in2o3_total_z, in2o3_molar_mass)

    print("\n[Physics] Target Material: In2O3")
    print(f" - Molar Mass  : {in2o3_molar_mass:.3f} g/mol")
    print(f" - Total Z     : {in2o3_total_z}")
    print(f" - Theo Density: {target_density} g/cm³")
    print(f" - Theo SLD    : {theo_sld:.2f} (x10⁻⁶ Å⁻²)") 
    print(f" - Theo e-Dens : {theo_rho_e:.3f} e/Å³")
    print("-" * 40)

    # =========================================================================
    # [Step 1] 데이터 로드 및 전처리
    # =========================================================================
    print("[1] Loading Data...")
    # 경로 수정: 실제 데이터가 있는 경로로 설정
    data_dir = Path("test_data") / "In2O3_Ar_30nm"
    file_path = data_dir / "#1_xrr.dat"
    
    # 만약 파일이 없으면 예외 처리 대신 더미 데이터 생성 혹은 안내
    if not file_path.exists():
        print(f"❌ Error: File not found at {file_path}")
        print("   Please check the path or put your .dat file there.")
        return

    df = reflecto.read_dat(file_path)
    R_raw = df["intensity"].to_numpy()
    tth_deg = df["tth"].to_numpy()

    # Preprocessing
    R = i0_normalize(R_raw)
    qs = tth2q(tth_deg)

    # =========================================================================
    # [Step 2] AI 예측 + GenX 피팅 실행
    # =========================================================================
    print("[2] Running Analysis (AI + GenX)...")
    # analyze 함수 내부에서 AI 추론 후 GenX 피팅까지 수행
    result = reflecto.analyze(qs, R, fit=True, verbose=True)

    # =========================================================================
    # [Step 3] 결과 리포트 및 물성 검증
    # =========================================================================
    print("\n" + "="*50)
    print(" 📊 ANALYSIS REPORT")
    print("="*50)
    
    fit = result['fit_params']
    fit_sld = fit['set_f_sld']
    fit_d = fit['set_f_d']
    fit_sig = fit['set_f_sig']

    print(f"Final Fit Parameters:")
    print(f"  - Thickness : {fit_d:.2f} Å")
    print(f"  - Roughness : {fit_sig:.2f} Å")
    print(f"  - SLD       : {fit_sld:.2f} (x10⁻⁶ Å⁻²)")
    
    # [검증] 피팅된 SLD를 다시 밀도로 환산하여 품질 평가
    calc_rho_e, calc_rho_mass = sld_to_material_properties(fit_sld, in2o3_total_z, in2o3_molar_mass)

    print("-" * 30)
    print(f"Material Quality Check (SLD -> Density):")
    print(f"  - Calc Mass Dens : {calc_rho_mass:.2f} g/cm³")
    print(f"  - Calc e-Dens    : {calc_rho_e:.3f} e/Å³")
    
    dens_diff = abs(calc_rho_mass - target_density)
    dens_ratio = (calc_rho_mass / target_density) * 100
    
    print(f"  - Reference Dens : {target_density:.2f} g/cm³")
    print(f"  - Difference     : {dens_diff:.2f} g/cm³ ({dens_ratio:.1f}% of Bulk)")
    
    # 판정 로직 (5% 오차 기준)
    threshold = target_density * 0.05 
    
    if dens_diff > threshold:
        if calc_rho_mass < target_density:
            print("  >> Warning: Low Density. (Possible porous film or stoichiometry issue)")
        else:
            print("  >> Warning: High Density. (Check if metallic In exists or measurement error)")
    else:
        print("  >> Pass: High quality film. Matches theoretical In2O3 properties.")

    print(f"FOM (Fit Error)    : {result['fom']:.2e}")
    print("="*50)

    # =========================================================================
    # [Step 4] 시각화 (피팅 커브 & 전자밀도 프로파일)
    # =========================================================================
    print("[3] Plotting Result...")
    
    # 프로파일 생성을 위한 레이어 정의
    ambient = ParamSet.from_genx_layer(AIR)
    final_film = ParamSet(fit_d, fit_sig, fit_sld)
    final_sio2 = ParamSet(fit['set_s_d'], fit['set_s_sig'], fit['set_s_sld'])
    final_substrate = ParamSet.from_genx_layer(SUBSTRATE_SI)
    
    layers = [ambient, final_film, final_sio2, final_substrate]

    # 1. XRR Curve Fitting Plot
    reflecto.plot_analysis_result(qs, R, result)

    # 2. Electron Density Profile Plot
    z, sld_profile = calc_profile(layers)
    plot_profile(z, sld_profile, title=f"Electron Density Profile (In2O3, {dens_ratio:.0f}%)")

if __name__ == "__main__":
    main()