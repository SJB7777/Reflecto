import matplotlib.pyplot as plt
import numpy as np

from .utils.physics import sld_to_density


def plot_analysis_result(q: np.ndarray, R_measured: np.ndarray, result: dict, title: str = "XRR Analysis"):
    """
    분석 결과를 받아 비교 그래프를 그립니다.

    Args:
        q: q 벡터
        R_measured: 측정된 반사율
        result: reflecto.analyze()의 반환값
    """
    plt.figure(figsize=(10, 6))

    # 1. 측정 데이터 (검은 점)
    plt.plot(q, R_measured, 'ko', label='Measured', markersize=3, alpha=0.6)

    # 2. AI 예측 곡선 (파란 점선)
    if 'ai_curve' in result:
        plt.plot(q, result['ai_curve'], 'b--', label='AI Guess', linewidth=1.5, alpha=0.8)

    # 3. 최종 피팅 곡선 (빨간 실선)
    if 'fit_curve' in result:
        plt.plot(q, result['fit_curve'], 'r-', label='Final Fit', linewidth=2)

    # 그래프 스타일링 (XRR 특화)
    plt.yscale('log')
    plt.xlabel(r'$q \ [\AA^{-1}]$')
    plt.ylabel('Reflectivity')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    
    # 텍스트 정보 추가 (피팅 결과)
    if 'fit_params' in result:
        fp = result['fit_params']
        info_text = (
            f"Thickness: {fp['set_f_d']:.2f} Å\n"
            f"Roughness: {fp['set_f_sig']:.2f} Å\n"
            f"SLD: {fp['set_f_sld']:.2f}"
        )
        # 그래프 오른쪽 상단에 텍스트 박스 배치
        plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


def plot_profile(z: np.ndarray, sld_profile: np.ndarray, title: str = "Electron Density Profile"):
    """
    SLD 프로파일을 받아 Electron Density 그래프로 예쁘게 그립니다.
    """
    # 1. 물리량 변환 (SLD -> Electron Density)
    density_cm3 = sld_to_density(sld_profile)
    
    # 2. 데이터 스케일링 (x10^23)
    y_plot = density_cm3 / 1e23

    # 3. 그래프 스타일 설정
    styles = {
        'title_fontsize': 16, 'title_fontweight': 'bold',
        'label_fontsize': 14, 'label_fontweight': 'bold',
        'tick_labelsize': 12, 'tick_width': 1.5,
        'spine_width': 2.0, 'legend_fontsize': 12
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    # 메인 플롯
    ax.plot(z, y_plot, color='black', lw=2.0, label='Model')
    
    # 영역 채우기 (시각적 효과)
    ax.fill_between(z, y_plot, 0, color='gray', alpha=0.1)

    # --- [스타일 상세 적용] ---
    ax.set_title(title, fontsize=styles['title_fontsize'], weight=styles['title_fontweight'], pad=15)
    ax.set_xlabel(r'Depth from Surface ($\AA$)', fontsize=styles['label_fontsize'], weight=styles['label_fontweight'])
    ax.set_ylabel(r'Electron Density ($e/\mathrm{cm}^3$)', fontsize=styles['label_fontsize'], weight=styles['label_fontweight'])

    # Annotation (x10^23)
    ax.annotate(r'$\times10^{23}$', xy=(0, 1), xytext=(5, 10),
                xycoords='axes fraction', textcoords='offset points',
                fontsize=15, ha='left', va='center')

    # 눈금 스타일 (Inward ticks)
    ax.tick_params(axis='both', which='major', labelsize=styles['tick_labelsize'], 
                   width=styles['tick_width'], length=6, direction='in', top=True, right=True)
    ax.tick_params(axis='both', which='minor', width=1.0, length=3, direction='in', top=True, right=True)

    # 테두리 두께
    for spine in ax.spines.values():
        spine.set_linewidth(styles['spine_width'])

    # 격자
    ax.grid(True, which='major', linestyle='--', linewidth=1.0, alpha=0.5)

    # 축 범위 자동 설정 (0부터 시작하도록)
    ax.set_xlim(z.min(), z.max())
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()
