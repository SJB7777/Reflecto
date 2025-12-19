import json

from pathlib import Path
import numpy as np
import torch

from .model import XRR1DRegressor
from ..utils.math_utils import powerspace


class XRRPreprocessor:
    """
    Shared class for XRR data preprocessing and inverse transformation.
    (Used by both Dataset and InferenceEngine)
    """
    def __init__(self,
        qs: np.ndarray,
        stats_file: Path | str | None = None,
        device: torch.device = torch.device('cpu')
    ):
        # 1. Set up Master Grid
        self.target_q = qs
        self.device = device
        self.param_mean = None
        self.param_std = None

        if stats_file and Path(stats_file).exists():
            self.load_stats(stats_file)

    def load_stats(self, stats_file):
        """Load statistics file"""
        stats = torch.load(stats_file, map_location=self.device)
        self.param_mean = stats["param_mean"]
        self.param_std = stats["param_std"]

    def process_input(self, q_raw, R_raw):
        """
        Raw Data (Linear q) -> Model Input Tensor (Power q)
        """
        # 1. 데이터 정제 (NaN 방지)

        R_raw = np.nan_to_num(R_raw, nan=1e-15, posinf=1e-15, neginf=1e-15)

        # 2. Normalize (Max=1.0) -> Log Scale
        R_max = np.max(R_raw)
        if R_max <= 0:
            R_max = 1.0 # 0 나누기 방지

        R_norm = R_raw / R_max
        R_log = np.log10(np.maximum(R_norm, 1e-15))

        # 3. 오름차순 정렬 (np.interp는 x가 정렬되어 있어야 함)
        if q_raw[0] > q_raw[-1]:
            q_raw = q_raw[::-1]
            R_log = R_log[::-1]

        padding_val = -15.0

        R_interp = np.interp(self.target_q, q_raw, R_log, left=padding_val, right=padding_val)

        # 모델 그리드(target_q) 중 실측 데이터(q_raw) 범위 안에 있는 것만 유효(1)
        q_valid_mask = (self.target_q >= np.min(q_raw)) & (self.target_q <= np.max(q_raw))

        # 5. Tensor 변환
        R_tensor = torch.from_numpy(R_interp.astype(np.float32))
        mask_tensor = torch.from_numpy(q_valid_mask.astype(np.float32))

        # (2, N) 형태로 반환 [LogR, Mask]
        return torch.stack([R_tensor, mask_tensor], dim=0)

    def denormalize_params(self, params_norm):
        """
        Model Output (Norm) -> Physical Values
        """
        if self.param_mean is None:
            raise ValueError("Statistics file is not loaded.")
        if isinstance(params_norm, torch.Tensor):
            params_norm = params_norm.detach().cpu().numpy()

        # CPU Numpy Operation (Convert to Numpy if mean/std are Tensors)
        # Assumes mean/std are loaded as Tensors in load_stats
        mean = self.param_mean.cpu().numpy() if isinstance(self.param_mean, torch.Tensor) else self.param_mean
        std = self.param_std.cpu().numpy() if isinstance(self.param_std, torch.Tensor) else self.param_std

        return params_norm * std + mean

    def normalize_parameters(self, params_real):
        """Physical Values -> Model Target (Norm)"""
        mean = self.param_mean.numpy() if isinstance(self.param_mean, torch.Tensor) else self.param_mean
        std = self.param_std.numpy() if isinstance(self.param_std, torch.Tensor) else self.param_std

        params_norm = (params_real - mean) / std
        return torch.from_numpy(params_norm.astype(np.float32))


class XRRInferenceEngine:
    def __init__(self, exp_dir: str | Path, device: str | torch.device = None):
        """
        학습된 모델 폴더(exp_dir)를 읽어 추론 엔진을 초기화합니다.
        외부 전역 변수(CONFIG)에 의존하지 않습니다.
        
        Args:
            exp_dir: 모델 파일(best.pt), 설정(config.json), 통계(stats.pt)가 있는 폴더 경로
            device: 실행할 장치 (None이면 자동 설정)
        """
        # 1. 장치 설정
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"[Inference] Device: {self.device}")

        # 2. 경로 유효성 검사
        self.exp_dir = Path(exp_dir)
        if not self.exp_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.exp_dir}")

        self.stats_file = self.exp_dir / "stats.pt"
        self.checkpoint_file = self.exp_dir / "best.pt"
        self.config_file = self.exp_dir / "config.json"

        # 3. 설정 로드 (필수)
        # CONFIG fallback을 제거하고, config.json이 없으면 에러를 띄우는 게 안전합니다.
        if not self.config_file.exists():
            raise FileNotFoundError(
                f"config.json not found in {self.exp_dir}.\n"
                "Inference requires the configuration used during training."
            )
            
        with open(self.config_file, "r") as f:
            self.model_config = json.load(f)
        
        print(f"[Inference] Config loaded from {self.config_file.name}")

        # 4. Master Grid 복원 (config.json 기반)
        # 학습 당시의 Grid 설정을 그대로 가져옵니다.
        try:
            sim_conf = self.model_config["simulation"]
            q_min = sim_conf["q_min"]
            q_max = sim_conf["q_max"]
            n_points = sim_conf["q_points"]
            power = sim_conf["power"]
        except KeyError as e:
            raise KeyError(f"Invalid config.json: Missing key {e}")

        # 모델이 학습된 Grid 생성
        self.target_qs = powerspace(q_min, q_max, n_points, power=power).astype(np.float32)

        # 5. 전처리기 초기화
        # stats.pt 파일이 존재하는지 확인
        if not self.stats_file.exists():
             raise FileNotFoundError(f"Statistics file (stats.pt) not found in {self.exp_dir}")

        self.processor = XRRPreprocessor(
            qs=self.target_qs,
            stats_file=self.stats_file,
            device=self.device
        )

        # 6. 모델 로드
        self._load_model()

    def _load_model(self):
        if not self.checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file (best.pt) not found in {self.exp_dir}")

        # map_location을 사용하여 CPU/GPU 호환성 확보
        ckpt = torch.load(self.checkpoint_file, map_location=self.device, weights_only=True)

        # 모델 아키텍처 파라미터 추출
        # config.json의 "model" 섹션을 사용
        try:
            m_conf = self.model_config["model"]
            model_args = {
                'q_len': len(self.target_qs),
                'input_channels': 2, # 고정값이거나 config에 있다면 m_conf.get('input_channels', 2)
                'n_channels': m_conf["n_channels"],
                'depth': m_conf["depth"],
                'mlp_hidden': m_conf["mlp_hidden"],
                # dropout은 추론 시 사용 안 하므로 기본값 둬도 무방하나 config값 주입 가능
                'dropout': m_conf.get("dropout", 0.0) 
            }
        except KeyError as e:
             raise KeyError(f"Invalid config.json: Missing model parameter {e}")

        # 모델 생성 및 가중치 로드
        self.model = XRR1DRegressor(**model_args).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() # 추론 모드 확정
        
        print("[Inference] Model loaded successfully.")

    def predict(self, q_raw, R_raw) -> tuple[float, float, float]:
        """
        q_raw: 실측 데이터의 q (1D array)
        R_raw: 실측 데이터의 R (1D array)
        Returns: (thickness, roughness, sld)
        """
        # 1. 전처리 (Resampling & Norm)
        # target_qs(학습 Grid)에 맞춰 보간
        x = self.processor.process_input(q_raw, R_raw).unsqueeze(0).to(self.device)

        # 2. 추론
        with torch.no_grad():
            y_pred_norm = self.model(x).squeeze(0)

        # 3. 역정규화 (Denormalize)
        y_pred = self.processor.denormalize_params(y_pred_norm)

        # 4. 결과 반환 (Numpy float or Python float)
        return tuple(y_pred.tolist())
