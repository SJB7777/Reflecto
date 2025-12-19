from pathlib import Path
import numpy as np

from .engines.inference import XRRInferenceEngine
from .engines.fitting import GenXFitter
from .simulate.simulate import ParamSet, param2refl 

PACKAGE_DIR = Path(__file__).parent
DEFAULT_MODEL_DIR = PACKAGE_DIR / "assets"

_GLOBAL_ENGINE = None

def load_default_engine(device: str = None) -> XRRInferenceEngine:
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        if not (DEFAULT_MODEL_DIR / "best.pt").exists():
            raise FileNotFoundError(f"Model weights not found in {DEFAULT_MODEL_DIR}")
        _GLOBAL_ENGINE = XRRInferenceEngine(DEFAULT_MODEL_DIR, device=device)
    return _GLOBAL_ENGINE

def analyze(
    q: np.ndarray, 
    R: np.ndarray, 
    engine: XRRInferenceEngine | None = None,
    fit: bool = True,
    verbose: bool = False
) -> dict:

    if engine is None:
        engine = load_default_engine()

    # 1. AI 추론 (Prediction)
    if verbose:
        print("Running AI Inference...")
    d, sig, sld = engine.predict(q, R)

    ai_params_obj = [ParamSet(d, sig, sld)]
    ai_curve = param2refl(q, ai_params_obj, sio2_param=None)

    result = {
        "prediction": {
            "thickness": float(d),
            "roughness": float(sig),
            "sld": float(sld)
        },
        "ai_curve": ai_curve,
    }

    if fit:
        if verbose:
            print("Refining with GenX...")
        initial_params = ParamSet(d, sig, sld)
        fitter = GenXFitter(q, R, initial_params)
        fit_res = fitter.run(verbose=verbose)
        
        result["fit_params"] = fit_res
        result["fit_curve"] = fitter.model.data[0].y_sim
        result["fom"] = float(fitter.model.fom)

    return result