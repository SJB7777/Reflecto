import numpy as np
import time
from dataclasses import dataclass
from genx import fom_funcs
from genx.data import DataList, DataSet
from genx.model import Model
from genx.parameters import Parameters
from .script_builder import GenXScriptBuilder

@dataclass
class XRRConfig:
    """XRR 분석을 위한 물리학적 범용 하이퍼파라미터 설정"""
    wavelength: float = 1.5406
    beam_width: float = 0.1
    sample_len_init: float = 15.0
    
    # Optimizer 설정
    steps_instrument: int = 300
    steps_thickness: int = 800 
    steps_structure: int = 1000
    steps_fine: int = 1200
    pop_size: int = 40

class GenXFitter:
    def __init__(self, q: np.ndarray, R: np.ndarray, nn_params, config: XRRConfig = None):
        self.q = q
        max_r = np.nanmax(R) if np.nanmax(R) > 0 else 1.0
        self.R = np.nan_to_num(R / max_r, nan=1e-12, posinf=1e-12)
        self.nn_params = nn_params
        self.config = config if config else XRRConfig()
        self.model = self._initialize_genx_model()
        self.pars_map = {}

    def _initialize_genx_model(self) -> Model:
        ds = DataSet(name="Reflecto_Ultimate_Fitter")
        ds.x_raw, ds.y_raw = self.q, self.R
        ds.error_raw = np.maximum(self.R * 0.1, 1e-9)
        ds.run_command()
        
        model = Model()
        model.data = DataList([ds])

        init_vals = {
            'f_d': float(self.nn_params.thickness),
            'f_sig': 3.0,
            'f_sld': float(self.nn_params.sld),
            's_len': self.config.sample_len_init,
            'beam_w': self.config.beam_width,
            'i0': 1.0,
            'ibkg': 1e-7
        }
        builder = GenXScriptBuilder()
        model.set_script(builder.build(init_vals))
        model.compile_script()
        return model

    def _setup_parameters(self):
        pars = Parameters()
        model, cfg = self.model, self.config
        d_ai = float(self.nn_params.thickness)
        sld_ai = float(self.nn_params.sld)

        def add_par(name, val, min_v, max_v, fit=True):
            p = pars.append(name, model)
            p.min, p.max = min_v, max_v
            p.value = np.clip(val, min_v, max_v)
            p.fit = fit
            clean_name = name.replace("v.set_", "set_").replace("v.", "")
            self.pars_map[clean_name] = p
            return p

        # [1] Main Film
        add_par("v.set_f_d", d_ai, max(10.0, d_ai-150.0), d_ai+150.0)
        add_par("v.set_f_sig", 3.0, 0.0, 20.0)
        add_par("v.set_f_sld", sld_ai, max(5.0, sld_ai-15.0), min(150.0, sld_ai+15.0))

        # [2] Substrate Oxide (SiO2) - 로깅 대상
        add_par("v.set_s_d",   15.0, 5.0, 50.0) 
        add_par("v.set_s_sig", 3.0,  0.5, 12.0)
        add_par("v.set_s_sld", 18.8, 16.5, 21.0) 

        # [3] Instrument
        add_par("v.set_i0", 1.0, 0.5, 2.5)
        add_par("v.set_ibkg", 1e-7, 1e-10, 1e-4)
        add_par("v.set_s_len", cfg.sample_len_init, 10.0, 45.0) 

        model.parameters = pars

    def run(self, verbose=True):
        self._setup_parameters()
        model, cfg = self.model, self.config
        if verbose: self._print_header()

        # Step 1: BASELINE
        start = time.time()
        model.set_fom_func(fom_funcs.logR1) 
        self._set_active_params(["set_i0", "set_s_len", "set_ibkg"])
        model.bumps_update_parameters(model.bumps_fit(method="de", steps=cfg.steps_instrument, pop=15))
        if verbose: self._print_status("BASELINE", time.time() - start)

        # Step 2: THICKNESS
        start = time.time()
        self._set_active_params(["set_f_d", "set_ibkg"]) 
        model.bumps_update_parameters(model.bumps_fit(method="de", steps=cfg.steps_thickness, pop=cfg.pop_size))
        if verbose: self._print_status("THICKNESS", time.time() - start)

        # Step 3: STRUCTURE
        start = time.time()
        self.pars_map['set_f_sld'].min, self.pars_map['set_f_sld'].max = 5.0, 150.0
        self._set_active_params(["set_f_d", "set_f_sld", "set_f_sig", "set_s_d"])
        model.bumps_update_parameters(model.bumps_fit(method="de", steps=cfg.steps_structure, pop=cfg.pop_size))
        if verbose: self._print_status("STRUCTURE", time.time() - start)

        # Step 4: FINAL
        start = time.time()
        self._set_active_params(["set_f_d", "set_f_sld", "set_f_sig", "set_s_d", "set_s_sld", "set_s_sig", "set_i0", "set_ibkg", "set_s_len"])
        model.bumps_update_parameters(model.bumps_fit(method="de", steps=cfg.steps_fine, pop=cfg.pop_size))
        if verbose: 
            self._print_status("FINAL", time.time() - start)
            print("="*180 + "\n")

        return self._collect_results()

    def _set_active_params(self, active_names: list[str]):
        for p in self.model.parameters: p.fit = False
        for name in active_names:
            if name in self.pars_map: self.pars_map[name].fit = True

    def _print_header(self):
        # SiO2 컬럼 추가로 인한 헤더 확장
        print("\n" + "="*180)
        print(f"{'Fitting Step':^15} | {'FOM (logR1)':^20} | {'Thick':^7} | {'Rough':^5} | {'SLD':^5} | {'SiO2_d':^6} | {'SiO2_s':^6} | {'SiO2_sld':^8} | {'I0':^5} | {'L':^4} | {'Bkg':^5} | {'Time':^7}")
        print("-" * 180)

    def _print_status(self, step_name: str, elapsed: float):
        self.model.evaluate_sim_func()
        p = self.pars_map
        fom_val = self.model.fom
        log_fom = np.log10(fom_val) if fom_val > 0 else -np.inf
        b_val = np.log10(max(1e-12, p['set_ibkg'].value))
        
        fom_str = f"{fom_val:.4e} ({log_fom:5.2f})"
        
        # Film & SiO2 & Instrument 통합 로깅
        print(f"   >> [{step_name:^12}] {fom_str:^20} | "
              f"{p['set_f_d'].value:7.2f} | {p['set_f_sig'].value:5.2f} | {p['set_f_sld'].value:5.2f} | "
              f"{p['set_s_d'].value:6.2f} | {p['set_s_sig'].value:6.2f} | {p['set_s_sld'].value:8.2f} | "
              f"{p['set_i0'].value:5.2f} | {p['set_s_len'].value:4.1f} | {b_val:.1f} | {elapsed:6.2f}s")

    def _collect_results(self) -> dict:
        self.model.evaluate_sim_func()
        results = {name: p.value for name, p in self.pars_map.items()}
        results.update({'R_sim': self.model.data[0].y_sim, 'fom': self.model.fom})
        return results