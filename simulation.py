# simulation.py
"""
Simulador Thompson (adaptado para usar core.py).

"""
from typing import Dict, Any
import math
import os
import pandas as pd

# importar funções do core.py (assume que core.py está no mesmo diretório / disponível no PYTHONPATH)
from core import (
    thompson_AB,
    passo_thompson_incremental,
    calcular_temperatura_equilibrio,
    cp_produto_kJkgK,
    hfg_kJkg,
    ue_percent_bs_por_produto_from_UR,
    calcular_propriedades_psicrometricas,
    densidade_grao_kg_m3,
    calcular_fluxo_ar_m3_min_m2,
    carga_transmissao_kJ_min,
    carga_infiltracao_kJ_h_FTA,
    moist_air_cp_kJkgK
)


class ThompsonSimulation:
    """
    Config (exemplo de chaves esperadas em cfg):
      - grao: "Milho" / "Arroz" / "Soja"
      - U0_percent: teor inicial % b.s.
      - U_final_target_percent: objetivo % b.s.
      - T_air: temperatura do ar de secagem (°C)
      - UR_percent: umidade relativa do ar (%) ambiente/entrada
      - P_atm_kPa: pressão atmosférica (kPa)
      - dt_minutes: passo em minutos
      - airflow_m3_min_m2: vazão de ar (m3/min por m2 de área)
      - r1, r2, L, ki: geometria / propriedades de transmissão
      - Tg0: temperatura inicial do grão (°C)
      - area_m2: área da camada (opcional; calculada se r1 fornecido)
      - max_steps, tol_U: controle numérico
    """

    def __init__(self, cfg: Dict[str, Any], out_dir: str = "out"):
        self.cfg = dict(cfg)
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def _prepare_params(self) -> Dict[str, Any]:
        c = self.cfg
        produto = c.get("grao", "Milho")
        U0 = float(c.get("U0_percent", 18.0))
        U_target = float(c.get("U_final_target_percent", 12.0))
        dt_min = float(c.get("dt_minutes", 5.0))
        dt_min = max(dt_min, 1e-6)
        dt_h = dt_min / 60.0
        max_steps = int(c.get("max_steps", 200000))
        tol_U = float(c.get("tol_U", 1e-4))

        T_air = float(c.get("T_air", 60.0))
        UR_percent = float(c.get("UR_percent", 30.0))
        P_atm_kPa = float(c.get("P_atm_kPa", 101.325))

        # psicrometria de entrada
        psych = calcular_propriedades_psicrometricas(T_air, UR_percent, P_atm_kPa)
        W_in = psych["W_kg_kg"]
        rho_air = psych["rho_kg_m3"]
        cp_air_in = moist_air_cp_kJkgK(W_in)

        airflow_m3_min_m2 = float(c.get("airflow_m3_min_m2", 1.0))

        # geometria: prefer r1 + altura_camada; se area fornecida, usa-a
        r1 = float(c.get("r1", 1.0))
        altura_camada = float(c.get("altura_camada", 1.0))
        area_m2 = float(c.get("area_m2", math.pi * r1 * r1))

        V_cam_m3 = area_m2 * altura_camada

        rho_grao = densidade_grao_kg_m3(produto)
        GT = rho_grao * V_cam_m3  # massa total (kg) assumida
        dry_mass = GT / (1.0 + U0 / 100.0)

        Ue = ue_percent_bs_por_produto_from_UR(T_air, UR_percent, produto)

        # geometria transmissao
        r2 = float(c.get("r2", r1 * 1.2))
        L = float(c.get("L", 1.0))
        ki = float(c.get("ki", 0.5))

        Tg0 = float(c.get("Tg0", 25.0))

        return {
            "produto": produto,
            "U0": U0,
            "U_target": U_target,
            "dt_min": dt_min,
            "dt_h": dt_h,
            "max_steps": max_steps,
            "tol_U": tol_U,
            "T_air": T_air,
            "UR_percent": UR_percent,
            "P_atm_kPa": P_atm_kPa,
            "W_in": W_in,
            "rho_air": rho_air,
            "cp_air_in": cp_air_in,
            "airflow_m3_min_m2": airflow_m3_min_m2,
            "area_m2": area_m2,
            "V_cam_m3": V_cam_m3,
            "GT": GT,
            "dry_mass": dry_mass,
            "Ue": Ue,
            "r1": r1,
            "r2": r2,
            "L": L,
            "ki": ki,
            "Tg0": Tg0
        }

    def run(self) -> Dict[str, Any]:
        p = self._prepare_params()

        U = p["U0"]
        Tg = p["Tg0"]
        Ta = p["T_air"]
        t_min = 0.0

        hist = {
            "tempo_min": [],
            "U_percent_bs": [],
            "Tg_C": [],
            "Ta_C": [],
            "Te_C": [],
            "Cp_kJkgK": [],
            "hfg_kJkg": [],
            "W_kg_kg": [],
            "rho_air_kg_m3": [],
            "Q_lat_kJ_h": [],
            "Q_sens_grain_kJ_h": [],
            "Q_trn_kJ_h": [],
            "Q_infil_kJ_h": [],
            "Q_total_kJ_h": []
        }

        convergiu = False
        motivo = "max steps"

        for step in range(p["max_steps"]):
            # propriedades do produto no estado atual
            Cp = cp_produto_kJkgK(U, p["produto"])          # kJ/kg·°C
            hfg = hfg_kJkg(Tg, U, p["produto"])             # kJ/kg

            # calcular Te usando função do core (forma do texto)
            Te = calcular_temperatura_equilibrio(p["T_air"], p["W_in"], Tg, Cp)

            # obter A,B de Thompson (camada fina)
            A, B = thompson_AB(p["T_air"], p["produto"])

            # passo de Thompson (usa U0 real)
            U_new = passo_thompson_incremental(U, p["Ue"], p["U0"], p["dt_h"], A, B)

            # massa de ar que atravessa a área na duração do passo (kg ar seco aproximado)
            m_air_kg = p["airflow_m3_min_m2"] * p["area_m2"] * p["rho_air"] * p["dt_min"]

            # energia sensível disponível no ar durante o passo (kJ)
            # Q_air_sens = m_air * cp_air * (T_air - Tg)  (kJ)
            Q_air_sens_kJ = m_air_kg * p["cp_air_in"] * max(p["T_air"] - Tg, 0.0)

            # umidade evaporada do grão no passo (kg água)
            dU_percent = max(0.0, U - U_new)                    # % b.s.
            m_evap_kg = (dU_percent / 100.0) * p["dry_mass"]    # kg água evaporada

            # energia latente requerida (kJ)
            Q_lat_kJ = m_evap_kg * hfg

            # energia sensível para o grão (kJ) = Q_air_sens - Q_lat (pode ser negativa -> resfriamento)
            Q_sens_grain_kJ = Q_air_sens_kJ - Q_lat_kJ

            # atualização da temperatura do grão (ΔT = Q_sens / (massa_total * Cp))
            if p["GT"] > 0 and Cp > 0:
                dTg = Q_sens_grain_kJ / (p["GT"] * Cp)
            else:
                dTg = 0.0

            Tg_new = Tg + dTg

            # limitar Tg_new entre Te e T_air (não sobe além do ar)
            Tg_new = min(Tg_new, p["T_air"])
            Tg_new = max(Tg_new, Te)

            # atualizar temperatura do ar na camada de forma simples (tempo de resposta arbitrário)
            Ta_new = Ta + (Tg_new - Ta) * (1 - math.exp(-p["dt_h"] / 0.1))

            # converter energias do passo (kJ) para kJ/h para relatório:
            # fator = 60 / dt_min  (pois passo corresponde a dt_min minutos)
            factor = 60.0 / p["dt_min"]
            Q_lat_kJ_h = Q_lat_kJ * factor
            Q_sens_grain_kJ_h = Q_sens_grain_kJ * factor

            # carga por transmissão radial (core retorna kJ/min) -> kJ/h
            Q_trn_kJ_min = carga_transmissao_kJ_min(p["ki"], p["L"], Tg - Ta, p["r1"], p["r2"])
            Q_trn_kJ_h = Q_trn_kJ_min * 60.0

            # infiltração: estimativa rápida (usar ar ambiente 25°C 60% como referência)
            props_amb = calcular_propriedades_psicrometricas(25.0, 60.0, p["P_atm_kPa"])
            props_cam = calcular_propriedades_psicrometricas(Ta, p["UR_percent"], p["P_atm_kPa"])
            delta_H_kJkg = abs(props_amb["H_kJ_kg"] - props_cam["H_kJ_kg"])
            # número de trocas por hora: assume 1 troca/h por padrão (pode ser parâmetro)
            n_change_h = float(self.cfg.get("n_change_h", 1.0))
            Q_infil_kJ_h = carga_infiltracao_kJ_h_FTA(p["V_cam_m3"], n_change_h, delta_H_kJkg, p["rho_air"])

            Q_total_kJ_h = Q_lat_kJ_h + Q_sens_grain_kJ_h + Q_trn_kJ_h + Q_infil_kJ_h

            # salvar histórico (estado antes de atualizar para próximo passo)
            hist["tempo_min"].append(t_min)
            hist["U_percent_bs"].append(U)
            hist["Tg_C"].append(Tg)
            hist["Ta_C"].append(Ta)
            hist["Te_C"].append(Te)
            hist["Cp_kJkgK"].append(Cp)
            hist["hfg_kJkg"].append(hfg)
            hist["W_kg_kg"].append(p["W_in"])
            hist["rho_air_kg_m3"].append(p["rho_air"])
            hist["Q_lat_kJ_h"].append(Q_lat_kJ_h)
            hist["Q_sens_grain_kJ_h"].append(Q_sens_grain_kJ_h)
            hist["Q_trn_kJ_h"].append(Q_trn_kJ_h)
            hist["Q_infil_kJ_h"].append(Q_infil_kJ_h)
            hist["Q_total_kJ_h"].append(Q_total_kJ_h)

            # critérios de parada
            if U_new <= p["U_target"] + p["tol_U"]:
                U_new = p["U_target"]
                convergiu = True
                motivo = f"Umidade alvo {p['U_target']} % b.s. atingida"
                # avançar tempo final e registrar estado final
                t_min += p["dt_min"]
                U = U_new; Tg = Tg_new; Ta = Ta_new
                break

            if U_new <= p["Ue"] + p["tol_U"]:
                U_new = p["Ue"]
                convergiu = True
                motivo = "Umidade de equilíbrio atingida"
                t_min += p["dt_min"]
                U = U_new; Tg = Tg_new; Ta = Ta_new
                break

            # atualizar variáveis para próxima iteração
            U = U_new
            Tg = Tg_new
            Ta = Ta_new
            t_min += p["dt_min"]

        # registrar último ponto se convergiu e não registrado
        if convergiu:
            hist["tempo_min"].append(t_min)
            hist["U_percent_bs"].append(U)
            hist["Tg_C"].append(Tg)
            hist["Ta_C"].append(Ta)
            hist["Te_C"].append(calcular_temperatura_equilibrio(p["T_air"], p["W_in"], Tg, cp_produto_kJkgK(U, p["produto"])))
            hist["Cp_kJkgK"].append(cp_produto_kJkgK(U, p["produto"]))
            hist["hfg_kJkg"].append(hfg_kJkg(Tg, U, p["produto"]))
            hist["W_kg_kg"].append(p["W_in"])
            hist["rho_air_kg_m3"].append(p["rho_air"])
            # recompute last fluxes (best-effort)
            hist["Q_lat_kJ_h"].append(Q_lat_kJ_h)
            hist["Q_sens_grain_kJ_h"].append(Q_sens_grain_kJ_h)
            hist["Q_trn_kJ_h"].append(Q_trn_kJ_h)
            hist["Q_infil_kJ_h"].append(Q_infil_kJ_h)
            hist["Q_total_kJ_h"].append(Q_total_kJ_h)

        df = pd.DataFrame(hist)

        summary = {
            "produto": p["produto"],
            "n_steps": int(len(df)),
            "tempo_final_min": float(df["tempo_min"].iloc[-1]) if not df.empty else t_min,
            "U_final_percent_bs": float(df["U_percent_bs"].iloc[-1]) if not df.empty else U,
            "convergiu": convergiu,
            "motivo_parada": motivo
        }

        return {"df": df, "summary": summary, "plots": []}


# pequeno utilitário de teste (quando executado diretamente)
if __name__ == "__main__":
    default_cfg = {
        "grao": "Milho",
        "U0_percent": 20.0,
        "U_final_target_percent": 14.0,
        "T_air": 60.0,
        "UR_percent": 30.0,
        "P_atm_kPa": 101.325,
        "dt_minutes": 5.0,
        "airflow_m3_min_m2": 1.0,
        "r1": 1.0,
        "altura_camada": 1.0,
        "L": 1.0,
        "ki": 0.5,
        "Tg0": 25.0
    }
    sim = ThompsonSimulation(default_cfg, out_dir="out_test")
    res = sim.run()
    print("Summary:", res["summary"])
    print(res["df"].head())

