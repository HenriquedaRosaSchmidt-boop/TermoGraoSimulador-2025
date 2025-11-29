# core.py
"""
Módulo core de utilitários para o modelo de Thompson.
Inclui:
 - Função passo_thompson_incremental (corrigida para usar U0 real)
 - calcular_temperatura_equilibrio (forma do texto)
 - propriedades específicas dos produtos (Cp, h_fg, Ue por produto)
 - funções psicrométricas simplificadas (retornam W, rho, H)
 - funções de carga (transmissão, infiltração)
 - helpers gerais (clamp, thompson_AB)
Unidades documentadas em cada função.
"""

import math
from typing import Tuple

# ---------------------------
# Helpers genéricos
# ---------------------------
def clamp(x, a, b):
    """Limita x entre [a,b]."""
    return max(min(x, b), a)

# ---------------------------
# Equações A e B de Thompson (camada fina)
# ---------------------------
def thompson_AB(T_air_c: float, produto: str) -> Tuple[float, float]:
    """
    Retorna A e B de Thompson para camada fina dependendo do produto e T (°C).
    Baseado nas equações das tabelas.
    """
    p = produto.lower()
    T = float(T_air_c)
    if "milho" in p:
        A = -1.7060 + 0.00880 * T
        B = 148.70 * math.exp(-0.0590 * T)
    elif "arroz" in p:
        # as fórmulas na tabela tinham divisão por 60 conforme texto original
        A = (-2445.060 + 82.790 * T - 1.0230 * T ** 2 + 0.0042670 * T ** 3) / 60.0
        B = (-449.680 + 14.520 * T - 0.1820 * T ** 2 + 0.0007560 * T ** 3) / 60.0
    elif "soja" in p:
        A = 0.173590 - 0.02970 * T + 0.00060270 * T ** 2 - 0.0000033330 * T ** 3
        B = -2.26610 + 0.21160 * T - 0.004540 * T ** 2 + 0.000025490 * T ** 3
    else:
        # fallback conservador
        A, B = -1.0, 100.0
    return A, B

# ---------------------------
# Passo de Thompson (corrigido)
# ---------------------------
def passo_thompson_incremental(U_atual_bs: float, Ue_bs: float, U0_bs: float,
                               dt_h: float, A: float, B: float) -> float:
    """
    Realiza um passo incremental do modelo de Thompson usando U0 real.
    Entradas:
      U_atual_bs, Ue_bs, U0_bs: % base seca
      dt_h: passo em horas
      A, B: constantes de Thompson
    Retorna U_next em % base seca (clampado entre Ue e U_atual).
    Observação: segue a formula t = A ln(RU) + B [ln(RU)]^2 com RU=(U-Ue)/(U0-Ue).
    """
    if dt_h <= 0.0:
        return U_atual_bs
    denom = (U0_bs - Ue_bs)
    if abs(denom) < 1e-12:
        # sem diferença entre U0 e Ue -> nada a fazer
        return U_atual_bs

    RU_current = (U_atual_bs - Ue_bs) / denom
    RU_current = max(RU_current, 1e-12)  # proteger log
    ln_RU_current = math.log(RU_current)
    t_current = A * ln_RU_current + B * (ln_RU_current ** 2)

    t_next = t_current + dt_h

    # resolver quadrática em x = ln(RU_next): B x^2 + A x - t_next = 0
    delta = A * A + 4.0 * B * t_next
    if delta < 0:
        # numérico inválido -> não altera
        return U_atual_bs

    # escolher a raiz que produz ln(RU) negativo para reduzir RU (secagem)
    ln_RU_next = (-A - math.sqrt(delta)) / (2.0 * B)
    RU_next = math.exp(ln_RU_next)

    U_next = Ue_bs + RU_next * denom

    # garantir monotonicidade: U deve decrescer em secagem (ou permanecer)
    U_next_clamped = clamp(U_next, Ue_bs, U_atual_bs)
    return U_next_clamped

# ---------------------------
# Temperatura de equilíbrio (Te) conforme o texto
# ---------------------------
def calcular_temperatura_equilibrio(T_air: float, W_in: float, Tg: float, Cp: float) -> float:
    """
    Calcula Te pela expressão do texto:
      Te = ((0.240 + 0.450 W) T_air + Cp * Tg) / (0.240 - 0.450 W + Cp)
    Entradas:
      T_air, Tg em °C
      Cp em kJ/kg·°C
      W_in em kg água/kg ar seco (razão de umidade absoluta)
    Retorna Te em °C.
    Proteção contra denominador ≈ 0.
    """
    num = (0.240 + 0.450 * W_in) * T_air + Cp * Tg
    den = 0.240 - 0.450 * W_in + Cp
    if abs(den) < 1e-12:
        den = 1e-12 if den >= 0 else -1e-12
    return num / den

# ---------------------------
# Propriedades do produto: Cp, h_fg, Ue (por produto)
# Unidades: Cp kJ/kg·°C ; h_fg kJ/kg ; Ue % b.s.
# ---------------------------
def cp_produto_kJkgK(U_percent_bs: float, produto: str) -> float:
    """Calor específico aproximado em kJ/kg·°C com base nas tabelas (U em % b.s.)."""
    p = produto.lower()
    U = float(U_percent_bs)
    # converter U como usado nas fórmulas (U/(100+U) ou U/100+U dependendo)
    if "milho" in p:
        # tabela: C_p = 0,350 + 0,851 × (U/100+U)
        # interpretar como U/(100+U)
        val = 0.350 + 0.851 * (U / (100.0 + U))
    elif "arroz" in p:
        val = 0.2780 + 0.960 * (U / (100.0 + U))
    elif "soja" in p:
        val = 0.3910 + 0.4610 * (U / (100.0 + U))
    else:
        # fallback: valor médio
        val = 0.350 + 0.8 * (U / (100.0 + U))
    return float(val)

def hfg_kJkg(T_c: float, U_percent_bs: float, produto: str) -> float:
    """
    Calor latente de vaporização corrigido por produto.
    Fórmulas aproximadas baseadas na tabela.
    T_c: temperatura em °C ; U_percent_bs: % b.s. (pode influenciar expo)
    """
    p = produto.lower()
    t = float(T_c)
    m = float(U_percent_bs)
    if "milho" in p:
        # hfg = (597.60 - 0.570 t) * (1 + 4.350 * exp(-0.28352 m))
        val = (597.60 - 0.570 * t) * (1.0 + 4.350 * math.exp(-0.28352 * m))
    elif "arroz" in p:
        val = (597.60 - 0.570 * t) * (1.0 + 2.06920 * math.exp(-0.21740 * m))
    elif "soja" in p:
        val = (597.60 - 0.570 * t) * (1.0 + 0.70010 * math.exp(-0.14970 * m))
    else:
        val = (597.60 - 0.570 * t) * (1.0 + 1.0 * math.exp(-0.2 * m))
    return float(val)

def ue_percent_bs_por_produto_from_UR(T_c: float, UR_percent: float, produto: str) -> float:
    """
    Calcula Ue (teor de equilíbrio % b.s.) usando as equações do texto.
    Entrada: T_c (°C), UR_percent (0-100)
    Retorno: Ue em % base seca
    Observação: aplica as fórmulas dadas nas tabelas por produto.
    """
    UR = float(UR_percent) / 100.0
    T = float(T_c)
    p = produto.lower()
    # evitar log(0) quando UR ≈ 1
    one_minus_UR = max(1.0 - UR, 1e-12)

    if "milho" in p:
        # U_e = 120.60 * [(-ln(1-UR) / (T + 45.60))]^{0.50}
        base = (-math.log(one_minus_UR) / (T + 45.60))
        base = max(base, 1e-12)
        Ue = 120.60 * (base ** 0.50)
    elif "arroz" in p:
        # U_e=[ln(1-UR)/(1,91870e-5 * (T+51.1610))]^{0.4090}
        denom = 1.91870e-5 * (T + 51.1610)
        # note o ln(1-UR) é negativo; para ajustar tomo -ln(1-UR)
        base = (-math.log(one_minus_UR) / denom)
        base = max(base, 1e-12)
        Ue = (base ** 0.4090)
    elif "soja" in p:
        denom = 30.5330e-5 * (T + 134.130)
        base = (-math.log(one_minus_UR) / denom)
        base = max(base, 1e-12)
        Ue = (base ** 0.82210)
    else:
        # fallback empírico
        base = (-math.log(one_minus_UR) / (T + 50.0))
        Ue = (100.0 * (base ** 0.5))
    return float(Ue)

# ---------------------------
# Psicrometria simplificada
# ---------------------------
def moist_air_cp_kJkgK(W_kgkg: float) -> float:
    """
    Calor específico molhado do ar (kJ/kg·K) aproximado como função de W.
    W em kg água / kg ar seco.
    cp_air_dry ≈ 1.005 kJ/kg·K; cp_vapor ≈ 1.86 kJ/kg·K
    cp_moist = cp_dry*(1) + W*cp_vapor -> valor médio por kg de massa de mistura.
    """
    cp_dry = 1.005
    cp_vapor = 1.86
    return cp_dry + W_kgkg * cp_vapor

def calcular_propriedades_psicrometricas(T_c: float, UR_percent: float, P_kPa: float) -> dict:
    """
    Versão simplificada que retorna:
      W_kg_kg: razão de mistura (kg água/kg ar seco) approximada
      rho_kg_m3: densidade do ar húmido aproximada (kg/m3)
      H_kJ_kg: entalpia total em kJ/kg (aprox)
    Observações: não é substituto de uma sub-rotina psicrométrica completa,
    mas adequada para integração no simulador.
    """
    # aproximações:
    T_k = T_c + 273.15
    # pressão de vapor de saturação (Pa) — fórmula de Antoine simplificada (Tetens)
    es_Pa = 610.78 * math.exp((17.2694 * T_c) / (T_c + 237.3))
    Pa = P_kPa * 1000.0
    Pv = es_Pa * (UR_percent / 100.0)
    # razão de mistura W (kg/kg) aproximada: W = 0.622 * Pv / (Pa - Pv)
    denom = max(Pa - Pv, 1e-6)
    W = 0.622 * Pv / denom
    # densidade do ar húmido: rho = Pa / (R_specific * T)
    R_dry = 287.058
    rho = Pa / (R_dry * T_k)  # kg/m3 para ar seco approximado
    # entalpia aproximada: h = cp_dry*(T_c) + W*(2501 + cp_vap*T_c)  (kJ/kg)
    cp_dry = 1.005
    cp_vap = 1.86
    h = cp_dry * T_c + W * (2501.0 + cp_vap * T_c)
    return {"W_kg_kg": W, "rho_kg_m3": rho, "H_kJ_kg": h}

# ---------------------------
# Geometria / densidades
# ---------------------------
def densidade_grao_kg_m3(produto: str) -> float:
    """
    Densidade aparente do grão (kg/m3) — valores típicos (aprox).
    Ajuste conforme dados experimentais.
    """
    p = produto.lower()
    if "milho" in p:
        return 720.0
    elif "arroz" in p:
        return 600.0
    elif "soja" in p:
        return 760.0
    else:
        return 700.0

def calcular_fluxo_ar_m3_min_m2(airflow_input) -> float:
    """
    Conversão direta/identidade: se entrada já for m3/min/m2, retorna.
    Caso contrário (por ex. m3/h/m2), tente converter — aqui assumimos que
    usuário passou valor em m3/min/m2.
    """
    return float(airflow_input)

# ---------------------------
# Cargas térmicas
# ---------------------------
def carga_transmissao_kJ_min(k_kW_mK: float, L_m: float, deltaT_C: float, r1_m: float, r2_m: float) -> float:
    """
    Estimativa de carga por condução radial (kJ/min).
    Usa Q = 2π k L deltaT / ln(r2/r1). k em kW/m·K -> 1 kW = 60 kJ/min.
    Retorna kJ/min.
    """
    # converter k (kW/mK) para kJ/min·m·K: 1 kW = 60 kJ/min
    k_kJ_min_mK = k_kW_mK * 60.0
    denom = math.log(max(r2_m, r1_m + 1e-12) / max(r1_m, 1e-12))
    if abs(denom) < 1e-12:
        return 0.0
    Q_kJ_min = 2.0 * math.pi * k_kJ_min_mK * L_m * deltaT_C / denom
    return float(Q_kJ_min)

def carga_infiltracao_kJ_h_FTA(V_cam_m3: float, n_air_changes_per_h: float, delta_H_kJ_kg: float, rho_air_kg_m3: float) -> float:
    """
    Estima carga de infiltração/ventilação:
      Q_infil (kJ/h) = V_cam * n_changes/h * rho_air * delta_H (kJ/kg)
    V_cam_m3: volume do compartimento; n_air_changes_per_h: trocas/h
    delta_H_kJ_kg: diferença de entalpia entre ar externo e interno (kJ/kg)
    rho_air_kg_m3: densidade do ar
    """
    m_air_h = V_cam_m3 * n_air_changes_per_h * rho_air_kg_m3
    Q_kJ_h = m_air_h * delta_H_kJ_kg
    return float(Q_kJ_h)

# ---------------------------
# Conveniências / testes
# ---------------------------
if __name__ == "__main__":
    # pequeno teste de sanidade
    print("Teste rápido core.py")
    A, B = thompson_AB(60.0, "milho")
    print("A,B (milho,60°C):", A, B)
    U0 = 18.0
    Ue = ue_percent_bs_por_produto_from_UR(60.0, 30.0, "milho")
    U_next = passo_thompson_incremental(U0, Ue, U0, 0.1, A, B)
    print("Ue:", Ue, "U_next (0.1h):", U_next)
    print("Te exemplo:", calcular_temperatura_equilibrio(60.0, 0.01, 25.0, cp_produto_kJkgK(18.0, "milho")))
