# vertisec_gui_new.py
#
# Dependências: PyQt5, numpy, pandas, matplotlib, psySI, simulation
# Salve como vertisec_gui_new.py e execute no diretório com os módulos do projeto.

from __future__ import annotations
import sys
import os
import math
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QTabWidget, QTableWidget, QTableWidgetItem, QFileDialog,
    QGridLayout, QComboBox, QDoubleSpinBox, QSpinBox, QMessageBox,
    QCheckBox, QProgressBar, QGroupBox, QFormLayout
)

# Tentativa de importar backend / utilitários; usamos com fallback seguro
try:
    from simulation import ThompsonSimulator
except Exception:
    ThompsonSimulator = None

try:
    import psySI
except Exception:
    psySI = None

try:
    from core import carga_transmissao_kJ_min, calcular_propriedades_psicrometricas
except Exception:
    carga_transmissao_kJ_min = None
    try:
        from core import calcular_propriedades_psicrometricas
    except Exception:
        calcular_propriedades_psicrometricas = None

# DEFAULT rcParams -> tema escuro industrial (mantemos para inicialização, será alternado)
matplotlib.rcParams.update({
    "axes.edgecolor": "#A0A0A0",
    "axes.labelcolor": "#E0E0E0",
    "axes.facecolor": "#1B1B1B",
    "grid.color": "#2E2E2E",
    "grid.linestyle": ":",
    "text.color": "#E0E0E0",
    "xtick.color": "#E0E0E0",
    "ytick.color": "#E0E0E0",
    "font.family": "Segoe UI",
    "figure.facecolor": "#1B1B1B"
})

# Nome do projeto conforme solicitado
PROJECT_TITLE = "TermoGrãoSimulador 2025"
APP_VERSION = "2025.1"

# ---------------- Thread para rodar a simulação ----------------
class SimulationThread(QThread):
    finished_signal = pyqtSignal(dict)
    log_signal = pyqtSignal(str)

    def __init__(self, cfg: Dict[str, Any], outdir: str):
        super().__init__()
        self.cfg = cfg
        self.outdir = outdir

    def run(self):
        try:
            if ThompsonSimulator is None:
                raise RuntimeError("Módulo 'simulation' não encontrado.")
            self.log_signal.emit("Iniciando ThompsonSimulator...")
            model = ThompsonSimulator(self.cfg, self.outdir)
            res = model.run()
            self.finished_signal.emit({"result": res})
        except Exception as e:
            import traceback
            self.finished_signal.emit({"error": str(e), "trace": traceback.format_exc()})

# ---------------- GUI ----------------
class VertiSecNew(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{PROJECT_TITLE} — v{APP_VERSION}")
        self.resize(1200, 820)
        self.output_root = os.path.join(os.getcwd(), "termograo_runs")
        os.makedirs(self.output_root, exist_ok=True)
        self.last_result: Optional[Dict[str, Any]] = None
        self._build_ui()
        # iniciar com modo escuro (industrial) aplicado
        self._apply_industrial_theme()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        # Cabeçalho
        header = QHBoxLayout()
        lbl_title = QLabel(f"<b>{PROJECT_TITLE}</b>")
        lbl_title.setStyleSheet("font-size:16px; color:#E0E0E0;")
        header.addWidget(lbl_title)
        header.addStretch()
        self.chk_dark = QCheckBox("Modo escuro")
        self.chk_dark.setChecked(True)
        self.chk_dark.stateChanged.connect(self._toggle_dark)
        header.addWidget(self.chk_dark)
        main_layout.addLayout(header)

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        self.tab_guide = QWidget()
        self.tab_entry = QWidget()
        self.tab_results = QWidget()
        self.tab_plots = QWidget()
        self.tab_export = QWidget()
        self.tabs.addTab(self.tab_guide, "Guia do Usuário")
        self.tabs.addTab(self.tab_entry, "Entrada de Dados")
        self.tabs.addTab(self.tab_results, "Resultados")
        self.tabs.addTab(self.tab_plots, "Gráficos")
        self.tabs.addTab(self.tab_export, "Exportar")

        # Construir abas
        self._build_tab_guide()
        self._build_tab_entry()
        self._build_tab_results()
        self._build_tab_plots()
        self._build_tab_export()

        # Console e progresso
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(140)
        main_layout.addWidget(QLabel("Console Técnico:"))
        main_layout.addWidget(self.log_box)
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        self.setLayout(main_layout)

    # ---------------- Guia do Usuário ----------------
    def _build_tab_guide(self):
        lay = QVBoxLayout(self.tab_guide)

        # Descrições listadas verticalmente (removida a palavra "Tolerância")
        desc_layout = QVBoxLayout()
        descriptions = [
            "Espécie do Grão: Milho, Arroz, Soja, Trigo",
            "Umidade Inicial U₀ [% b.s.] — Umidade do grão (base seca).",
            "Umidade Alvo U_final [% b.s.] — Umidade desejada ao final da secagem.",
            "Temperatura Inicial do Grão Tg₀ [°C].",
            "Temperatura do Ar Tar [°C].",
            "Pressão Atmosférica Patm [kPa].",
            "Umidade Relativa UR [%].",
            "Fluxo de Ar [m³/min/m²].",
            "Geometria: Raio Interno r₁ [m], Raio Externo r₂ [m], Altura da Camada L [m].",
            "K_i (Condutividade térmica do material do secador) [kJ/°C·h·m].",
            "Parâmetros Numéricos: Δt, Máximo de Passos."
        ]
        for line in descriptions:
            lbl = QLabel(line)
            lbl.setStyleSheet("color:#E0E0E0;")
            desc_layout.addWidget(lbl)

        # Container: descrições (esquerda) e cortesia (direita)
        container = QHBoxLayout()
        # esquerda: descrições
        left_widget = QWidget()
        left_widget.setLayout(desc_layout)
        container.addWidget(left_widget, 3)

        # direita: cortesia psicrometria (baseada no HRS_psy_GUI, sem observações nem gráfico)
        psy_group = QGroupBox("Cortesia psicrometria (psySI)")
        psy_form = QFormLayout()
        self.guide_psy_T = QDoubleSpinBox(); self.guide_psy_T.setRange(-50, 200); self.guide_psy_T.setDecimals(7); self.guide_psy_T.setValue(25.0000000)
        self.guide_psy_P = QDoubleSpinBox(); self.guide_psy_P.setRange(20, 500); self.guide_psy_P.setDecimals(7); self.guide_psy_P.setValue(101.3250000)
        self.guide_psy_RH = QDoubleSpinBox(); self.guide_psy_RH.setRange(0, 100); self.guide_psy_RH.setDecimals(7); self.guide_psy_RH.setValue(50.0000000)
        psy_form.addRow("Temperatura DBT (°C):", self.guide_psy_T)
        psy_form.addRow("Pressão (kPa):", self.guide_psy_P)
        psy_form.addRow("Umidade Relativa (%):", self.guide_psy_RH)
        btn_calc = QPushButton("Calcular psicrometria")
        btn_calc.clicked.connect(self._guide_psy_calculate)
        psy_form.addRow(btn_calc)
        self.lbl_psy_out = QLabel("—")
        self.lbl_psy_out.setStyleSheet("color:#E0E0E0;")
        psy_form.addRow(self.lbl_psy_out)
        psy_group.setLayout(psy_form)
        container.addWidget(psy_group, 1)

        lay.addLayout(container)
        self.tab_guide.setLayout(lay)

    def _guide_psy_calculate(self):
        T_c = float(self.guide_psy_T.value())
        P_kpa = float(self.guide_psy_P.value())
        RH_percent = float(self.guide_psy_RH.value())
        if calcular_propriedades_psicrometricas is None and psySI is None:
            QMessageBox.information(self, "psySI ausente", "psySI/core psicrometria não disponível no ambiente.")
            return
        try:
            if calcular_propriedades_psicrometricas is not None:
                props = calcular_propriedades_psicrometricas(T_c, RH_percent, P_kpa)
                # Exibir com 7 casas decimais
                dbt = props.get('DBT_C', T_c)
                W = props.get('W_kg_kg', 0.0)
                H = props.get('H_kJ_kg', 0.0)
                rho = props.get('rho_kg_m3', 0.0)
                self.lbl_psy_out.setText(
                    f"DBT: {dbt:.7f} °C\nW: {W:.7f} kg/kg\nH: {H:.7f} kJ/kg\nrho: {rho:.7f} kg/m³"
                )
            else:
                out = psySI.state("DBT", T_c + 273.15, "RH", RH_percent / 100.0, P_kpa * 1000.0)
                # out expected: DBT_K, H, RH_out, V, W, WBT  (depends on psySI impl)
                DBT_K = out[0]
                H = out[1]
                W = out[4]
                self.lbl_psy_out.setText(
                    f"DBT: {(DBT_K - 273.15):.7f} °C\nW: {W:.7f} kg/kg\nH: {H:.7f} kJ/kg"
                )
        except Exception as e:
            self.lbl_psy_out.setText(f"Erro: {e}")

    # ---------------- Entrada de Dados ----------------
    def _build_tab_entry(self):
        layout = QVBoxLayout(self.tab_entry)

        # Espécie
        form_top = QFormLayout()
        self.cbx_grain = QComboBox(); self.cbx_grain.addItems(["Milho", "Arroz", "Soja", "Trigo"])
        form_top.addRow("Espécie do Grão:", self.cbx_grain)
        layout.addLayout(form_top)

        # Subsecção: Ar de Secagem
        grp_air = QGroupBox("Ar de Secagem")
        f_air = QFormLayout()
        self.spin_T_air = QDoubleSpinBox(); self.spin_T_air.setRange(-50, 300); self.spin_T_air.setDecimals(7); self.spin_T_air.setValue(60.0000000)
        self.spin_P_atm = QDoubleSpinBox(); self.spin_P_atm.setRange(20, 500); self.spin_P_atm.setDecimals(7); self.spin_P_atm.setValue(101.3250000)
        self.spin_UR = QDoubleSpinBox(); self.spin_UR.setRange(0, 100); self.spin_UR.setDecimals(7); self.spin_UR.setValue(35.0000000)
        self.spin_airflow = QDoubleSpinBox(); self.spin_airflow.setRange(0, 100); self.spin_airflow.setDecimals(7); self.spin_airflow.setValue(1.0000000)
        f_air.addRow("Temperatura do Ar (°C):", self.spin_T_air)
        f_air.addRow("Pressão Atmosférica (kPa):", self.spin_P_atm)
        f_air.addRow("Umidade Relativa (%):", self.spin_UR)
        f_air.addRow("Fluxo de Ar (m³/min/m²):", self.spin_airflow)
        grp_air.setLayout(f_air)
        layout.addWidget(grp_air)

        # Subsecção: Propriedades do grão
        grp_grain = QGroupBox("Propriedades do grão")
        f_grain = QFormLayout()
        self.spin_U0 = QDoubleSpinBox(); self.spin_U0.setRange(0, 100); self.spin_U0.setDecimals(7); self.spin_U0.setValue(25.0000000)
        self.spin_U_target = QDoubleSpinBox(); self.spin_U_target.setRange(0, 100); self.spin_U_target.setDecimals(7); self.spin_U_target.setValue(12.0000000)
        self.spin_Tg0 = QDoubleSpinBox(); self.spin_Tg0.setRange(-50, 200); self.spin_Tg0.setDecimals(7); self.spin_Tg0.setValue(25.0000000)
        f_grain.addRow("Umidade Inicial U₀ (% b.s.):", self.spin_U0)
        f_grain.addRow("Umidade Alvo U_final (% b.s.):", self.spin_U_target)
        f_grain.addRow("Temperatura Inicial do Grão Tg₀ (°C):", self.spin_Tg0)
        grp_grain.setLayout(f_grain)
        layout.addWidget(grp_grain)

        # Geometria / Dados da câmara
        grp_geom = QGroupBox("Geometria / Dados da Câmara")
        f_geom = QFormLayout()
        self.spin_r1 = QDoubleSpinBox(); self.spin_r1.setRange(0.001, 100); self.spin_r1.setDecimals(7); self.spin_r1.setValue(1.0000000)
        self.spin_r2 = QDoubleSpinBox(); self.spin_r2.setRange(0.002, 100); self.spin_r2.setDecimals(7); self.spin_r2.setValue(1.2000000)
        self.spin_h_cam = QDoubleSpinBox(); self.spin_h_cam.setRange(0.01, 50); self.spin_h_cam.setDecimals(7); self.spin_h_cam.setValue(2.0000000)
        self.spin_ki = QDoubleSpinBox(); self.spin_ki.setRange(1e-12, 1e6); self.spin_ki.setDecimals(7); self.spin_ki.setValue(0.5000000)
        f_geom.addRow("Raio Interno r₁ (m):", self.spin_r1)
        f_geom.addRow("Raio Externo r₂ (m):", self.spin_r2)
        f_geom.addRow("Altura da Camada L (m):", self.spin_h_cam)
        f_geom.addRow("K_i (kJ/°C·h·m):", self.spin_ki)
        grp_geom.setLayout(f_geom)
        layout.addWidget(grp_geom)

        # Parâmetros numéricos
        grp_num = QGroupBox("Parâmetros Numéricos")
        f_num = QFormLayout()
        self.spin_dt_min = QDoubleSpinBox(); self.spin_dt_min.setRange(0.01, 1440); self.spin_dt_min.setDecimals(7); self.spin_dt_min.setValue(5.0000000)
        self.spin_max_steps = QSpinBox(); self.spin_max_steps.setRange(1, 10000000); self.spin_max_steps.setValue(20000)
        f_num.addRow("Δt (min):", self.spin_dt_min)
        f_num.addRow("Máximo de Passos:", self.spin_max_steps)
        grp_num.setLayout(f_num)
        layout.addWidget(grp_num)

        # Botões
        hbtn = QHBoxLayout()
        self.btn_diagnose = QPushButton("Diagnosticar Entradas")
        self.btn_diagnose.clicked.connect(self._diagnose)
        self.btn_run = QPushButton("▶ Iniciar Simulação")
        self.btn_run.clicked.connect(self._start_simulation)
        hbtn.addWidget(self.btn_diagnose)
        hbtn.addWidget(self.btn_run)
        layout.addLayout(hbtn)

        self.tab_entry.setLayout(layout)

    def _diagnose(self):
        msgs = []
        if self.spin_U0.value() <= self.spin_U_target.value():
            msgs.append("U₀ ≤ U_final — secagem pode não reduzir umidade.")
        if not (10.0 <= self.spin_U_target.value() <= 14.0):
            msgs.append("Umidade alvo fora da faixa recomendada (10–14%).")
        if msgs:
            QMessageBox.warning(self, "Diagnóstico", "\n".join(msgs))
            self._log("⚠️ Diagnóstico: " + " | ".join(msgs))
        else:
            QMessageBox.information(self, "Diagnóstico", "Parâmetros OK.")
            self._log("✅ Diagnóstico: OK")

    # ---------------- Resultados ----------------
    def _build_tab_results(self):
        lay = QVBoxLayout(self.tab_results)
        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setFixedHeight(220)
        lay.addWidget(QLabel("Resumo da Simulação:"))
        lay.addWidget(self.summary_box)
        self.table = QTableWidget()
        lay.addWidget(QLabel("Tabela Iterativa (Resumo dos passos)"))
        lay.addWidget(self.table)
        self.tab_results.setLayout(lay)

    def _format_and_show_summary(self, resumo: Dict[str, Any]):
        texto_resumo = (
            f"⏱️ Tempo Total de Secagem: {resumo['tempo_total_h']:.7f} horas ({resumo['tempo_total_min']:.7f} minutos)\n"
            f"💧 Umidade Inicial: {resumo['umidade_inicial']:.7f}% b.s.\n"
            f"💧 Umidade Final: {resumo['umidade_final']:.7f}% b.s.\n"
            f"🌡️ Temperatura Inicial do Grão: {resumo['temp_inicial_grao']:.7f}°C\n"
            f"🌡️ Temperatura Final do Grão: {resumo['temp_final_grao']:.7f}°C\n"
            f"🌡️ Temperatura de Equilíbrio Final: {resumo['temp_equilibrio_final']:.7f}°C\n"
            f"🔄 Número de Iterações: {resumo['num_iteracoes']}\n"
            f"🎯 Critério de Parada: Umidade alvo do grão atingida.\n"
        )
        self.summary_box.setPlainText(texto_resumo)

    def _populate_table(self, df: pd.DataFrame):
        if df is None or df.empty:
            self.table.clear()
            return
        cols = list(df.columns)
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(df))
        for i, row in df.iterrows():
            for j, col in enumerate(cols):
                self.table.setItem(i, j, QTableWidgetItem(f"{row[col]}"))

    # ---------------- Gráficos ----------------
    def _build_tab_plots(self):
        lay = QVBoxLayout(self.tab_plots)
        # três subplots: umidade, temperaturas, cargas
        self.fig, axs = plt.subplots(3, 1, figsize=(9, 12), constrained_layout=True)
        self.axs = axs
        self.canvas = FigureCanvas(self.fig)
        lay.addWidget(self.canvas)
        self.tab_plots.setLayout(lay)

    def _display_plots(self, df: pd.DataFrame):
        if df is None or df.empty:
            self._log("Sem dados para plotar.")
            return

        # ajustar rcParams e estilos conforme modo
        if self.chk_dark.isChecked():
            plt.style.use('dark_background')
            color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
            line_styles = ['-', '--', '-.', ':', '-', '--']
            markers = [None] * 6
        else:
            # modo claro científico P/B
            matplotlib.rcParams.update({
                "axes.edgecolor": "#000000",
                "axes.labelcolor": "#000000",
                "axes.facecolor": "#FFFFFF",
                "grid.color": "#CCCCCC",
                "grid.linestyle": "-",
                "text.color": "#000000",
                "xtick.color": "#000000",
                "ytick.color": "#000000",
                "font.family": "Segoe UI",
                "figure.facecolor": "#FFFFFF"
            })
            plt.style.use('classic')
            color_cycle = ['#000000'] * 6
            line_styles = ['-', '--', ':', '-.', '-', '--']
            markers = ['o', 's', '^', 'x', 'D', 'v']

        # converte tempo para horas se existir coluna "tempo_min" ou "tempo"
        if "tempo_min" in df.columns:
            t_h = np.array(df["tempo_min"]) / 60.0
        elif "tempo" in df.columns:
            t_h = np.array(df["tempo"]) / 3600.0
        else:
            # fallback: usar índice como tempo em horas
            t_h = np.arange(len(df))

        # Umidade do grão vs tempo + linha alvo
        ax = self.axs[0]; ax.clear()
        if "U_percent_bs" in df.columns:
            ax.plot(t_h, df["U_percent_bs"], label="Umidade do Grão", color=color_cycle[0], linestyle=line_styles[0], marker=markers[0])
        else:
            # fallback: procurar colunas parecidas
            for col in df.columns:
                if "U" in col and "percent" in col.lower():
                    ax.plot(t_h, df[col], label="Umidade do Grão", color=color_cycle[0], linestyle=line_styles[0], marker=markers[0])
                    break
        # linha umidade alvo
        ua = float(self.spin_U_target.value())
        ax.axhline(y=ua, label="Umidade Alvo", color="#FF4500" if self.chk_dark.isChecked() else 'k', linestyle='--')
        ax.set_xlabel("Tempo (h)")
        ax.set_ylabel("Umidade (% b.s.)")
        ax.grid(True)
        ax.legend()

        # Temperaturas: Tg, Ta, Te
        ax = self.axs[1]; ax.clear()
        # tentativas de colunas
        Tg_col = "Tg_C" if "Tg_C" in df.columns else ("Tg" if "Tg" in df.columns else None)
        Ta_col = "Ta_C" if "Ta_C" in df.columns else ("Ta" if "Ta" in df.columns else None)
        Te_col = "Te_C" if "Te_C" in df.columns else ("Te" if "Te" in df.columns else None)
        if Tg_col: ax.plot(t_h, df[Tg_col], label="Tg (Grão)", color=color_cycle[0], linestyle=line_styles[0], marker=markers[0])
        if Ta_col: ax.plot(t_h, df[Ta_col], label="Ta (Ar)", color=color_cycle[1], linestyle=line_styles[1], marker=markers[1])
        if Te_col: ax.plot(t_h, df[Te_col], label="Te (Equilíbrio)", color=color_cycle[2], linestyle=line_styles[2], marker=markers[2])
        ax.set_xlabel("Tempo (h)"); ax.set_ylabel("Temperatura (°C)")
        ax.grid(True); ax.legend()

        # Cargas térmicas: Q_infil (Q_inf), Q_prod, Q_trn, Q_total, Q_med
        ax = self.axs[2]; ax.clear()
        # obter arrays com fallback zeros
        def get_arr(name):
            return np.array(df[name]) if name in df.columns else np.zeros(len(t_h))

        Q_infil = get_arr("Q_infil_kJ_h") if "Q_infil_kJ_h" in df.columns else get_arr("Q_inf_kJ_h") if "Q_inf_kJ_h" in df.columns else np.zeros(len(t_h))
        Q_prod = get_arr("Q_prod_kJ_h") if "Q_prod_kJ_h" in df.columns else np.zeros(len(t_h))
        Q_trn = get_arr("Q_trn_kJ_h") if "Q_trn_kJ_h" in df.columns else np.zeros(len(t_h))
        Q_total = get_arr("Q_total_kJ_h") if "Q_total_kJ_h" in df.columns else np.zeros(len(t_h))

        # Recalcular Q_trn robusto usando L=altura da camada e deltaT = Te - Tg; evitar ln <= 0
        try:
            ki = float(self.spin_ki.value())
            L = float(self.spin_h_cam.value())
            r1 = float(self.spin_r1.value()); r2 = float(self.spin_r2.value())
            # usar Te_col e Tg_col quando presentes
            if Te_col is not None and Tg_col is not None:
                deltaT = np.array(df[Te_col]) - np.array(df[Tg_col])
                eps = 1e-12
                # Para uso em fórmulas logarítmicas, garantir valores válidos:
                deltaT_abs = np.maximum(np.abs(deltaT), eps)
                Q_trn_recomp = []
                for dT in deltaT_abs:
                    if carga_transmissao_kJ_min is not None:
                        # função retorna kJ/min presumivelmente — convertemos para kJ/h
                        qmin = carga_transmissao_kJ_min(ki, L, dT, r1, r2)
                        qh = qmin * 60.0
                    else:
                        # aproximação simples: Q = k * A / L * dT  (kJ/h) (A = lateral área ~ 2π r1 L)
                        A = 2 * math.pi * max(r1, eps) * L
                        k = max(ki, eps)
                        qh = (k * A * dT) / max(L, eps)
                    Q_trn_recomp.append(qh)
                Q_trn_recomp = np.array(Q_trn_recomp)
                # substituir somente se shapes baterem
                if Q_trn_recomp.shape == Q_trn.shape:
                    Q_trn = Q_trn_recomp
        except Exception as e:
            self._log(f"Aviso ao recomputar Q_trn: {e}")

        # Q_med = média elementwise de (Q_infil, Q_prod, Q_trn)
        Q_med = (np.nan_to_num(Q_infil) + np.nan_to_num(Q_prod) + np.nan_to_num(Q_trn)) / 3.0

        # Plotagem com estilos distintos conforme modo
        if self.chk_dark.isChecked():
            ax.plot(t_h, Q_infil, label="Q_inf", color="#9467bd", linestyle=line_styles[3], marker=markers[3])
            ax.plot(t_h, Q_prod, label="Q_prod", color="#2ca02c", linestyle=line_styles[2], marker=markers[2])
            ax.plot(t_h, Q_trn, label="Q_trn", color="#d62728", linestyle=line_styles[1], marker=markers[1])
            ax.plot(t_h, Q_total, label="Q_total", color="#1f77b4", linestyle=line_styles[0], marker=markers[0])
            ax.plot(t_h, Q_med, label="Q_med (média)", color="#d62728", linestyle=':', linewidth=2)  # vermelho pontilhado
        else:
            # modo claro: PB científico — linhas pretas com estilos/markers; Q_med vermelho pontilhado
            ax.plot(t_h, Q_infil, label="Q_inf", color='k', linestyle=line_styles[0], marker=markers[0])
            ax.plot(t_h, Q_prod, label="Q_prod", color='k', linestyle=line_styles[1], marker=markers[1])
            ax.plot(t_h, Q_trn, label="Q_trn", color='k', linestyle=line_styles[2], marker=markers[2])
            ax.plot(t_h, Q_total, label="Q_total", color='k', linestyle=line_styles[3], marker=markers[3])
            ax.plot(t_h, Q_med, label="Q_med (média)", color='r', linestyle=':', linewidth=2)  # sempre vermelho pontilhado

        ax.set_xlabel("Tempo (h)")
        ax.set_ylabel("Carga térmica (kJ/h)")
        ax.grid(True)
        ax.legend()

        self.canvas.draw()
        self._log("Gráficos atualizados.")

    # ---------------- Export ----------------
    def _build_tab_export(self):
        layout = QVBoxLayout(self.tab_export)
        self.btn_save_csv = QPushButton("Salvar .csv"); self.btn_save_csv.clicked.connect(self._export_csv)
        self.btn_save_txt = QPushButton("Salvar .txt"); self.btn_save_txt.clicked.connect(self._export_txt)
        self.btn_save_imgs = QPushButton("Salvar imagens (.png)"); self.btn_save_imgs.clicked.connect(self._export_images)
        layout.addWidget(self.btn_save_csv)
        layout.addWidget(self.btn_save_txt)
        layout.addWidget(self.btn_save_imgs)
        self.tab_export.setLayout(layout)

    def _export_csv(self):
        if not self.last_result:
            QMessageBox.warning(self, "Nenhum resultado", "Execute uma simulação primeiro.")
            return
        df = self.last_result.get("df")
        if df is None or df.empty:
            QMessageBox.warning(self, "Sem dados", "DataFrame vazio.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Salvar CSV", os.getcwd(), "CSV files (*.csv)")
        if path:
            df.to_csv(path, index=False)
            QMessageBox.information(self, "Exportado", f"CSV salvo em: {path}")

    def _export_txt(self):
        if not self.last_result:
            QMessageBox.warning(self, "Nenhum resultado", "Execute uma simulação primeiro.")
            return
        summary = self.last_result.get("summary")
        if summary is None:
            QMessageBox.warning(self, "Sem resumo", "Nenhum resumo disponível.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Salvar TXT", os.getcwd(), "Text files (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(str(summary))
            QMessageBox.information(self, "Exportado", f"TXT salvo em: {path}")

    # ---------------- Simulação ----------------
    def _start_simulation(self):
        cfg = {
            "grao": self.cbx_grain.currentText(),
            "U0_percent": float(self.spin_U0.value()),
            "U_final_target_percent": float(self.spin_U_target.value()),
            "Tg0": float(self.spin_Tg0.value()),
            "T_air": float(self.spin_T_air.value()),
            "UR_percent": float(self.spin_UR.value()),
            "P_atm_kPa": float(self.spin_P_atm.value()),
            "airflow_m3_min_m2": float(self.spin_airflow.value()),
            "r1": float(self.spin_r1.value()),
            "r2": float(self.spin_r2.value()),
            "altura_camada": float(self.spin_h_cam.value()),
            "ki": float(self.spin_ki.value()),
            "dt_minutes": float(self.spin_dt_min.value()),
            "max_steps": int(self.spin_max_steps.value()),
            "tol_U": 1e-4
        }
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = os.path.join(self.output_root, f"run_{now}")
        os.makedirs(outdir, exist_ok=True)
        self._log(f"Iniciando simulação — saída: {outdir}")
        self.btn_run.setEnabled(False)
        self.thread = SimulationThread(cfg, outdir)
        self.thread.log_signal.connect(self._log)
        self.thread.finished_signal.connect(self._on_sim_finished)
        self.thread.start()
        self.progress_bar.setRange(0, 0)

    def _on_sim_finished(self, payload):
        self.progress_bar.setRange(0, 1)
        self.btn_run.setEnabled(True)
        if "error" in payload:
            QMessageBox.critical(self, "Erro de Simulação", payload.get("error", "Erro desconhecido"))
            self._log(f"Erro: {payload.get('error')}")
            return
        res = payload["result"]
        df = res.get("df")
        summary = res.get("summary", {})
        self.last_result = {"df": df, "summary": summary}
        # mapear campos para resumo (garantir presença e formato)
        tempo_min = float(summary.get("tempo_final_min", 0.0))
        resumo_for_text = {
            "tempo_total_h": tempo_min / 60.0,
            "tempo_total_min": tempo_min,
            "umidade_inicial": float(summary.get("U0", float(self.spin_U0.value()))),
            "umidade_final": float(summary.get("U_final", float(self.spin_U_target.value()))),
            "temp_inicial_grao": float(summary.get("Tg0", float(self.spin_Tg0.value()))),
            "temp_final_grao": float(summary.get("Tg_final", df["Tg_C"].iloc[-1] if df is not None and "Tg_C" in df.columns else float(self.spin_Tg0.value()))),
            "temp_equilibrio_final": float(summary.get("Ta_final", df["Te_C"].iloc[-1] if df is not None and "Te_C" in df.columns else float(self.spin_T_air.value()))),
            "num_iteracoes": int(summary.get("n_steps", 0))
        }
        self._format_and_show_summary(resumo_for_text)
        self._populate_table(df)
        self._display_plots(df)
        self._log("Simulação finalizada com sucesso.")
    
    ######################################################
    ####
    # ---------------- A FUNÇÃO ADICIONADA: salvar imagens ----------------
    def _export_images(self):
        """
        Salva imagens da simulação para uma pasta escolhida pelo usuário.
        Comportamento:
         - Se self.last_result['plots'] existir e contiver caminhos de arquivos, copia-os.
         - Caso contrário, salva a figura atual do canvas inteira e também grava
           três PNGs individuais (umidade, temperaturas, cargas) baseados nos eixos atuais.
        Mantém o restante da interface inalterada.
        """
        if not self.last_result:
            QMessageBox.warning(self, "Nenhum resultado", "Execute uma simulação primeiro.")
            return

        choose_dir = QFileDialog.getExistingDirectory(self, "Escolha pasta para salvar imagens", os.getcwd())
        if not choose_dir:
            return

        saved = []
        plots: List[str] = self.last_result.get("plots") or []

        # 1) copiar plots retornados pela simulação (se houver)
        if plots:
            for p in plots:
                try:
                    if os.path.isfile(p):
                        dest = os.path.join(choose_dir, os.path.basename(p))
                        import shutil
                        shutil.copy2(p, dest)
                        saved.append(dest)
                except Exception:
                    # ignorar problemas na cópia de arquivos externos
                    pass

        # 2) se não existirem plots ou para garantir, salvar a figura atual do canvas
        try:
            # salvar figura inteira
            fn_base = os.path.join(choose_dir, f"termograo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            fig_path = fn_base + "_fig.png"
            self.fig.savefig(fig_path, dpi=200, bbox_inches='tight')
            saved.append(fig_path)

            # salvar subplots individuais (umidade, temperaturas, cargas)
            # Subplot 0 -> umidade
            fig0, ax0 = plt.subplots(figsize=(8,4))
            ax_src = self.axs[0]
            for line in ax_src.get_lines():
                ax0.plot(line.get_xdata(), line.get_ydata(), linestyle=line.get_linestyle(), label=line.get_label())
            ax0.set_title("Umidade"); ax0.set_xlabel("Tempo (h)"); ax0.set_ylabel("Umidade (% b.s.)"); ax0.legend(); ax0.grid(True)
            p1 = fn_base + "_umidade.png"; fig0.tight_layout(); fig0.savefig(p1, dpi=150); plt.close(fig0); saved.append(p1)

            # Subplot 1 -> temperaturas
            fig1, ax1 = plt.subplots(figsize=(8,4))
            ax_src = self.axs[1]
            for line in ax_src.get_lines():
                ax1.plot(line.get_xdata(), line.get_ydata(), linestyle=line.get_linestyle(), label=line.get_label())
            ax1.set_title("Temperaturas"); ax1.set_xlabel("Tempo (h)"); ax1.set_ylabel("°C"); ax1.legend(); ax1.grid(True)
            p2 = fn_base + "_temperaturas.png"; fig1.tight_layout(); fig1.savefig(p2, dpi=150); plt.close(fig1); saved.append(p2)

            # Subplot 2 -> cargas
            fig2, ax2 = plt.subplots(figsize=(8,4))
            ax_src = self.axs[2]
            for line in ax_src.get_lines():
                ax2.plot(line.get_xdata(), line.get_ydata(), linestyle=line.get_linestyle(), label=line.get_label())
            ax2.set_title("Cargas"); ax2.set_xlabel("Tempo (h)"); ax2.set_ylabel("kJ/h"); ax2.legend(); ax2.grid(True)
            p3 = fn_base + "_cargas.png"; fig2.tight_layout(); fig2.savefig(p3, dpi=150); plt.close(fig2); saved.append(p3)
        except Exception:
            # se houver erro ao salvar imagens do canvas, ignoramos silenciosamente (não alteramos UI)
            pass

        if saved:
            QMessageBox.information(self, "Imagens salvas", f"{len(saved)} imagens salvas na pasta:\n{choose_dir}")
            self._log(f"Imagens salvas: {saved}")
        else:
            QMessageBox.warning(self, "Nenhuma imagem", "Não foi possível salvar imagens.")

    
    ####
    #######################################################

    # ---------------- Temas e utilitários ----------------
    def _apply_industrial_theme(self):
        # tema industrial (modo escuro)
        self.setStyleSheet("""
            QWidget { background-color:#1B1B1B; color:#E0E0E0; font-family:'Segoe UI'; font-size:11pt; }
            QPushButton { background-color:#3A506B; color:#EEEEEE; border:1px solid #506070; border-radius:4px; padding:6px 12px; }
            QPushButton:hover { background-color:#425C77; }
            QTextEdit { background-color:#2A2A2A; color:#E0E0E0; border:1px solid #444; }
            QGroupBox { border:1px solid #333; margin-top:6px; padding:8px; }
            QGroupBox::title { color:#E0E0E0; subcontrol-origin: margin; left:8px; padding:0 3px 0 3px; }
            QLabel { color:#E0E0E0; }
            QTabBar::tab:selected { background:#3A3A3A; color:#FFFFFF; }
        """)
        # matplotlib rcParams para modo escuro (industrial)
        matplotlib.rcParams.update({
            "axes.edgecolor": "#A0A0A0",
            "axes.labelcolor": "#E0E0E0",
            "axes.facecolor": "#1B1B1B",
            "grid.color": "#2E2E2E",
            "grid.linestyle": ":",
            "text.color": "#E0E0E0",
            "xtick.color": "#E0E0E0",
            "ytick.color": "#E0E0E0",
            "font.family": "Segoe UI",
            "figure.facecolor": "#1B1B1B"
        })

    def _apply_light_theme(self):
        # tema científico / modo claro (conforme solicitado)
        self.setStyleSheet("""
            /* modo claro — tema científico / padrão */
            QWidget {
                background-color: #FFFFFF;
                color: #000000;
                font-family: 'Segoe UI';
                font-size: 11pt;
            }

            QPushButton {
                background-color: #E7E7E7;
                color: #000000;
                border: 1px solid #AAAAAA;
                border-radius: 4px;
                padding: 6px 12px;
            }

            QPushButton:hover {
                background-color: #125ea8;
            }

            QTextEdit {
                background-color: #FFFFFF;
                color: #000000;
                border: 1px solid #DDD;
            }

            QGroupBox {
                border: 1px solid #DDD;
                margin-top: 6px;
                padding: 8px;
            }

            QTabBar::tab:selected {
                background: #DDDDDD;
                color: #000000;
            }
        """)
        # matplotlib rcParams — modo claro científico P/B
        matplotlib.rcParams.update({
            "axes.edgecolor": "#000000",
            "axes.labelcolor": "#000000",
            "axes.facecolor": "#FFFFFF",
            "grid.color": "#CCCCCC",
            "grid.linestyle": "-",
            "text.color": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "font.family": "Segoe UI",
            "figure.facecolor": "#FFFFFF"
        })

    def _toggle_dark(self):
        if self.chk_dark.isChecked():
            self._apply_industrial_theme()
        else:
            self._apply_light_theme()
        # atualizar plots com novo estilo
        if self.last_result and self.last_result.get("df") is not None:
            self._display_plots(self.last_result["df"])

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.append(f"[{ts}] {msg}")

# ---------------- Entrypoint ----------------
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = VertiSecNew()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()


