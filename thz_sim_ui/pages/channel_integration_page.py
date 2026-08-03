from __future__ import annotations

import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget,
)

from thz_sim_ui.widgets.forms import combo, dspin, line, make_form_group, spin
from thz_sim_ui.widgets.workbench import WorkbenchPage

PLOT_FONT_FAMILY = [
    "Times New Roman",
    "DejaVu Serif",
    "STIXGeneral",
    # 上述西文字体没有中文字形；后两项仅作为 CJK glyph fallback。
    "Noto Serif SC",
    "Microsoft YaHei",
]
PLOT_STYLE = {
    "font.family": PLOT_FONT_FAMILY,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
}
plt.rcParams.update(PLOT_STYLE)

CH_MODULES = ["AWGN模块", "多径模块", "CFO模块", "IQ异常模块", "功率放大器模块"]


class ChannelIntegrationPage(WorkbenchPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("信道集成", "信道模块可视化设计器，支持模块启停与参数编辑。", parent)

        # ── 模块总览 ──
        self.module_checks: dict[str, QCheckBox] = {}
        self._selected_module = CH_MODULES[0]

        modules_group = QGroupBox("信道模块总览")
        mod_layout = QVBoxLayout(modules_group)
        for name in CH_MODULES:
            row = QHBoxLayout()
            row.setSpacing(10)
            cb = QCheckBox(name)
            cb.setProperty("optionCard", True)
            cb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            cb.setChecked(False)
            row.addWidget(cb)
            btn = QPushButton("编辑")
            btn.setProperty("compact", True)
            btn.setMinimumWidth(64)
            btn.setFixedHeight(32)
            btn.clicked.connect(lambda _, n=name: self._on_select_module(n))
            row.addWidget(btn)
            mod_layout.addLayout(row)
            self.module_checks[name] = cb
        self.module_checks["AWGN模块"].setChecked(True)
        self.add_left_widget(modules_group)

        # ── 参数面板 ──
        self.param_stack = QStackedWidget()
        self.param_widgets: dict[str, QWidget] = {}
        for name in CH_MODULES:
            w = self._build_param_panel(name)
            self.param_widgets[name] = w
            self.param_stack.addWidget(w)
        param_wrapper = QWidget()
        pw_layout = QVBoxLayout(param_wrapper)
        pw_layout.setContentsMargins(0, 0, 0, 0)
        title_label = QLabel("当前选中模块参数")
        title_label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 4px 0;")
        pw_layout.addWidget(title_label)
        pw_layout.addWidget(self.param_stack)
        self.add_left_widget(param_wrapper)
        self.add_left_stretch()

        # ── 右侧：星座影响 (含绘制按钮) ──
        right = QWidget()
        r_layout = QVBoxLayout(right)
        r_layout.setContentsMargins(8, 8, 8, 8)
        self.constellation_label = QLabel("点击「绘制」查看信道对星座图的影响")
        self.constellation_label.setAlignment(Qt.AlignCenter)
        self.constellation_label.setMinimumHeight(320)
        self.constellation_label.setStyleSheet(
            "background: #FAFBFE; border: 1px solid #E6ECF6; font-size: 12px;")
        self.constellation_label.setScaledContents(False)
        self._original_pixmap: QPixmap | None = None

        btn_row = QHBoxLayout()
        self.plot_btn = QPushButton("绘制星座")
        self.plot_btn.setFixedHeight(32)
        self.plot_btn.clicked.connect(self._on_plot_constellation)
        btn_row.addStretch()
        btn_row.addWidget(self.plot_btn)
        btn_row.addStretch()

        r_layout.addWidget(self.constellation_label)
        r_layout.addLayout(btn_row)
        self.add_right_widget(right)
        self._on_mp_source_changed()

    # =================================================================
    # 参数面板
    # =================================================================
    def _build_param_panel(self, name: str) -> QWidget:
        if name == "AWGN模块":
            self.awgn_mode = combo(["信噪比 (SNR)", "热噪声模型"])
            self.awgn_mode.currentIndexChanged.connect(self._on_awgn_mode_changed)
            self.awgn_snr = dspin(-20, 80, 24, suffix="dB")
            self.awgn_noise_temp = dspin(0, 10000, 290, suffix="K")
            self.awgn_noise_temp.setEnabled(False)
            self.awgn_noise_fig = dspin(0, 20, 2, suffix="dB")
            self.awgn_noise_fig.setEnabled(False)
            return make_form_group("AWGN 参数", [
                ("噪声模型", self.awgn_mode),
                ("信噪比 (SNRdB)", self.awgn_snr),
                ("噪声温度 (K)", self.awgn_noise_temp),
                ("噪声系数 (dB)", self.awgn_noise_fig),
            ])
        elif name == "多径模块":
            self.mp_source = combo(["仿真模型", "实测确定性回放", "实测PDP-Rayleigh"])
            self.mp_tdl_model = combo(["无", "TDL-A", "TDL-B", "TDL-C", "TDL-D", "TDL-E"])
            self.mp_tdl_ds = dspin(0.01, 1000, 30, suffix="ns")
            self.mp_tdl_vel = dspin(0, 120, 0, suffix="m/s")
            self.mp_fading = combo(["static", "frame", "block"])
            self.mp_normalize = combo(["否", "是"])
            self.mp_jakes = spin(4, 200, 48)
            self.mp_paths_cfg = line(
                '[{"delay_samples": 0, "gain_type": "static", "gain": "1+0j"},'
                ' {"delay_samples": 4, "gain_type": "static", "gain": "0.45+0j"},'
                ' {"delay_samples": 12, "gain_type": "static", "gain": "0.25+0j"}]')
            self.mp_measured_scenario = combo(["8 cm", "12 cm", "50 cm"])
            self.mp_measured_retained = dspin(50, 100, 95, decimals=1, suffix="%")
            self.mp_measured_siso_link = combo([
                "TX0 → RX0 (h00)", "TX1 → RX0 (h01)",
                "TX0 → RX1 (h10)", "TX1 → RX1 (h11)",
            ])
            self.mp_measured_info = QLabel()
            self.mp_measured_info.setWordWrap(True)
            self.mp_measured_info.setStyleSheet(
                "color: #475569; background: #F8FAFC; padding: 6px;"
            )
            self.mp_source.currentIndexChanged.connect(self._on_mp_source_changed)
            self.mp_measured_scenario.currentIndexChanged.connect(
                self._on_measured_scenario_changed
            )
            return make_form_group("多径参数", [
                ("信道来源", self.mp_source),
                ("TDL模型 (tdl_model)", self.mp_tdl_model),
                ("RMS时延扩展 (ns)", self.mp_tdl_ds),
                ("移动速度 (m/s)", self.mp_tdl_vel),
                ("衰落模型 (fading_model)", self.mp_fading),
                ("功率归一化", self.mp_normalize),
                ("Jakes正弦波数", self.mp_jakes),
                ("自定义路径 (multipath_paths)", self.mp_paths_cfg),
                ("实测场景", self.mp_measured_scenario),
                ("累计功率保留率", self.mp_measured_retained),
                ("SISO子链路", self.mp_measured_siso_link),
                ("实测信道信息", self.mp_measured_info),
            ])
        elif name == "CFO模块":
            self.cfo_ppm = dspin(0, 100, 1, suffix="ppm")
            self.cfo_enable_phase_noise = combo(["否", "是"])
            self.cfo_pn_std = dspin(0, 1, 0.01, suffix="rad")
            self.cfo_pn_bw = dspin(0.1, 10000, 100, suffix="kHz")
            return make_form_group("CFO 与相位噪声参数", [
                ("载频偏移 (ppm)", self.cfo_ppm),
                ("启用相位噪声", self.cfo_enable_phase_noise),
                ("相位噪声标准差 (rad)", self.cfo_pn_std),
                ("相位噪声带宽 (kHz)", self.cfo_pn_bw),
            ])
        elif name == "IQ异常模块":
            self.iq_pos = combo(["rx", "tx", "both"])
            self.iq_model = combo(["fid", "fd"])
            self.iq_rx_gain = dspin(0, 10, 2, suffix="dB")
            self.iq_rx_phase = dspin(0, 30, 5, suffix="°")
            self.iq_rx_gI = line("[1.0, 0.08, -0.03]")
            self.iq_rx_gQ = line("[1.0, -0.12, 0.04]")
            self.iq_tx_gain = dspin(0, 10, 0, suffix="dB")
            self.iq_tx_phase = dspin(0, 30, 0, suffix="°")
            self.iq_tx_gI = line("[1.0]")
            self.iq_tx_gQ = line("[1.0]")
            return make_form_group("IQ 不平衡参数", [
                ("不平衡位置", self.iq_pos),
                ("IQ模型 (fid/fd)", self.iq_model),
                ("── RX ──", None),
                ("RX增益不平衡 (dB)", self.iq_rx_gain),
                ("RX相位不平衡 (°)", self.iq_rx_phase),
                ("RX I路FIR系数", self.iq_rx_gI),
                ("RX Q路FIR系数", self.iq_rx_gQ),
                ("── TX ──", None),
                ("TX增益不平衡 (dB)", self.iq_tx_gain),
                ("TX相位不平衡 (°)", self.iq_tx_phase),
                ("TX I路FIR系数", self.iq_tx_gI),
                ("TX Q路FIR系数", self.iq_tx_gQ),
            ])
        else:  # 非线性放大
            self.pa_model = combo(["modified_rapp"])
            self.pa_param_source = combo(["开源数据集", "手动参数"])
            self.pa_fc = dspin(1, 1000, 300, decimals=1, suffix="GHz")
            self.pa_nearest_fc = combo(["否", "是"])
            self.pa_auto_scaling = combo(["是", "否"])
            self.pa_power = dspin(-40, 20, -13.2, suffix="dBm")
            self.pa_G = dspin(0, 100, 11.2616, decimals=4)
            self.pa_Vsat = dspin(0.0001, 10, 0.0628, decimals=4, suffix="V")
            self.pa_p = dspin(0.01, 100, 1.0013, decimals=4)
            self.pa_A = dspin(-1e9, 1e9, -6.2038e4, decimals=4)
            self.pa_B = dspin(0.000001, 1000, 0.0160, decimals=6)
            self.pa_q1 = dspin(0.01, 1000, 1.7344, decimals=4)
            self.pa_q2 = dspin(0.01, 1000, 1.8972, decimals=4)
            self.pa_param_source.currentIndexChanged.connect(self._on_pa_source_changed)
            self._on_pa_source_changed()
            return make_form_group("非线性放大参数", [
                ("PA模型", self.pa_model),
                ("参数来源", self.pa_param_source),
                ("PA载频", self.pa_fc),
                ("无精确载频时取最近值", self.pa_nearest_fc),
                ("自动缩放至输入功率", self.pa_auto_scaling),
                ("输入功率 (dBm)", self.pa_power),
                ("线性增益 G", self.pa_G),
                ("饱和电压 Vsat", self.pa_Vsat),
                ("平滑因子 p", self.pa_p),
                ("AM-PM 系数 A", self.pa_A),
                ("AM-PM 系数 B", self.pa_B),
                ("AM-PM 指数 q1", self.pa_q1),
                ("AM-PM 指数 q2", self.pa_q2),
            ])

    # =================================================================
    # 槽函数
    # =================================================================
    def _on_select_module(self, name: str) -> None:
        self._selected_module = name
        if name in self.param_widgets:
            self.param_stack.setCurrentWidget(self.param_widgets[name])

    def _on_awgn_mode_changed(self, _idx: int = 0) -> None:
        is_snr = (self.awgn_mode.currentText() == "信噪比 (SNR)")
        self.awgn_snr.setEnabled(is_snr)
        self.awgn_noise_temp.setEnabled(not is_snr)
        self.awgn_noise_fig.setEnabled(not is_snr)

    def _on_pa_source_changed(self, _idx: int = 0) -> None:
        from_dataset = self.pa_param_source.currentText() == "开源数据集"
        self.pa_nearest_fc.setEnabled(from_dataset)
        for widget in (self.pa_G, self.pa_Vsat, self.pa_p, self.pa_A,
                       self.pa_B, self.pa_q1, self.pa_q2):
            widget.setEnabled(not from_dataset)

    def _on_mp_source_changed(self, _idx: int = 0) -> None:
        if not hasattr(self, "mp_source"):
            return
        measured = self.mp_source.currentText() != "仿真模型"
        for widget in (
            self.mp_tdl_model, self.mp_tdl_ds, self.mp_tdl_vel,
            self.mp_fading, self.mp_normalize, self.mp_jakes, self.mp_paths_cfg,
        ):
            widget.setEnabled(not measured)
        for widget in (
            self.mp_measured_scenario, self.mp_measured_siso_link,
            self.mp_measured_info,
        ):
            widget.setEnabled(measured)
        self.mp_measured_retained.setEnabled(
            self.mp_source.currentText() == "实测PDP-Rayleigh"
        )
        if measured:
            self.module_checks["多径模块"].setChecked(True)
            self.awgn_mode.setCurrentIndex(0)
        self.awgn_mode.setEnabled(not measured)
        self._on_awgn_mode_changed()
        self._on_measured_scenario_changed()
        if hasattr(self, "plot_btn"):
            self.plot_btn.setText("预览实测PDP" if measured else "绘制星座")

    def _on_measured_scenario_changed(self, _idx: int = 0) -> None:
        if not hasattr(self, "mp_measured_info"):
            return
        scene = self.mp_measured_scenario.currentText()
        info = {
            "8 cm": ("THz_MIMO_08cm_2.npz", 1.6, 49, 64),
            "12 cm": ("THz_MIMO_12cm_2.npz", 1.2, 37, 64),
            "50 cm": ("THz_MIMO_50cm_2.npz", 0.5, 16, 32),
        }[scene]
        self.mp_measured_info.setText(
            f"{info[0]}\n有效时延 {info[1]:g} ns，{info[2]} 抽头，"
            f"运行时自动采用 GI={info[3]}。信道功率统一归一化。"
        )

    # =================================================================
    # 绘制星座
    # =================================================================
    def _on_plot_constellation(self) -> None:
        """用少量测试样本生成信道影响后的星座图"""
        if (
            hasattr(self, "mp_source")
            and self.mp_source.currentText() != "仿真模型"
        ):
            self._plot_measured_channel()
            return
        from params.PHYParams import PHYParams
        from transmitter.THzTransmitter import THzTransmitter
        from channel.THzChannel import THzChannel
        from receiver.MatchedFilter import RxMatchedFilter
        from receiver.sync.CoarseSync import CoarseSync

        self.plot_btn.setEnabled(False)
        self.plot_btn.setText("生成中...")
        try:
            params = PHYParams()
            ch_cfg = self.get_channel_params()

            # 只传递 PHYParams 支持的键
            valid_keys = set(params._params) | PHYParams._LEGACY_MULTIPATH_KEYS
            for k, v in ch_cfg.items():
                if v is not None and k in valid_keys:
                    params.update(**{k: v})

            # 读取参数配置页面的调制方式
            mod_ncbps = 6  # 默认 64QAM
            try:
                window = self.window()
                param_page = window.pages.get("parameter_config") if hasattr(window, "pages") else None
                if param_page and hasattr(param_page, "modulation"):
                    mod_map = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "16QAM": 4, "64QAM": 6}
                    mod_ncbps = mod_map.get(param_page.modulation.currentText(), 6)
            except Exception:
                pass

            # 确保关键参数不为 None（防止 int(None) 报错）
            if params.get("oversampling") is None:
                params.update(oversampling=4)
            if params.get("filter_length") is None:
                params.update(filter_length=32)
            params.update(duration=1e-6, link_mode="sc-fde", Preamble_type="short",
                          NCBPS=mod_ncbps)

            tx = THzTransmitter(params); tx.run()

            # TX → Channel → MF → CoarseSync → 下采样 → 去前导码
            # 始终走完整 RX 链，确保有无多径时对比对等
            ch = THzChannel(params)
            rx = ch.run(tx.tx_signal_dict)
            mf = RxMatchedFilter(tx); sig = mf.matched_filter(rx)
            cs = CoarseSync(tx); sig = cs.detect_sync(sig)
            sps = params.get("oversampling", 4) or 4
            preamble_len_sym = 3328
            start = preamble_len_sym * sps
            y = np.asarray(sig["signal_stream"])
            payload = y[start::sps]

            # 绘图
            plt.rcParams.update(PLOT_STYLE)
            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            n = min(3000, len(payload))
            ax.scatter(np.real(payload[:n]), np.imag(payload[:n]), s=3, alpha=0.5, c="#D95F02")
            ax.set_xlabel("I"); ax.set_ylabel("Q")
            ax.set_title("Channel constellation (payload)", fontsize=10, pad=4)
            ax.axis("equal")
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)

            pixmap = QPixmap()
            pixmap.loadFromData(buf.read(), "PNG")
            self._original_pixmap = pixmap
            self._update_pixmap()
            self.constellation_label.setText("")
        except Exception as e:
            self.constellation_label.setText(f"绘制失败: {e}")
        finally:
            self.plot_btn.setEnabled(True)
            self.plot_btn.setText("绘制星座")

    def _plot_measured_channel(self) -> None:
        """预览原始PDP、物理时延窗和最终选中的随机TDL抽头。"""
        from channel.MeasuredChannel import MeasuredChannel

        self.plot_btn.setEnabled(False)
        self.plot_btn.setText("生成中...")
        try:
            params = self.get_channel_params()
            measured = MeasuredChannel(params)
            preview = measured.preview_data()
            raw = preview["raw_pdp"]
            cropped = preview["cropped_pdp"]
            selected = preview["selected_pdp"]
            delays = preview["delays_ns"]
            raw_delays = np.arange(len(raw)) / measured.SOURCE_SAMPLE_RATE_HZ * 1e9
            eps = np.finfo(float).tiny

            # 其他图表可能修改过全局 rcParams；每次绘制前恢复统一字体和样式。
            plt.rcParams.update(PLOT_STYLE)
            fig, ax = plt.subplots(figsize=(6.8, 4.8))
            ax.plot(
                raw_delays,
                10 * np.log10(np.maximum(raw / np.max(raw), eps)),
                color="#94A3B8", linewidth=0.8, label="原始聚合PDP",
            )
            ax.plot(
                delays,
                10 * np.log10(np.maximum(cropped / np.max(cropped), eps)),
                color="#2563EB", linewidth=1.4, label="去噪/截断PDP",
            )
            indexes = preview["selected_indexes"]
            ax.scatter(
                delays[indexes],
                10 * np.log10(np.maximum(selected[indexes] / np.max(selected), eps)),
                color="#DC2626", s=24, zorder=3, label="选中主抽头",
            )
            noise_floor = float(np.mean(measured.noise_floor_per_link))
            noise_floor_db = 10 * np.log10(max(noise_floor / np.max(raw), eps))
            ax.axhline(
                noise_floor_db, color="#64748B", linestyle=":", linewidth=0.9,
                label=f"估计噪声底 {noise_floor_db:.1f} dB",
            )
            max_delay = measured.scenario_info["max_delay_ns"]
            ax.axvline(max_delay, color="#111827", linestyle="--", linewidth=0.9,
                       label=f"最大时延 {max_delay:g} ns")
            ax.set_xlim(0, max(2.0, max_delay * 1.35))
            ax.set_ylim(-50, 3)
            ax.set_xlabel("时延 (ns)")
            ax.set_ylabel("相对功率 (dB)")
            ax.set_title(f"{measured.scenario} 实测信道 PDP")
            ax.grid(True, alpha=0.25, linestyle="--")
            ax.legend(fontsize=8)
            outside = measured.diagnostics["outside_window_peak_relative_db"]
            above_noise = measured.diagnostics["outside_window_peak_above_noise_db"]
            judgment = (
                "存在显著窗外强径"
                if measured.diagnostics["significant_power_outside_window"]
                else "无显著窗外强径"
            )
            ax.text(
                0.98, 0.03,
                f"窗外最强径：{outside:.1f} dB（高于噪声底 {above_noise:.1f} dB）\n"
                f"判定：{judgment}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85,
                      "edgecolor": "#CBD5E1"},
            )
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            pixmap = QPixmap()
            pixmap.loadFromData(buf.read(), "PNG")
            self._original_pixmap = pixmap
            self._update_pixmap()
            self.constellation_label.setText("")
        except Exception as exc:
            self.constellation_label.setText(f"预览失败: {exc}")
        finally:
            self.plot_btn.setEnabled(True)
            self.plot_btn.setText("预览实测PDP")

    def _update_pixmap(self) -> None:
        if self._original_pixmap is None:
            return
        sz = self.constellation_label.size()
        if sz.width() <= 0 or sz.height() <= 0:
            return
        scaled = self._original_pixmap.scaled(sz, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.constellation_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_pixmap()

    # =================================================================
    # 参数收集
    # =================================================================
    def get_channel_params(self) -> dict[str, object]:
        p: dict[str, object] = {}
        p["enable_awgn"] = self.module_checks["AWGN模块"].isChecked()
        is_snr = (self.awgn_mode.currentText() == "信噪比 (SNR)")
        if is_snr:
            p["SNRdB"] = self.awgn_snr.value()
            p["noise_temperature"] = None
            p["noise_figure_db"] = None
        else:
            p["SNRdB"] = None
            p["noise_temperature"] = self.awgn_noise_temp.value()
            p["noise_figure_db"] = self.awgn_noise_fig.value()

        mp_on = self.module_checks["多径模块"].isChecked()
        p["enable_multipath"] = mp_on
        p["multipath_source"] = "simulated"
        if mp_on:
            measured = self.mp_source.currentText() != "仿真模型"
            if measured:
                p["multipath_source"] = "measured"
                p["measured_channel_mode"] = (
                    "pdp_rayleigh"
                    if self.mp_source.currentText() == "实测PDP-Rayleigh"
                    else "deterministic"
                )
                p["measured_channel_scenario"] = {
                    "8 cm": "8cm", "12 cm": "12cm", "50 cm": "50cm",
                }[self.mp_measured_scenario.currentText()]
                p["measured_channel_retained_power"] = (
                    self.mp_measured_retained.value() / 100.0
                )
                p["measured_channel_tx_index"], p["measured_channel_rx_index"] = {
                    "TX0 → RX0 (h00)": (0, 0),
                    "TX1 → RX0 (h01)": (1, 0),
                    "TX0 → RX1 (h10)": (0, 1),
                    "TX1 → RX1 (h11)": (1, 1),
                }[self.mp_measured_siso_link.currentText()]
            else:
                tdl = self.mp_tdl_model.currentText()
                if tdl != "无":
                    p["tdl_model"] = tdl
                p["tdl_delay_spread_ns"] = self.mp_tdl_ds.value()
                p["tdl_velocity_mps"] = self.mp_tdl_vel.value()
                p["fading_model"] = self.mp_fading.currentText()
                p["normalize_channel_power"] = (self.mp_normalize.currentText() == "是")
                p["tdl_jakes_num_sinusoids"] = self.mp_jakes.value()
                import ast
                try:
                    p["multipath_paths"] = ast.literal_eval(self.mp_paths_cfg.text())
                except (ValueError, SyntaxError):
                    pass

        cfo_on = self.module_checks["CFO模块"].isChecked()
        p["enable_cfo"] = cfo_on
        p["ppm"] = self.cfo_ppm.value()
        pn_on = (self.cfo_enable_phase_noise.currentText() == "是")
        p["enable_phase_noise"] = pn_on
        if pn_on:
            p["phase_noise_std"] = self.cfo_pn_std.value()
            p["phase_noise_bw"] = self.cfo_pn_bw.value() * 1e3

        iq_on = self.module_checks["IQ异常模块"].isChecked()
        p["enable_iq_imbalance"] = iq_on
        if iq_on:
            p["iq_imbalance_position"] = self.iq_pos.currentText()
            p["iq_imbalance_model"] = self.iq_model.currentText()
            p["rx_iq_gain_imbalance_db"] = self.iq_rx_gain.value()
            p["rx_iq_phase_imbalance_deg"] = self.iq_rx_phase.value()
            import ast
            for src, key in [(self.iq_rx_gI, "rx_iq_gI_taps"),
                              (self.iq_rx_gQ, "rx_iq_gQ_taps"),
                              (self.iq_tx_gI, "tx_iq_gI_taps"),
                              (self.iq_tx_gQ, "tx_iq_gQ_taps")]:
                try:
                    p[key] = ast.literal_eval(src.text())
                except (ValueError, SyntaxError):
                    p[key] = [1.0]

        pa_on = self.module_checks["功率放大器模块"].isChecked()
        p["enable_pa"] = pa_on
        if p["enable_pa"]:
            p["pa_model"] = self.pa_model.currentText()
            p["pa_load_params_from_dataset"] = (
                self.pa_param_source.currentText() == "开源数据集"
            )
            p["pa_fc_Hz"] = self.pa_fc.value() * 1e9
            p["pa_use_nearest_fc"] = self.pa_nearest_fc.currentText() == "是"
            p["pa_auto_input_scaling"] = self.pa_auto_scaling.currentText() == "是"
            p["pa_input_power_dbm"] = self.pa_power.value()
            p["pa_G"] = self.pa_G.value()
            p["pa_Vsat"] = self.pa_Vsat.value()
            p["pa_p"] = self.pa_p.value()
            p["pa_A"] = self.pa_A.value()
            p["pa_B"] = self.pa_B.value()
            p["pa_q1"] = self.pa_q1.value()
            p["pa_q2"] = self.pa_q2.value()

        return p

    def get_all_parameters(self) -> dict[str, object]:
        params = self.get_channel_params()
        params["信道模块启用"] = {n: self.module_checks[n].isChecked() for n in CH_MODULES}
        params["当前选中模块"] = self._selected_module
        return params

    def set_channel_params(self, ch: dict[str, object]) -> None:
        if not ch:
            return
        self.module_checks["AWGN模块"].setChecked(ch.get("enable_awgn", True))
        self.awgn_snr.setValue(float(ch.get("SNRdB", 24)))
        self.awgn_noise_temp.setValue(float(ch.get("noise_temperature", 0) or 0))
        self.awgn_noise_fig.setValue(float(ch.get("noise_figure_db", 0) or 0))
        mp_on = ch.get("enable_multipath", False)
        self.module_checks["多径模块"].setChecked(bool(mp_on))
        measured_source = str(ch.get("multipath_source", "simulated")) == "measured"
        measured_mode = str(ch.get("measured_channel_mode", "deterministic"))
        _set_combo(
            self.mp_source,
            ("实测PDP-Rayleigh" if measured_mode == "pdp_rayleigh"
             else "实测确定性回放") if measured_source else "仿真模型",
        )
        tdl = str(ch.get("tdl_model", "无") or "无")
        _set_combo(self.mp_tdl_model, tdl)
        self.mp_tdl_ds.setValue(float(ch.get("tdl_delay_spread_ns", 30)))
        self.mp_tdl_vel.setValue(float(ch.get("tdl_velocity_mps", 0)))
        _set_combo(self.mp_fading, str(ch.get("fading_model", "static")))
        _set_combo(self.mp_normalize, "是" if ch.get("normalize_channel_power") else "否")
        self.mp_jakes.setValue(int(ch.get("tdl_jakes_num_sinusoids", 48)))
        _set_combo(
            self.mp_measured_scenario,
            {"8cm": "8 cm", "12cm": "12 cm", "50cm": "50 cm"}.get(
                str(ch.get("measured_channel_scenario", "8cm")), "8 cm"
            ),
        )
        self.mp_measured_retained.setValue(
            float(ch.get("measured_channel_retained_power", 0.95)) * 100.0
        )
        tx_index = int(ch.get("measured_channel_tx_index", 0))
        rx_index = int(ch.get("measured_channel_rx_index", 0))
        _set_combo(self.mp_measured_siso_link, {
            (0, 0): "TX0 → RX0 (h00)",
            (1, 0): "TX1 → RX0 (h01)",
            (0, 1): "TX0 → RX1 (h10)",
            (1, 1): "TX1 → RX1 (h11)",
        }.get((tx_index, rx_index), "TX0 → RX0 (h00)"))
        self._on_mp_source_changed()
        cfo_on = ch.get("enable_cfo", False)
        self.module_checks["CFO模块"].setChecked(bool(cfo_on))
        self.cfo_ppm.setValue(float(ch.get("ppm", 1)))
        pn_on = ch.get("enable_phase_noise", False)
        _set_combo(self.cfo_enable_phase_noise, "是" if pn_on else "否")
        self.cfo_pn_std.setValue(float(ch.get("phase_noise_std", 0.01)))
        self.cfo_pn_bw.setValue(float(ch.get("phase_noise_bw", 100e3) or 100e3) / 1e3)
        iq_on = ch.get("enable_iq_imbalance", False)
        self.module_checks["IQ异常模块"].setChecked(bool(iq_on))
        _set_combo(self.iq_pos, str(ch.get("iq_imbalance_position", "rx")))
        _set_combo(self.iq_model, str(ch.get("iq_imbalance_model", "fid")))
        self.iq_rx_gain.setValue(float(ch.get("rx_iq_gain_imbalance_db", 2)))
        self.iq_rx_phase.setValue(float(ch.get("rx_iq_phase_imbalance_deg", 5)))
        self.iq_tx_gain.setValue(float(ch.get("tx_iq_gain_imbalance_db", 0)))
        self.iq_tx_phase.setValue(float(ch.get("tx_iq_phase_imbalance_deg", 0)))
        pa_on = ch.get("enable_pa", False)
        self.module_checks["功率放大器模块"].setChecked(bool(pa_on))
        pa = str(ch.get("pa_model", "modified_rapp") or "modified_rapp")
        _set_combo(self.pa_model, pa)
        _set_combo(
            self.pa_param_source,
            "开源数据集" if ch.get("pa_load_params_from_dataset", True) else "手动参数",
        )
        self.pa_fc.setValue(float(ch.get("pa_fc_Hz", 300e9)) / 1e9)
        _set_combo(self.pa_nearest_fc, "是" if ch.get("pa_use_nearest_fc", False) else "否")
        _set_combo(self.pa_auto_scaling, "是" if ch.get("pa_auto_input_scaling", True) else "否")
        self.pa_power.setValue(float(ch.get("pa_input_power_dbm", -13.2)))
        self.pa_G.setValue(float(ch.get("pa_G", 11.2616)))
        self.pa_Vsat.setValue(float(ch.get("pa_Vsat", 0.0628)))
        self.pa_p.setValue(float(ch.get("pa_p", 1.0013)))
        self.pa_A.setValue(float(ch.get("pa_A", -6.2038e4)))
        self.pa_B.setValue(float(ch.get("pa_B", 0.0160)))
        self.pa_q1.setValue(float(ch.get("pa_q1", 1.7344)))
        self.pa_q2.setValue(float(ch.get("pa_q2", 1.8972)))


def _set_combo(widget, text: str) -> None:
    idx = widget.findText(text)
    if idx >= 0:
        widget.setCurrentIndex(idx)
