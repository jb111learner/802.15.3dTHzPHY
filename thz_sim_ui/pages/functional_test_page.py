"""
功能测试页面 — 调制、波形、编码、浮点精度、物理层速率与功能校准。

八个功能可独立选择运行：页面自带「运行测试」按钮，测试在后台
FunctionalTestWorker(QThread) 中执行，结果经 Qt 信号回传并渲染到右侧结果栏。
"""
from __future__ import annotations

from pathlib import Path
import json
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from thz_sim_ui.constants import STATUS_COLORS
from thz_sim_ui.services.ber_test_service import (
    BER_CONFIG_OPTIONS,
    ber_option_link_text,
    ber_option_snr_default,
)
from thz_sim_ui.services.functional_test_service import (
    FunctionalTestWorker,
    qam_constellation_points,
)
from thz_sim_ui.widgets.charts import TextSummaryCard
from thz_sim_ui.widgets.common import CardWidget, StatusBadge
from thz_sim_ui.widgets.forms import combo, dspin, make_form_group, make_radio_group, spin
from thz_sim_ui.widgets.workbench import WorkbenchPage

_PLACEHOLDER_STYLE = (
    "background: #FAFBFE; border: 1px dashed #CAD4E5; border-radius: 8px;"
    " color: #7C869B; min-height: 200px;"
)
_MONO_STYLE = "font-family: Consolas, monospace;"

_TEST_KEYS = (
    "modulation", "waveform_time", "waveform_spectrum", "codec", "precision",
    "single_link_rate", "total_phy_rate", "function_calibration", "ber",
)


class _UpdatableSummaryCard(TextSummaryCard):
    """支持运行时刷新行内容的汇总卡。"""

    def __init__(self, title: str, lines: List[str], parent: QWidget | None = None) -> None:
        super().__init__(title, lines, parent)
        self.body_label = self.layout.itemAt(self.layout.count() - 1).widget()

    def set_lines(self, lines: List[str]) -> None:
        self.body_label.setText("\n".join(f"• {line}" for line in lines))


class FunctionalTestPage(WorkbenchPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("功能测试", "链路模块功能验证：调制 / 波形 / 编码 / 精度 / 速率 / 校准。", parent)
        self._worker: Optional[FunctionalTestWorker] = None
        self._codec_detail_lines: List[str] = []
        self._codec_case_results: List[dict] = []
        self._modulation_pixmaps: List[tuple] = []  # [(QLabel, QPixmap 原图)]
        self._wave_time_pixmaps: List[tuple] = []
        self._wave_spectrum_pixmaps: List[tuple] = []
        self._calibration_pixmaps: List[tuple] = []
        self._ber_pixmaps: List[tuple] = []

        self._build_left_panel()
        self._build_right_panel()
        self._sync_stacks()

    # ==================== 左栏 ====================

    def _build_left_panel(self) -> None:
        radio_box = make_radio_group("测试项目选择", [
            "调制方式测试",
            "时域波形测试",
            "功率谱测试",
            "编码技术测试",
            "浮点精度验证",
            "单链路物理层速率测试",
            "总物理层速率测试",
            "校准测试",
            "误码率测试",
        ])
        self.test_radios: List[QRadioButton] = radio_box.findChildren(QRadioButton)
        for rb in self.test_radios:
            rb.toggled.connect(self._on_radio_toggled)
        self.add_left_widget(radio_box)

        self.param_stack = QStackedWidget()
        self.param_stack.addWidget(self._build_modulation_params())
        self.param_stack.addWidget(self._build_waveform_time_params())
        self.param_stack.addWidget(self._build_waveform_spectrum_params())
        self.param_stack.addWidget(self._build_codec_params())
        self.param_stack.addWidget(self._build_precision_params())
        self.param_stack.addWidget(self._build_single_rate_params())
        self.param_stack.addWidget(self._build_total_rate_params())
        self.param_stack.addWidget(self._build_calibration_params())
        self.param_stack.addWidget(self._build_ber_params())
        self.add_left_widget(self.param_stack)

        self.run_button = QPushButton("运行测试")
        self.run_button.setProperty("role", "primary")
        self.run_button.clicked.connect(self._on_run_clicked)
        self.stop_button = QPushButton("停止测试")
        self.stop_button.setEnabled(False)
        self.stop_button.hide()
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.status_badge = StatusBadge("待运行")
        run_row = QWidget()
        run_layout = QHBoxLayout(run_row)
        run_layout.setContentsMargins(0, 0, 0, 0)
        run_layout.addWidget(self.run_button)
        run_layout.addWidget(self.stop_button)
        run_layout.addWidget(self.status_badge)
        run_layout.addStretch(1)
        self.add_left_footer_widget(run_row)

        self.add_left_stretch()

    def _build_modulation_params(self) -> QWidget:
        self.modulation_type = combo(["QPSK", "16QAM", "64QAM"])
        self.modulation_symbols = spin(64, 1048576, 1024)
        self.modulation_show = spin(64, 10000, 1024)
        self.modulation_seed = spin(0, 2147483647, 2026)
        box = make_form_group("调制方式测试参数", [
            ("调制方式", self.modulation_type),
            ("调制符号数", self.modulation_symbols),
            ("星座图显示点数", self.modulation_show),
            ("随机种子", self.modulation_seed),
        ])
        hint = QLabel(
            "调用发射端调制器生成调制符号并展示实际输出星座图；"
            "星座图显示点数可小于调制符号数，随机种子用于生成输入比特流。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("CardHint")
        return self._wrap_with_hint(box, hint)

    def _build_waveform_time_params(self) -> QWidget:
        self.wave_time_mode = combo(["单载波", "OFDM"])

        self.wave_time_sc_modulation = combo(["QPSK", "16QAM", "64QAM"])
        self.wave_time_sc_pulses = spin(2, 3, 2)
        self.wave_time_sc_filter_type = combo(["RRC", "RC", "RECT"])
        self.wave_time_sc_filter_len = spin(8, 128, 32)
        self.wave_time_sc_sps = spin(1, 16, 4)
        self.wave_time_sc_rolloff = dspin(0.0, 1.0, 0.22, decimals=3)

        self.wave_time_ofdm_symbols = spin(2, 3, 2)
        self.wave_time_ofdm_start = spin(-1024, 1023, -6)
        self.wave_time_ofdm_step = spin(1, 1024, 11)
        self.wave_time_ofdm_nsc = spin(64, 2048, 512)
        self.wave_time_ofdm_nsc.setSingleStep(64)
        self.wave_time_ofdm_sps = spin(1, 16, 4)
        self.wave_time_ofdm_cp = spin(0, 1024, 32)

        self.wave_time_ofdm_heat_symbols = spin(8, 512, 48)
        self.wave_time_ofdm_heat_start = spin(-1024, 1023, -24)
        self.wave_time_ofdm_heat_step = spin(1, 128, 1)
        self.wave_time_ofdm_heat_margin = spin(0, 64, 4)
        self.wave_time_ofdm_heat_floor = dspin(-120.0, -20.0, -45.0, decimals=1, suffix="dB")

        mode_box = make_form_group("时域波形测试模式", [("链路模式", self.wave_time_mode)])
        self.wave_time_sc_box = make_form_group("单载波时域波形参数", [
            ("调制符号", self.wave_time_sc_modulation),
            ("完整脉冲数", self.wave_time_sc_pulses),
            ("滤波器类型", self.wave_time_sc_filter_type),
            ("滤波器长度", self.wave_time_sc_filter_len),
            ("过采样率", self.wave_time_sc_sps),
            ("滚降系数", self.wave_time_sc_rolloff),
        ])
        self.wave_time_symbol_box = QGroupBox("各脉冲星座点选择")
        self.wave_time_symbol_layout = QVBoxLayout(self.wave_time_symbol_box)
        self.wave_time_symbol_layout.setContentsMargins(14, 18, 14, 14)
        self.wave_time_symbol_layout.setSpacing(10)
        self.wave_time_symbol_selectors: List[tuple] = []  # [(QSpinBox, QLabel)]

        self.wave_time_ofdm_box = make_form_group("OFDM 时域波形参数", [
            ("显示 OFDM 符号数", self.wave_time_ofdm_symbols),
            ("起始子载波 k", self.wave_time_ofdm_start),
            ("子载波步长", self.wave_time_ofdm_step),
            ("子载波数", self.wave_time_ofdm_nsc),
            ("过采样率", self.wave_time_ofdm_sps),
            ("CP 长度", self.wave_time_ofdm_cp),
        ])
        self.wave_time_ofdm_heat_box = make_form_group("OFDM 逐符号频谱热图参数", [
            ("扫描符号数", self.wave_time_ofdm_heat_symbols),
            ("起始子载波 k", self.wave_time_ofdm_heat_start),
            ("扫描步长", self.wave_time_ofdm_heat_step),
            ("横轴边距", self.wave_time_ofdm_heat_margin),
            ("色阶下限", self.wave_time_ofdm_heat_floor),
        ])
        hint = QLabel(
            "单载波：从所选调制（QPSK/16QAM/64QAM）的星座图中为每个完整脉冲"
            "挑选星座点，时域图分别显示 I/Q 实际与理想波形。"
            "OFDM 使用确定性典型 16QAM 单音扫描，显示含 CP 时域图与逐符号频谱热图。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("CardHint")

        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        for widget in (mode_box, self.wave_time_sc_box, self.wave_time_symbol_box,
                       self.wave_time_ofdm_box, self.wave_time_ofdm_heat_box, hint):
            layout.addWidget(widget)
        self.wave_time_mode.currentTextChanged.connect(self._on_wave_time_mode_changed)
        self.wave_time_sc_modulation.currentIndexChanged.connect(
            lambda _index: self._rebuild_wave_time_symbol_selectors())
        self.wave_time_sc_pulses.valueChanged.connect(
            lambda _value: self._rebuild_wave_time_symbol_selectors())
        self._rebuild_wave_time_symbol_selectors()
        self._on_wave_time_mode_changed(self.wave_time_mode.currentText())
        return wrap

    def _on_wave_time_mode_changed(self, text: str) -> None:
        is_ofdm = text == "OFDM"
        self.wave_time_sc_box.setVisible(not is_ofdm)
        self.wave_time_symbol_box.setVisible(not is_ofdm)
        self.wave_time_ofdm_box.setVisible(is_ofdm)
        self.wave_time_ofdm_heat_box.setVisible(is_ofdm)
        self._fit_param_stack_height()

    def _rebuild_wave_time_symbol_selectors(self) -> None:
        """按调制方式与完整脉冲数重建每个脉冲的星座点选择控件。

        每行用容器 QWidget 承载（QFormLayout 的 addRow(文本, 控件) 内部
        布局项无法稳定取得 widget，重建清理会残留/误删控件，导致 I/Q 值
        显示重叠错乱）；重建时整行容器销毁，selectors 列表始终指向新控件。
        """
        while self.wave_time_symbol_layout.count():
            item = self.wave_time_symbol_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.wave_time_symbol_selectors = []
        points = qam_constellation_points(self.wave_time_sc_modulation.currentText())
        last_index = len(points) - 1
        for index in range(self.wave_time_sc_pulses.value()):
            selector = spin(0, last_index, min(index, last_index))
            value_label = QLabel()
            row_widget = QWidget()
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            caption = QLabel(f"脉冲 {index + 1} 星座点")
            caption.setMinimumWidth(96)
            row.addWidget(caption)
            row.addWidget(selector)
            row.addWidget(value_label)
            row.addStretch(1)
            self.wave_time_symbol_layout.addWidget(row_widget)

            def _update(_value=None, sel=selector, lab=value_label, pts=points):
                point = pts[sel.value()]
                lab.setText(f"I = {point.real:+.3f}，Q = {point.imag:+.3f}")

            selector.valueChanged.connect(_update)
            _update()
            self.wave_time_symbol_selectors.append((selector, value_label))
        self._fit_param_stack_height()

    def _build_waveform_spectrum_params(self) -> QWidget:
        self.wave_spectrum_mode = combo(["单载波", "OFDM"])

        self.wave_spectrum_sc_modulation = combo(["QPSK", "BPSK", "16QAM", "64QAM", "256QAM"])
        self.wave_spectrum_sc_filter_type = combo(["RRC", "RC", "RECT"])
        self.wave_spectrum_sc_filter_len = spin(8, 128, 32)
        self.wave_spectrum_sc_sps = spin(1, 16, 4)
        self.wave_spectrum_sc_rolloff = dspin(0.0, 1.0, 0.22, decimals=3)
        self.wave_spectrum_sc_symbols = spin(2048, 131072, 8192)
        self.wave_spectrum_sc_symbols.setSingleStep(2048)
        self.wave_spectrum_sc_scale = combo(["对数功率 dB", "线性归一化功率"])

        self.wave_spectrum_ofdm_scale = combo(["对数功率 dB", "线性归一化功率"])
        self.wave_spectrum_ofdm_nsc = spin(64, 2048, 512)
        self.wave_spectrum_ofdm_nsc.setSingleStep(64)
        self.wave_spectrum_ofdm_sps = spin(1, 16, 4)
        self.wave_spectrum_ofdm_cp = spin(0, 1024, 32)
        self.wave_spectrum_ofdm_symbols = spin(16, 2048, 128)
        self.wave_spectrum_ofdm_symbols.setSingleStep(16)
        self.wave_spectrum_ofdm_seed = spin(0, 2147483647, 2026)

        mode_box = make_form_group("功率谱测试模式", [("链路模式", self.wave_spectrum_mode)])
        self.wave_spectrum_sc_box = make_form_group("单载波功率谱参数", [
            ("调制符号", self.wave_spectrum_sc_modulation),
            ("滤波器类型", self.wave_spectrum_sc_filter_type),
            ("滤波器长度", self.wave_spectrum_sc_filter_len),
            ("过采样率", self.wave_spectrum_sc_sps),
            ("滚降系数", self.wave_spectrum_sc_rolloff),
            ("频谱符号数", self.wave_spectrum_sc_symbols),
            ("纵坐标", self.wave_spectrum_sc_scale),
        ])
        self.wave_spectrum_ofdm_box = make_form_group("OFDM 功率谱参数", [
            ("纵坐标", self.wave_spectrum_ofdm_scale),
            ("子载波数", self.wave_spectrum_ofdm_nsc),
            ("过采样率", self.wave_spectrum_ofdm_sps),
            ("CP 长度", self.wave_spectrum_ofdm_cp),
            ("随机 OFDM 符号数", self.wave_spectrum_ofdm_symbols),
            ("随机种子", self.wave_spectrum_ofdm_seed),
        ])
        hint = QLabel(
            "单载波：实测 Welch 功率谱叠加理想无限长滤波器的解析滚降响应，"
            "纵坐标支持对数功率 dB 与线性归一化功率两种显示；"
            "OFDM：随机 16QAM-OFDM 连续时域数据经分段 FFT 平均计算功率谱。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("CardHint")

        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        for widget in (mode_box, self.wave_spectrum_sc_box,
                       self.wave_spectrum_ofdm_box, hint):
            layout.addWidget(widget)
        self.wave_spectrum_mode.currentTextChanged.connect(
            self._on_wave_spectrum_mode_changed)
        self._on_wave_spectrum_mode_changed(self.wave_spectrum_mode.currentText())
        return wrap

    def _on_wave_spectrum_mode_changed(self, text: str) -> None:
        is_ofdm = text == "OFDM"
        self.wave_spectrum_sc_box.setVisible(not is_ofdm)
        self.wave_spectrum_ofdm_box.setVisible(is_ofdm)
        self._fit_param_stack_height()

    def _build_codec_params(self) -> QWidget:
        self.codec_type_combo = combo(["RS (GF符号)", "LDPC (十六进制)"])
        self.codec_preset_combo = combo([
            "示例用例集", "无错误闭环", "单点错误", "纠错边界 / 压力测试", "自定义"
        ])

        self.codec_rs_n = spin(3, 255, 15)
        self.codec_rs_k = spin(1, 254, 11)
        self.codec_rs_m = combo(["4", "8"])
        self.codec_ldpc_rate = combo(["14/15", "11/15"])
        self.codec_param_stack = QStackedWidget()
        self.codec_param_stack.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.codec_param_stack.setMaximumHeight(190)
        self.codec_param_stack.addWidget(make_form_group("RS 编码参数", [
            ("码长 n", self.codec_rs_n),
            ("信息符号 k", self.codec_rs_k),
            ("有限域 GF(2^m)", self.codec_rs_m),
        ]))
        self.codec_param_stack.addWidget(make_form_group("LDPC 编码参数", [
            ("标准码率", self.codec_ldpc_rate),
        ]))

        self.codec_input_editor = QPlainTextEdit()
        self.codec_input_editor.setMaximumHeight(92)
        self.codec_input_editor.setPlaceholderText("RS：输入逗号分隔的 GF 符号；LDPC：输入十六进制串")
        self.codec_input_editor.setStyleSheet(_MONO_STYLE)
        self.codec_error_positions = QLineEdit()
        self.codec_error_positions.setPlaceholderText("例如：0, 3, 12；留空表示不注入错误")

        self.codec_editor = QPlainTextEdit()
        self.codec_editor.setStyleSheet(_MONO_STYLE)
        self.codec_editor.setMinimumHeight(170)
        self.codec_type_combo.currentTextChanged.connect(self._on_codec_type_changed)

        load_file_btn = QPushButton("选择文件")
        load_file_btn.clicked.connect(self._on_load_file)
        demo_btn = QPushButton("加载示例")
        demo_btn.clicked.connect(self._on_load_demo)

        self.codec_advanced_box = QGroupBox("高级配置：JSON 导入与编辑")
        self.codec_advanced_box.setCheckable(True)
        self.codec_advanced_box.setChecked(False)
        self.codec_advanced_box.setMaximumHeight(42)
        advanced_layout = QVBoxLayout(self.codec_advanced_box)
        advanced_buttons = QHBoxLayout()
        advanced_buttons.addWidget(load_file_btn)
        advanced_buttons.addWidget(demo_btn)
        advanced_buttons.addStretch(1)
        advanced_layout.addLayout(advanced_buttons)
        advanced_layout.addWidget(self.codec_editor)
        for widget in (load_file_btn, demo_btn, self.codec_editor):
            widget.setVisible(False)
            self.codec_advanced_box.toggled.connect(widget.setVisible)
        self.codec_advanced_box.toggled.connect(
            lambda checked: self.codec_advanced_box.setMaximumHeight(320 if checked else 42)
        )
        self.codec_advanced_box.toggled.connect(self._on_codec_advanced_toggled)

        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(make_form_group("测试方案", [
            ("编码类型", self.codec_type_combo),
            ("测试预设", self.codec_preset_combo),
        ]))
        layout.addWidget(self.codec_param_stack)
        self.codec_case_box = make_form_group("自定义算例", [
            ("输入数据", self.codec_input_editor),
            ("错误位置", self.codec_error_positions),
        ])
        layout.addWidget(self.codec_case_box)
        layout.addWidget(self.codec_advanced_box)
        hint = QLabel(
            "推荐先使用预设观察纠错过程；选择“自定义”后可填写输入和错误位置。"
            "原始 JSON 文件仍可在高级配置中导入、编辑并直接运行。"
            "MATLAB 离线结果可写入每个算例的 matlab_reference.encoded / decoded 字段。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("CardHint")
        layout.addWidget(hint)
        wrap.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self._on_load_demo()
        self.codec_preset_combo.currentTextChanged.connect(self._on_codec_preset_changed)
        self._on_codec_preset_changed(self.codec_preset_combo.currentText())
        return wrap

    def _build_precision_params(self) -> QWidget:
        self.prec_mode = combo(["SC", "OFDM"])
        self.prec_duration = dspin(1e-7, 1e-4, 1e-6, decimals=7)
        self.prec_snr = dspin(-20.0, 60.0, 24.0, decimals=1)
        box = make_form_group("浮点精度验证参数", [
            ("链路模式", self.prec_mode),
            ("时长 (s)", self.prec_duration),
            ("SNR (dB)", self.prec_snr),
        ])
        hint = QLabel(
            "运行一次短链路（TX→信道→RX），逐模块校验输出信号数据类型"
            "（期望：比特 uint8、波形符号 complex128、LLR float64）。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("CardHint")
        return self._wrap_with_hint(box, hint)

    def _build_single_rate_params(self) -> QWidget:
        self.single_rate_mode = combo(["OFDM", "SC"])
        self.single_rate_modulation = combo(["64QAM", "QPSK", "16QAM", "256QAM"])
        self.single_rate_code = combo(["LDPC", "RS"])
        self.single_rate_symbol_rate = dspin(1.0, 200.0, 30.0, decimals=1, suffix=" GHz")
        self.single_rate_duration = dspin(1e-7, 1e-4, 1e-6, decimals=7, suffix=" s")
        self.single_rate_threshold = dspin(1.0, 1000.0, 50.0, decimals=1, suffix=" Gbps")
        box = make_form_group("单链路物理层速率参数", [
            ("链路模式", self.single_rate_mode),
            ("调制方式", self.single_rate_modulation),
            ("编码方式", self.single_rate_code),
            ("符号率", self.single_rate_symbol_rate),
            ("测试时长", self.single_rate_duration),
            ("验收门限", self.single_rate_threshold),
        ])
        hint = QLabel("判定口径：实际速率 = 实际发送的信息比特数 / 实际发射波形持续时间。")
        hint.setWordWrap(True)
        hint.setObjectName("CardHint")
        return self._wrap_with_hint(box, hint)

    def _build_total_rate_params(self) -> QWidget:
        self.total_rate_configuration = combo([
            "256QAM / 1024 子载波 / 2×2 MIMO / LDPC(11/15)"
        ])
        self.total_rate_configuration.setEnabled(False)
        self.total_rate_threshold = dspin(0.1, 2.0, 0.5, decimals=3, suffix=" Tbps")
        box = make_form_group("总物理层速率固定验收配置", [
            ("PHY 配置", self.total_rate_configuration),
            ("验收门限", self.total_rate_threshold),
        ])
        hint = QLabel(
            "固定采用 60 GBd、256QAM、1024 子载波、2×2 双空间流、"
            "LDPC(1440,1056) 11/15，CP 长度 32；"
            "实际速率 = 信息比特数 ÷ 波形持续时间。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("CardHint")
        return self._wrap_with_hint(box, hint)

    def _build_calibration_params(self) -> QWidget:
        self.calibration_bits = spin(10000, 10000000, 200000)
        self.calibration_seed = spin(0, 2147483647, 2026)
        box = make_form_group("校准测试参数", [
            ("每个 BER 点比特数", self.calibration_bits),
            ("随机种子", self.calibration_seed),
        ])
        hint = QLabel(
            "QPSK / 16QAM / 64QAM 与 AWGN 理论 BER 曲线比较；"
            "LDPC(1440,1056) 与 RS(255,192) 的 QPSK 编码链路 BER 校准："
            "虚线为同 Eb/N0 下未编码 QPSK 理论，点划线为编码后理论曲线"
            "（RS 为硬判决解析式，LDPC 来自 MATLAB 官方工具同配置仿真）。"
            "每条曲线独立绘图，图下附仿真点明细表格。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("CardHint")
        return self._wrap_with_hint(box, hint)

    def _build_ber_params(self) -> QWidget:
        self.ber_config_combo = combo(
            [option["label"] for option in BER_CONFIG_OPTIONS])
        for index, option in enumerate(BER_CONFIG_OPTIONS):
            self.ber_config_combo.setItemData(index, option["key"])

        self.ber_link_config = QPlainTextEdit()
        self.ber_link_config.setReadOnly(True)
        self.ber_link_config.setStyleSheet(_MONO_STYLE)
        self.ber_link_config.setMinimumHeight(150)
        self.ber_link_config.setMaximumHeight(210)
        self.ber_link_box = QGroupBox("链路配置显示")
        link_layout = QVBoxLayout(self.ber_link_box)
        link_layout.setContentsMargins(14, 18, 14, 14)
        link_layout.addWidget(self.ber_link_config)

        self.ber_snr_min = dspin(-20.0, 100.0, 14.0, decimals=1, suffix=" dB")
        self.ber_snr_max = dspin(-20.0, 100.0, 20.0, decimals=1, suffix=" dB")
        self.ber_snr_step = dspin(0.1, 10.0, 1.0, decimals=1, suffix=" dB")
        self.ber_seed = spin(0, 2147483647, 2026)
        self.ber_quick_bits = spin(100000, 10000000, 3000000)
        self.ber_quick_bits.setSingleStep(100000)
        self.ber_quick_errors = spin(10, 10000, 100)
        self.ber_quick_errors.setSingleStep(10)

        config_box = make_form_group("测试配置选择", [
            ("链路配置", self.ber_config_combo)])
        snr_box = make_form_group("SNR 扫描范围", [
            ("最小值", self.ber_snr_min),
            ("最大值", self.ber_snr_max),
            ("步长", self.ber_snr_step),
        ])
        scan_box = make_form_group("扫描控制", [
            ("每点最大比特数", self.ber_quick_bits),
            ("每点最少误码数", self.ber_quick_errors),
            ("随机种子", self.ber_seed),
        ])
        hint = QLabel(
            "严格按 SNR 扫描范围逐点仿真（无早停、无 95% 置信验证）："
            "每个点按每点最大比特数与最少误码数逐试次累计实测误码率，"
            "0 误码点在曲线上以空心圆标识，目标 BER=1e-6 以红色虚线标出。"
            "切换配置会载入对应链路说明并恢复默认 SNR 扫描范围（可自由修改）。"
            "可随时停止，已完成点会保存到 simulation_results。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("CardHint")

        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        for widget in (config_box, self.ber_link_box, snr_box, scan_box, hint):
            layout.addWidget(widget)
        self.ber_config_combo.currentIndexChanged.connect(self._on_ber_config_changed)
        self._on_ber_config_changed(0)
        return wrap

    def _on_ber_config_changed(self, _index: int) -> None:
        """切换链路配置：刷新链路配置显示并恢复默认 SNR 扫描范围。"""
        option_key = self.ber_config_combo.currentData()
        try:
            text = ber_option_link_text(option_key)
            snr_min, snr_max, snr_step = ber_option_snr_default(option_key)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.ber_link_config.setPlainText(f"链路配置读取失败：{exc}")
            return
        self.ber_link_config.setPlainText(text)
        self.ber_snr_min.setValue(snr_min)
        self.ber_snr_max.setValue(snr_max)
        self.ber_snr_step.setValue(snr_step)

    @staticmethod
    def _wrap_with_hint(box: QWidget, hint: QWidget) -> QWidget:
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(box)
        layout.addWidget(hint)
        wrap.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        return wrap

    # ==================== 右栏 ====================

    def _build_right_panel(self) -> None:
        self.result_stack = QStackedWidget()
        self.result_stack.addWidget(self._build_modulation_results())
        self.result_stack.addWidget(self._build_waveform_time_results())
        self.result_stack.addWidget(self._build_waveform_spectrum_results())
        self.result_stack.addWidget(self._build_codec_results())
        self.result_stack.addWidget(self._build_precision_results())
        self.result_stack.addWidget(self._build_table_results(
            "single_rate", "单链路速率测试", ["指标", "测量值", "门限/参考", "判定"]))
        self.result_stack.addWidget(self._build_table_results(
            "total_rate", "总物理层速率测试", ["指标", "测量值", "门限/参考", "判定"]))
        self.result_stack.addWidget(self._build_calibration_results())
        self.result_stack.addWidget(self._build_ber_results())
        self.add_right_widget(self.result_stack)

    def _make_placeholder(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        label.setStyleSheet(_PLACEHOLDER_STYLE)
        return label

    def _build_modulation_results(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.modulation_placeholder = self._make_placeholder(
            "点击左侧「运行测试」生成发射端星座图")
        layout.addWidget(self.modulation_placeholder)

        self.modulation_card = CardWidget("发射端星座图")
        self.modulation_plot = QLabel()
        self.modulation_plot.setAlignment(Qt.AlignCenter)
        # 高度与星座点网格表、汇总卡整体平衡，保证最小窗口下整页无滚动条
        self.modulation_plot.setMinimumHeight(330)
        self.modulation_card.layout.addWidget(self.modulation_plot)
        self.modulation_card.hide()
        layout.addWidget(self.modulation_card)

        self.modulation_constellation_table = self._make_table(["星座点", "I 分量", "Q 分量"])
        self.modulation_constellation_table.hide()
        layout.addWidget(self.modulation_constellation_table)

        self.modulation_summary_card = _UpdatableSummaryCard("调制信息", ["尚未运行"])
        layout.addWidget(self.modulation_summary_card)
        layout.addStretch(1)
        return page

    def _build_waveform_time_results(self) -> QWidget:
        return self._build_wave_results_page(
            "wave_time", "点击左侧「运行测试」生成时域波形")

    def _build_waveform_spectrum_results(self) -> QWidget:
        return self._build_wave_results_page(
            "wave_spectrum", "点击左侧「运行测试」生成功率谱")

    def _build_wave_results_page(self, prefix: str, placeholder_text: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        placeholder = self._make_placeholder(placeholder_text)
        setattr(self, f"{prefix}_placeholder", placeholder)
        layout.addWidget(placeholder)

        # 结果图卡片容器：按每次运行产出的图数量动态生成（时域/热图/频谱等）
        cards_container = QWidget()
        cards_layout = QVBoxLayout(cards_container)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(12)
        setattr(self, f"{prefix}_cards_container", cards_container)
        setattr(self, f"{prefix}_cards_layout", cards_layout)
        layout.addWidget(cards_container)

        checks_card = _UpdatableSummaryCard("验证结果", ["尚未运行"])
        setattr(self, f"{prefix}_checks_card", checks_card)
        layout.addWidget(checks_card)
        layout.addStretch(1)
        return page

    def _build_codec_results(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.codec_placeholder = self._make_placeholder(
            "选择编码方案和测试预设，然后点击左侧「运行测试」\n"
            "结果将按 输入 → 编码 → 注错 → 译码 → 比对 展示"
        )
        self.codec_placeholder.setMaximumHeight(260)
        layout.addWidget(self.codec_placeholder)

        self.codec_overview_card = _UpdatableSummaryCard("本次测试概览", ["尚未运行"])
        self.codec_overview_card.hide()
        layout.addWidget(self.codec_overview_card)

        self.codec_flow_card = CardWidget("编译码流程")
        self.codec_flow_label = QLabel("输入  →  分包 / 分块  →  编码  →  注入错误  →  译码  →  一致性比对")
        self.codec_flow_label.setAlignment(Qt.AlignCenter)
        self.codec_flow_label.setWordWrap(True)
        self.codec_flow_label.setStyleSheet(
            "padding: 14px; border-radius: 8px; background: #F5F8FF; color: #41516B; font-weight: 600;"
        )
        self.codec_flow_card.layout.addWidget(self.codec_flow_label)
        self.codec_flow_card.hide()
        layout.addWidget(self.codec_flow_card)

        self.codec_table = self._make_table([
            "算例", "输入长度", "编码长度", "包 / 块", "错误数", "MATLAB 对比", "判定"
        ])
        self.codec_table.hide()
        layout.addWidget(self.codec_table, 1)

        self.codec_detail_tabs = QTabWidget()
        self.codec_detail_overview = self._readonly_codec_detail("选中算例后查看执行摘要")
        self.codec_detail_data = self._readonly_codec_detail("选中算例后查看输入、编码流和译码结果")
        self.codec_detail_diagnostics = self._readonly_codec_detail("选中算例后查看错误与译码诊断")
        self.codec_detail_tabs.addTab(self.codec_detail_overview, "概览")
        self.codec_detail_tabs.addTab(self.codec_detail_data, "数据对比")
        self.codec_detail_tabs.addTab(self.codec_detail_diagnostics, "错误与诊断")
        self.codec_detail_tabs.setMaximumHeight(230)
        self.codec_detail_tabs.hide()
        self.codec_table.itemSelectionChanged.connect(self._on_codec_row_selected)
        layout.addWidget(self.codec_detail_tabs)

        self.codec_summary_card = _UpdatableSummaryCard("用例统计", ["尚未运行"])
        self.codec_summary_card.hide()
        layout.addWidget(self.codec_summary_card)
        return page

    @staticmethod
    def _readonly_codec_detail(placeholder: str) -> QPlainTextEdit:
        editor = QPlainTextEdit()
        editor.setReadOnly(True)
        editor.setPlaceholderText(placeholder)
        editor.setStyleSheet(_MONO_STYLE)
        return editor

    def _build_precision_results(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.prec_placeholder = self._make_placeholder("点击左侧「运行测试」执行全链路精度校验")
        layout.addWidget(self.prec_placeholder)

        self.prec_table = self._make_table(["模块", "输出信号", "数据类型", "字节数", "判定"])
        layout.addWidget(self.prec_table, 1)

        self.prec_summary_card = _UpdatableSummaryCard("精度总览", ["尚未运行"])
        layout.addWidget(self.prec_summary_card)
        layout.addStretch(1)
        return page

    def _build_table_results(self, prefix: str, title: str, columns: List[str]) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        placeholder = self._make_placeholder(f"点击左侧「运行测试」执行{title}")
        table = self._make_table(columns)
        summary = _UpdatableSummaryCard(title, ["尚未运行"])
        setattr(self, f"{prefix}_placeholder", placeholder)
        setattr(self, f"{prefix}_table", table)
        setattr(self, f"{prefix}_summary_card", summary)
        layout.addWidget(placeholder)
        layout.addWidget(table, 1)
        layout.addWidget(summary)
        layout.addStretch(1)
        return page

    def _build_calibration_results(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        self.calibration_placeholder = self._make_placeholder(
            "点击左侧「运行测试」执行调制与编码 BER 校准")
        # 每条校准曲线一张图，图下附仿真点明细表格
        self.calibration_cards_container = QWidget()
        self.calibration_cards_layout = QVBoxLayout(self.calibration_cards_container)
        self.calibration_cards_layout.setContentsMargins(0, 0, 0, 0)
        self.calibration_cards_layout.setSpacing(12)
        self.calibration_summary_card = _UpdatableSummaryCard("校准总览", ["尚未运行"])
        layout.addWidget(self.calibration_placeholder)
        layout.addWidget(self.calibration_cards_container)
        layout.addWidget(self.calibration_summary_card)
        return page

    def _build_ber_results(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        self.ber_placeholder = self._make_placeholder(
            "点击左侧「运行测试」执行全链路 BER 扫描")
        self.ber_progress_label = QLabel("待运行")
        self.ber_progress_label.setWordWrap(True)
        self.ber_progress = QProgressBar()
        self.ber_progress.setRange(0, 100)
        self.ber_progress.setValue(0)
        progress_card = CardWidget("运行进度")
        progress_card.layout.addWidget(self.ber_progress_label)
        progress_card.layout.addWidget(self.ber_progress)
        self.ber_plot_card = CardWidget("BER 曲线")
        self.ber_plot = QLabel()
        self.ber_plot.setAlignment(Qt.AlignCenter)
        self.ber_plot.setMinimumHeight(390)
        self.ber_plot_card.layout.addWidget(self.ber_plot)
        self.ber_plot_card.hide()
        self.ber_table = self._make_table([
            "SNR/dB", "Eb/N0/dB", "误码数", "比特数",
            "实测 BER", "运行次数",
        ])
        self.ber_summary_card = _UpdatableSummaryCard("BER 扫描总览", ["尚未运行"])
        layout.addWidget(self.ber_placeholder)
        layout.addWidget(progress_card)
        layout.addWidget(self.ber_plot_card)
        layout.addWidget(self.ber_table, 1)
        layout.addWidget(self.ber_summary_card)
        return page

    @staticmethod
    def _make_table(columns: List[str]) -> QTableWidget:
        table = QTableWidget(0, len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    # ==================== 交互 ====================

    def _selected_index(self) -> int:
        for idx, rb in enumerate(self.test_radios):
            if rb.isChecked():
                return idx
        return 0

    def _selected_test(self) -> str:
        return _TEST_KEYS[self._selected_index()]

    def _on_radio_toggled(self, checked: bool) -> None:
        if checked:
            self._sync_stacks()

    def _sync_stacks(self) -> None:
        idx = self._selected_index()
        self.param_stack.setCurrentIndex(idx)
        self.result_stack.setCurrentIndex(idx)
        self.stop_button.setVisible(idx == 8)
        if idx == 3:
            self.param_stack.setMaximumHeight(760 if self.codec_advanced_box.isChecked() else 440)
        else:
            # QStackedWidget 默认按所有页面中最大的 sizeHint 请求高度，导致
            # 参数较少的速率页出现大片空白；这里只采用当前页的紧凑高度。
            self._fit_param_stack_height()

    def _fit_param_stack_height(self) -> None:
        """把参数区高度贴合当前页面的 sizeHint（编码页固定高度除外）。

        波形测试页切换链路模式会增删参数组，需要重新贴合，否则内容会被
        压进旧模式的页面高度，出现参数行挤在一起的现象。
        """
        if self._selected_test() == "codec":
            return
        current = self.param_stack.currentWidget()
        if current is None:
            return
        current.adjustSize()
        self.param_stack.setMaximumHeight(max(120, current.sizeHint().height()))

    def _demo_path(self) -> Path:
        name = "rs_demo.json" if "RS" in self.codec_type_combo.currentText() else "ldpc_demo.json"
        return Path(__file__).resolve().parent.parent.parent / "test_cases" / name

    def _on_load_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择测试用例文件", "", "JSON 文件 (*.json)")
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            self._set_badge("已失败")
            self.codec_placeholder.setText(f"文件读取失败：{exc}")
            self.codec_placeholder.show()
            return
        self.codec_editor.setPlainText(text)
        self.codec_advanced_box.setChecked(True)

    def _on_load_demo(self) -> None:
        try:
            text = self._demo_path().read_text(encoding="utf-8")
        except OSError as exc:
            self.codec_editor.setPlainText(
                f"// 示例文件读取失败：{exc}\n// 可点击「选择文件」手动加载用例")
            return
        self.codec_editor.setPlainText(text)
        if hasattr(self, "codec_preset_combo"):
            self.codec_preset_combo.setCurrentIndex(0)

    def _on_codec_type_changed(self, _text: str) -> None:
        """切换编码类型时同步加载对应示例，避免界面选择与实际 JSON 用例不一致。"""
        self.codec_param_stack.setCurrentIndex(0 if "RS" in _text else 1)
        self._on_load_demo()

    def _on_codec_advanced_toggled(self, checked: bool) -> None:
        if self._selected_test() == "codec":
            self.param_stack.setMaximumHeight(760 if checked else 440)

    def _on_codec_preset_changed(self, text: str) -> None:
        is_custom = text == "自定义"
        self.codec_case_box.setVisible(is_custom)
        if not is_custom:
            return
        if "RS" in self.codec_type_combo.currentText():
            m = int(self.codec_rs_m.currentText())
            values = [str(i % (2 ** m)) for i in range(self.codec_rs_k.value())]
            self.codec_input_editor.setPlainText(", ".join(values))
        else:
            rate = self.codec_ldpc_rate.currentText()
            byte_count = 168 if rate == "14/15" else 132
            seed = b"THz-802.15.3d|"
            raw = (seed * ((byte_count + len(seed) - 1) // len(seed)))[:byte_count]
            self.codec_input_editor.setPlainText(raw.hex().upper())
        self.codec_error_positions.clear()

    @staticmethod
    def _parse_codec_positions(text: str) -> List[int]:
        clean = text.replace("，", ",").strip()
        if not clean:
            return []
        try:
            values = [int(part.strip()) for part in clean.split(",") if part.strip()]
        except ValueError as exc:
            raise ValueError("错误位置必须是以逗号分隔的整数，例如：0, 3, 12") from exc
        if any(value < 0 for value in values):
            raise ValueError("错误位置不能为负数")
        return values

    def _codec_config_from_form(self) -> dict:
        preset = self.codec_preset_combo.currentText()
        is_rs = "RS" in self.codec_type_combo.currentText()
        if preset == "示例用例集":
            # 每次运行都重新读取磁盘示例，避免已打开页面仍持有旧版示例内容。
            return json.loads(self._demo_path().read_text(encoding="utf-8"))

        if is_rs:
            n, k, m = self.codec_rs_n.value(), self.codec_rs_k.value(), int(self.codec_rs_m.currentText())
            if n <= k:
                raise ValueError(f"RS 参数需满足 n > k，当前 n={n}、k={k}")
            if n > 2 ** m - 1:
                raise ValueError(f"GF(2^{m}) 下 RS 码长 n 不能超过 {2 ** m - 1}")
            if preset == "自定义":
                raw = self.codec_input_editor.toPlainText().replace("，", ",")
                try:
                    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
                except ValueError as exc:
                    raise ValueError("RS 输入必须是以逗号分隔的 GF 符号整数") from exc
                positions = self._parse_codec_positions(self.codec_error_positions.text())
            else:
                values = [i % (2 ** m) for i in range(k)]
                positions = [] if preset == "无错误闭环" else [0]
                if preset == "纠错边界 / 压力测试":
                    positions = list(range(max(1, (n - k) // 2)))
            if not values:
                raise ValueError("RS 输入不能为空")
            if any(value < 0 or value >= 2 ** m for value in values):
                raise ValueError(f"RS 输入符号必须位于 0～{2 ** m - 1}")
            encoded_len = ((len(values) + k - 1) // k) * n
            if any(pos >= encoded_len for pos in positions):
                raise ValueError(f"错误位置必须小于编码流长度 {encoded_len}")
            errors = [{"pos": pos, "value": (pos + 1) % (2 ** m)} for pos in positions]
            return {
                "code_type": "RS",
                "rs_params": {"n": n, "k": k, "m": m},
                "cases": [{"name": preset, "input": values, "errors": errors}],
            }

        rate = self.codec_ldpc_rate.currentText()
        k_bits = 1344 if rate == "14/15" else 1056
        if preset == "自定义":
            hex_in = "".join(self.codec_input_editor.toPlainText().split()).upper()
            positions = self._parse_codec_positions(self.codec_error_positions.text())
        else:
            byte_count = k_bits // 8
            seed = b"THz-802.15.3d|"
            raw = (seed * ((byte_count + len(seed) - 1) // len(seed)))[:byte_count]
            hex_in = raw.hex().upper()
            positions = [] if preset == "无错误闭环" else [0]
            if preset == "纠错边界 / 压力测试":
                positions = list(range(0, 128, 8))
        if not hex_in or len(hex_in) % 2 or any(ch not in "0123456789ABCDEF" for ch in hex_in):
            raise ValueError("LDPC 输入必须是非空、偶数长度的十六进制串")
        encoded_len = ((len(hex_in) * 4 + k_bits - 1) // k_bits) * 1440
        if any(pos >= encoded_len for pos in positions):
            raise ValueError(f"错误位置必须小于编码流长度 {encoded_len}")
        return {
            "code_type": "LDPC",
            "ldpc_params": {"rate": rate},
            "cases": [{"name": preset, "input_hex": hex_in, "error_bits": positions}],
        }

    def _on_run_clicked(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        test_type = self._selected_test()
        try:
            payload = self._build_payload(test_type)
        except (ValueError, json.JSONDecodeError) as exc:
            self._set_badge("已失败")
            if test_type == "codec":
                self.codec_placeholder.setText(f"配置有误：{exc}")
                self.codec_placeholder.show()
            return
        self._reset_results(test_type)
        self._set_badge("运行中")
        self.run_button.setEnabled(False)
        self._worker = FunctionalTestWorker(test_type, payload, parent=self)
        self._worker.result_ready.connect(self._on_result)
        self._worker.failed.connect(self._on_failed)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self.stop_button.setEnabled(test_type == "ber")
        self._worker.start()

    def _on_stop_clicked(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self.stop_button.setEnabled(False)
            self.ber_progress_label.setText("正在安全停止；当前帧完成后保存已有结果…")

    def _on_progress(self, progress: dict) -> None:
        if self._selected_test() != "ber":
            return
        # 单配置扫描：进度按当前链路配置已完成的扫描点计算。
        point_index = int(progress.get("point_index", 0))
        point_count = max(1, int(progress.get("point_count", 11)))
        local = min(0.98, point_index / point_count) if point_index else 0.0
        percent = int(100.0 * local)
        self.ber_progress.setValue(min(percent, 99))
        self.ber_progress_label.setText(
            f"{progress.get('config_label', '-')}｜{progress.get('phase', '-')}｜"
            f"SNR={float(progress.get('snr_db', 0.0)):.1f} dB｜"
            f"误码 {int(progress.get('total_errors', 0))} / "
            f"{int(progress.get('total_bits', 0)):,} bit"
        )

    def _build_payload(self, test_type: str) -> dict:
        if test_type == "modulation":
            return {
                "modulation": self.modulation_type.currentText(),
                "num_symbols": self.modulation_symbols.value(),
                "show_points": self.modulation_show.value(),
                "random_seed": self.modulation_seed.value(),
            }
        if test_type == "waveform_time":
            is_ofdm = self.wave_time_mode.currentText() == "OFDM"
            payload = {"link_mode": "ofdm" if is_ofdm else "sc-fde"}
            if is_ofdm:
                payload.update({
                    "ofdm_subcarriers": self.wave_time_ofdm_nsc.value(),
                    "ofdm_oversampling": self.wave_time_ofdm_sps.value(),
                    "ofdm_cp_length": self.wave_time_ofdm_cp.value(),
                    "ofdm_time_symbol_count": self.wave_time_ofdm_symbols.value(),
                    "ofdm_time_start_subcarrier": self.wave_time_ofdm_start.value(),
                    "ofdm_time_subcarrier_step": self.wave_time_ofdm_step.value(),
                    "ofdm_heatmap_symbol_count": self.wave_time_ofdm_heat_symbols.value(),
                    "ofdm_heatmap_start_subcarrier": self.wave_time_ofdm_heat_start.value(),
                    "ofdm_heatmap_subcarrier_step": self.wave_time_ofdm_heat_step.value(),
                    "ofdm_heatmap_axis_margin": self.wave_time_ofdm_heat_margin.value(),
                    "ofdm_heatmap_floor_db": self.wave_time_ofdm_heat_floor.value(),
                })
            else:
                payload.update({
                    "sc_modulation": self.wave_time_sc_modulation.currentText(),
                    "sc_symbol_indexes": tuple(
                        selector.value()
                        for selector, _label in self.wave_time_symbol_selectors),
                    "sc_pulse_count": self.wave_time_sc_pulses.value(),
                    "sc_filter_type": self.wave_time_sc_filter_type.currentText().lower(),
                    "sc_filter_length": self.wave_time_sc_filter_len.value(),
                    "sc_oversampling": self.wave_time_sc_sps.value(),
                    "sc_rolloff": self.wave_time_sc_rolloff.value(),
                })
            return payload
        if test_type == "waveform_spectrum":
            is_ofdm = self.wave_spectrum_mode.currentText() == "OFDM"
            payload = {"link_mode": "ofdm" if is_ofdm else "sc-fde"}
            if is_ofdm:
                payload.update({
                    "ofdm_spectrum_subcarriers": self.wave_spectrum_ofdm_nsc.value(),
                    "ofdm_spectrum_oversampling": self.wave_spectrum_ofdm_sps.value(),
                    "ofdm_spectrum_cp_length": self.wave_spectrum_ofdm_cp.value(),
                    "ofdm_spectrum_scale": self.wave_spectrum_ofdm_scale.currentText(),
                    "ofdm_spectrum_num_symbols": self.wave_spectrum_ofdm_symbols.value(),
                    "ofdm_spectrum_random_seed": self.wave_spectrum_ofdm_seed.value(),
                })
            else:
                payload.update({
                    "sc_modulation": self.wave_spectrum_sc_modulation.currentText(),
                    "sc_filter_type": self.wave_spectrum_sc_filter_type.currentText().lower(),
                    "sc_filter_length": self.wave_spectrum_sc_filter_len.value(),
                    "sc_oversampling": self.wave_spectrum_sc_sps.value(),
                    "sc_rolloff": self.wave_spectrum_sc_rolloff.value(),
                    "sc_num_symbols": self.wave_spectrum_sc_symbols.value(),
                    "sc_scale": self.wave_spectrum_sc_scale.currentText(),
                })
            return payload
        if test_type == "codec":
            if self.codec_advanced_box.isChecked():
                config_text = self.codec_editor.toPlainText()
            else:
                config = self._codec_config_from_form()
                config_text = json.dumps(config, ensure_ascii=False, indent=2)
                self.codec_editor.setPlainText(config_text)
            return {"config_text": config_text}
        if test_type == "precision":
            return {
                "link_mode": "sc-fde" if self.prec_mode.currentText() == "SC" else "ofdm",
                "duration": self.prec_duration.value(),
                "SNRdB": self.prec_snr.value(),
            }
        if test_type == "single_link_rate":
            return {
                "link_mode": "sc-fde" if self.single_rate_mode.currentText() == "SC" else "ofdm",
                "modulation": self.single_rate_modulation.currentText(),
                "code_type": self.single_rate_code.currentText(),
                "symbol_rate_ghz": self.single_rate_symbol_rate.value(),
                "duration": self.single_rate_duration.value(),
                "threshold_gbps": self.single_rate_threshold.value(),
            }
        if test_type == "total_phy_rate":
            return {"threshold_tbps": self.total_rate_threshold.value()}
        if test_type == "ber":
            return {
                "config_key": self.ber_config_combo.currentData(),
                "snr_min": self.ber_snr_min.value(),
                "snr_max": self.ber_snr_max.value(),
                "snr_step": self.ber_snr_step.value(),
                "quick_max_bits": self.ber_quick_bits.value(),
                "quick_min_errors": self.ber_quick_errors.value(),
                "random_seed": self.ber_seed.value(),
            }
        return {
            "bits_per_point": self.calibration_bits.value(),
            "random_seed": self.calibration_seed.value(),
        }

    def _on_result(self, test_type: str, result: dict) -> None:
        self._set_badge("已中断" if result.get("cancelled") else
                        ("已完成" if result.get("ok") else "已失败"))
        if test_type == "modulation":
            self._render_modulation(result)
        elif test_type == "waveform_time":
            self._render_waveform(result, "wave_time")
        elif test_type == "waveform_spectrum":
            self._render_waveform(result, "wave_spectrum")
        elif test_type == "codec":
            self._render_codec(result)
        elif test_type == "precision":
            self._render_table_result(result, self.prec_table,
                                      self.prec_placeholder, self.prec_summary_card)
        elif test_type == "single_link_rate":
            self._render_table_result(result, self.single_rate_table,
                                      self.single_rate_placeholder, self.single_rate_summary_card)
        elif test_type == "total_phy_rate":
            self._render_table_result(result, self.total_rate_table,
                                      self.total_rate_placeholder, self.total_rate_summary_card)
        elif test_type == "function_calibration":
            self._render_calibration(result)
        else:
            self._render_ber(result)

    def _on_failed(self, test_type: str, message: str) -> None:
        self._set_badge("已失败")
        target = {
            "modulation": self.modulation_placeholder,
            "waveform_time": self.wave_time_placeholder,
            "waveform_spectrum": self.wave_spectrum_placeholder,
            "codec": self.codec_placeholder,
            "precision": self.prec_placeholder,
            "single_link_rate": self.single_rate_placeholder,
            "total_phy_rate": self.total_rate_placeholder,
            "function_calibration": self.calibration_placeholder,
            "ber": self.ber_placeholder,
        }[test_type]
        target.setText(f"测试失败：{message}")
        target.show()

    def _on_worker_finished(self) -> None:
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self._worker is not None:
            self._worker.deleteLater()
        self._worker = None

    def _set_badge(self, text: str) -> None:
        self.status_badge.setText(text)
        color = STATUS_COLORS.get(text, "#7C869B")
        self.status_badge.setStyleSheet(
            f"padding: 4px 10px; border-radius: 12px; background: {color}22;"
            f" color: {color}; font-weight: 700;")

    # ==================== 结果渲染 ====================

    def _reset_results(self, test_type: str) -> None:
        if test_type == "modulation":
            self.modulation_placeholder.setText("运行中，请稍候…")
            self.modulation_placeholder.show()
            self.modulation_card.hide()
            self.modulation_plot.clear()
            self.modulation_constellation_table.hide()
            self.modulation_constellation_table.setRowCount(0)
            self._modulation_pixmaps = []
            self.modulation_summary_card.set_lines(["运行中…"])
        elif test_type == "waveform_time":
            self.wave_time_placeholder.setText("运行中，请稍候…")
            self.wave_time_placeholder.show()
            self._clear_wave_cards("wave_time")
            self._wave_time_pixmaps = []
            self.wave_time_checks_card.set_lines(["运行中…"])
        elif test_type == "waveform_spectrum":
            self.wave_spectrum_placeholder.setText("运行中，请稍候…")
            self.wave_spectrum_placeholder.show()
            self._clear_wave_cards("wave_spectrum")
            self._wave_spectrum_pixmaps = []
            self.wave_spectrum_checks_card.set_lines(["运行中…"])
        elif test_type == "codec":
            self.codec_placeholder.setText("运行中，请稍候…")
            self.codec_placeholder.show()
            self.codec_table.setRowCount(0)
            self.codec_detail_overview.clear()
            self.codec_detail_data.clear()
            self.codec_detail_diagnostics.clear()
            self._codec_detail_lines = []
            self._codec_case_results = []
            self.codec_overview_card.set_lines(["运行中…"])
            self.codec_overview_card.show()
            self.codec_flow_card.show()
            self.codec_table.hide()
            self.codec_detail_tabs.hide()
            self.codec_flow_label.setText("输入  →  分包 / 分块  →  编码  →  注入错误  →  译码  →  一致性比对")
            self.codec_summary_card.set_lines(["运行中…"])
        elif test_type == "precision":
            self.prec_placeholder.setText("运行中，请稍候…")
            self.prec_placeholder.show()
            self.prec_table.setRowCount(0)
            self.prec_summary_card.set_lines(["运行中…"])
        elif test_type == "single_link_rate":
            self.single_rate_placeholder.setText("运行中，请稍候…")
            self.single_rate_placeholder.show()
            self.single_rate_table.setRowCount(0)
            self.single_rate_summary_card.set_lines(["运行中…"])
        elif test_type == "total_phy_rate":
            self.total_rate_placeholder.setText("运行中，请稍候…")
            self.total_rate_placeholder.show()
            self.total_rate_table.setRowCount(0)
            self.total_rate_summary_card.set_lines(["运行中…"])
        elif test_type == "function_calibration":
            self.calibration_placeholder.setText("运行中，请稍候…")
            self.calibration_placeholder.show()
            self._clear_calibration_cards()
            self._calibration_pixmaps = []
            self.calibration_summary_card.set_lines(["运行中…"])
        else:
            self.ber_placeholder.setText("运行中，请稍候…")
            self.ber_placeholder.show()
            self.ber_plot_card.hide()
            self.ber_plot.clear()
            self._ber_pixmaps = []
            self.ber_table.setRowCount(0)
            self.ber_progress.setValue(0)
            self.ber_progress_label.setText("正在初始化 BER 扫描…")
            self.ber_summary_card.set_lines(["运行中…"])

    def _render_modulation(self, result: dict) -> None:
        self._modulation_pixmaps = []
        plots = result.get("plots", [])
        if plots:
            pm = QPixmap()
            if pm.loadFromData(plots[0].get("png", b""), "PNG"):
                self._modulation_pixmaps.append((self.modulation_plot, pm))
                self.modulation_card.show()
        self.modulation_placeholder.hide()
        self._apply_result_pixmaps(self._modulation_pixmaps)
        lines = [f"{item['label']}：{item['value']}" for item in result.get("summary", [])]
        self.modulation_summary_card.set_lines(lines or ["-"])

        # 星座点 IQ 网格表：行=Q（自上而下递减）、列=I（自左向右递增），
        # 每个星座点一个单元格，直接显示 (Q, I) 坐标值（QPSK 2×2、
        # 16QAM 4×4、64QAM 8×8），无需滚动即可看全。
        grid = (result.get("data") or {}).get("constellation_grid") or {}
        grid_cells = grid.get("cells", [])
        table_widget = self.modulation_constellation_table
        if grid_cells:
            i_levels = grid.get("i_levels", [])
            q_levels = grid.get("q_levels", [])
            table_widget.verticalHeader().setVisible(True)
            table_widget.setColumnCount(len(i_levels))
            table_widget.setHorizontalHeaderLabels([f"I={value:+.3f}" for value in i_levels])
            table_widget.setRowCount(len(q_levels))
            table_widget.setVerticalHeaderLabels([f"Q={value:+.3f}" for value in q_levels])
            for row_index, row_cells in enumerate(grid_cells):
                for column_index, cell in enumerate(row_cells):
                    # 单元格按 (Q, I) 坐标形式两行显示：第一行 Q、第二行 I
                    text = "—" if cell is None else f"{cell[0]:+.3f}\n{cell[1]:+.3f}"
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignCenter)
                    table_widget.setItem(row_index, column_index, item)
            table_widget.resizeRowsToContents()
            for row_index in range(table_widget.rowCount()):
                table_widget.setRowHeight(row_index, table_widget.rowHeight(row_index) + 3)
            table_widget.resizeColumnsToContents()
            # 网格列均分拉伸：等宽列与星座图方型排布对应，且保证任意
            # 窗口宽度下都不出现横向滚动条
            table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            # 表格高度设为完整内容高度，全部行直接显示，不出现表内纵向滚动条
            table_widget.setMinimumHeight(
                table_widget.horizontalHeader().height()
                + sum(table_widget.rowHeight(r) for r in range(table_widget.rowCount()))
                + 2 * table_widget.frameWidth() + 2)
            table_widget.setVisible(True)
        else:
            # 旧结果回退：逐点 I/Q 列表
            table = result.get("table") or {}
            columns = table.get("columns", ["星座点", "I 分量", "Q 分量"])
            rows = table.get("rows", [])
            table_widget.verticalHeader().setVisible(False)
            table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
            table_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            table_widget.setColumnCount(len(columns))
            table_widget.setHorizontalHeaderLabels(columns)
            table_widget.setRowCount(len(rows))
            for row_index, row in enumerate(rows):
                for column_index, value in enumerate(row):
                    table_widget.setItem(
                        row_index, column_index, QTableWidgetItem(str(value)))
            table_widget.resizeColumnsToContents()
            table_widget.setVisible(bool(rows))

    def _clear_wave_cards(self, prefix: str) -> None:
        layout = getattr(self, f"{prefix}_cards_layout")
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _render_waveform(self, result: dict, prefix: str) -> None:
        self._clear_wave_cards(prefix)
        pixmaps: List[tuple] = []
        layout = getattr(self, f"{prefix}_cards_layout")
        for plot in result.get("plots", []):
            card = CardWidget(plot.get("title", ""))
            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            # 高分辨率源图按控件尺寸平滑缩小；较高的展示区避免密集波形与标注
            # 在纵向被过度压缩。
            label.setMinimumHeight(320)
            card.layout.addWidget(label)
            layout.addWidget(card)
            pm = QPixmap()
            if pm.loadFromData(plot.get("png", b""), "PNG"):
                pixmaps.append((label, pm))
        setattr(self, f"_{prefix}_pixmaps", pixmaps)
        getattr(self, f"{prefix}_placeholder").hide()
        self._apply_result_pixmaps(pixmaps)
        lines = [
            f"{'通过' if c['ok'] else '失败'} | {c['name']}：{c['detail']}"
            for c in result.get("checks", [])
        ]
        getattr(self, f"{prefix}_checks_card").set_lines(lines or ["无校验项"])

    @staticmethod
    def _apply_result_pixmaps(pixmaps: List[tuple]) -> None:
        for label, pm in pixmaps:
            size = label.size()
            if size.width() > 0 and size.height() > 0:
                label.setPixmap(pm.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _render_table_result(self, result: dict, table: QTableWidget,
                             placeholder: QLabel, summary_card: _UpdatableSummaryCard) -> None:
        placeholder.hide()
        columns = result.get("table", {}).get("columns", [])
        rows = result.get("table", {}).get("rows", [])
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                table.setItem(r, c, QTableWidgetItem(str(value)))
        table.resizeColumnsToContents()
        for c in range(len(columns)):
            # 上限放宽以容纳速率表的“实际速率计算”公式列（~520px）
            table.setColumnWidth(c, min(table.columnWidth(c), 560))
        self._codec_detail_lines = result.get("detail_lines", [])
        lines = [f"{item['label']}：{item['value']}" for item in result.get("summary", [])]
        summary_card.set_lines(lines or ["-"])

    def _clear_calibration_cards(self) -> None:
        while self.calibration_cards_layout.count():
            item = self.calibration_cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _render_calibration(self, result: dict) -> None:
        self._clear_calibration_cards()
        self._calibration_pixmaps = []
        plots = result.get("plots", [])
        tables = result.get("tables", [])
        for index, plot in enumerate(plots):
            # 校准图卡片
            card = CardWidget(plot.get("title", ""))
            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumHeight(340)
            card.layout.addWidget(label)
            self.calibration_cards_layout.addWidget(card)
            pm = QPixmap()
            if pm.loadFromData(plot.get("png", b""), "PNG"):
                self._calibration_pixmaps.append((label, pm))
            # 图下仿真点明细表格
            if index < len(tables):
                table_info = tables[index]
                table = self._make_table(table_info.get("columns", []))
                rows = table_info.get("rows", [])
                table.setRowCount(len(rows))
                for row_index, row in enumerate(rows):
                    for column_index, value in enumerate(row):
                        table.setItem(row_index, column_index,
                                     QTableWidgetItem(str(value)))
                table.resizeColumnsToContents()
                table.setMaximumHeight(36 + 30 * len(rows))
                self.calibration_cards_layout.addWidget(table)
        self.calibration_placeholder.hide()
        lines = [f"{item['label']}：{item['value']}"
                 for item in result.get("summary", [])]
        self.calibration_summary_card.set_lines(lines or ["-"])
        self._apply_result_pixmaps(self._calibration_pixmaps)

    def _render_ber(self, result: dict) -> None:
        self._ber_pixmaps = []
        plots = result.get("plots", [])
        if plots:
            pm = QPixmap()
            if pm.loadFromData(plots[0].get("png", b""), "PNG"):
                self._ber_pixmaps.append((self.ber_plot, pm))
                self.ber_plot_card.show()
        self._render_table_result(
            result, self.ber_table, self.ber_placeholder, self.ber_summary_card)
        self.ber_progress.setValue(100 if not result.get("cancelled") else self.ber_progress.value())
        self.ber_progress_label.setText(
            "测试已停止，已完成结果已保存" if result.get("cancelled") else
            "测试完成，全部扫描点已仿真")
        self._apply_result_pixmaps(self._ber_pixmaps)

    def _render_codec(self, result: dict) -> None:
        self.codec_placeholder.hide()
        self.codec_overview_card.show()
        self.codec_flow_card.show()
        self.codec_table.show()
        self.codec_detail_tabs.show()
        self._codec_detail_lines = result.get("detail_lines", [])
        self._codec_case_results = result.get("case_results", [])
        summary = {item.get("label", ""): item.get("value", "") for item in result.get("summary", [])}
        elapsed = result.get("elapsed_ms", 0.0)
        self.codec_overview_card.set_lines([
            f"方案：{summary.get('编码方案', '-')}    用例：{summary.get('算例总数', '0')}",
            f"通过：{summary.get('通过', '0')}    失败：{summary.get('失败', '0')}    用时：{elapsed:.1f} ms",
        ])

        if self._codec_case_results:
            columns = ["算例", "输入长度", "编码长度", "包 / 块", "错误数", "MATLAB 对比", "判定"]
            rows = [[
                case.get("name", ""), case.get("input_size", ""), case.get("encoded_size", ""),
                case.get("block_count", ""), case.get("error_count", ""),
                case.get("matlab_comparison", "未提供 MATLAB 参考结果"), case.get("status", ""),
            ] for case in self._codec_case_results]
        else:
            columns = result.get("table", {}).get("columns", [])
            rows = result.get("table", {}).get("rows", [])

        self.codec_table.setColumnCount(len(columns))
        self.codec_table.setHorizontalHeaderLabels(columns)
        self.codec_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row):
                self.codec_table.setItem(row_index, column_index, QTableWidgetItem(str(value)))
        self.codec_table.resizeColumnsToContents()
        self.codec_table.horizontalHeader().setStretchLastSection(True)
        if rows:
            self.codec_table.selectRow(0)

    def _show_codec_case(self, case: dict) -> None:
        status = case.get("status", "未知")
        mark = "✓" if status == "通过" else "✕"
        self.codec_flow_label.setText(
            f"输入 {case.get('input_size', '-')}  →  "
            f"{case.get('block_count', '-')} 包 / 块  →  "
            f"编码 {case.get('encoded_size', '-')}  →  "
            f"注错 {case.get('error_count', 0)}  →  译码  →  {mark} {status}"
        )
        self.codec_detail_overview.setPlainText(case.get("overview", ""))
        self.codec_detail_data.setPlainText(
            f"原始输入\n{case.get('input_data', '-')}\n\n"
            f"本地 Python 编码结果\n{case.get('encoded_data', '-')}\n\n"
            f"MATLAB 编码结果\n{case.get('matlab_encoded', '未提供')}\n\n"
            f"接收数据（注错后）\n{case.get('received_data', '-')}\n\n"
            f"本地 Python 译码结果\n{case.get('decoded_data', '-')}\n\n"
            f"MATLAB 译码结果\n{case.get('matlab_decoded', '未提供')}\n\n"
            f"对比结论\n{case.get('matlab_comparison', '未提供 MATLAB 参考结果')}"
        )
        self.codec_detail_diagnostics.setPlainText(
            f"错误注入\n{case.get('errors', '无')}\n\n"
            f"译码诊断\n{case.get('diagnostics', '-')}"
        )

    def _on_codec_row_selected(self) -> None:
        row = self.codec_table.currentRow()
        if 0 <= row < len(self._codec_case_results):
            self._show_codec_case(self._codec_case_results[row])
        elif 0 <= row < len(self._codec_detail_lines):
            self.codec_detail_overview.setPlainText(self._codec_detail_lines[row])

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_result_pixmaps(self._modulation_pixmaps)
        self._apply_result_pixmaps(self._wave_time_pixmaps)
        self._apply_result_pixmaps(self._wave_spectrum_pixmaps)
        self._apply_result_pixmaps(self._calibration_pixmaps)
        self._apply_result_pixmaps(self._ber_pixmaps)

    # ==================== 参数收集（与其他页面对齐） ====================

    def get_all_parameters(self) -> dict:
        return {
            "功能选择": self._selected_test(),
            "调制方式测试": {
                "调制方式": self.modulation_type.currentText(),
                "调制符号数": self.modulation_symbols.value(),
                "星座图显示点数": self.modulation_show.value(),
                "随机种子": self.modulation_seed.value(),
            },
            "时域波形测试": self._build_payload("waveform_time"),
            "功率谱测试": self._build_payload("waveform_spectrum"),
            "编码类型": self.codec_type_combo.currentText(),
            "精度验证": {
                "链路模式": self.prec_mode.currentText(),
                "时长": self.prec_duration.value(),
                "SNRdB": self.prec_snr.value(),
            },
            "单链路物理层速率测试": self._build_payload("single_link_rate"),
            "总物理层速率测试": self._build_payload("total_phy_rate"),
            "校准测试": self._build_payload("function_calibration"),
            "误码率测试": {
                "测试配置": self.ber_config_combo.currentText(),
                **self._build_payload("ber"),
            },
        }
