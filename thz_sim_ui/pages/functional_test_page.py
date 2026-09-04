"""
功能测试页面 — 调制方式测试 / 波形测试 / 编码技术测试 / 浮点精度验证。

四个功能可独立选择运行：页面自带「运行测试」按钮，测试在后台
FunctionalTestWorker(QThread) 中执行，结果经 Qt 信号回传并渲染到右侧结果栏。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from thz_sim_ui.constants import STATUS_COLORS
from thz_sim_ui.services.functional_test_service import FunctionalTestWorker
from thz_sim_ui.widgets.charts import TextSummaryCard
from thz_sim_ui.widgets.common import CardWidget, StatusBadge
from thz_sim_ui.widgets.forms import combo, dspin, make_form_group, make_radio_group, spin
from thz_sim_ui.widgets.workbench import WorkbenchPage

_PLACEHOLDER_STYLE = (
    "background: #FAFBFE; border: 1px dashed #CAD4E5; border-radius: 8px;"
    " color: #7C869B; min-height: 200px;"
)
_MONO_STYLE = "font-family: Consolas, monospace;"

_TEST_KEYS = ("modulation", "waveform", "codec", "precision")


class _UpdatableSummaryCard(TextSummaryCard):
    """支持运行时刷新行内容的汇总卡。"""

    def __init__(self, title: str, lines: List[str], parent: QWidget | None = None) -> None:
        super().__init__(title, lines, parent)
        self.body_label = self.layout.itemAt(self.layout.count() - 1).widget()

    def set_lines(self, lines: List[str]) -> None:
        self.body_label.setText("\n".join(f"• {line}" for line in lines))


class FunctionalTestPage(WorkbenchPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("功能测试", "链路模块功能验证：调制 / 波形 / 编码 / 浮点精度。", parent)
        self._worker: Optional[FunctionalTestWorker] = None
        self._codec_detail_lines: List[str] = []
        self._modulation_pixmaps: List[tuple] = []  # [(QLabel, QPixmap 原图)]
        self._wave_pixmaps: List[tuple] = []  # [(QLabel, QPixmap 原图)]

        self._build_left_panel()
        self._build_right_panel()
        self._sync_stacks()

    # ==================== 左栏 ====================

    def _build_left_panel(self) -> None:
        radio_box = make_radio_group("测试项目选择", [
            "调制方式测试 (QPSK/16QAM/64QAM)",
            "波形测试 (SC/OFDM)",
            "编码技术测试 (RS/LDPC)",
            "浮点精度验证",
        ])
        self.test_radios: List[QRadioButton] = radio_box.findChildren(QRadioButton)
        for rb in self.test_radios:
            rb.toggled.connect(self._on_radio_toggled)
        self.add_left_widget(radio_box)

        self.param_stack = QStackedWidget()
        self.param_stack.addWidget(self._build_modulation_params())
        self.param_stack.addWidget(self._build_waveform_params())
        self.param_stack.addWidget(self._build_codec_params())
        self.param_stack.addWidget(self._build_precision_params())
        self.add_left_widget(self.param_stack)

        self.run_button = QPushButton("运行测试")
        self.run_button.setProperty("role", "primary")
        self.run_button.clicked.connect(self._on_run_clicked)
        self.status_badge = StatusBadge("待运行")
        run_row = QWidget()
        run_layout = QHBoxLayout(run_row)
        run_layout.setContentsMargins(0, 0, 0, 0)
        run_layout.addWidget(self.run_button)
        run_layout.addWidget(self.status_badge)
        run_layout.addStretch(1)
        self.add_left_widget(run_row)

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

    def _build_waveform_params(self) -> QWidget:
        self.wave_mode = combo(["SC", "OFDM"])
        self.wave_filter_len = spin(8, 128, 32)
        self.wave_sps = spin(1, 16, 4)
        self.wave_rolloff = dspin(0.0, 1.0, 0.22, decimals=3)
        self.wave_symbols = spin(64, 1024, 128)
        self.wave_show = spin(64, 4096, 600)
        self.wave_papr_syms = spin(16384, 1048576, 262144)
        self.wave_papr_syms.setSingleStep(16384)
        box = make_form_group("波形测试参数", [
            ("链路模式", self.wave_mode),
            ("滚降滤波器长度", self.wave_filter_len),
            ("过采样率", self.wave_sps),
            ("滚降系数", self.wave_rolloff),
            ("符号数", self.wave_symbols),
            ("显示点数", self.wave_show),
            ("PAPR 符号数/调制", self.wave_papr_syms),
        ])
        hint = QLabel(
            "SC：±1 脉冲点相隔 2×滤波器长度，期望正负交替的滚降波形；"
            "OFDM：每个符号仅单个子载波有值且索引逐符号递增，期望频率渐升正弦"
            "（过采样由频域补零 IFFT 完成）。"
            "另生成多调制 PAPR CCDF 曲线（BPSK/QPSK/16QAM/64QAM/256QAM 叠加，"
            "符号数越多曲线越精确但越耗时）与成型后功率谱（SC 标滤波器截止频率，"
            "OFDM 验证补零 IFFT 理想带限）。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("CardHint")
        return self._wrap_with_hint(box, hint)

    def _build_codec_params(self) -> QWidget:
        self.codec_type_combo = combo(["RS (GF符号)", "LDPC (十六进制)"])
        self.codec_editor = QPlainTextEdit()
        self.codec_editor.setStyleSheet(_MONO_STYLE)
        self.codec_editor.setMinimumHeight(180)
        self.codec_type_combo.currentTextChanged.connect(self._on_codec_type_changed)

        load_file_btn = QPushButton("选择文件")
        load_file_btn.clicked.connect(self._on_load_file)
        demo_btn = QPushButton("加载示例")
        demo_btn.clicked.connect(self._on_load_demo)

        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(make_form_group(
            "编码技术测试 (测试用例文件)", [("编码类型", self.codec_type_combo)]))
        btn_row = QHBoxLayout()
        btn_row.addWidget(load_file_btn)
        btn_row.addWidget(demo_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)
        layout.addWidget(self.codec_editor)
        hint = QLabel(
            "用例为 JSON 文件：RS 输入为 GF 符号数组（可带 errors 注入符号错误），"
            "LDPC 输入为 input_hex 十六进制串（可带 error_bits 注入比特错误）。"
            "每组算例的输入 / 编码结果 / 译码结果显示在右侧。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("CardHint")
        layout.addWidget(hint)
        self._on_load_demo()
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

    @staticmethod
    def _wrap_with_hint(box: QWidget, hint: QWidget) -> QWidget:
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(box)
        layout.addWidget(hint)
        return wrap

    # ==================== 右栏 ====================

    def _build_right_panel(self) -> None:
        self.result_stack = QStackedWidget()
        self.result_stack.addWidget(self._build_modulation_results())
        self.result_stack.addWidget(self._build_waveform_results())
        self.result_stack.addWidget(self._build_codec_results())
        self.result_stack.addWidget(self._build_precision_results())
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
        self.modulation_plot.setMinimumHeight(320)
        self.modulation_card.layout.addWidget(self.modulation_plot)
        self.modulation_card.hide()
        layout.addWidget(self.modulation_card)

        self.modulation_summary_card = _UpdatableSummaryCard("调制信息", ["尚未运行"])
        layout.addWidget(self.modulation_summary_card)
        layout.addStretch(1)
        return page

    def _build_waveform_results(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.wave_placeholder = self._make_placeholder("点击左侧「运行测试」生成波形")
        layout.addWidget(self.wave_placeholder)

        # 结果图卡片容器：按每次运行产出的图数量动态生成（时域/星座/CCDF/频谱等）
        self.wave_cards_container = QWidget()
        self.wave_cards_layout = QVBoxLayout(self.wave_cards_container)
        self.wave_cards_layout.setContentsMargins(0, 0, 0, 0)
        self.wave_cards_layout.setSpacing(12)
        layout.addWidget(self.wave_cards_container)

        self.wave_checks_card = _UpdatableSummaryCard("验证结果", ["尚未运行"])
        layout.addWidget(self.wave_checks_card)
        layout.addStretch(1)
        return page

    def _build_codec_results(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.codec_placeholder = self._make_placeholder("点击左侧「运行测试」执行编译码用例")
        layout.addWidget(self.codec_placeholder)

        self.codec_table = self._make_table(["算例", "输入", "编码结果", "注入错误", "译码结果", "判定"])
        layout.addWidget(self.codec_table, 1)

        self.codec_detail = QPlainTextEdit()
        self.codec_detail.setReadOnly(True)
        self.codec_detail.setPlaceholderText("选中某行查看算例详情（含完整数据）")
        self.codec_detail.setStyleSheet(_MONO_STYLE)
        self.codec_detail.setMaximumHeight(140)
        self.codec_table.itemSelectionChanged.connect(self._on_codec_row_selected)
        layout.addWidget(self.codec_detail)

        self.codec_summary_card = _UpdatableSummaryCard("用例统计", ["尚未运行"])
        layout.addWidget(self.codec_summary_card)
        return page

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

    def _on_load_demo(self) -> None:
        try:
            text = self._demo_path().read_text(encoding="utf-8")
        except OSError as exc:
            self.codec_editor.setPlainText(
                f"// 示例文件读取失败：{exc}\n// 可点击「选择文件」手动加载用例")
            return
        self.codec_editor.setPlainText(text)

    def _on_codec_type_changed(self, _text: str) -> None:
        """切换编码类型时同步加载对应示例，避免界面选择与实际 JSON 用例不一致。"""
        self._on_load_demo()

    def _on_run_clicked(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        test_type = self._selected_test()
        payload = self._build_payload(test_type)
        self._reset_results(test_type)
        self._set_badge("运行中")
        self.run_button.setEnabled(False)
        self._worker = FunctionalTestWorker(test_type, payload, parent=self)
        self._worker.result_ready.connect(self._on_result)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _build_payload(self, test_type: str) -> dict:
        if test_type == "modulation":
            return {
                "modulation": self.modulation_type.currentText(),
                "num_symbols": self.modulation_symbols.value(),
                "show_points": self.modulation_show.value(),
                "random_seed": self.modulation_seed.value(),
            }
        if test_type == "waveform":
            return {
                "link_mode": "sc-fde" if self.wave_mode.currentText() == "SC" else "ofdm",
                "filter_length": self.wave_filter_len.value(),
                "oversampling": self.wave_sps.value(),
                "rolloff": self.wave_rolloff.value(),
                "num_symbols": self.wave_symbols.value(),
                "show_points": self.wave_show.value(),
                "papr_symbols_per_mod": self.wave_papr_syms.value(),
            }
        if test_type == "codec":
            return {"config_text": self.codec_editor.toPlainText()}
        return {
            "link_mode": "sc-fde" if self.prec_mode.currentText() == "SC" else "ofdm",
            "duration": self.prec_duration.value(),
            "SNRdB": self.prec_snr.value(),
        }

    def _on_result(self, test_type: str, result: dict) -> None:
        self._set_badge("已完成" if result.get("ok") else "已失败")
        if test_type == "modulation":
            self._render_modulation(result)
        elif test_type == "waveform":
            self._render_waveform(result)
        elif test_type == "codec":
            self._render_table_result(result, self.codec_table,
                                      self.codec_placeholder, self.codec_summary_card)
        else:
            self._render_table_result(result, self.prec_table,
                                      self.prec_placeholder, self.prec_summary_card)

    def _on_failed(self, test_type: str, message: str) -> None:
        self._set_badge("已失败")
        target = {
            "modulation": self.modulation_placeholder,
            "waveform": self.wave_placeholder,
            "codec": self.codec_placeholder,
            "precision": self.prec_placeholder,
        }[test_type]
        target.setText(f"测试失败：{message}")
        target.show()

    def _on_worker_finished(self) -> None:
        self.run_button.setEnabled(True)
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
            self._modulation_pixmaps = []
            self.modulation_summary_card.set_lines(["运行中…"])
        elif test_type == "waveform":
            self.wave_placeholder.setText("运行中，请稍候…")
            self.wave_placeholder.show()
            self._clear_wave_cards()
            self._wave_pixmaps = []
            self.wave_checks_card.set_lines(["运行中…"])
        elif test_type == "codec":
            self.codec_placeholder.setText("运行中，请稍候…")
            self.codec_placeholder.show()
            self.codec_table.setRowCount(0)
            self.codec_detail.clear()
            self._codec_detail_lines = []
            self.codec_summary_card.set_lines(["运行中…"])
        else:
            self.prec_placeholder.setText("运行中，请稍候…")
            self.prec_placeholder.show()
            self.prec_table.setRowCount(0)
            self.prec_summary_card.set_lines(["运行中…"])

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

    def _clear_wave_cards(self) -> None:
        while self.wave_cards_layout.count():
            item = self.wave_cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _render_waveform(self, result: dict) -> None:
        self._clear_wave_cards()
        self._wave_pixmaps = []
        for plot in result.get("plots", []):
            card = CardWidget(plot.get("title", ""))
            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumHeight(240)
            card.layout.addWidget(label)
            self.wave_cards_layout.addWidget(card)
            pm = QPixmap()
            if pm.loadFromData(plot.get("png", b""), "PNG"):
                self._wave_pixmaps.append((label, pm))
        self.wave_placeholder.hide()
        self._apply_wave_pixmaps()
        lines = [
            f"{'通过' if c['ok'] else '失败'} | {c['name']}：{c['detail']}"
            for c in result.get("checks", [])
        ]
        self.wave_checks_card.set_lines(lines or ["无校验项"])

    def _apply_wave_pixmaps(self) -> None:
        self._apply_result_pixmaps(self._wave_pixmaps)

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
            table.setColumnWidth(c, min(table.columnWidth(c), 320))
        self._codec_detail_lines = result.get("detail_lines", [])
        lines = [f"{item['label']}：{item['value']}" for item in result.get("summary", [])]
        summary_card.set_lines(lines or ["-"])

    def _on_codec_row_selected(self) -> None:
        row = self.codec_table.currentRow()
        if 0 <= row < len(self._codec_detail_lines):
            self.codec_detail.setPlainText(self._codec_detail_lines[row])

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_result_pixmaps(self._modulation_pixmaps)
        self._apply_wave_pixmaps()

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
            "波形测试": {
                "链路模式": self.wave_mode.currentText(),
                "滚降滤波器长度": self.wave_filter_len.value(),
                "过采样率": self.wave_sps.value(),
                "滚降系数": self.wave_rolloff.value(),
                "符号数": self.wave_symbols.value(),
                "显示点数": self.wave_show.value(),
                "PAPR符号数": self.wave_papr_syms.value(),
            },
            "编码类型": self.codec_type_combo.currentText(),
            "精度验证": {
                "链路模式": self.prec_mode.currentText(),
                "时长": self.prec_duration.value(),
                "SNRdB": self.prec_snr.value(),
            },
        }
