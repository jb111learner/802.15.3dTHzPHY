from __future__ import annotations

import ast

from PySide6.QtWidgets import (
    QFileDialog, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from thz_sim_ui.widgets.charts import TextSummaryCard
from thz_sim_ui.widgets.common import PlaceholderList, TwoColumnMetricGrid
from thz_sim_ui.widgets.forms import combo, dspin, line, make_form_group, spin
from thz_sim_ui.widgets.workbench import WorkbenchPage

# ── P01–P12 参数测试模版 ──────────────────────────────────────────
PRESET_TABLE = {
    "P01 — SC + QPSK + RS + AWGN":       {"波形": "单载波SC", "调制": "QPSK",  "编码": "RS（192,255）",     "信道": "AWGN"},
    "P02 — SC + QPSK + LDPC + AWGN":     {"波形": "单载波SC", "调制": "QPSK",  "编码": "LDPC（1344,1440）", "信道": "AWGN"},
    "P03 — SC + 16QAM + RS + AWGN":      {"波形": "单载波SC", "调制": "16QAM", "编码": "RS（192,255）",     "信道": "AWGN"},
    "P04 — SC + 16QAM + LDPC + AWGN":    {"波形": "单载波SC", "调制": "16QAM", "编码": "LDPC（1344,1440）", "信道": "AWGN"},
    "P05 — SC + 64QAM + RS + AWGN":      {"波形": "单载波SC", "调制": "64QAM", "编码": "RS（192,255）",     "信道": "AWGN"},
    "P06 — SC + 64QAM + LDPC + AWGN":    {"波形": "单载波SC", "调制": "64QAM", "编码": "LDPC（1344,1440）", "信道": "AWGN"},
    "P07 — OFDM + QPSK + RS + AWGN":     {"波形": "多载波OFDM", "调制": "QPSK",  "编码": "RS（11,15）",      "信道": "AWGN"},
    "P08 — OFDM + QPSK + LDPC + AWGN":   {"波形": "多载波OFDM", "调制": "QPSK",  "编码": "LDPC（1344,1440）", "信道": "AWGN"},
    "P09 — OFDM + 16QAM + RS + AWGN":    {"波形": "多载波OFDM", "调制": "16QAM", "编码": "RS（11,15）",      "信道": "AWGN"},
    "P10 — OFDM + 16QAM + LDPC + AWGN":  {"波形": "多载波OFDM", "调制": "16QAM", "编码": "LDPC（1344,1440）", "信道": "AWGN"},
    "P11 — OFDM + 64QAM + RS + AWGN":    {"波形": "多载波OFDM", "调制": "64QAM", "编码": "RS（11,15）",      "信道": "AWGN"},
    "P12 — OFDM + 64QAM + LDPC + AWGN":  {"波形": "多载波OFDM", "调制": "64QAM", "编码": "LDPC（1344,1440）", "信道": "AWGN"},
}
PRESET_KEYS = list(PRESET_TABLE.keys())

# ── 调制阶数映射 ──────────────────────────────────────────────────
MOD_ORDER = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "16QAM": 4, "64QAM": 6}

# ── 编码效率映射 ──────────────────────────────────────────────────
CODE_RATE = {
    "RS（192,255）":     (192, 255),
    "RS（11,15）":       (11, 15),
    "LDPC（1344,1440）": (1344, 1440),
}

# ── 前导码长度 (SYNC + SFD + CES) ─────────────────────────────────
# Short: SYNC=14×128=1792, SFD=128, CES=b128(128)+a512(512)+b512(512)+a256(256)=1408
# Long:  SYNC=28×128=3584, SFD=128, CES=1408
PREAMBLE_LEN = {"短前导": 1792 + 128 + 1408, "长前导": 3584 + 128 + 1408}  # 3328 / 5120


class ParameterConfigPage(WorkbenchPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("参数配置", "链路参数工作台，右侧实时刷新摘要、速率、校验和帧结构。", parent)

        # =================================================================
        # 工程参数
        # =================================================================
        self.project_name = line("")
        self.preset_mode = combo(PRESET_KEYS)
        self.preset_mode.currentIndexChanged.connect(self._on_preset_changed)
        self.link_name = line("THz_Link_Main")

        # =================================================================
        # 链路参数
        # =================================================================
        self.waveform_type = combo(["单载波SC", "多载波OFDM"])
        self.bandwidth = dspin(1, 200, 30, suffix="GHz")
        self.fc = dspin(0.1, 10000, 1000, suffix="GHz")

        self.bit_source = combo(["PRBS", "文件输入"])
        self.bit_source.currentIndexChanged.connect(self._on_bit_source_changed)
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("选择数据文件...")
        self.file_path_edit.setEnabled(False)
        self.file_path_btn = QPushButton("浏览")
        self.file_path_btn.setToolTip("选择数据文件；选择成功后将自动切换为“文件输入”")
        self.file_path_btn.clicked.connect(self._on_browse_file)
        file_row = QHBoxLayout()
        file_row.addWidget(self.file_path_edit)
        file_row.addWidget(self.file_path_btn)
        self.file_path_widget = QWidget()
        self.file_path_widget.setLayout(file_row)

        self.subframe_length = spin(1, 2000, 480, "symbols")
        self.subframe_num = spin(1, 100, 51)
        self.cp_length = spin(0, 512, 32, "symbols")

        self.modulation = combo(["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"])
        self.coding = combo(list(CODE_RATE.keys()))
        self.scrambling = combo(["启用", "关闭"])

        self.rs_decode_mode = combo(["硬译码", "Chase软译码"])
        self.rs_decode_mode.currentIndexChanged.connect(self._on_rs_decode_changed)
        self.chase_num = spin(1, 10, 3)
        self.chase_num.setEnabled(False)

        self.preamble = combo(["短前导", "长前导"])

        self.ofdm_subcarriers = combo(["64", "128", "256", "512"])
        self.ofdm_subcarriers.setCurrentIndex(3)
        self.ofdm_symbols = spin(1, 100, 48)
        self.pilot_indexes = line("[0, 16, 32]")

        self.pulse_shaping = combo(["rc", "rrc", "rect"])
        self.pulse_shaping.setCurrentText("rrc")
        self.rolloff_factor = dspin(0, 1, 0.22)
        self.rolloff_factor.setSingleStep(0.01)
        self.filter_span = spin(1, 128, 32, "samples")
        self.oversampling_factor = combo(["1x", "2x", "4x", "8x"])
        self.oversampling_factor.setCurrentText("4x")

        self.sample_rate = dspin(1, 3000000, 30000, suffix="MBd")
        self.sample_rate.setSingleStep(1000)
        self.duration = dspin(0.01, 1000, 0.01, suffix="ms")
        self.duration.setSingleStep(0.01)

        # 接收端补偿开关
        self.enable_cfo_comp = combo(["是", "否"])
        self.enable_cfo_comp.setCurrentText("否")
        self.enable_iq_comp = combo(["是", "否"])
        self.enable_iq_comp.setCurrentText("否")
        self.enable_ch_est = combo(["是", "否"])
        self.enable_ch_est.setCurrentText("是")

        self.seed_strategy = combo(["固定种子", "递增种子", "时间种子"])

        # =================================================================
        # 左侧表单
        # =================================================================
        self.add_left_widget(
            make_form_group("工程参数", [
                ("工程名称", self.project_name),
                ("参数测试模版", self.preset_mode),
                ("链路名称", self.link_name),
            ])
        )

        self.add_left_widget(
            make_form_group("链路参数", [
                ("波形类型", self.waveform_type),
                ("带宽", self.bandwidth),
                ("载频", self.fc),
                ("数据源配置", self.bit_source),
                ("文件地址", self.file_path_widget),
                ("数据子帧长度", self.subframe_length),
                ("单帧数据子帧数量", self.subframe_num),
                ("CP长度", self.cp_length),
                ("调制方式", self.modulation),
                ("信道编码类型", self.coding),
                ("RS译码方式", self.rs_decode_mode),
                ("Chase译码试探数", self.chase_num),
                ("扰码配置", self.scrambling),
                ("前导码配置", self.preamble),
                ("OFDM子载波数", self.ofdm_subcarriers),
                ("单帧OFDM符号数", self.ofdm_symbols),
                ("块状导频索引", self.pilot_indexes),
                ("波形成形", self.pulse_shaping),
                ("滚降因子", self.rolloff_factor),
                ("滤波器跨度", self.filter_span),
                ("过采样率", self.oversampling_factor),
                ("采样率", self.sample_rate),
                ("时长", self.duration),
                ("频偏补偿", self.enable_cfo_comp),
                ("IQ补偿", self.enable_iq_comp),
                ("信道估计与均衡", self.enable_ch_est),
            ])
        )

        self.add_left_widget(
            make_form_group("任务运行参数", [
                ("随机种子策略", self.seed_strategy),
            ])
        )

        # =================================================================
        # 右侧 — 四大信息区块
        # =================================================================
        self._ov_container = QWidget()
        self._ov_layout = QGridLayout(self._ov_container)
        self._ov_layout.setContentsMargins(8, 8, 8, 8)
        self._ov_layout.setSpacing(8)

        self._rate_container = QWidget()
        self._rate_layout = QGridLayout(self._rate_container)
        self._rate_layout.setContentsMargins(8, 8, 8, 8)
        self._rate_layout.setSpacing(8)

        self._val_table = QTableWidget(8, 4)
        self._val_table.setHorizontalHeaderLabels(["级别", "参数", "说明", "建议"])
        self._val_table.verticalHeader().setVisible(False)

        self._fs_table = QTableWidget(8, 2)
        self._fs_table.setHorizontalHeaderLabels(["项目", "符号数"])
        self._fs_table.verticalHeader().setVisible(False)

        right_root = QVBoxLayout()
        right_root.setContentsMargins(0, 0, 0, 0)
        right_root.setSpacing(10)

        right_root.addWidget(self._build_overview_block())
        right_root.addWidget(self._build_rate_block())
        right_root.addWidget(self._build_validation_block())
        right_root.addWidget(self._build_frame_structure_block())
        right_root.addStretch()

        container = QWidget()
        container.setLayout(right_root)
        self.add_right_widget(container)

        # ── 信号连接：参数变更 → 延迟刷新（不阻塞 UI）──
        self._refresh_pending = False
        self._refresh_guard = False
        for w in self._refresh_sources():
            if hasattr(w, "currentIndexChanged"):
                w.currentIndexChanged.connect(self._schedule_refresh)
            elif hasattr(w, "valueChanged"):
                w.valueChanged.connect(self._schedule_refresh)
            elif hasattr(w, "textChanged"):
                w.textChanged.connect(self._schedule_refresh)
        self._refresh_all()

    # =================================================================
    # 右侧区块构建
    # =================================================================
    def _build_overview_block(self) -> QWidget:
        return self._ov_container

    def _build_rate_block(self) -> QWidget:
        return self._rate_container

    def _build_validation_block(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        title = QLabel("参数校验提示")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(title)
        layout.addWidget(self._val_table)
        return panel

    def _build_frame_structure_block(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        title = QLabel("帧结构总览")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(title)
        layout.addWidget(self._fs_table)
        return panel

    # =================================================================
    # 刷新核心（轻量 QLabel/QTable 文本更新，不影响仿真）
    # =================================================================
    def _schedule_refresh(self, *_args) -> None:
        """合并短时间内的多次变更，只执行一次刷新"""
        if self._refresh_pending:
            return
        self._refresh_pending = True
        from PySide6.QtCore import QTimer
        QTimer.singleShot(20, self._do_refresh)

    def _do_refresh(self) -> None:
        self._refresh_pending = False
        if self._refresh_guard:
            return
        self._refresh_guard = True
        try:
            p = self._collect_params()
            self._refresh_overview(p)
            self._refresh_rate(p)
            self._refresh_validation(p)
            self._refresh_frame_structure(p)
        finally:
            self._refresh_guard = False

    def _refresh_all(self) -> None:
        """立即刷新（初始化时调用）"""
        self._refresh_guard = True
        try:
            p = self._collect_params()
            self._refresh_overview(p)
            self._refresh_rate(p)
            self._refresh_validation(p)
            self._refresh_frame_structure(p)
        finally:
            self._refresh_guard = False

    def _collect_params(self) -> dict:
        """收集所有当前参数值"""
        is_ofdm = (self.waveform_type.currentText() == "多载波OFDM")
        mod_str = self.modulation.currentText()
        code_str = self.coding.currentText()
        nbps = MOD_ORDER.get(mod_str, 1)
        k, n = CODE_RATE.get(code_str, (1, 1))
        eta_c = k / n

        # 帧结构
        preamble_len = PREAMBLE_LEN.get(self.preamble.currentText(), 5120)
        cp_len = self.cp_length.value()

        if is_ofdm:
            n_sc = int(self.ofdm_subcarriers.currentText())
            n_sym = self.ofdm_symbols.value()
            try:
                pilots = ast.literal_eval(self.pilot_indexes.text())
                n_pilots = len(pilots) if isinstance(pilots, list) else 0
            except (ValueError, SyntaxError):
                n_pilots = 0
            n_data_sym = n_sym - n_pilots
            n_data = n_data_sym * n_sc
            n_frame = preamble_len + (n_sc + cp_len) * n_sym
        else:
            n_sc = self.subframe_length.value()  # SC 块大小
            n_sym = self.subframe_num.value()
            n_data = n_sc * n_sym
            n_frame = preamble_len + (n_sc + cp_len) * n_sym
            n_pilots = 0

        eta_f = n_data / n_frame if n_frame > 0 else 0
        eta = eta_c * eta_f
        rs = self.sample_rate.value() * 1e6  # MBd → Bd

        # frame_bit_num (对齐 DataProcesser.frame_process 的校验逻辑)
        if is_ofdm:
            frame_bits_raw = nbps * (n_sym - n_pilots) * n_sc * eta_c
        else:
            frame_bits_raw = nbps * n_sc * n_sym * eta_c
        frame_bit_num = round(frame_bits_raw)
        need_pad = abs(frame_bits_raw - frame_bit_num) > 1e-9

        # 补零计算 (参考 DataProcesser.frame_process)
        total_bits = int(rs * nbps * self.duration.value() * 1e-3)  # Rs(MBd) × Nb × dur(ms)
        pad_bits = 0 if not need_pad else (frame_bit_num - (total_bits % frame_bit_num)) % frame_bit_num

        # ── 理论谱效 ──
        if is_ofdm:
            eta_cp = n_sc / (n_sc + cp_len) if (n_sc + cp_len) > 0 else 0.0
            eta_pilot = (n_sym - n_pilots) / n_sym if n_sym > 0 else 0.0
        else:
            eta_cp = n_sc / (n_sc + cp_len) if (n_sc + cp_len) > 0 else 0.0
            eta_pilot = 1.0
        eta_se_theory = eta_c * nbps * eta_cp * eta_pilot

        return {
            "is_ofdm": is_ofdm, "mod_str": mod_str, "code_str": code_str,
            "nbps": nbps, "K": k, "N": n, "eta_c": eta_c,
            "preamble_len": preamble_len, "cp_len": cp_len,
            "n_sc": n_sc, "n_sym": n_sym, "n_data": n_data,
            "n_frame": n_frame, "n_pilots": n_pilots,
            "eta_f": eta_f, "eta": eta, "rs": rs,
            "eta_cp": eta_cp, "eta_pilot": eta_pilot,
            "eta_se_theory": eta_se_theory,
            "bandwidth": self.bandwidth.value(), "fc": self.fc.value(),
            "waveform": self.waveform_type.currentText(),
            "modulation": self.modulation.currentText(),
            "coding": self.coding.currentText(),
            "preamble": self.preamble.currentText(),
            "scrambling": self.scrambling.currentText(),
            "ofdm_subcarriers": self.ofdm_subcarriers.currentText(),
            "frame_bits_raw": frame_bits_raw, "frame_bit_num": frame_bit_num,
            "need_pad": need_pad, "total_bits": total_bits, "pad_bits": pad_bits,
        }

    # ── 参数总览 ──
    def _refresh_overview(self, p: dict) -> None:
        # 清空重建
        while self._ov_layout.count():
            item = self._ov_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        R0 = p["rs"] * p["nbps"]
        R = R0 * p["eta"]

        metrics = TwoColumnMetricGrid([
            ("理论峰值速率", f"{R0 / 1e9:.2f} Gbps"),
            ("净有效速率", f"{R / 1e9:.2f} Gbps"),
            ("带宽设置", f"{p['bandwidth']:.1f} GHz"),
            ("理论谱效", f"{p['eta_se_theory']:.2f} bit/s/Hz"),
        ])
        summary = PlaceholderList("当前配置摘要", [
            p["waveform"], p["modulation"], p["coding"],
            f"前导: {p['preamble']}", f"扰码: {p['scrambling']}",
        ])
        notes = TextSummaryCard("当前链路说明", [
            f"调制阶数 {p['nbps']} bits/sym，编码效率 {p['eta_c']:.3f}",
            f"帧效率 {p['eta_f']:.3f}，综合效率 {p['eta']:.3f}",
            f"符号速率 {p['rs']/1e6:.0f} MBd，R₀ = {R0/1e9:.2f} Gbps",
        ])
        self._ov_layout.addWidget(metrics, 0, 0, 1, 2)
        self._ov_layout.addWidget(summary, 1, 0)
        self._ov_layout.addWidget(notes, 1, 1)

    # ── 理论速率计算 ──
    def _refresh_rate(self, p: dict) -> None:
        while self._rate_layout.count():
            item = self._rate_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        R0 = p["rs"] * p["nbps"]
        R = R0 * p["eta"]
        peak_card = TextSummaryCard("理论峰值速率 R₀", [
            f"Rs = {p['rs']/1e6:.0f} MBd",
            f"log₂M = {p['nbps']} bits/sym",
            f"R₀ = {R0/1e9:.2f} Gbps",
        ])
        net_card = TextSummaryCard("净有效速率 R", [
            f"前导码: {p['preamble']} → {p['preamble_len']} symbols",
            f"编码效率 ηc = {p['eta_c']:.4f}",
            f"帧效率 ηf = {p['eta_f']:.4f}",
            f"综合效率 η = {p['eta']:.4f}",
            f"R = {R/1e9:.2f} Gbps",
        ])
        self._rate_layout.addWidget(peak_card, 0, 0)
        self._rate_layout.addWidget(net_card, 0, 1)

        se_card = TextSummaryCard("理论谱效 η_SE", [
            f"CP/GI效率 ηcp = {p['eta_cp']:.4f}",
            f"导频效率 ηpilot = {p['eta_pilot']:.4f}",
            f"编码效率 ηc = {p['eta_c']:.4f}",
            f"调制阶数 log₂M = {p['nbps']} bits/sym",
            f"η_SE = ηc × log₂M × ηcp × ηpilot = {p['eta_se_theory']:.4f} bit/s/Hz",
        ])
        self._rate_layout.addWidget(se_card, 1, 0, 1, 2)

    # ── 参数校验 ──
    def _refresh_validation(self, p: dict) -> None:
        rows = []
        # ① 帧比特数整数校验 (对齐 DataProcesser.frame_process)
        if p["need_pad"]:
            rows.append(("错误", "帧比特数",
                         f"frame_bit_num = {p['frame_bits_raw']:.3f} 非整数",
                         "调整子帧数/OFDM符号数使 frame_bit_num 为整数"))
        else:
            rows.append(("通过", "帧比特数",
                         f"frame_bit_num = {p['frame_bit_num']}",
                         "单帧比特数为整数，可正常组帧"))

        # ② 补零校验
        if p["pad_bits"] > 0:
            rows.append(("提示", "补零",
                         f"总比特 {p['total_bits']}，需补零 {p['pad_bits']} bits "
                         f"(占 {100*p['pad_bits']/max(p['total_bits']+p['pad_bits'],1):.1f}%)",
                         "增大时长或调整帧结构可减少补零比例"))
        else:
            rows.append(("通过", "帧对齐",
                         f"总比特 {p['total_bits']} 恰好对齐帧边界，无需补零",
                         ""))

        # ③ 帧数
        total_frames = (p["total_bits"] + p["pad_bits"]) // max(p["frame_bit_num"], 1)
        rows.append(("通过", "帧数",
                     f"约 {total_frames} 帧 ({p['total_bits']+p['pad_bits']} bits)",
                     f"每帧 {p['frame_bit_num']} bits"))

        # ④ CP 校验
        rows.append(("通过", "CP长度", f"CP = {p['cp_len']} symbols",
                     "用于吸收多径时延" if p["cp_len"] > 0 else "无保护间隔"))

        # ⑤ OFDM 导频校验
        if p["is_ofdm"] and p["n_pilots"] > 0:
            try:
                max_pilot = max(ast.literal_eval(self.pilot_indexes.text()))
                if max_pilot >= p["n_sym"]:
                    rows.append(("错误", "导频索引",
                                 f"最大值 {max_pilot} ≥ OFDM符号数 {p['n_sym']}",
                                 "减小导频索引或增加OFDM符号数"))
                else:
                    rows.append(("通过", "导频索引",
                                 f"{p['n_pilots']} 个导频, max={max_pilot} < {p['n_sym']}",
                                 "范围合法"))
            except (ValueError, SyntaxError):
                rows.append(("警告", "导频索引", "无法解析导频索引列表", "检查格式，如 [0, 16, 32]"))

        # ⑥ 编码匹配校验
        if p["is_ofdm"] and "RS（192,255）" in p["code_str"]:
            rows.append(("警告", "编码匹配",
                         "OFDM 模式建议使用 RS（11,15）",
                         "切换编码或使用参数测试模版 P07–P12"))

        self._val_table.setRowCount(len(rows))
        for r, (level, param, desc, suggestion) in enumerate(rows):
            self._val_table.setItem(r, 0, QTableWidgetItem(level))
            self._val_table.setItem(r, 1, QTableWidgetItem(param))
            self._val_table.setItem(r, 2, QTableWidgetItem(desc))
            self._val_table.setItem(r, 3, QTableWidgetItem(suggestion))

    # ── 帧结构总览 ──
    def _refresh_frame_structure(self, p: dict) -> None:
        cp_total = p["cp_len"] * p["n_sym"]
        if p["is_ofdm"]:
            rows = [
                ("前导码 (SYNC+SFD+CES)", f"{p['preamble_len']}"),
                (f"CP ({p['cp_len']} × {p['n_sym']} 符号)", f"{cp_total}"),
                (f"OFDM 数据 ({p['n_sym']} × {p['n_sc']})", f"{p['n_sym'] * p['n_sc']}"),
                (f"导频符号 ({p['n_pilots']} × {p['n_sc']})",
                 f"{p['n_pilots'] * p['n_sc']}"),
                ("帧总符号数", f"{p['n_frame']}"),
                (f"净数据符号 ({p['n_sym'] - p['n_pilots']} × {p['n_sc']})",
                 f"{p['n_data']}"),
            ]
        else:
            rows = [
                ("前导码 (SYNC+SFD+CES)", f"{p['preamble_len']}"),
                (f"CP ({p['cp_len']} × {p['n_sym']} blocks)", f"{cp_total}"),
                (f"数据块 ({p['n_sym']} × {p['n_sc']})", f"{p['n_sym'] * p['n_sc']}"),
                ("帧总符号数", f"{p['n_frame']}"),
                (f"净数据符号 ({p['n_sym']} × {p['n_sc']})", f"{p['n_data']}"),
            ]
        self._fs_table.setRowCount(len(rows))
        for r, (name, val) in enumerate(rows):
            self._fs_table.setItem(r, 0, QTableWidgetItem(name))
            self._fs_table.setItem(r, 1, QTableWidgetItem(val))

    # =================================================================
    # 信号源列表
    # =================================================================
    def _refresh_sources(self):
        return [
            self.waveform_type, self.modulation, self.coding,
            self.scrambling, self.preamble,
            self.ofdm_subcarriers, self.pilot_indexes,
            self.rs_decode_mode, self.bit_source,
            self.seed_strategy,
            self.pulse_shaping, self.oversampling_factor,
            self.bandwidth, self.fc,
            self.sample_rate, self.duration, self.rolloff_factor,
            self.subframe_length, self.subframe_num, self.cp_length,
            self.ofdm_symbols, self.chase_num, self.filter_span,
            self.enable_cfo_comp, self.enable_iq_comp, self.enable_ch_est,
            self.project_name, self.link_name, self.file_path_edit,
        ]

    # =================================================================
    # 槽函数
    # =================================================================
    def _on_preset_changed(self, idx: int) -> None:
        if idx < 0 or idx >= len(PRESET_KEYS):
            return
        preset = PRESET_TABLE[PRESET_KEYS[idx]]
        for name, widget in [("波形", self.waveform_type),
                              ("调制", self.modulation),
                              ("编码", self.coding)]:
            i = widget.findText(preset[name])
            if i >= 0:
                widget.setCurrentIndex(i)

    def _on_bit_source_changed(self, _idx: int = 0) -> None:
        is_file = (self.bit_source.currentText() == "文件输入")
        self.file_path_edit.setEnabled(is_file)

    def _on_browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "所有文件 (*)")
        if path:
            self.bit_source.setCurrentText("文件输入")
            self.file_path_edit.setText(path)

    def _on_rs_decode_changed(self, _idx: int = 0) -> None:
        is_chase = (self.rs_decode_mode.currentText() == "Chase软译码")
        self.chase_num.setEnabled(is_chase)

    # =================================================================
    # 参数读写
    # =================================================================
    def get_basic_parameters(self) -> dict[str, object]:
        return {
            "工程名称": self.project_name.text(),
            "参数测试模版": self.preset_mode.currentText(),
            "链路名称": self.link_name.text(),
            "波形类型": self.waveform_type.currentText(),
            "带宽": self.bandwidth.value(),
            "载频": self.fc.value(),
            "数据源配置": self.bit_source.currentText(),
            "文件地址": self.file_path_edit.text(),
            "数据子帧长度": self.subframe_length.value(),
            "单帧数据子帧数量": self.subframe_num.value(),
            "CP长度": self.cp_length.value(),
            "调制方式": self.modulation.currentText(),
            "信道编码类型": self.coding.currentText(),
            "RS译码方式": self.rs_decode_mode.currentText(),
            "Chase译码试探数": self.chase_num.value(),
            "扰码配置": self.scrambling.currentText(),
            "前导码配置": self.preamble.currentText(),
            "OFDM子载波数": self.ofdm_subcarriers.currentText(),
            "单帧OFDM符号数": self.ofdm_symbols.value(),
            "块状导频索引": self.pilot_indexes.text(),
            "波形成形": self.pulse_shaping.currentText(),
            "滚降因子": self.rolloff_factor.value(),
            "滤波器跨度": self.filter_span.value(),
            "过采样率": self.oversampling_factor.currentText(),
            "采样率": self.sample_rate.value(),
            "时长": self.duration.value(),
            "频偏补偿": self.enable_cfo_comp.currentText(),
            "IQ补偿": self.enable_iq_comp.currentText(),
            "信道估计与均衡": self.enable_ch_est.currentText(),
        }

    def get_run_parameters(self) -> dict[str, object]:
        return {
            "随机种子策略": self.seed_strategy.currentText(),
        }

    def get_all_parameters(self) -> dict[str, object]:
        result = {}
        result.update(self.get_basic_parameters())
        result.update(self.get_run_parameters())
        return result

    def set_all_parameters(self, params: dict[str, object]) -> None:
        # 批量设置时阻塞信号，最后统一刷新一次
        sources = self._refresh_sources()
        for w in sources:
            w.blockSignals(True)
        try:
            self._apply_params(params)
        finally:
            for w in sources:
                w.blockSignals(False)
            self._refresh_all()

    def _apply_params(self, params: dict[str, object]) -> None:
        _set_combo(params, "参数测试模版", self.preset_mode)
        _set_text(params, "工程名称", self.project_name)
        _set_text(params, "链路名称", self.link_name)
        _set_combo(params, "波形类型", self.waveform_type)
        _set_dspin(params, "带宽", self.bandwidth)
        _set_dspin(params, "载频", self.fc)
        _set_combo(params, "数据源配置", self.bit_source)
        _set_text(params, "文件地址", self.file_path_edit)
        _set_spin(params, "数据子帧长度", self.subframe_length)
        _set_spin(params, "单帧数据子帧数量", self.subframe_num)
        _set_spin(params, "CP长度", self.cp_length)
        _set_combo(params, "调制方式", self.modulation)
        _set_combo(params, "信道编码类型", self.coding)
        _set_combo(params, "RS译码方式", self.rs_decode_mode)
        _set_spin(params, "Chase译码试探数", self.chase_num)
        _set_combo(params, "扰码配置", self.scrambling)
        _set_combo(params, "前导码配置", self.preamble)
        _set_combo(params, "OFDM子载波数", self.ofdm_subcarriers)
        _set_spin(params, "单帧OFDM符号数", self.ofdm_symbols)
        _set_text(params, "块状导频索引", self.pilot_indexes)
        _set_combo(params, "波形成形", self.pulse_shaping)
        _set_dspin(params, "滚降因子", self.rolloff_factor)
        _set_spin(params, "滤波器跨度", self.filter_span)
        _set_combo(params, "过采样率", self.oversampling_factor)
        _set_dspin(params, "采样率", self.sample_rate)
        _set_dspin(params, "时长", self.duration)
        _set_combo(params, "频偏补偿", self.enable_cfo_comp)
        _set_combo(params, "IQ补偿", self.enable_iq_comp)
        _set_combo(params, "信道估计与均衡", self.enable_ch_est)
        _set_combo(params, "随机种子策略", self.seed_strategy)


# -----------------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------------
def _set_combo(params: dict, key: str, widget) -> None:
    if key in params:
        idx = widget.findText(str(params[key]))
        if idx >= 0:
            widget.setCurrentIndex(idx)


def _set_text(params: dict, key: str, widget: QLineEdit) -> None:
    if key in params:
        widget.setText(str(params[key]))


def _set_spin(params: dict, key: str, widget) -> None:
    if key in params:
        widget.setValue(int(params[key]))


def _set_dspin(params: dict, key: str, widget) -> None:
    if key in params:
        widget.setValue(float(params[key]))
