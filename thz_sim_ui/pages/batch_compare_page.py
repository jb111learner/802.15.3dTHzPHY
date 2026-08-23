from __future__ import annotations

import json
import os
from typing import List

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from thz_sim_ui.widgets.common import PlaceholderList, StatusBadge
from thz_sim_ui.widgets.forms import combo, line, make_form_group, spin, dspin
from thz_sim_ui.widgets.workbench import WorkbenchPage


def get_project_list() -> List[str]:
    workspace_path = os.path.join(os.path.dirname(__file__), '..', '..', 'projects/single')
    workspace_path = os.path.abspath(workspace_path)
    if os.path.exists(workspace_path):
        return sorted(
            d for d in os.listdir(workspace_path)
            if os.path.isdir(os.path.join(workspace_path, d))
        )
    return []


def load_project_config(project_name: str) -> dict | None:
    """加载单方案工程的 config.json"""
    workspace_path = os.path.join(os.path.dirname(__file__), '..', '..', 'projects/single')
    config_path = os.path.abspath(os.path.join(workspace_path, project_name, 'config.json'))
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# 对比表中展示的参数（不含自变量 SNR）
MATRIX_PARAMS = [
    ("波形类型", "波形"),
    ("调制方式", "调制"),
    ("信道编码类型", "编码"),
    ("数据子帧长度", "子帧长"),
    ("CP长度", "CP"),
    ("前导码配置", "前导"),
    ("OFDM子载波数", "子载波"),
    ("波形成形", "成型"),
    ("过采样率", "过采样"),
    ("滤波器跨度", "滤波器"),
    ("频偏补偿", "CFO补偿"),
    ("IQ补偿", "IQ补偿"),
    ("信道估计与均衡", "信道均衡"),
]


class BatchComparePage(WorkbenchPage):

    def set_all_parameters(self, params: dict[str, object]) -> None:
        if '工程名称' in params:
            self.project_name_input.setText(str(params['工程名称']))
        if 'SNR最小值' in params:
            self.snr_min.setValue(float(params['SNR最小值']))
        if 'SNR最大值' in params:
            self.snr_max.setValue(float(params['SNR最大值']))
        if 'SNR步长' in params:
            self.snr_step.setValue(float(params['SNR步长']))
        if '每点最大帧数' in params:
            self.max_frames.setValue(int(params['每点最大帧数']))
        if '每点最少独立运行次数' in params:
            self.min_independent_runs.setValue(int(params['每点最少独立运行次数']))
        if '每点最少错误比特' in params:
            self.min_errors.setValue(int(params['每点最少错误比特']))
        configs = params.get('配置方案', [])
        while self.config_groups:
            cfg = self.config_groups.pop(0)
            cfg['group'].deleteLater()
        for config in configs:
            self._add_config_scheme(
                config.get('工程', ''),
                config.get('结果标签', ''),
            )
        self._refresh_matrix()
        # 导入后尝试加载已有结果图表
        self._try_load_existing_chart(str(params.get('工程名称', '')))

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('批量对比', parent)

        self.config_layout = QVBoxLayout()
        self.config_groups: list[dict] = []
        self.scheme_table = None
        self.matrix_table = None
        self.ber_chart = None

        self.project_name_input = line('BatchCompare_01')
        self.add_config_button = QPushButton('新增配置')
        self.add_config_button.clicked.connect(self._add_config_scheme)

        self.config_container = QGroupBox('对比配置设置')
        self.config_container.setLayout(self.config_layout)

        project_name_layout = QFormLayout()
        project_name_layout.setContentsMargins(14, 18, 14, 10)
        project_name_layout.setSpacing(10)
        project_name_layout.addRow(QLabel('工程名称'), self.project_name_input)
        pw = QWidget(); pw.setLayout(project_name_layout)
        self.config_layout.addWidget(pw)
        self.config_layout.addWidget(self.add_config_button)

        # ── 自变量选择 ──
        self.independent_var_combo = combo(['信噪比'])
        self.snr_min = dspin(-20, 100, 0, decimals=1, suffix=' dB')
        self.snr_max = dspin(-20, 100, 30, decimals=1, suffix=' dB')
        self.snr_step = dspin(0.1, 10, 2, decimals=1, suffix=' dB')
        snr_range = QWidget()
        sg = QGridLayout(snr_range); sg.setContentsMargins(0, 0, 0, 0); sg.setSpacing(6)
        sg.addWidget(QLabel('最小值'), 0, 0); sg.addWidget(self.snr_min, 0, 1)
        sg.addWidget(QLabel('最大值'), 1, 0); sg.addWidget(self.snr_max, 1, 1)
        sg.addWidget(QLabel('步长'), 2, 0); sg.addWidget(self.snr_step, 2, 1)
        ivg = QGroupBox('自变量选择')
        ivl = QFormLayout(ivg); ivl.setContentsMargins(14, 18, 14, 14); ivl.setSpacing(10)
        ivl.addRow(QLabel('自变量参数'), self.independent_var_combo)
        ivl.addRow(QLabel('SNR 范围设置'), snr_range)
        self.add_left_widget(ivg)
        self.add_left_widget(self.config_container)

        # ── 调度策略 ──
        self.max_frames = spin(1, 1000, 300)
        self.max_frames.setSingleStep(10)
        self.min_independent_runs = spin(1, 1000, 20)
        self.min_independent_runs.setSingleStep(5)
        self.min_errors = spin(1, 10000, 5000)
        self.min_errors.setSingleStep(100)
        self.add_left_widget(make_form_group('调度策略', [
            ('每点最大帧数', self.max_frames),
            ('每点最少独立运行次数', self.min_independent_runs),
            ('每点最少错误比特', self.min_errors),
        ]))
        self.add_left_widget(PlaceholderList('说明', [
            '从单方案工程读取配置进行对比仿真',
            '每点先满足最少独立运行次数，再按错误比特或最大帧数结束',
            '不保存过程图像，仅保存 BER 数据',
        ]))
        self.add_left_stretch()

        # ── 右侧 Tabs ──
        tabs = QTabWidget()
        tabs.addTab(self._scheme_list_table(), '对比方案列表')
        self.matrix_table = QTableWidget()
        tabs.addTab(self.matrix_table, '参数矩阵预览')
        tabs.addTab(self._overlay_tab(), '结果叠图')
        self.add_right_widget(tabs)

        self._add_config_scheme()
        self._refresh_matrix()

    # =================================================================
    # 方案列表
    # =================================================================
    def _scheme_list_table(self) -> QTableWidget:
        self.scheme_table = QTableWidget(0, 6)
        self.scheme_table.setHorizontalHeaderLabels([
            '方案名称', '关联工程', 'SNR范围', '仿真点数', '进度', '状态'])
        self.scheme_table.verticalHeader().setVisible(False)
        self.scheme_table.setEditTriggers(QTableWidget.NoEditTriggers)
        return self.scheme_table

    def prepare_for_simulation(self):
        if self.scheme_table is None:
            return
        self.scheme_table.setRowCount(len(self.config_groups))
        snr_min = self.snr_min.value()
        snr_max = self.snr_max.value()
        snr_step = self.snr_step.value()
        snr_count = max(1, int((snr_max - snr_min) / snr_step) + 1)
        for row, cfg in enumerate(self.config_groups):
            self.scheme_table.setItem(row, 0, QTableWidgetItem(cfg['label'].text()))
            self.scheme_table.setItem(row, 1, QTableWidgetItem(cfg['project'].currentText()))
            self.scheme_table.setItem(row, 2, QTableWidgetItem(f'{snr_min}~{snr_max}dB'))
            self.scheme_table.setItem(row, 3, QTableWidgetItem(str(snr_count)))
            pb = QProgressBar(); pb.setValue(0)
            self.scheme_table.setCellWidget(row, 4, pb)
            self.scheme_table.setCellWidget(row, 5, StatusBadge('待运行'))

    def update_simulation_progress(self, scheme_index, progress):
        if self.scheme_table and scheme_index < self.scheme_table.rowCount():
            w = self.scheme_table.cellWidget(scheme_index, 4)
            if w:
                w.setValue(progress)
            if progress == 100:
                self.scheme_table.setCellWidget(scheme_index, 5, StatusBadge('已完成'))

    # =================================================================
    # 添加/删除方案
    # =================================================================
    def _add_config_scheme(self, project_name: str = '', result_label_text: str = '') -> None:
        idx = len(self.config_groups) + 1
        projects = get_project_list()
        project_combo = combo(projects)
        if project_name and project_name in projects:
            project_combo.setCurrentText(project_name)
        if not result_label_text:
            result_label_text = f'方案{idx}'
        result_label = line(result_label_text)
        remove_btn = QPushButton('删除')
        remove_btn.setStyleSheet('color: #E54B4F;')

        group = QGroupBox(f'对比配置 {idx}')
        layout = QFormLayout(group)
        layout.setContentsMargins(14, 18, 14, 14); layout.setSpacing(10)
        layout.addRow(QLabel('工程'), project_combo)
        layout.addRow(QLabel('结果标签'), result_label)
        bl = QHBoxLayout(); bl.addWidget(remove_btn); bl.addStretch()
        layout.addRow(bl)

        self.config_groups.append({
            'group': group, 'project': project_combo, 'label': result_label, 'remove': remove_btn,
        })
        self.config_layout.insertWidget(len(self.config_groups), group)

        def remove():
            for i, c in enumerate(self.config_groups):
                if c['group'] == group:
                    self.config_groups.pop(i); group.deleteLater()
                    for j, cc in enumerate(self.config_groups):
                        cc['group'].setTitle(f'对比配置 {j + 1}')
                    break
            self._refresh_matrix()
        remove_btn.clicked.connect(remove)
        # 工程切换时刷新矩阵
        project_combo.currentTextChanged.connect(lambda _: self._refresh_matrix())

    # =================================================================
    # 参数矩阵预览 — 读取各方案 config.json
    # =================================================================
    def _refresh_matrix(self):
        if self.matrix_table is None:
            return
        headers = ['参数'] + [cfg['label'].text() for cfg in self.config_groups]
        n_cols = len(headers)
        self.matrix_table.setColumnCount(n_cols)
        self.matrix_table.setHorizontalHeaderLabels(headers)
        self.matrix_table.setRowCount(len(MATRIX_PARAMS))

        for r, (param_key, display_name) in enumerate(MATRIX_PARAMS):
            self.matrix_table.setItem(r, 0, QTableWidgetItem(display_name))
            for c, cfg in enumerate(self.config_groups):
                proj = cfg['project'].currentText()
                conf = load_project_config(proj) or {}
                val = conf.get(param_key, '—')
                self.matrix_table.setItem(r, c + 1, QTableWidgetItem(str(val)))

    def _overlay_tab(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0)

        # BER
        ber_panel = QWidget()
        ber_layout = QGridLayout(ber_panel)
        ber_layout.setContentsMargins(12, 12, 12, 12); ber_layout.setSpacing(12)
        self.ber_status = QLabel('仿真未完成')
        self.ber_status.setAlignment(Qt.AlignCenter)
        self.ber_status.setMinimumHeight(300)
        self.ber_status.setStyleSheet(
            'background: #FAFBFE; border: 1px solid #E6ECF6; font-size: 14px; color: #888;')
        ber_layout.addWidget(self.ber_status, 0, 0)
        self.ber_chart_label = QLabel()
        self.ber_chart_label.setAlignment(Qt.AlignCenter)
        self.ber_chart_label.setMinimumHeight(300)
        self.ber_chart_label.hide()
        ber_layout.addWidget(self.ber_chart_label, 0, 0)

        # SE
        se_panel = QWidget()
        se_layout = QGridLayout(se_panel)
        se_layout.setContentsMargins(12, 12, 12, 12); se_layout.setSpacing(12)
        self.se_status = QLabel('仿真未完成')
        self.se_status.setAlignment(Qt.AlignCenter)
        self.se_status.setMinimumHeight(300)
        self.se_status.setStyleSheet(
            'background: #FAFBFE; border: 1px solid #E6ECF6; font-size: 14px; color: #888;')
        se_layout.addWidget(self.se_status, 0, 0)
        self.se_chart_label = QLabel()
        self.se_chart_label.setAlignment(Qt.AlignCenter)
        self.se_chart_label.setMinimumHeight(300)
        self.se_chart_label.hide()
        se_layout.addWidget(self.se_chart_label, 0, 0)

        overlay_tabs = QTabWidget()
        overlay_tabs.addTab(ber_panel, "BER")
        overlay_tabs.addTab(se_panel, "谱效")
        layout.addWidget(overlay_tabs, 0, 0)
        return panel

    def _try_load_existing_chart(self, project_name: str) -> None:
        if not project_name:
            return
        # 从各方案 JSON 结果读 BER/SE 数据直接绘图
        root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', 'projects', 'batch'))
        ber_data = []
        se_data = []
        for cfg in self.config_groups:
            scheme_name = cfg['label'].text()
            json_path = os.path.join(root, project_name, scheme_name, 'scheme_results.json')
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                all_snrs, snrs, bers, ebn0s, all_ebn0s = [], [], [], [], []
                snrs_se, ses = [], []
                for item in results:
                    r = item.get('result', {})
                    snr = item.get('snr')
                    if isinstance(r, dict):
                        ber = r.get('BER')
                        if ber is not None:
                            all_snrs.append(snr)
                            eb = r.get('EbN0_dB')
                            if eb is not None and np.isfinite(eb):
                                all_ebn0s.append(eb)
                            if ber > 0:
                                snrs.append(snr)
                                bers.append(ber)
                                if eb is not None and np.isfinite(eb):
                                    ebn0s.append(eb)
                        se = r.get('spectral_efficiency_bps_per_hz')
                        if se is not None and np.isfinite(se):
                            snrs_se.append(snr)
                            ses.append(se)
                if all_snrs:
                    ber_data.append((scheme_name, snrs, bers, ebn0s if ebn0s else None,
                                     all_snrs, all_ebn0s if all_ebn0s else None))
                if snrs_se:
                    se_data.append((scheme_name, snrs_se, ses))
        if ber_data:
            self._plot_ber_overlay(ber_data)
        if se_data:
            self._plot_se_overlay(se_data)

    def _plot_ber_overlay(self, ber_data: list) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io

        plt.rcParams.update({
            "font.family": "serif", "axes.unicode_minus": False,
            "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
            "xtick.major.size": 4, "ytick.major.size": 4,
            "grid.alpha": 0.25, "grid.linestyle": "--", "grid.linewidth": 0.4,
        })
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.set_facecolor("white"); fig.patch.set_facecolor("white")
        ax2 = ax.twiny()
        colors = ['#2C68B4', '#D95F02', '#3A9D3A', '#9467BD', '#E54B4F', '#8C6B4F']
        for i, item in enumerate(ber_data):
            name, snrs, bers = item[0], item[1], item[2]
            color = colors[i % len(colors)]
            if snrs:
                ax.semilogy(snrs, bers, marker='o', ms=6, lw=1.3,
                            color=color, label=name)
                # 垂直下降线
                all_snrs = item[4] if len(item) > 4 else snrs
                if len(all_snrs) > len(snrs):
                    zero_snrs = [s for s in all_snrs if s not in set(snrs)]
                    if zero_snrs:
                        lx, lb = snrs[-1], bers[-1]
                        zx = min(zero_snrs)
                        ax.plot([lx, zx], [lb, lb], ':', color=color, lw=0.8)
                        ax.plot([zx, zx], [lb, 1e-10], ':', color=color, lw=0.8)
                        ax.plot(zero_snrs, [1e-10]*len(zero_snrs), marker='o', ms=6,
                                color=color, fillstyle='none', linestyle='none')
        ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BER")
        ax2.set_xlabel("Eₐ/N₀ (dB)")
        ax.set_title("BER Comparison", fontsize=11, pad=6)
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, which="major", alpha=0.25, ls="--", lw=0.4)
        ax.grid(True, which="minor", alpha=0.10, ls="--", lw=0.3)

        # Eb/N0 上横轴刻度（含 BER=0 的全部点）
        if ber_data:
            all_ebn0s = ber_data[0][5] if len(ber_data[0]) > 5 else None
            all_snrs0 = ber_data[0][4] if len(ber_data[0]) > 4 else ber_data[0][1]
            if all_ebn0s and len(all_ebn0s) == len(all_snrs0):
                ax2.set_xlim(ax.get_xlim())
                n_pts = len(all_snrs0)
                idx = list(range(n_pts)) if n_pts <= 8 else list(
                    np.linspace(0, n_pts - 1, min(10, n_pts), dtype=int)
                )
                ax2.set_xticks([all_snrs0[i] for i in idx])
                ax2.set_xticklabels([f'{all_ebn0s[i]:.1f}' for i in idx])

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.read(), "PNG")
        self.ber_status.hide()
        self.ber_chart_label.setPixmap(pixmap.scaled(
            self.ber_chart_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.ber_chart_label.show()

    def _plot_se_overlay(self, se_data: list) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io

        plt.rcParams.update({
            "font.family": "serif", "axes.unicode_minus": False,
            "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
            "xtick.major.size": 4, "ytick.major.size": 4,
            "grid.alpha": 0.25, "grid.linestyle": "--", "grid.linewidth": 0.4,
        })
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.set_facecolor("white"); fig.patch.set_facecolor("white")
        colors = ['#2C68B4', '#D95F02', '#3A9D3A', '#9467BD', '#E54B4F', '#8C6B4F']
        for i, (name, snrs, ses) in enumerate(se_data):
            ax.plot(snrs, ses, marker='o', ms=6, lw=1.3,
                    color=colors[i % len(colors)], label=name)
        ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Spectral Efficiency (bit/s/Hz)")
        ax.set_title("Spectral Efficiency Comparison", fontsize=11, pad=6)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, which="major", alpha=0.25, ls="--", lw=0.4)
        ax.grid(True, which="minor", alpha=0.10, ls="--", lw=0.3)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.read(), "PNG")
        self.se_status.hide()
        if self.se_chart_label.width() > 0 and self.se_chart_label.height() > 0:
            self.se_chart_label.setPixmap(pixmap.scaled(
                self.se_chart_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.se_chart_label.setPixmap(pixmap)
        self.se_chart_label.show()

    def update_ber_chart(self, image_path: str) -> None:
        """兼容旧版信号（直接加载已有 PNG）"""
        if not image_path or not os.path.exists(image_path):
            return
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return
        self.ber_status.hide()
        self.ber_chart_label.setPixmap(pixmap.scaled(
            self.ber_chart_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.ber_chart_label.show()

    # =================================================================
    # 参数收集
    # =================================================================
    def get_all_parameters(self) -> dict[str, object]:
        configs = []
        for cfg in self.config_groups:
            configs.append({
                '工程': cfg['project'].currentText(),
                '结果标签': cfg['label'].text(),
            })
        return {
            '工程名称': self.project_name_input.text(),
            '自变量参数': self.independent_var_combo.currentText(),
            'SNR最小值': self.snr_min.value(),
            'SNR最大值': self.snr_max.value(),
            'SNR步长': self.snr_step.value(),
            '配置方案': configs,
            '每点最大帧数': self.max_frames.value(),
            '每点最少独立运行次数': self.min_independent_runs.value(),
            '每点最少错误比特': self.min_errors.value(),
        }
