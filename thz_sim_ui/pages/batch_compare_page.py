from __future__ import annotations

import os
from typing import List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QRadioButton, QTableWidget, QTableWidgetItem, QTabWidget, QWidget, QPushButton, QVBoxLayout, QGroupBox, QLabel, QFormLayout, QHBoxLayout, QProgressBar

from thz_sim_ui.widgets.charts import ChartPlaceholder, TextSummaryCard
from thz_sim_ui.widgets.common import PlaceholderList, StatusBadge
from thz_sim_ui.widgets.forms import combo, line, make_form_group, make_radio_group, spin, dspin
from thz_sim_ui.widgets.workbench import WorkbenchPage


def get_project_list() -> List[str]:
    workspace_path = os.path.join(os.path.dirname(__file__), '..', '..', 'projects/single')
    workspace_path = os.path.abspath(workspace_path)
    projects = []
    if os.path.exists(workspace_path):
        for item in os.listdir(workspace_path):
            item_path = os.path.join(workspace_path, item)
            if os.path.isdir(item_path):
                projects.append(item)
    if not projects:
        projects = ['THz_Demo_Project', 'MIMO_Baseline', '1Tbps_PreStudy']
    return sorted(projects)


class BatchComparePage(WorkbenchPage):

    def set_all_parameters(self, params: dict[str, object]) -> None:
        if '工程名称' in params:
            self.project_name_input.setText(str(params['工程名称']))

        if '自变量参数' in params:
            index = self.independent_var_combo.findText(str(params['自变量参数']))
            if index >= 0:
                self.independent_var_combo.setCurrentIndex(index)

        if 'SNR最小值' in params:
            self.snr_min.setValue(float(params['SNR最小值']))
        if 'SNR最大值' in params:
            self.snr_max.setValue(float(params['SNR最大值']))
        if 'SNR步长' in params:
            self.snr_step.setValue(float(params['SNR步长']))

        if '运行方式' in params:
            index = self.run_mode.findText(str(params['运行方式']))
            if index >= 0:
                self.run_mode.setCurrentIndex(index)
        if '优先级' in params:
            index = self.priority.findText(str(params['优先级']))
            if index >= 0:
                self.priority.setCurrentIndex(index)
        if '最大并发数' in params:
            self.max_concurrency.setValue(int(params['最大并发数']))
        if '异常重试次数' in params:
            self.retry_count.setValue(int(params['异常重试次数']))
        if '保存中间结果' in params:
            index = self.save_intermediate.findText(str(params['保存中间结果']))
            if index >= 0:
                self.save_intermediate.setCurrentIndex(index)

        configs = params.get('配置方案', [])
        while self.config_groups:
            cfg = self.config_groups.pop(0)
            cfg['group'].deleteLater()

        for config in configs:
            project = config.get('工程', '')
            label = config.get('结果标签', '')
            priority = config.get('运行优先级', '中')
            self._add_config_scheme(project, label, priority)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('批量对比', parent)

        self.config_layout = QVBoxLayout()
        self.config_groups = []
        self.scheme_table = None
        self.simulation_results = {}
        self.ber_chart = None  # 保存 BER 曲线叠图引用

        self.project_name_input = line('BatchCompare_01')

        self.add_config_button = QPushButton('新增配置')
        self.add_config_button.clicked.connect(self._add_config_scheme)

        self.config_container = QGroupBox('对比配置设置')
        self.config_container.setLayout(self.config_layout)

        project_name_layout = QFormLayout()
        project_name_layout.setContentsMargins(14, 18, 14, 10)
        project_name_layout.setSpacing(10)
        project_name_label = QLabel('工程名称')
        project_name_label.setToolTip('工程名称：设置批量对比工程的名称，保存时会以此名称创建文件夹。')
        project_name_layout.addRow(project_name_label, self.project_name_input)

        project_name_widget = QWidget()
        project_name_widget.setLayout(project_name_layout)
        self.config_layout.addWidget(project_name_widget)

        self.config_layout.addWidget(self.add_config_button)

        self.independent_var_combo = combo(['信噪比'])

        self.snr_min = dspin(-20, 100, 0, decimals=1, suffix=' dB')
        self.snr_max = dspin(-20, 100, 30, decimals=1, suffix=' dB')
        self.snr_step = dspin(0.1, 10, 2, decimals=1, suffix=' dB')

        snr_range_layout = QWidget()
        snr_range_grid = QGridLayout(snr_range_layout)
        snr_range_grid.setContentsMargins(0, 0, 0, 0)
        snr_range_grid.setSpacing(8)

        snr_range_grid.addWidget(QLabel('最小值'), 0, 0)
        snr_range_grid.addWidget(self.snr_min, 0, 1)
        snr_range_grid.addWidget(QLabel('最大值'), 1, 0)
        snr_range_grid.addWidget(self.snr_max, 1, 1)
        snr_range_grid.addWidget(QLabel('步长'), 2, 0)
        snr_range_grid.addWidget(self.snr_step, 2, 1)

        self.independent_var_group = QGroupBox('自变量选择')
        iv_layout = QFormLayout(self.independent_var_group)
        iv_layout.setContentsMargins(14, 18, 14, 14)
        iv_layout.setSpacing(10)

        var_label = QLabel('自变量参数')
        var_label.setToolTip('自变量参数：接口已预留，后续可绑定真实参数模型。')
        iv_layout.addRow(var_label, self.independent_var_combo)

        range_label = QLabel('SNR 范围设置')
        range_label.setToolTip('SNR 范围设置：设置信噪比的扫描范围。')
        iv_layout.addRow(range_label, snr_range_layout)

        self.add_left_widget(self.independent_var_group)

        self.add_left_widget(self.config_container)

        self.run_mode = combo(['串行运行', '并行运行'])
        self.priority = combo(['高', '中', '低'])
        self.max_concurrency = spin(1, 128, 8)
        self.retry_count = spin(0, 10, 1)
        self.save_intermediate = combo(['是', '否'])

        self.add_left_widget(make_form_group('调度策略', [
            ('运行方式', self.run_mode),
            ('优先级', self.priority),
            ('最大并发数', self.max_concurrency),
            ('异常重试次数', self.retry_count),
            ('保存中间结果', self.save_intermediate),
        ]))
        self.add_left_widget(PlaceholderList('页面目标', ['一次性设计多组实验任务', '避免重复手工配置', '为结果叠图和排行准备结构化数据']))
        self.add_left_stretch()

        tabs = QTabWidget()
        tabs.addTab(self._scheme_list(), '对比方案列表')
        tabs.addTab(self._matrix_tab(), '参数矩阵预览')
        tabs.addTab(self._estimate_tab(), '任务规模预估')
        tabs.addTab(self._overlay_tab(), '结果叠图')
        tabs.addTab(self._ranking_tab(), '指标排行')
        tabs.addTab(TextSummaryCard('结论摘要', ['适合作为单次实验中的多参数、多波形与多体制对比页面。', '后续建议增加"导出对比报告"和"保存为对比模板"。']), '结论摘要')
        self.add_right_widget(tabs)

        self._add_config_scheme()

    def _scheme_list(self) -> QWidget:
        self.scheme_table = QTableWidget(0, 7)
        self.scheme_table.setHorizontalHeaderLabels(['方案名称', '关联工程', '运行优先级', 'SNR范围', '预计仿真点', '进度', '状态'])
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
        snr_count = int((snr_max - snr_min) / snr_step) + 1

        for row, cfg in enumerate(self.config_groups):
            label = cfg['label'].text()
            project = cfg['project'].currentText()
            priority = cfg['priority'].currentText()

            self.scheme_table.setItem(row, 0, QTableWidgetItem(label))
            self.scheme_table.setItem(row, 1, QTableWidgetItem(project))
            self.scheme_table.setItem(row, 2, QTableWidgetItem(priority))
            self.scheme_table.setItem(row, 3, QTableWidgetItem(f'{snr_min}~{snr_max}dB'))
            self.scheme_table.setItem(row, 4, QTableWidgetItem(str(snr_count)))

            progress_widget = QProgressBar()
            progress_widget.setValue(0)
            self.scheme_table.setCellWidget(row, 5, progress_widget)

            status_widget = StatusBadge('运行中')
            self.scheme_table.setCellWidget(row, 6, status_widget)

    def _add_config_scheme(self, project_name: str = '', result_label_text: str = '', priority_text: str = '中') -> None:
        scheme_index = len(self.config_groups) + 1
        project_list = get_project_list()
        project_combo = combo(project_list)

        if project_name and project_name in project_list:
            project_combo.setCurrentText(project_name)

        if not result_label_text:
            result_label_text = f'方案{scheme_index}'
        result_label = line(result_label_text)

        priority_combo = combo(['高', '中', '低'])
        if priority_text in ['高', '中', '低']:
            priority_combo.setCurrentText(priority_text)

        remove_button = QPushButton('删除')
        remove_button.setStyleSheet('color: #E54B4F;')

        group = QGroupBox(f'对比配置 {scheme_index}')
        layout = QFormLayout(group)
        layout.setContentsMargins(14, 18, 14, 14)
        layout.setSpacing(10)

        project_label = QLabel('工程')
        project_label.setToolTip('工程：从工作空间中选择项目。')
        layout.addRow(project_label, project_combo)

        label_label = QLabel('结果标签')
        label_label.setToolTip('结果标签：标识此配置方案的结果。')
        layout.addRow(label_label, result_label)

        priority_label = QLabel('运行优先级')
        priority_label.setToolTip('运行优先级：设置此方案的执行优先级。')
        layout.addRow(priority_label, priority_combo)

        button_layout = QGridLayout()
        button_layout.addWidget(remove_button, 0, 0)
        layout.addRow(button_layout)

        self.config_groups.append({
            'group': group,
            'project': project_combo,
            'label': result_label,
            'priority': priority_combo,
            'remove': remove_button
        })

        self.config_layout.insertWidget(len(self.config_groups), group)

        def remove_scheme():
            index = None
            for i, cfg in enumerate(self.config_groups):
                if cfg['group'] == group:
                    index = i
                    break
            if index is not None:
                self.config_groups.pop(index)
                group.deleteLater()
                for i, cfg in enumerate(self.config_groups):
                    cfg['group'].setTitle(f'对比配置 {i + 1}')

        remove_button.clicked.connect(remove_scheme)

    def update_simulation_progress(self, scheme_index, progress):
        if self.scheme_table is not None and scheme_index < self.scheme_table.rowCount():
            progress_widget = self.scheme_table.cellWidget(scheme_index, 5)
            if progress_widget is not None:
                progress_widget.setValue(progress)
                if progress == 100:
                    status_widget = StatusBadge('已完成')
                    self.scheme_table.setCellWidget(scheme_index, 6, status_widget)

    def set_simulation_results(self, results):
        self.simulation_results = results

    def _matrix_tab(self) -> QWidget:
        table = QTableWidget(5, 5)
        table.setHorizontalHeaderLabels(['参数', '方案-01', '方案-02', '方案-03', '方案-04'])
        table.verticalHeader().setVisible(False)
        rows = [
            ('SNR', '12', '18', '18', '18'),
            ('调制', 'QPSK', 'QPSK', '16QAM', '64QAM'),
            ('编码率', '0.67', '0.67', '0.75', '0.83'),
            ('MIMO', 'off', 'off', 'off', 'on'),
            ('波形', '经典', '经典', '多载波', '新波形A'),
        ]
        for r, values in enumerate(rows):
            for c, value in enumerate(values):
                table.setItem(r, c, QTableWidgetItem(value))
        return table

    def _estimate_tab(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(ChartPlaceholder('预计运行量', '用于估算整体任务规模和并发压力。', mode='bar'), 0, 0)
        layout.addWidget(TextSummaryCard('调度建议', ['当前共 6 个方案，建议最大并发数 4。', 'MIMO 和 1Tbps 方案应分配更高优先级的资源。']), 0, 1)
        return panel

    def _overlay_tab(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self.ber_chart = ChartPlaceholder('BER 曲线叠图', '展示各方案在不同SNR下的BER对比。', mode='semilog')
        layout.addWidget(self.ber_chart, 0, 0)

        return panel

    def update_ber_chart(self, image_path: str) -> None:
        """更新 BER 曲线叠图显示"""
        if self.ber_chart is not None:
            self.ber_chart.set_image(image_path)

    def _ranking_tab(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(ChartPlaceholder('核心指标排行', '按 BER、吞吐率和复杂度进行综合排序。', mode='bar'), 0, 0)
        layout.addWidget(PlaceholderList('结论候选', ['方案-05 吞吐率最高', '方案-02 BER 最优', '方案-04 频谱效率更优', '方案-03 复杂度更低']), 0, 1)
        return panel

    def _get_selected_radio_text(self, group: QWidget) -> str:
        for rb in group.findChildren(QRadioButton):
            if rb.isChecked():
                return rb.text()
        return ''

    def get_all_parameters(self) -> dict[str, object]:
        configs = []
        for cfg in self.config_groups:
            configs.append({
                '工程': cfg['project'].currentText(),
                '结果标签': cfg['label'].text(),
                '运行优先级': cfg['priority'].currentText(),
            })

        return {
            '工程名称': self.project_name_input.text(),
            '自变量参数': self.independent_var_combo.currentText(),
            'SNR最小值': self.snr_min.value(),
            'SNR最大值': self.snr_max.value(),
            'SNR步长': self.snr_step.value(),
            '配置方案': configs,
            '运行方式': self.run_mode.currentText(),
            '优先级': self.priority.currentText(),
            '最大并发数': self.max_concurrency.value(),
            '异常重试次数': self.retry_count.value(),
            '保存中间结果': self.save_intermediate.currentText(),
        }
