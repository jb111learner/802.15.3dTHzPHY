from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QListWidget, QProgressBar, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget, QPushButton

from thz_sim_ui.data.mock_data import RECENT_TASKS
from thz_sim_ui.widgets.charts import TextSummaryCard
from thz_sim_ui.widgets.common import CardWidget, StatusBadge
from thz_sim_ui.widgets.workbench import WorkbenchPage


class TaskCenterPage(WorkbenchPage):
    open_recovery_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('任务中心', '用于承载运行管理、任务筛选、详情查看和恢复入口。', parent)
        self.task_table = None

        filter_list = QListWidget()
        filter_list.addItems(['全部任务', '运行中', '已完成', '已暂停', '已中断', '待恢复', '已失败', '已归档'])
        filter_list.setCurrentRow(0)
        self.add_left_widget(filter_list)

        hint_card = CardWidget('筛选说明', '后续可绑定任务查询接口，实现按工程、日期、状态与标签多条件过滤。')
        jump_button = QPushButton('打开断点续跑页')
        jump_button.setProperty('role', 'primary')
        jump_button.clicked.connect(self.open_recovery_requested.emit)
        hint_card.layout.addWidget(jump_button)
        self.add_left_widget(hint_card)
        self.add_left_stretch()

        tabs = QTabWidget()
        tabs.addTab(self._task_list_tab(), '任务列表')
        tabs.addTab(self._detail_tab(), '任务详情')
        self.add_right_widget(tabs)

    def _task_list_tab(self) -> QWidget:
        self.task_table = QTableWidget(len(RECENT_TASKS), 9)
        self.task_table.setHorizontalHeaderLabels(['任务名称', '工程归属', '运行模式', '关键参数', '阶段', '进度', '已用时长', '预计剩余', '状态'])
        self.task_table.verticalHeader().setVisible(False)
        self.task_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._refresh_task_table(RECENT_TASKS)
        return self.task_table

    def _refresh_task_table(self, tasks: list) -> None:
        if self.task_table is None:
            return
        self.task_table.setRowCount(len(tasks))
        for row, item in enumerate(tasks):
            # 基本文本单元复用
            for col, value in enumerate((item.name, item.project, item.mode, item.tags, item.stage)):
                cell = self.task_table.item(row, col)
                if cell is None:
                    cell = QTableWidgetItem(value)
                    self.task_table.setItem(row, col, cell)
                elif cell.text() != value:
                    cell.setText(value)

            # 进度条复用
            progress_widget = self.task_table.cellWidget(row, 5)
            if progress_widget is None:
                progress_widget = QProgressBar()
                self.task_table.setCellWidget(row, 5, progress_widget)
            if int(progress_widget.value()) != int(item.progress):
                progress_widget.setValue(item.progress)

            for col, value in enumerate((item.elapsed, item.eta), start=6):
                cell = self.task_table.item(row, col)
                if cell is None:
                    cell = QTableWidgetItem(value)
                    self.task_table.setItem(row, col, cell)
                elif cell.text() != value:
                    cell.setText(value)

            status_widget = self.task_table.cellWidget(row, 8)
            if status_widget is None or status_widget.text() != item.status:
                self.task_table.setCellWidget(row, 8, StatusBadge(item.status))

    def _detail_tab(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(TextSummaryCard('配置快照', ['链路模式：2×2 MIMO', '编码：LDPC', '调制：16QAM', '信道：多径 + CFO']), 0, 0)
        layout.addWidget(TextSummaryCard('执行日志', ['[16:21:00] 初始化完成', '[16:21:03] 进入同步阶段', '[16:22:11] 保存快照点 #3']), 0, 1)
        layout.addWidget(TextSummaryCard('中间结果预览', ['BER 曲线：已生成', '星座图：已生成', 'PDP：待生成']), 1, 0)
        layout.addWidget(TextSummaryCard('恢复建议', ['可以从参数点级恢复', '当前接收机阶段已执行到均衡', '建议保留最近一次快照']), 1, 1)
        return panel

    def update_tasks(self, tasks: list) -> None:
        # 更新任务列表表
        self._refresh_task_table(tasks)

