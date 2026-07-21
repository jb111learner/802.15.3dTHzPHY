from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QProgressBar, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from thz_sim_ui.data.mock_data import CAPABILITIES, RECENT_RESULTS, RECENT_TASKS
from thz_sim_ui.widgets.common import CardWidget, ChipList, SectionHeader, StatusBadge


class HomePage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        layout.addWidget(
            SectionHeader(
                '平台总览',
                '用于承载快速进入、状态总览、任务恢复和模板调用的首页入口。',
            )
        )

        quick_row = QGridLayout()
        quick_row.setSpacing(12)
        quick_items = [
            ('新建单链路仿真', '从链路模板快速启动单条链路验证。'),
            ('新建批量对比实验', '创建多参数、多方案并行对比任务。'),
            ('新建 1Tbps 专项工程', '面向超高速链路论证的专项入口。'),
            ('从标准模板创建', '调用标准体制或组内模板。'),
            ('从历史工程复制', '复用历史参数并生成新版本。'),
        ]
        for idx, (title, desc) in enumerate(quick_items):
            card = CardWidget(title, desc)
            enter = QLabel('点击后可接入实际创建逻辑 →')
            enter.setObjectName('CardHint')
            card.layout.addWidget(enter)
            quick_row.addWidget(card, 0, idx)
        layout.addLayout(quick_row)

        capability_card = CardWidget('平台能力总览', '面向链路搭建、批量实验、工程运行与展示汇报的统一工作台能力。')
        capability_card.layout.addWidget(ChipList(CAPABILITIES, columns=5))
        layout.addWidget(capability_card)

        bottom = QHBoxLayout()
        bottom.setSpacing(12)

        task_card = CardWidget('最近任务', '保留恢复入口与状态摘要，后续可直接绑定任务中心数据。')
        task_table = QTableWidget(len(RECENT_TASKS), 6)
        task_table.setHorizontalHeaderLabels(['任务名称', '运行模式', '参数标签', '状态', '进度', '更新时间'])
        task_table.verticalHeader().setVisible(False)
        task_table.setEditTriggers(QTableWidget.NoEditTriggers)
        task_table.setSelectionBehavior(QTableWidget.SelectRows)
        task_table.horizontalHeader().setStretchLastSection(True)
        for row, item in enumerate(RECENT_TASKS):
            task_table.setItem(row, 0, QTableWidgetItem(item.name))
            task_table.setItem(row, 1, QTableWidgetItem(item.mode))
            task_table.setItem(row, 2, QTableWidgetItem(item.tags))
            task_table.setCellWidget(row, 3, StatusBadge(item.status))
            progress = QProgressBar()
            progress.setValue(item.progress)
            task_table.setCellWidget(row, 4, progress)
            task_table.setItem(row, 5, QTableWidgetItem('2026-03-16 16:30'))
        task_card.layout.addWidget(task_table)
        bottom.addWidget(task_card, 2)

        result_card = CardWidget('最近结果', '保留图表入口和报告导出位置。')
        result_grid = QGridLayout()
        result_grid.setSpacing(10)
        for idx, title in enumerate(RECENT_RESULTS):
            tile = CardWidget(title, '图表组件已留位，后续接入真实结果即可。')
            result_grid.addWidget(tile, idx // 2, idx % 2)
        result_card.layout.addLayout(result_grid)
        bottom.addWidget(result_card, 1)

        layout.addLayout(bottom, 1)

        self.task_table = task_table

    def update_tasks(self, tasks: list) -> None:
        # 更新任务表，仅更新变化字段，避免频繁重绘导致卡顿
        if not hasattr(self, 'task_table') or self.task_table is None:
            return

        self.task_table.setRowCount(len(tasks))

        for row, item in enumerate(tasks):
            # 基本单元格复用
            for col, value in enumerate((item.name, item.mode, item.tags)):
                cell_item = self.task_table.item(row, col)
                if cell_item is None:
                    cell_item = QTableWidgetItem(value)
                    self.task_table.setItem(row, col, cell_item)
                else:
                    if cell_item.text() != value:
                        cell_item.setText(value)

            # 状态Badge复用
            status_widget = self.task_table.cellWidget(row, 3)
            if status_widget is None or status_widget.text() != item.status:
                self.task_table.setCellWidget(row, 3, StatusBadge(item.status))

            # 进度条复用
            progress_widget = self.task_table.cellWidget(row, 4)
            if progress_widget is None:
                progress_widget = QProgressBar()
                self.task_table.setCellWidget(row, 4, progress_widget)
            if int(progress_widget.value()) != int(item.progress):
                progress_widget.setValue(item.progress)

            # 运行时间显示当前 elapsed
            time_item = self.task_table.item(row, 5)
            time_text = item.elapsed
            if time_item is None:
                self.task_table.setItem(row, 5, QTableWidgetItem(time_text))
            elif time_item.text() != time_text:
                time_item.setText(time_text)

