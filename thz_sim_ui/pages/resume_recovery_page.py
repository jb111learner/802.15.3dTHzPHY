from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QTableWidget, QTableWidgetItem, QTabWidget, QWidget

from thz_sim_ui.widgets.charts import ChartPlaceholder, TextSummaryCard
from thz_sim_ui.widgets.common import PlaceholderList
from thz_sim_ui.widgets.forms import combo, line, make_form_group
from thz_sim_ui.widgets.workbench import WorkbenchPage


class ResumeRecoveryPage(WorkbenchPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('中断保护与断点续跑', '面向长任务恢复，展示快照、差异、恢复路径与无损恢复判断。', parent)

        self.add_left_widget(make_form_group('任务中断信息', [
            ('任务名称', line('STD-802153d-002')),
            ('中断原因', combo(['人工暂停', '异常退出', '资源不足'])),
            ('中断时间', line('2026-03-16 15:58:22')),
            ('最近快照时间', line('2026-03-16 15:57:10')),
            ('当前恢复状态', combo(['待恢复', '已校验', '恢复中'])),
            ('可恢复层级', combo(['任务级恢复', '参数点级恢复', '链路阶段级恢复', '图表结果续生成'])),
        ]))
        self.add_left_widget(PlaceholderList('恢复目标', ['清楚看到任务执行到哪里', '看见保存了哪些状态', '知道可从何处恢复']))
        self.add_left_stretch()

        tabs = QTabWidget()
        tabs.addTab(self._snapshot_tab(), '快照列表')
        tabs.addTab(TextSummaryCard('参数快照', ['当前配置与中断前一致', '导频密度：密集导频', '信道估计：LMMSE', '恢复后建议保持随机种子不变']), '参数快照')
        tabs.addTab(TextSummaryCard('中间数据', ['已完成扫描点：12 / 20', '未完成扫描点：8', '已保存中间结果：BER、PDP、估计误差图']), '中间数据')
        tabs.addTab(self._path_tab(), '恢复路径')
        tabs.addTab(self._diff_tab(), '差异校验')
        self.add_right_widget(tabs)

    def _snapshot_tab(self) -> QWidget:
        table = QTableWidget(4, 5)
        table.setHorizontalHeaderLabels(['快照点', '保存时间', '阶段', '数据范围', '可恢复性'])
        table.verticalHeader().setVisible(False)
        rows = [
            ('#1', '15:20:03', '同步', '训练序列', '可恢复'),
            ('#2', '15:34:55', '频偏估计', '多参数点 1-8', '可恢复'),
            ('#3', '15:57:10', '均衡', '多参数点 1-12', '推荐恢复'),
            ('#4', '15:58:00', '译码', '多参数点 1-12', '需校验'),
        ]
        for r, values in enumerate(rows):
            for c, value in enumerate(values):
                table.setItem(r, c, QTableWidgetItem(value))
        return table

    def _path_tab(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(TextSummaryCard('恢复路径建议', [
            '优先从快照点 #3 恢复。',
            '恢复后先执行参数差异校验，再重新挂接结果导出流程。',
            '若恢复失败，可降级为参数点级恢复。',
        ]), 0, 0)
        layout.addWidget(ChartPlaceholder('恢复路径图', '后续可替换为阶段状态机或流程图。', mode='line'), 0, 1)
        return panel

    def _diff_tab(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(ChartPlaceholder('差异热力图', '用于比较当前配置与快照配置差异。', mode='heatmap'), 0, 0)
        layout.addWidget(PlaceholderList('差异校验结果', ['配置文件一致', '信道种子一致', 'MIMO 模式一致', '允许无损恢复：是']), 0, 1)
        return panel
