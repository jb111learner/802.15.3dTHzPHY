from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QGridLayout, QTabWidget, QWidget

from thz_sim_ui.data.mock_data import CHANNEL_MODULES
from thz_sim_ui.widgets.charts import ChartPlaceholder, TextSummaryCard
from thz_sim_ui.widgets.common import PlaceholderList
from thz_sim_ui.widgets.forms import combo, dspin, make_form_group, make_toggle_list, spin
from thz_sim_ui.widgets.workbench import WorkbenchPage


class ChannelIntegrationPage(WorkbenchPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('信道集成', '作为可视化信道设计器，支持模块启停、排序、范围配置与对比标记。', parent)

        self.channel_modules_group = make_toggle_list('信道模块总览', CHANNEL_MODULES)
        self.add_left_widget(self.channel_modules_group)

        self.scope = combo(['全链路', '仅训练序列', '仅数据区'])
        self.priority = spin(1, 20, 5)
        self.snr = dspin(-10, 80, 18, suffix='dB')
        self.multipath_count = spin(1, 32, 4)
        self.delay_spread = dspin(0, 500, 42, suffix='ns')
        self.cfo = dspin(0, 500, 12.5, suffix='MHz')
        self.absorption = combo(['低', '中', '高'])

        self.add_left_widget(make_form_group('当前选中模块参数', [
            ('作用范围', self.scope),
            ('优先级', self.priority),
            ('SNR', self.snr),
            ('多径路径数', self.multipath_count),
            ('路径时延扩展', self.delay_spread),
            ('载频偏移', self.cfo),
            ('吸收强度', self.absorption),
        ]))
        self.add_left_widget(PlaceholderList('操作说明', ['支持开启/关闭', '支持参数录入', '支持优先级排序', '支持作用范围配置', '支持批量对比标记']))
        self.add_left_stretch()

        tabs = QTabWidget()
        tabs.addTab(self._overview_tab(), '信道框图')
        tabs.addTab(self._grid_two('时域响应', '频域响应', 'line', 'line'), '时域响应')
        tabs.addTab(self._grid_two('CIR / PDP', '多径能量分布', 'bar', 'bar'), 'CIR / PDP')
        tabs.addTab(self._grid_two('星座影响', 'CFO 相位旋转', 'scatter', 'line'), '星座影响')
        tabs.addTab(self._grid_two('角域分布', '波束视图', 'heatmap', 'bar'), '角域分布')
        tabs.addTab(TextSummaryCard('扩展说明', ['建议后续为每个信道模块增加独立参数面板与小型预览图。', '可与任务中心打通，实现“信道模板库”和“实验方案库”。']), '波束视图')
        self.add_right_widget(tabs)

    def _overview_tab(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(TextSummaryCard('信道设计器说明', [
            '该页面不只是参数页，更是可视化信道模块拼装器。',
            '后续可接入拖拽排序、依赖连线和模块模板库。',
        ]), 0, 0)
        layout.addWidget(ChartPlaceholder('信道框图', '展示 AWGN、多径、CFO、THz 吸收和 MIMO 模块组合结构。', mode='heatmap'), 0, 1)
        return panel

    def _grid_two(self, left_title: str, right_title: str, left_mode: str, right_mode: str) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(ChartPlaceholder(left_title, '图表占位，可替换为真实仿真输出。', mode=left_mode), 0, 0)
        layout.addWidget(ChartPlaceholder(right_title, '图表占位，可替换为真实仿真输出。', mode=right_mode), 0, 1)
        return panel

    def get_all_parameters(self) -> dict[str, object]:
        enabled_modules = [cb.text() for cb in self.channel_modules_group.findChildren(QCheckBox) if cb.isChecked()]
        return {
            '信道模块总览': enabled_modules,
            '作用范围': self.scope.currentText(),
            '优先级': self.priority.value(),
            'SNR': self.snr.value(),
            '多径路径数': self.multipath_count.value(),
            '路径时延扩展': self.delay_spread.value(),
            '载频偏移': self.cfo.value(),
            '吸收强度': self.absorption.currentText(),
        }
