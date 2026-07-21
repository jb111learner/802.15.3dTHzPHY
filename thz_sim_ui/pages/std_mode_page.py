from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QTabWidget, QWidget

from thz_sim_ui.widgets.charts import ChartPlaceholder, TextSummaryCard
from thz_sim_ui.widgets.common import PlaceholderList
from thz_sim_ui.widgets.forms import combo, make_form_group
from thz_sim_ui.widgets.workbench import WorkbenchPage


class StandardModePage(WorkbenchPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('802.15.3d 扩展模式', '承载标准体制扩展场景，便于标准化研究与模式切换。', parent)

        self.standard_template = combo(['模板 A', '模板 B', '模板 C'])
        self.standard_link_mode = combo(['基础模式', '增强模式', '自定义模式'])
        self.compliance_check = combo(['启用', '关闭'])
        self.performance_template = combo(['BER 优先', '吞吐率优先', '综合评价'])

        self.add_left_widget(make_form_group('标准参数与模板配置', [
            ('标准模板', self.standard_template),
            ('标准链路模式', self.standard_link_mode),
            ('合规性检查', self.compliance_check),
            ('性能分析模板', self.performance_template),
        ]))
        self.add_left_widget(PlaceholderList('标准模式能力', ['支持标准参数模板调用', '支持标准链路配置展示', '支持标准模式与自定义模式对比', '支持标准合规性检查']))
        self.add_left_stretch()

        tabs = QTabWidget()
        tabs.addTab(self._overview_tab(), '标准链路说明')
        tabs.addTab(self._pair_tab('性能指标', '标准性能指标', '模式差异对比', 'bar', 'line'), '性能指标')
        tabs.addTab(TextSummaryCard('合规状态展示', ['当前模板满足主要参数范围约束。', '若切换到自定义模式，应重新执行合规性校验。']), '合规状态展示')
        self.add_right_widget(tabs)

    def _overview_tab(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(TextSummaryCard('标准模式说明', ['该页面用于标准参数模板调用与标准模式对比。', '后续可接入标准条文映射、合规规则引擎和自动报告。']), 0, 0)
        layout.addWidget(ChartPlaceholder('标准链路配置示意', '占位图表，可替换为标准链路框图。', mode='line'), 0, 1)
        return panel

    def _pair_tab(self, _, left_title: str, right_title: str, left_mode: str, right_mode: str) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(ChartPlaceholder(left_title, '标准模式占位图表。', mode=left_mode), 0, 0)
        layout.addWidget(ChartPlaceholder(right_title, '标准模式占位图表。', mode=right_mode), 0, 1)
        return panel

    def get_all_parameters(self) -> dict[str, object]:
        return {
            '标准模板': self.standard_template.currentText(),
            '标准链路模式': self.standard_link_mode.currentText(),
            '合规性检查': self.compliance_check.currentText(),
            '性能分析模板': self.performance_template.currentText(),
        }
