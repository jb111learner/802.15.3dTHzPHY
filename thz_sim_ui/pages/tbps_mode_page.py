from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QTabWidget, QWidget

from thz_sim_ui.widgets.charts import ChartPlaceholder, TextSummaryCard
from thz_sim_ui.widgets.common import PlaceholderList, TwoColumnMetricGrid
from thz_sim_ui.widgets.forms import combo, dspin, make_form_group, spin
from thz_sim_ui.widgets.workbench import WorkbenchPage


class TbpsModePage(WorkbenchPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('1Tbps 专项模式', '作为一级能力入口，面向超高速链路方案设计、参数约束分析和新波形验证。', parent)

        self.new_waveform = combo(['新波形 A', '新波形 B', '自定义波形'])
        self.target_rate = dspin(100, 2000, 1000, suffix='Gbps')
        self.modulation_order = combo(['QPSK', '16QAM', '64QAM', '256QAM'])
        self.coding_scheme = combo(['LDPC', 'Polar', 'Hybrid'])
        self.bandwidth_config = dspin(1, 200, 48, suffix='GHz')
        self.parallelism = spin(1, 64, 16)
        self.frame_structure = combo(['高速短帧', '均衡长帧'])
        self.pulse_shaping = combo(['RRC', '自定义'])
        self.antenna_beam = combo(['窄波束', '自适应波束', '混合波束'])
        self.channel_condition = combo(['理想', '温和', '严苛'])

        self.add_left_widget(make_form_group('1Tbps 专项参数', [
            ('新波形选择', self.new_waveform),
            ('目标速率设置', self.target_rate),
            ('调制阶数', self.modulation_order),
            ('编码方式', self.coding_scheme),
            ('带宽配置', self.bandwidth_config),
            ('并行度设置', self.parallelism),
            ('帧结构参数', self.frame_structure),
            ('波形成形参数', self.pulse_shaping),
            ('天线与波束参数', self.antenna_beam),
            ('信道条件选择', self.channel_condition),
        ]))
        self.add_left_widget(PlaceholderList('专项目标', ['理论峰值速率', '净有效速率', '是否达到 1Tbps', '关键瓶颈提示']))
        self.add_left_stretch()

        tabs = QTabWidget()
        tabs.addTab(self._rate_tab(), '速率计算')
        tabs.addTab(self._double_chart('波形结构', '波形结构示意', '资源消耗', 'line', 'bar'), '波形结构')
        tabs.addTab(self._double_chart('链路预算', '链路预算', '可行性分析', 'bar', 'line'), '链路预算')
        tabs.addTab(self._double_chart('资源消耗', '资源消耗', '对比评估', 'bar', 'line'), '资源消耗')
        tabs.addTab(TextSummaryCard('方案优选建议', ['当前参数下理论峰值速率可达 1.18 Tbps。', '瓶颈主要集中在相位噪声和导频开销。', '建议进一步比较新波形 A 与自定义波形在复杂信道下的净速率。']), '对比评估')
        self.add_right_widget(tabs)

    def _rate_tab(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(TwoColumnMetricGrid([
            ('理论峰值速率', '1.18 Tbps'),
            ('净有效速率', '0.93 Tbps'),
            ('是否达到 1Tbps', '理论可达 / 净速率待优化'),
            ('关键瓶颈', '导频开销 + 相位噪声'),
        ]), 0, 0, 1, 2)
        layout.addWidget(ChartPlaceholder('速率计算曲线', '根据带宽、调制、并行度实时估算。', mode='line'), 1, 0)
        layout.addWidget(PlaceholderList('参数灵敏度分析', ['带宽对速率最敏感', '高阶调制受相位噪声影响显著', '并行度提升受实现复杂度约束']), 1, 1)
        return panel

    def _double_chart(self, _, left_title: str, right_title: str, left_mode: str, right_mode: str) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(ChartPlaceholder(left_title, '专项占位图表。', mode=left_mode), 0, 0)
        layout.addWidget(ChartPlaceholder(right_title, '专项占位图表。', mode=right_mode), 0, 1)
        return panel

    def get_all_parameters(self) -> dict[str, object]:
        return {
            '新波形选择': self.new_waveform.currentText(),
            '目标速率设置': self.target_rate.value(),
            '调制阶数': self.modulation_order.currentText(),
            '编码方式': self.coding_scheme.currentText(),
            '带宽配置': self.bandwidth_config.value(),
            '并行度设置': self.parallelism.value(),
            '帧结构参数': self.frame_structure.currentText(),
            '波形成形参数': self.pulse_shaping.currentText(),
            '天线与波束参数': self.antenna_beam.currentText(),
            '信道条件选择': self.channel_condition.currentText(),
        }
