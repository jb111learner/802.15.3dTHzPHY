from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QRadioButton, QTabWidget, QWidget

from thz_sim_ui.data.mock_data import LINK_FLOW
from thz_sim_ui.widgets.charts import ChartPlaceholder, TextSummaryCard
from thz_sim_ui.widgets.common import CardWidget, PlaceholderList, SimpleFlowWidget, TwoColumnMetricGrid
from thz_sim_ui.widgets.forms import make_radio_group
from thz_sim_ui.widgets.workbench import WorkbenchPage


class LinkDesignPage(WorkbenchPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('仿真链路设计', '选择链路模式、实现精度、空间模式和波形方案，构建系统级链路入口。', parent)

        self.link_mode_group = make_radio_group('链路模式分区', ['单载波模式', '多载波模式', '1Tbps 专项链路模式', '802.15.3d 链路模式'])
        self.add_left_widget(self.link_mode_group)

        self.precision_group = make_radio_group('实现精度分区', ['浮点仿真', '定点仿真'])
        self.add_left_widget(self.precision_group)

        self.spatial_mode_group = make_radio_group('空间模式分区', ['非 MIMO', '2×2 MIMO'])
        self.add_left_widget(self.spatial_mode_group)

        self.waveform_group = make_radio_group('波形方案分区', ['经典波形', '多载波波形', '新波形方案 A', '新波形方案 B', '自定义波形'])
        self.add_left_widget(self.waveform_group)
        self.add_left_stretch()

        tabs = QTabWidget()
        tabs.addTab(self._build_diagram_tab(), '链路框图')
        tabs.addTab(self._build_view_tab('数据流视图', 'line'), '数据流视图')
        tabs.addTab(self._build_view_tab('物理层结构', 'bar'), '物理层结构')
        tabs.addTab(self._build_summary_tab(), '关键指标摘要')
        tabs.addTab(TextSummaryCard('模式说明', [
            '该页面用于快速确定要跑的链路、所选体制、MIMO 状态与是否启用 1Tbps 专项模式。',
            '后续可在右侧增加链路模板缩略图、模式差异对比和依赖告警。',
        ]), '模式说明')
        self.add_right_widget(tabs)

    def _build_diagram_tab(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        flow_card = CardWidget('默认链路框图', '遵循“发射端—信道—接收端—结果统计”的完整流程。')
        flow_card.layout.addWidget(SimpleFlowWidget(LINK_FLOW))
        layout.addWidget(flow_card, 0, 0, 1, 2)
        layout.addWidget(ChartPlaceholder('数据吞吐示意', '占位图表示例，可替换为真实数据流动画。'), 1, 0)
        layout.addWidget(ChartPlaceholder('模块资源占用', '展示模式切换后的复杂度变化。', mode='bar'), 1, 1)
        return panel

    def _build_view_tab(self, title: str, mode: str) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(ChartPlaceholder(title, '后续可接入链路过程动画或真实拓扑图。', mode=mode), 0, 0)
        layout.addWidget(PlaceholderList('设计要点', ['链路模式切换', 'MIMO 开关', '定点/浮点精度', '波形方案选择']), 0, 1)
        return panel

    def _build_summary_tab(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        metrics = [
            ('当前模式', '多载波 / 浮点 / 2×2 MIMO'),
            ('理论速率', '932 Gbps'),
            ('主要瓶颈', 'CFO 与信道估计复杂度'),
            ('可扩展入口', '1Tbps / 802.15.3d'),
        ]
        layout.addWidget(TwoColumnMetricGrid(metrics), 0, 0, 1, 2)
        layout.addWidget(PlaceholderList('页面目标', ['让用户快速确定要跑什么链路', '采用什么体制', '是否为 MIMO', '逻辑结构是什么']), 1, 0)
        layout.addWidget(TextSummaryCard('接口预留说明', ['模式切换信号可在 MainWindow 中统一管理。', '链路图数据源建议由后端返回流程节点与依赖关系。']), 1, 1)
        return panel

    def _get_selected_radio_text(self, group: QWidget) -> str:
        for rb in group.findChildren(QRadioButton):
            if rb.isChecked():
                return rb.text()
        return ''

    def get_all_parameters(self) -> dict[str, object]:
        return {
            '链路模式分区': self._get_selected_radio_text(self.link_mode_group),
            '实现精度分区': self._get_selected_radio_text(self.precision_group),
            '空间模式分区': self._get_selected_radio_text(self.spatial_mode_group),
            '波形方案分区': self._get_selected_radio_text(self.waveform_group),
        }
