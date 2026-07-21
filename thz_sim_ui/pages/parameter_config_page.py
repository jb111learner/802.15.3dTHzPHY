from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QGridLayout, QTableWidget, QTableWidgetItem, QTabWidget, QWidget
from PySide6.QtWidgets import QVBoxLayout
from thz_sim_ui.widgets.charts import ChartPlaceholder, TextSummaryCard
from thz_sim_ui.widgets.common import PlaceholderList, TwoColumnMetricGrid
from thz_sim_ui.widgets.forms import combo, dspin, line, make_checkbox_grid, make_form_group, spin, text_area
from thz_sim_ui.widgets.workbench import WorkbenchPage


class ParameterConfigPage(WorkbenchPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('参数配置', '采用“大分组 + 子标签”的强交互参数工作台，右侧实时展示摘要、速率和校验。', parent)

        # -------------------------------------------------------------------------
        # 项目与链路基本信息
        # -------------------------------------------------------------------------
        ### 项目信息
        self.project_name = line('')
        self.standard_mode = combo(['自定义', '802.15.3d 模板 A', '802.15.3d 模板 B'])
        self.link_name = line('THz_Link_Main')

        ### 链路基本参数
        self.bandwidth = dspin(1, 200, 38.4, suffix='GHz')
        self.fc = dspin(0.1, 1000, 300, suffix='GHz')
        self.system_type = combo(['单载波', '多载波'])

        ### 仿真参数

        ## 未实现：
        self.run_times = spin(1, 1000, 10)
        self.save_intermediate = combo(['是', '否'])
        self.seed_strategy = combo(['固定种子', '递增种子', '时间种子'])
        self.arithmetic = combo(['浮点', '定点'])
        self.enable_resume = combo(['是', '否'])
        self.full_log = combo(['是', '否'])
        self.export_graph = combo(['是', '否'])


        # -------------------------------------------------------------------------
        # 系统级参数
        # -------------------------------------------------------------------------
        ### 发射端参数
        ## 已实现：
        self.bit_source = combo(['PRBS', '文件输入'])
        self.sample_rate = dspin(1, 3000000, 300000, suffix='MBd')
        self.sample_rate.setSingleStep(1000)
        self.duration = dspin(0.01, 1000, 0.01, suffix='ms')
        self.duration.setSingleStep(0.01)
        self.subframe_length = spin(1, 1000, 480, 'symbols')
        self.subframe_num = spin(1, 1000, 51, 'frames')
        self.gi_length = spin(0, 512, 32, 'symbols')
        self.modulation = combo(['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM'])
        self.coding = combo(['RS(255, 192)', 'LDPC'])
        self.pulse_shaping = combo(["rc", "rrc", "rect"])
        self.rolloff_factor = dspin(0, 1, 0.22)
        self.rolloff_factor.setSingleStep(0.01)
        self.filter_span = spin(1, 20, 8, 'symbols')
        self.oversampling_factor = combo(['1x', '2x', '4x', '8x'])
        self.scrambling = combo(['启用', '关闭'])

        ## 未实现：
        self.mimo_mode = combo(['非 MIMO', '2×2 MIMO'])
        # self.modulation_param = combo(['Gray 映射', 'Set Partitioning'])
        self.preamble = combo(['短前导', '长前导', '双前导'])
        self.pilot = combo(['稀疏导频', '密集导频', '梳状导频'])
        self.subcarrier_mapping = combo(['标准映射', '自定义映射'])
        self.tx_power = dspin(-30, 60, 10, suffix='dBm')

        ### 信道参数
        ## 已实现：
        self.snr = dspin(-10, 80, 5, suffix='dB')

        ## 尚未使用：
        self.noise_temperature = dspin(0, 1000, 290, suffix='K')
        self.noise_figure_db = dspin(0, 20, 2, suffix='dB')

        ## 未实现：
        self.path_loss = dspin(0, 200, 86, suffix='dB')
        self.multipath_count = spin(1, 32, 4)
        self.cfo = dspin(0, 0.1, 0.05, suffix='ppm')
        self.cfo.setSingleStep(0.01)
        self.phase_noise = dspin(0, 20, 1.2, suffix='deg')
        self.atmospheric_absorption = combo(['低', '中', '高'])
        self.antenna_pattern = combo(['定向', '宽波束', '自定义'])
        self.beam_params = line('az=12°, el=5°')
        self.mimo_matrix = combo(['Rayleigh', 'LoS', '测量导入'])

        ### 接收端参数
        ## 实现中：
        self.decoding = combo(['硬译码', '软译码'])

        ## 未实现：
        self.coarse_sync = combo(['相关峰检测', '能量检测'])
        self.fine_sync = combo(['滑窗优化', '最大似然'])
        self.carrier_sync = combo(['PLL', '数据辅助'])
        self.freq_estimation = combo(['导频辅助', '盲估计'])
        self.timing_recovery = combo(['Gardner', 'Mueller'])
        self.channel_estimation = combo(['LS', 'LMMSE', 'OMP'])
        self.equalization = combo(['ZF', 'MMSE', 'DFE'])


        self.add_left_widget(
            make_form_group('系统级参数', [
                ('工程名称', self.project_name),
                ('标准模式', self.standard_mode),
                ('链路名称', self.link_name),
                ('体制类型', self.system_type),
                ('定点/浮点', self.arithmetic),
                ('带宽', self.bandwidth),
                ('载频', self.fc),
            ])
        )

        self.add_left_widget(
            make_form_group('发射端参数', [
                ('比特源配置', self.bit_source),
                ('采样率', self.sample_rate),
                ('时长', self.duration),
                ('数据帧长度', self.subframe_length),
                ('单帧数据子帧数量', self.subframe_num),
                ('GI 长度', self.gi_length),
                ('调制方式', self.modulation),
                ('编码方式', self.coding),
                ('扰码配置', self.scrambling),
                # ('调制参数', self.modulation_param),
                # ('CP 配置', self.cp),
                ('前导码配置', self.preamble),
                ('导频配置', self.pilot),
                ('mimo 模式', self.mimo_mode),
                ('子载波映射', self.subcarrier_mapping),
                ('波形成形', self.pulse_shaping),
                ('滚降因子', self.rolloff_factor),
                ('滤波器跨度', self.filter_span),
                ('发送功率', self.tx_power),
            ])
        )

        self.add_left_widget(
            make_form_group('信道参数', [
                ('SNR', self.snr),
                ('路径损耗', self.path_loss),
                ('多径路径数', self.multipath_count),
                ('载频偏移', self.cfo),
                ('相位噪声', self.phase_noise),
                ('大气吸收', self.atmospheric_absorption),
                ('天线方向图', self.antenna_pattern),
                ('波束参数', self.beam_params),
                ('MIMO 矩阵', self.mimo_matrix),
            ])
        )

        self.add_left_widget(
            make_form_group('接收端参数', [
                ('粗同步参数', self.coarse_sync),
                ('细同步参数', self.fine_sync),
                ('载波同步参数', self.carrier_sync),
                ('频偏估计参数', self.freq_estimation),
                ('定时恢复参数', self.timing_recovery),
                ('信道估计参数', self.channel_estimation),
                ('均衡参数', self.equalization),
                ('译码参数', self.decoding),
            ])
        )
        self.add_left_widget(
            make_form_group('任务运行参数', [
                ('运行次数', self.run_times),
                ('随机种子策略', self.seed_strategy),
                ('保存中间结果', self.save_intermediate),
                ('启用断点续跑', self.enable_resume),
                ('输出完整日志', self.full_log),
                ('导出中间图表', self.export_graph),
            ])
        )
        # self.advanced_checkbox_group = make_checkbox_grid('高级能力勾选', ['参数数组编辑器', '高级结构参数弹窗', '复杂波形面板', '模板差异对比'])
        # self.add_left_widget(self.advanced_checkbox_group)
        # self.add_left_stretch()

        tabs = QTabWidget()
        tabs.addTab(self._overview_tab(), '参数总览')
        tabs.addTab(self._rate_tab(), '理论速率')
        tabs.addTab(self._budget_tab(), '链路预算')
        tabs.addTab(TextSummaryCard('模式说明', ['右侧区域可根据左侧实时配置刷新。', '建议后续接入参数依赖分析器与即时冲突提示引擎。']), '模式说明')
        tabs.addTab(self._dependency_tab(), '依赖关系')
        tabs.addTab(self._validation_tab(), '参数校验提示')
        self.add_right_widget(tabs)

    def _overview_tab(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(TwoColumnMetricGrid([
            ('理论峰值速率', '1.18 Tbps'),
            ('净有效速率', '0.93 Tbps'),
            ('频谱占用估算', '38.4 GHz'),
            ('运算复杂度估算', 'High'),
        ]), 0, 0, 1, 2)
        layout.addWidget(PlaceholderList('当前配置摘要卡片', ['多载波', '64QAM', 'LDPC', '2×2 MIMO', '启用断点续跑', '输出中间图表']), 1, 0)
        layout.addWidget(TextSummaryCard('当前链路说明', ['当前配置适合用于 1Tbps 目标验证前的预研。', '建议重点观察相位噪声、CFO 与导频密度之间的折衷。']), 1, 1)
        return panel

    def _rate_tab(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(ChartPlaceholder('理论峰值速率', '根据带宽、调制阶数、编码率和并行度估算。'), 0, 0)
        layout.addWidget(ChartPlaceholder('净有效速率', '剔除导频、前导和冗余开销后的净速率。', mode='bar'), 0, 1)
        return panel

    def _budget_tab(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(ChartPlaceholder('链路预算概览', '后续替换为功率、损耗、SNR 级联图。'), 0, 0)
        layout.addWidget(ChartPlaceholder('复杂度分解', '展示编解码、同步、均衡与估计模块开销。', mode='bar'), 0, 1)
        return panel

    def _dependency_tab(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(TextSummaryCard('主要依赖关系', [
            'MIMO 模式会联动激活 MIMO 信道矩阵参数与均衡参数。',
            '多载波模式会联动 CP、导频和子载波映射参数。',
            '1Tbps 模式会联动目标速率、并行度和带宽约束。',
        ]), 0, 0)
        layout.addWidget(ChartPlaceholder('参数依赖图', '可替换为关系图或 DAG 视图。', mode='heatmap'), 0, 1)
        return panel

    def _validation_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        table = QTableWidget(4, 4)
        table.setHorizontalHeaderLabels(['级别', '参数', '说明', '建议'])
        table.verticalHeader().setVisible(False)
        rows = [
            ('警告', '目标速率', '当前净速率与 1Tbps 目标仍有差距', '增大并行度或提高编码率'),
            ('提示', '导频配置', '高频偏场景建议提高导频密度', '切换至密集导频'),
            ('警告', '相位噪声', '高阶调制对相位噪声更敏感', '补充相位跟踪模块'),
            ('通过', '断点续跑', '当前任务已开启快照能力', '可继续保持'),
        ]
        for r, values in enumerate(rows):
            for c, value in enumerate(values):
                table.setItem(r, c, QTableWidgetItem(value))
        layout.addWidget(table)
        return panel

    def get_basic_parameters(self) -> dict[str, object]:
        return {
            '工程名称': self.project_name.text(),
            '标准模式': self.standard_mode.currentText(),
            '链路名称': self.link_name.text(),
            '体制类型': self.system_type.currentText(),
            '带宽': self.bandwidth.value(),
            '载频': self.fc.value(),
            '定点/浮点': self.arithmetic.currentText(),
            '比特源配置': self.bit_source.currentText(),
            '采样率': self.sample_rate.value(),
            '时长': self.duration.value(),
            '数据帧长度': self.subframe_length.value(),
            '单帧数据子帧数量': self.subframe_num.value(),
            'GI 长度': self.gi_length.value(),
            '调制方式': self.modulation.currentText(),
            '编码方式': self.coding.currentText(),
            '波形成形': self.pulse_shaping.currentText(),
            '滚降因子': self.rolloff_factor.value(),
            '滤波器跨度': self.filter_span.value(),
            '扰码配置': self.scrambling.currentText(),
        }

    def get_run_parameters(self) -> dict[str, object]:
        return {
            '运行次数': self.run_times.value(),
            '随机种子策略': self.seed_strategy.currentText(),
            '保存中间结果': self.save_intermediate.currentText(),
            '启用断点续跑': self.enable_resume.currentText(),
            '输出完整日志': self.full_log.currentText(),
            '导出中间图表': self.export_graph.currentText(),
            # '高级能力勾选': [cb.text() for cb in self.advanced_checkbox_group.findChildren(QCheckBox) if cb.isChecked()],
        }

    def get_all_parameters(self) -> dict[str, object]:
        result = {}
        result.update(self.get_basic_parameters())
        result.update(self.get_run_parameters())
        result.update({
            '前导码配置': self.preamble.currentText(),
            '导频配置': self.pilot.currentText(),
            'mimo 模式': self.mimo_mode.currentText(),
            '子载波映射': self.subcarrier_mapping.currentText(),
            '发送功率': self.tx_power.value(),
            'SNR': self.snr.value(),
            '路径损耗': self.path_loss.value(),
            '多径路径数': self.multipath_count.value(),
            '载频偏移': self.cfo.value(),
            '相位噪声': self.phase_noise.value(),
            '大气吸收': self.atmospheric_absorption.currentText(),
            '天线方向图': self.antenna_pattern.currentText(),
            '波束参数': self.beam_params.text(),
            'MIMO 矩阵': self.mimo_matrix.currentText(),
            '粗同步参数': self.coarse_sync.currentText(),
            '细同步参数': self.fine_sync.currentText(),
            '载波同步参数': self.carrier_sync.currentText(),
            '频偏估计参数': self.freq_estimation.currentText(),
            '定时恢复参数': self.timing_recovery.currentText(),
            '信道估计参数': self.channel_estimation.currentText(),
            '均衡参数': self.equalization.currentText(),
            '译码参数': self.decoding.currentText(),
        })
        return result

    def set_all_parameters(self, params: dict[str, object]) -> None:
        # 设置基本参数
        if '工程名称' in params:
            self.project_name.setText(str(params['工程名称']))
        if '标准模式' in params:
            index = self.standard_mode.findText(str(params['标准模式']))
            if index >= 0:
                self.standard_mode.setCurrentIndex(index)
        if '链路名称' in params:
            self.link_name.setText(str(params['链路名称']))
        if '体制类型' in params:
            index = self.system_type.findText(str(params['体制类型']))
            if index >= 0:
                self.system_type.setCurrentIndex(index)
        if '带宽' in params:
            self.bandwidth.setValue(float(params['带宽']))
        if '载频' in params:
            self.fc.setValue(float(params['载频']))
        if '定点/浮点' in params:
            index = self.arithmetic.findText(str(params['定点/浮点']))
            if index >= 0:
                self.arithmetic.setCurrentIndex(index)
        if '比特源配置' in params:
            index = self.bit_source.findText(str(params['比特源配置']))
            if index >= 0:
                self.bit_source.setCurrentIndex(index)
        if '采样率' in params:
            self.sample_rate.setValue(float(params['采样率']))
        if '时长' in params:
            self.duration.setValue(float(params['时长']))
        if '数据帧长度' in params:
            self.subframe_length.setValue(int(params['数据帧长度']))
        if '单帧数据子帧数量' in params:
            self.subframe_num.setValue(int(params['单帧数据子帧数量']))
        if 'GI 长度' in params:
            self.gi_length.setValue(int(params['GI 长度']))
        if '调制方式' in params:
            index = self.modulation.findText(str(params['调制方式']))
            if index >= 0:
                self.modulation.setCurrentIndex(index)
        if '编码方式' in params:
            index = self.coding.findText(str(params['编码方式']))
            if index >= 0:
                self.coding.setCurrentIndex(index)
        if '波形成形' in params:
            index = self.pulse_shaping.findText(str(params['波形成形']))
            if index >= 0:
                self.pulse_shaping.setCurrentIndex(index)
        if '滚降因子' in params:
            self.rolloff_factor.setValue(float(params['滚降因子']))
        if '滤波器跨度' in params:
            self.filter_span.setValue(int(params['滤波器跨度']))
        if '扰码配置' in params:
            index = self.scrambling.findText(str(params['扰码配置']))
            if index >= 0:
                self.scrambling.setCurrentIndex(index)
        # 运行参数
        if '运行次数' in params:
            self.run_times.setValue(int(params['运行次数']))
        if '随机种子策略' in params:
            index = self.seed_strategy.findText(str(params['随机种子策略']))
            if index >= 0:
                self.seed_strategy.setCurrentIndex(index)
        if '保存中间结果' in params:
            index = self.save_intermediate.findText(str(params['保存中间结果']))
            if index >= 0:
                self.save_intermediate.setCurrentIndex(index)
        if '启用断点续跑' in params:
            index = self.enable_resume.findText(str(params['启用断点续跑']))
            if index >= 0:
                self.enable_resume.setCurrentIndex(index)
        if '输出完整日志' in params:
            index = self.full_log.findText(str(params['输出完整日志']))
            if index >= 0:
                self.full_log.setCurrentIndex(index)
        if '导出中间图表' in params:
            index = self.export_graph.findText(str(params['导出中间图表']))
            if index >= 0:
                self.export_graph.setCurrentIndex(index)
        # 其他参数
        if '前导码配置' in params:
            index = self.preamble.findText(str(params['前导码配置']))
            if index >= 0:
                self.preamble.setCurrentIndex(index)
        if '导频配置' in params:
            index = self.pilot.findText(str(params['导频配置']))
            if index >= 0:
                self.pilot.setCurrentIndex(index)
        if 'mimo 模式' in params:
            index = self.mimo_mode.findText(str(params['mimo 模式']))
            if index >= 0:
                self.mimo_mode.setCurrentIndex(index)
        if '子载波映射' in params:
            index = self.subcarrier_mapping.findText(str(params['子载波映射']))
            if index >= 0:
                self.subcarrier_mapping.setCurrentIndex(index)
        if '发送功率' in params:
            self.tx_power.setValue(float(params['发送功率']))
        if 'SNR' in params:
            self.snr.setValue(float(params['SNR']))
        if '路径损耗' in params:
            self.path_loss.setValue(float(params['路径损耗']))
        if '多径路径数' in params:
            self.multipath_count.setValue(int(params['多径路径数']))
        if '载频偏移' in params:
            self.cfo.setValue(float(params['载频偏移']))
        if '相位噪声' in params:
            self.phase_noise.setValue(float(params['相位噪声']))
        if '大气吸收' in params:
            index = self.atmospheric_absorption.findText(str(params['大气吸收']))
            if index >= 0:
                self.atmospheric_absorption.setCurrentIndex(index)
        if '天线方向图' in params:
            index = self.antenna_pattern.findText(str(params['天线方向图']))
            if index >= 0:
                self.antenna_pattern.setCurrentIndex(index)
        if '波束参数' in params:
            self.beam_params.setText(str(params['波束参数']))
        if 'MIMO 矩阵' in params:
            index = self.mimo_matrix.findText(str(params['MIMO 矩阵']))
            if index >= 0:
                self.mimo_matrix.setCurrentIndex(index)
        if '粗同步参数' in params:
            index = self.coarse_sync.findText(str(params['粗同步参数']))
            if index >= 0:
                self.coarse_sync.setCurrentIndex(index)
        if '细同步参数' in params:
            index = self.fine_sync.findText(str(params['细同步参数']))
            if index >= 0:
                self.fine_sync.setCurrentIndex(index)
        if '载波同步参数' in params:
            index = self.carrier_sync.findText(str(params['载波同步参数']))
            if index >= 0:
                self.carrier_sync.setCurrentIndex(index)
        if '频偏估计参数' in params:
            index = self.freq_estimation.findText(str(params['频偏估计参数']))
            if index >= 0:
                self.freq_estimation.setCurrentIndex(index)
        if '定时恢复参数' in params:
            index = self.timing_recovery.findText(str(params['定时恢复参数']))
            if index >= 0:
                self.timing_recovery.setCurrentIndex(index)
        if '信道估计参数' in params:
            index = self.channel_estimation.findText(str(params['信道估计参数']))
            if index >= 0:
                self.channel_estimation.setCurrentIndex(index)
        if '均衡参数' in params:
            index = self.equalization.findText(str(params['均衡参数']))
            if index >= 0:
                self.equalization.setCurrentIndex(index)
        if '译码参数' in params:
            index = self.decoding.findText(str(params['译码参数']))
            if index >= 0:
                self.decoding.setCurrentIndex(index)

