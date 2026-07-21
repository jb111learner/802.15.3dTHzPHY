from __future__ import annotations

from pathlib import Path
import json

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QGridLayout, QTabWidget, QWidget, QLabel, QVBoxLayout

from thz_sim_ui.data.mock_data import RECENT_TASKS
from thz_sim_ui.widgets.charts import ChartPlaceholder, TextSummaryCard
from thz_sim_ui.widgets.common import CardWidget, PlaceholderList
from thz_sim_ui.widgets.forms import combo, line, make_form_group
from thz_sim_ui.widgets.workbench import WorkbenchPage


class ProjectImageCard(CardWidget):
    def __init__(self, title: str, image_name: str, parent: QWidget | None = None) -> None:
        super().__init__(title=title, parent=parent)
        self.image_name = image_name
        self.image_label = QLabel('等待加载图表...')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setWordWrap(True)
        self.image_label.setMinimumHeight(260)
        self.image_label.setStyleSheet('background: #FAFBFE; border: 1px solid #E6ECF6;')
        self.image_label.setScaledContents(False)
        self.layout.addWidget(self.image_label, 1)
        self._original_pixmap: QPixmap | None = None
        self._pending_path: Path | None = None
        self._reload_attempts = 0

    def set_image_path(self, path: Path | None) -> None:
        if path is None:
            self._pending_path = None
            self._reload_attempts = 0
            self._original_pixmap = None
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText(f'未找到：{self.image_name}')
            return

        if not hasattr(self, '_last_set_path') or self._last_set_path != str(path):
            self._reload_attempts = 0
        self._last_set_path = str(path)
        self._pending_path = path

        if not path.exists():
            self._original_pixmap = None
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText(f'未找到：{self.image_name}')
            return

        try:
            pixmap = QPixmap(str(path))
            if pixmap.isNull():
                self._original_pixmap = None
                self.image_label.setPixmap(QPixmap())
                self.image_label.setText(f'图表生成中：{self.image_name}')
                return

            self._original_pixmap = pixmap
            self.image_label.setText('')
            self._update_pixmap()
        except Exception:
            self._original_pixmap = None
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText(f'加载失败：{self.image_name}')

    def _retry_load_image(self) -> None:
        if self._pending_path is not None:
            self.set_image_path(self._pending_path)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self._original_pixmap is None:
            return
        target_size = self.image_label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        scaled = self._original_pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)


class ResultAnalysisPage(WorkbenchPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('结果分析', '具备结果汇总、过程查看、方案对比和报告输出四类能力。', parent)

        self.task = combo([])
        self._known_task_names: list[str] = []
        
        self.scheme_combo = combo(['方案1'])
        self.snr_point_combo = combo(['SNR_0 dB'])
        # self.scan_param = line('SNR=18 dB')
        self.export_format = combo(['PDF', 'Word', 'PNG', 'CSV'])

        self.mode_label = QLabel('任务模式：单方案')
        self.mode_label.setStyleSheet('font-weight: bold; color: #333;')

        self._batch_config_widgets = [
            self.scheme_combo,
            self.snr_point_combo
        ]

        self._refresh_task_options()

        self.task.currentTextChanged.connect(self.on_task_changed)
        self.scheme_combo.currentTextChanged.connect(self.load_task_results)
        self.snr_point_combo.currentTextChanged.connect(self.load_task_results)

        self.add_left_widget(make_form_group('结果筛选条件', [
            ('任务', self.task),
            ('方案', self.scheme_combo),
            ('仿真点', self.snr_point_combo),
            # ('扫描参数点', self.scan_param),
            ('导出格式', self.export_format),
        ]))
        self.add_left_widget(self.mode_label)
        self.add_left_widget(PlaceholderList('分析说明', ['仅显示已完成仿真任务', '任务选择后结果列表自动刷新', '发射端结果直接读取工程目录中的图表文件']))
        self.add_left_stretch()

        self.tx_waveform_chart = ProjectImageCard('基带波形图', 'transmitter_time.png')
        self.tx_spectrum_chart = ProjectImageCard('发射频谱图', 'transmitter_spectrum.png')
        self.tx_constellation_chart = ProjectImageCard('发射星座图', 'transmitter_constellation.png')
        self.tx_eye_chart = ProjectImageCard('发射眼图', 'transmitter_eye.png')

        self._project_image_cards = [
            self.tx_waveform_chart,
            self.tx_spectrum_chart,
            self.tx_constellation_chart,
            self.tx_eye_chart,
        ]
        self._channel_image_cards: list[ProjectImageCard] = []
        self._receiver_image_cards: list[ProjectImageCard] = []

        tabs = QTabWidget()
        tabs.addTab(self._overview_tab(), '综合总览')
        tabs.addTab(self._tx_results_tab(), '发射端结果')
        tabs.addTab(self._image_grid_tab('信道结果', [
            ('接收信号时域波形', 'channel_time.png'),
            ('接收信号频谱', 'channel_spectrum.png'),
            ('接收星座图', 'channel_constellation.png'),
            ('接收信号功率分布', 'channel_power.png'),
            ('接收信号眼图（实部）', 'channel_eye_real.png'),
        ]), '信道结果')
        tabs.addTab(self._image_grid_tab('接收端结果', [
            ('发射信号时域波形', 'matched_tx_time.png'),
            ('接收信号时域波形', 'matched_rx_time.png'),
            ('接收信号频谱', 'matched_rx_spectrum.png'),
            ('匹配滤波后频谱', 'matched_filtered_spectrum.png'),
            ('发射符号时域波形', 'matched_tx_symbols.png'),
            ('恢复符号时域波形', 'matched_recovered_symbols.png'),
        ]), '接收端结果')
        tabs.addTab(self._pair_tab('性能指标', 'BER 曲线', '吞吐率曲线', 'line', 'bar'), '性能指标')
        tabs.addTab(self._pair_tab('方案对比', '多方案叠图', '关键指标排行', 'line', 'bar'), '方案对比')
        tabs.addTab(TextSummaryCard('报告导出', ['导出面板已预留，可在后续接入 PDF/Word/图片导出逻辑。', '建议在后端统一生成报告，再由前端触发和下载。']), '报告导出')
        self.add_right_widget(tabs)

        self.load_last_completed_task_results()

    def _overview_tab(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(ChartPlaceholder('关键指标总览', '展示 BER、吞吐率、NMSE 和时延的核心摘要。', mode='bar'), 0, 0)
        layout.addWidget(PlaceholderList('推荐观察项', ['BER 曲线', '吞吐率曲线', '信道估计误差图', '均衡前后星座', '参数敏感性分析']), 0, 1)
        return panel

    def _tx_results_tab(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(self.tx_waveform_chart, 0, 0)
        layout.addWidget(self.tx_spectrum_chart, 0, 1)
        layout.addWidget(self.tx_constellation_chart, 1, 0)
        layout.addWidget(self.tx_eye_chart, 1, 1)
        return panel

    def _image_grid_tab(self, section: str, image_items: list[tuple[str, str]]) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        for index, (title, filename) in enumerate(image_items):
            card = ProjectImageCard(title, filename)
            self._project_image_cards.append(card)
            if section == '信道结果':
                self._channel_image_cards.append(card)
            elif section == '接收端结果':
                self._receiver_image_cards.append(card)
            layout.addWidget(card, index // 2, index % 2)

        return panel

    def _pair_tab(self, section: str, left_title: str, right_title: str, left_mode: str, right_mode: str) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(ChartPlaceholder(left_title, f'{section}占位图表。', mode=left_mode), 0, 0)
        layout.addWidget(ChartPlaceholder(right_title, f'{section}占位图表。', mode=right_mode), 0, 1)
        return panel

    def _is_batch_compare_task(self, task) -> bool:
        if hasattr(task, 'mode') and '批量对比' in task.mode:
            return True
        if task.result_path:
            result_path = Path(task.result_path)
            batch_config = result_path / 'batch_config.json'
            if batch_config.exists():
                return True
        return False

    def _get_batch_config(self, project_folder: Path) -> dict:
        try:
            batch_config = project_folder / 'batch_config.json'
            if batch_config.exists():
                with open(batch_config, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _refresh_task_options(self, force: bool = False) -> None:
        completed_tasks = [t for t in RECENT_TASKS if t.status == '已完成']
        tasks_for_project = [t.name for t in completed_tasks]
        if not tasks_for_project:
            tasks_for_project = ['TASK-001']

        if tasks_for_project == self._known_task_names and not force:
            if self.task.count() > 0 and hasattr(self, 'tx_waveform_chart'):
                self.load_last_completed_task_results()
            return
        self._known_task_names = tasks_for_project

        current_task = self.task.currentText() if hasattr(self, 'task') else ''
        self.task.clear()
        self.task.addItems(tasks_for_project)
        if current_task in tasks_for_project:
            self.task.setCurrentText(current_task)
        elif self.task.count() > 0:
            self.task.setCurrentIndex(0)

        if self.task.count() > 0 and hasattr(self, 'tx_waveform_chart'):
            self.load_last_completed_task_results()

    def on_task_changed(self) -> None:
        selected_task_name = self.task.currentText()
        completed = [t for t in RECENT_TASKS if t.status == '已完成']
        task = next((t for t in completed if t.name == selected_task_name), None)

        if task and self._is_batch_compare_task(task):
            self.mode_label.setText('任务模式：多方案对比')
            self._update_batch_options(task)
            self._update_widgets_visibility(True)
        else:
            self.mode_label.setText('任务模式：单方案')
            self._update_widgets_visibility(False)

        self.load_task_results()

    def _update_widgets_visibility(self, is_batch: bool) -> None:
        for widget in self._batch_config_widgets:
            widget.setVisible(is_batch)

    def _update_batch_options(self, task) -> None:
        project_folder = self._get_project_folder()
        if not project_folder:
            return

        batch_config = self._get_batch_config(project_folder)
        configs = batch_config.get('配置方案', [])

        scheme_names = [cfg.get('结果标签', f'方案{i+1}') for i, cfg in enumerate(configs)]
        if not scheme_names:
            scheme_names = ['方案1']

        snr_min = batch_config.get('SNR最小值', 0)
        snr_max = batch_config.get('SNR最大值', 30)
        snr_step = batch_config.get('SNR步长', 2)

        snr_points = []
        current_snr = snr_min
        while current_snr <= snr_max:
            snr_points.append(f'SNR_{current_snr} dB')
            current_snr += snr_step

        if not snr_points:
            snr_points = ['SNR_0 dB']

        current_scheme = self.scheme_combo.currentText()
        current_snr = self.snr_point_combo.currentText()

        self.scheme_combo.blockSignals(True)
        self.snr_point_combo.blockSignals(True)

        self.scheme_combo.clear()
        self.scheme_combo.addItems(scheme_names)

        self.snr_point_combo.clear()
        self.snr_point_combo.addItems(snr_points)

        if current_scheme in scheme_names:
            self.scheme_combo.setCurrentText(current_scheme)
        if current_snr in snr_points:
            self.snr_point_combo.setCurrentText(current_snr)

        self.scheme_combo.blockSignals(False)
        self.snr_point_combo.blockSignals(False)

    def load_last_completed_task_results(self) -> None:
        self.load_task_results()

    def load_task_results(self) -> None:
        project_folder = self._get_project_folder()
        if project_folder is None:
            self._clear_all_images()
            return

        selected_task_name = self.task.currentText()
        completed = [t for t in RECENT_TASKS if t.status == '已完成']
        task = next((t for t in completed if t.name == selected_task_name), None)

        if task and self._is_batch_compare_task(task):
            self._load_batch_project_images(project_folder)
        else:
            self._load_single_project_images(project_folder)

    def _get_project_folder(self) -> Path | None:
        selected_task_name = self.task.currentText()
        completed = [t for t in RECENT_TASKS if t.status == '已完成']
        task = next((t for t in completed if t.name == selected_task_name), None)
        if task is not None and task.result_path:
            folder = Path(task.result_path)
            if folder.exists():
                return folder

        window = self.window()
        if hasattr(window, 'current_project_folder') and window.current_project_folder:
            folder = Path(window.current_project_folder)
            if folder.exists():
                return folder
        if hasattr(window, 'backend') and hasattr(window.backend, 'current_project_folder') and window.backend.current_project_folder:
            folder = Path(window.backend.current_project_folder)
            if folder.exists():
                return folder
        if hasattr(window, 'current_batch_project_folder') and window.current_batch_project_folder:
            folder = Path(window.current_batch_project_folder)
            if folder.exists():
                return folder
        return None

    def _load_single_project_images(self, project_folder: Path | None) -> None:
        transmitter_dir = project_folder / 'transmitter' if project_folder is not None else None
        channel_dir = project_folder / 'channel' if project_folder is not None else None
        receiver_dir = project_folder / 'receiver' if project_folder is not None else None

        self.tx_waveform_chart.set_image_path(transmitter_dir / self.tx_waveform_chart.image_name if transmitter_dir is not None else None)
        self.tx_spectrum_chart.set_image_path(transmitter_dir / self.tx_spectrum_chart.image_name if transmitter_dir is not None else None)
        self.tx_constellation_chart.set_image_path(transmitter_dir / self.tx_constellation_chart.image_name if transmitter_dir is not None else None)
        self.tx_eye_chart.set_image_path(transmitter_dir / self.tx_eye_chart.image_name if transmitter_dir is not None else None)

        for card in self._channel_image_cards:
            card.set_image_path(channel_dir / card.image_name if channel_dir is not None else None)
        for card in self._receiver_image_cards:
            card.set_image_path(receiver_dir / card.image_name if receiver_dir is not None else None)

    def _load_batch_project_images(self, project_folder: Path | None) -> None:
        if project_folder is None:
            self._clear_all_images()
            return

        scheme_name = self.scheme_combo.currentText()
        snr_point = self.snr_point_combo.currentText()

        if not scheme_name or not snr_point:
            self._clear_all_images()
            return

        snr_value = snr_point.replace('SNR_', '').replace('dB', '').replace(' ', '').replace('_', '')
        snr_folder = project_folder / scheme_name / f'SNR_{snr_value}dB'

        if not snr_folder.exists():
            snr_folder_base = project_folder / scheme_name
            if snr_folder_base.exists():
                for folder in snr_folder_base.iterdir():
                    if folder.is_dir() and folder.name.startswith('SNR_'):
                        snr_folder = folder
                        break
            else:
                self._clear_all_images()
                return

        transmitter_dir = snr_folder / 'transmitter'
        channel_dir = snr_folder / 'channel'
        receiver_dir = snr_folder / 'receiver'

        self.tx_waveform_chart.set_image_path(transmitter_dir / self.tx_waveform_chart.image_name if transmitter_dir.exists() else None)
        self.tx_spectrum_chart.set_image_path(transmitter_dir / self.tx_spectrum_chart.image_name if transmitter_dir.exists() else None)
        self.tx_constellation_chart.set_image_path(transmitter_dir / self.tx_constellation_chart.image_name if transmitter_dir.exists() else None)
        self.tx_eye_chart.set_image_path(transmitter_dir / self.tx_eye_chart.image_name if transmitter_dir.exists() else None)

        for card in self._channel_image_cards:
            card.set_image_path(channel_dir / card.image_name if channel_dir.exists() else None)
        for card in self._receiver_image_cards:
            card.set_image_path(receiver_dir / card.image_name if receiver_dir.exists() else None)

    def _clear_all_images(self) -> None:
        for card in self._project_image_cards:
            card.set_image_path(None)

    def refresh_from_tasks(self, tasks: list) -> None:
        self._refresh_task_options(force=True)

    def get_all_parameters(self) -> dict[str, object]:
        return {
            '任务': self.task.currentText(),
            '方案': self.scheme_combo.currentText(),
            '仿真点': self.snr_point_combo.currentText(),
            '扫描参数点': self.scan_param.text(),
            '导出格式': self.export_format.currentText(),
        }