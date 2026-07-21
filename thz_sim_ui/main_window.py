from __future__ import annotations

from functools import partial
from pathlib import Path
import json
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from thz_sim_ui.constants import APP_NAME, NAV_ITEMS, WINDOW_MIN_SIZE
from thz_sim_ui.pages import (
    BatchComparePage,
    ChannelIntegrationPage,
    HomePage,
    LinkDesignPage,
    ParameterConfigPage,
    ResultAnalysisPage,
    ResumeRecoveryPage,
    SettingsPage,
    StandardModePage,
    TaskCenterPage,
    TbpsModePage,
)
from thz_sim_ui.services.backend import BackendService


class MainWindow(QMainWindow):
    def __init__(self, backend: BackendService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.backend = backend
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(*WINDOW_MIN_SIZE)

        self._nav_buttons: dict[str, QPushButton] = {}
        self._page_indices: dict[str, int] = {}
        self._pending_tasks: list | None = None
        self._task_update_timer = QTimer(self)
        self._task_update_timer.setInterval(100)
        self._task_update_timer.timeout.connect(self._flush_pending_tasks)
        
        self._default_single_project_root = Path(__file__).resolve().parent.parent / 'projects' / 'single'
        self._default_batch_project_root = Path(__file__).resolve().parent.parent / 'projects' / 'batch'
        self._default_single_project_root.mkdir(parents=True, exist_ok=True)
        self._default_batch_project_root.mkdir(parents=True, exist_ok=True)
        
        self.current_project_folder = None
        self.current_batch_project_folder = None

        self._build_ui()
        self._connect_signals()
        self._load_backend_state(backend.snapshot())
        self.switch_page('home')

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 10)
        root.setSpacing(12)

        root.addWidget(self._build_top_bar())
        root.addWidget(self._build_nav_bar())

        self.page_stack = QStackedWidget()
        root.addWidget(self.page_stack, 1)
        self.setCentralWidget(central)

        self._register_pages()
        self._build_status_bar()

    def _is_batch_compare_mode(self) -> bool:
        current_page = self.page_stack.currentWidget()
        return isinstance(current_page, BatchComparePage)

    def _save_project(self) -> None:
        if self._is_batch_compare_mode():
            self._save_batch_compare_project()
        else:
            self._save_single_project()

    def _save_single_project(self) -> None:
        param_page = self.pages['parameter_config']
        if hasattr(param_page, 'get_all_parameters'):
            params = param_page.get_all_parameters()
            project_name = params.get('工程名称', 'Unnamed_Project')
            if not project_name.strip():
                project_name = 'Unnamed_Project'
            if not self.current_project_folder or not Path(self.current_project_folder).exists():
                project_folder = self._default_single_project_root / project_name
            else:
                project_folder = Path(self.current_project_folder)
            project_folder.mkdir(parents=True, exist_ok=True)
            self.current_project_folder = str(project_folder)
            config_file = project_folder / "config.json"
            with open(config_file, "w", encoding='utf-8') as f:
                json.dump(params, f, ensure_ascii=False, indent=4)
            import time
            self.backend._state.project_name = project_name
            self.backend._state.last_saved_at = time.strftime("%Y-%m-%d %H:%M:%S")
            self.project_label.setText(f'工程：{project_name}')
            self.backend.message_emitted.emit(f'工程配置已保存到 {project_folder}')
            self.backend.state_changed.emit(self.backend.snapshot())

    def _save_batch_compare_project(self) -> None:
        batch_page = self.pages['batch_compare']
        if hasattr(batch_page, 'get_all_parameters'):
            params = batch_page.get_all_parameters()
            project_name = params.get('工程名称', f'BatchCompare_{len(params.get("配置方案", []))}Schemes')
            if not project_name.strip():
                project_name = f'BatchCompare_{len(params.get("配置方案", []))}Schemes'
            if not self.current_batch_project_folder or not Path(self.current_batch_project_folder).exists():
                project_folder = self._default_batch_project_root / project_name
            else:
                project_folder = Path(self.current_batch_project_folder)
            project_folder.mkdir(parents=True, exist_ok=True)
            self.current_batch_project_folder = str(project_folder)
            config_file = project_folder / "batch_config.json"
            with open(config_file, "w", encoding='utf-8') as f:
                json.dump(params, f, ensure_ascii=False, indent=4)
            import time
            self.backend._state.project_name = project_name
            self.backend._state.last_saved_at = time.strftime("%Y-%m-%d %H:%M:%S")
            self.project_label.setText(f'工程：{project_name}')
            self.backend.message_emitted.emit(f'批量对比配置已保存到 {project_folder}')
            self.backend.state_changed.emit(self.backend.snapshot())

    def _ensure_project_folder(self, project_name: str) -> bool:
        project_folder = self._default_single_project_root / project_name
        project_folder.mkdir(parents=True, exist_ok=True)
        self.current_project_folder = str(project_folder)
        self.backend._state.project_name = project_name
        self.project_label.setText(f'工程：{project_name}')
        return True

    def _import_config(self) -> None:
        if self._is_batch_compare_mode():
            self._import_batch_compare_config()
        else:
            self._import_single_config()

    def _import_single_config(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择单方案工程文件夹", str(self._default_single_project_root))
        if folder:
            project_folder = Path(folder)
            config_file = project_folder / "config.json"
            batch_config_file = project_folder / "batch_config.json"
            
            if batch_config_file.exists():
                QMessageBox.warning(self, "错误", "这是一个批量对比工程，请在批量对比页面导入")
                return
            
            if config_file.exists():
                with open(config_file, "r", encoding='utf-8') as f:
                    params = json.load(f)
                param_page = self.pages['parameter_config']
                if hasattr(param_page, 'set_all_parameters'):
                    param_page.set_all_parameters(params)
                project_name = params.get('工程名称', '')
                if not project_name.strip():
                    project_name = project_folder.name
                self.backend._state.project_name = project_name
                self.project_label.setText(f'工程：{project_name}')
                self.current_project_folder = str(project_folder)
                self.backend.message_emitted.emit(f'配置已从 {project_folder} 导入')
            else:
                QMessageBox.warning(self, "错误", "未找到配置文件 config.json")

    def _import_batch_compare_config(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择批量对比工程文件夹", str(self._default_batch_project_root))
        if folder:
            project_folder = Path(folder)
            config_file = project_folder / "batch_config.json"
            single_config_file = project_folder / "config.json"
            
            if single_config_file.exists() and not config_file.exists():
                QMessageBox.warning(self, "错误", "这是一个单方案工程，请在参数配置页面导入")
                return
            
            if config_file.exists():
                with open(config_file, "r", encoding='utf-8') as f:
                    params = json.load(f)
                batch_page = self.pages['batch_compare']
                if hasattr(batch_page, 'set_all_parameters'):
                    batch_page.set_all_parameters(params)
                self.current_batch_project_folder = str(project_folder)
                project_name = params.get('工程名称', '')
                if not project_name.strip():
                    project_name = project_folder.name
                self.backend._state.project_name = project_name
                self.project_label.setText(f'工程：{project_name}')
                self.backend.message_emitted.emit(f'批量对比配置已从 {project_folder} 导入')
            else:
                QMessageBox.warning(self, "错误", "未找到配置文件 batch_config.json")

    def _run_simulation(self) -> None:
        if self._is_batch_compare_mode():
            self._run_batch_compare_simulation()
        else:
            self._run_single_simulation()

    def _run_single_simulation(self) -> None:
        param_page = self.pages['parameter_config']
        if not hasattr(param_page, 'get_all_parameters'):
            return
        params = param_page.get_all_parameters()
        project_name = params.get('工程名称', 'Unnamed_Project')
        if not project_name.strip():
            project_name = 'Unnamed_Project'
        if not self.current_project_folder or not Path(self.current_project_folder).exists():
            if not self._ensure_project_folder(project_name):
                return
        self._save_single_project()
        self.backend._state.project_name = project_name
        self.project_label.setText(f'工程：{project_name}')
        self.backend.run_simulation(params, self.current_project_folder)

    def _run_batch_compare_simulation(self) -> None:
        batch_page = self.pages['batch_compare']
        if not hasattr(batch_page, 'get_all_parameters'):
            return
        
        if hasattr(batch_page, 'prepare_for_simulation'):
            batch_page.prepare_for_simulation()
        
        params = batch_page.get_all_parameters()
        project_name = params.get('工程名称', f'BatchCompare_{len(params.get("配置方案", []))}Schemes')
        if not project_name.strip():
            project_name = f'BatchCompare_{len(params.get("配置方案", []))}Schemes'
        
        if not self.current_batch_project_folder or not Path(self.current_batch_project_folder).exists():
            self._save_batch_compare_project()
        else:
            self._save_batch_compare_project()
        
        self.backend._state.project_name = project_name
        self.project_label.setText(f'工程：{project_name}')
        self.backend.run_batch_compare_simulation(params, self.current_batch_project_folder)

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName('TopBar')
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(12)

        left_box = QHBoxLayout()
        left_box.setSpacing(12)

        logo_label = QLabel()
        logo_label.setFixedSize(60, 60)
        logo_label.setAlignment(Qt.AlignCenter)

        logo_path = Path(__file__).resolve().parent / "assets" / "xjtu_logo.png"
        pixmap = QPixmap(str(logo_path))
        if not pixmap.isNull():
            logo_label.setPixmap(
                pixmap.scaled(56, 56, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

        title_box = QVBoxLayout()
        title_box.setSpacing(2)

        title = QLabel(APP_NAME)
        title.setObjectName('Title')

        subtitle = QLabel('西安交通大学无线通信研究所杜清河老师课题组研制')
        subtitle.setObjectName('Subtitle')

        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        left_box.addWidget(logo_label)
        left_box.addLayout(title_box)

        layout.addLayout(left_box)

        layout.addStretch(1)

        self.project_label = QLabel('工程：-')
        self.project_label.setStyleSheet('font-weight: 700;')
        self.user_label = QLabel('用户：李 喆')
        self.version_label = QLabel('版本：-')
        layout.addWidget(self.project_label)
        layout.addWidget(self.version_label)
        layout.addWidget(self.user_label)

        actions = [
            ('保存', self._save_project, ''),
            ('另存为模板', self.backend.save_as_template, ''),
            ('导入配置', self._import_config, ''),
            ('导出报告', self.backend.export_report, ''),
            ('运行', self._run_simulation, 'primary'),
            ('暂停', self.backend.pause_simulation, 'warning'),
            ('停止', self.backend.stop_simulation, 'danger'),
            ('系统通知', self._show_notices, ''),
        ]
        for text, slot, role in actions:
            button = QPushButton(text)
            if role:
                button.setProperty('role', role)
                button.style().unpolish(button)
                button.style().polish(button)
            button.clicked.connect(slot)
            layout.addWidget(button)
        return bar

    def _build_nav_bar(self) -> QWidget:
        nav = QWidget()
        nav.setObjectName('NavBar')
        layout = QHBoxLayout(nav)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        for page_key, title in NAV_ITEMS.items():
            button = QPushButton(title)
            button.setCheckable(True)
            button.setProperty('nav', 'true')
            button.clicked.connect(partial(self.switch_page, page_key))
            layout.addWidget(button)
            self._nav_buttons[page_key] = button
        layout.addStretch(1)
        return nav

    def _register_pages(self) -> None:
        pages = {
            'home': HomePage(),
            'link_design': LinkDesignPage(),
            'parameter_config': ParameterConfigPage(),
            'channel_integration': ChannelIntegrationPage(),
            'batch_compare': BatchComparePage(),
            'task_center': TaskCenterPage(),
            'result_analysis': ResultAnalysisPage(),
            'tbps_mode': TbpsModePage(),
            'std_mode': StandardModePage(),
            'settings': SettingsPage(),
            'resume_recovery': ResumeRecoveryPage(),
        }
        self.pages = pages

        task_page = pages['task_center']
        if isinstance(task_page, TaskCenterPage):
            task_page.open_recovery_requested.connect(lambda: self.switch_page('resume_recovery'))

        for key, widget in pages.items():
            self._page_indices[key] = self.page_stack.addWidget(widget)

    def _build_status_bar(self) -> None:
        status = QStatusBar()
        self.setStatusBar(status)
        self.task_id_label = QLabel('任务编号：-')
        self.phase_label = QLabel('阶段：-')
        self.progress_label = QLabel('进度：0%')
        self.saved_label = QLabel('最近保存：-')
        self.recovery_label = QLabel('恢复点：0')
        for widget in [self.task_id_label, self.phase_label, self.progress_label, self.saved_label, self.recovery_label]:
            status.addPermanentWidget(widget)
            status.addPermanentWidget(QLabel('   '))
        status.showMessage('界面原型已就绪，后端接口待接入。')

    def _connect_signals(self) -> None:
        self.backend.message_emitted.connect(self.statusBar().showMessage)
        self.backend.state_changed.connect(self._load_backend_state)
        self.backend.tasks_updated.connect(self._on_tasks_update_received)
        self.backend.scheme_progress_updated.connect(self._on_scheme_progress_updated)
        self.backend.compare_chart_saved.connect(self._on_compare_chart_saved)

    def _on_compare_chart_saved(self, image_path: str):
        """处理 BER 对比图片保存事件"""
        if 'batch_compare' in self.pages:
            batch_page = self.pages['batch_compare']
            if hasattr(batch_page, 'update_ber_chart'):
                batch_page.update_ber_chart(image_path)

    def _on_scheme_progress_updated(self, scheme_index: int, progress: int):
        if 'batch_compare' in self.pages:
            batch_page = self.pages['batch_compare']
            if hasattr(batch_page, 'update_simulation_progress'):
                batch_page.update_simulation_progress(scheme_index, progress)

    def _load_backend_state(self, state: dict) -> None:
        self.project_label.setText(f"工程：{state.get('project_name', '-')}")
        self.version_label.setText(f"版本：{state.get('project_version', '-')}")
        self.task_id_label.setText(f"任务编号：{state.get('current_task_id', '-')}")
        self.phase_label.setText(f"阶段：{state.get('phase', '-')}")
        self.progress_label.setText(f"进度：{state.get('progress', 0)}%")
        self.saved_label.setText(f"最近保存：{state.get('last_saved_at', '-')}")
        self.recovery_label.setText(f"恢复点：{state.get('recovery_points', 0)}")

    def _on_tasks_update_received(self, tasks: list) -> None:
        self._pending_tasks = tasks
        if not self._task_update_timer.isActive():
            self._task_update_timer.start()

    def _flush_pending_tasks(self) -> None:
        if self._pending_tasks is None:
            self._task_update_timer.stop()
            return
        tasks = self._pending_tasks
        self._pending_tasks = None
        if 'home' in self.pages:
            self.pages['home'].update_tasks(tasks)
        if 'task_center' in self.pages:
            self.pages['task_center'].update_tasks(tasks)
        if 'result_analysis' in self.pages and hasattr(self.pages['result_analysis'], 'refresh_from_tasks'):
            self.pages['result_analysis'].refresh_from_tasks(tasks)
        self._task_update_timer.stop()

    def switch_page(self, page_key: str) -> None:
        index = self._page_indices.get(page_key)
        if index is None:
            return
        self.page_stack.setCurrentIndex(index)
        for key, button in self._nav_buttons.items():
            button.setChecked(key == page_key)
        title = NAV_ITEMS.get(page_key, '断点续跑')
        self.statusBar().showMessage(f'已切换到：{title}')

    def _show_notices(self) -> None:
        notices = self.backend.snapshot().get('notices', ['暂无通知'])
        QMessageBox.information(self, '系统通知', '\n'.join(f'• {item}' for item in notices))

    def get_all_page_parameters(self) -> dict[str, dict[str, object]]:
        all_params = {}
        for i in range(self.page_stack.count()):
            page = self.page_stack.widget(i)
            if isinstance(page, (ParameterConfigPage, BatchComparePage, LinkDesignPage, ChannelIntegrationPage, ResultAnalysisPage, TbpsModePage, StandardModePage, SettingsPage)):
                page_name = type(page).__name__.replace('Page', '').lower()
                all_params[page_name] = page.get_all_parameters()
        return all_params