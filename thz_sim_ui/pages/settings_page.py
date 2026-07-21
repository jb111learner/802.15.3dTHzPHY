from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QWidget

from thz_sim_ui.widgets.charts import TextSummaryCard
from thz_sim_ui.widgets.forms import combo, line, make_form_group
from thz_sim_ui.widgets.workbench import WorkbenchPage


class SettingsPage(WorkbenchPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('系统设置', '用于配置界面风格、任务默认参数、日志级别和接口地址等系统级内容。', parent)

        self.theme_style = combo(['科研浅色', '工程深色（预留）'])
        self.default_home = combo(['总览首页', '参数配置', '任务中心'])
        self.display_language = combo(['中文', 'English'])

        self.add_left_widget(make_form_group('界面设置', [
            ('主题样式', self.theme_style),
            ('默认首页', self.default_home),
            ('显示语言', self.display_language),
        ]))

        self.default_log_level = combo(['INFO', 'DEBUG', 'WARNING', 'ERROR'])
        self.default_export_format = combo(['PDF', 'PNG', 'CSV'])
        self.default_concurrency = combo(['2', '4', '8', '16'])

        self.add_left_widget(make_form_group('任务默认设置', [
            ('默认日志级别', self.default_log_level),
            ('默认导出格式', self.default_export_format),
            ('默认并发数', self.default_concurrency),
        ]))

        self.config_service_url = line('http://127.0.0.1:8000/config')
        self.task_service_url = line('http://127.0.0.1:8000/task')
        self.result_service_url = line('http://127.0.0.1:8000/result')

        self.add_left_widget(make_form_group('后端接口预留', [
            ('配置服务地址', self.config_service_url),
            ('任务服务地址', self.task_service_url),
            ('结果服务地址', self.result_service_url),
        ]))
        self.add_left_stretch()

        self.add_right_widget(self._summary())

    def _summary(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(TextSummaryCard('系统设置说明', [
            '该页面主要预留全局设置入口。',
            '后续可增加用户权限、模板管理、日志查看和插件管理。',
            '也可把后端 API、文件存储路径、报告模板目录统一放在这里。',
        ]), 0, 0)
        layout.addWidget(TextSummaryCard('接口接入建议', [
            '建议将系统配置读写统一封装到 BackendService。',
            '配置变更后通过信号通知各页面刷新。',
            '后续可新增持久化配置文件或数据库。',
        ]), 0, 1)
        return panel

    def get_all_parameters(self) -> dict[str, object]:
        return {
            '主题样式': self.theme_style.currentText(),
            '默认首页': self.default_home.currentText(),
            '显示语言': self.display_language.currentText(),
            '默认日志级别': self.default_log_level.currentText(),
            '默认导出格式': self.default_export_format.currentText(),
            '默认并发数': self.default_concurrency.currentText(),
            '配置服务地址': self.config_service_url.text(),
            '任务服务地址': self.task_service_url.text(),
            '结果服务地址': self.result_service_url.text(),
        }
