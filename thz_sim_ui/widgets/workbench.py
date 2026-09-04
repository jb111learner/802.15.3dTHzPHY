from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QSplitter, QVBoxLayout, QWidget

from thz_sim_ui.widgets.common import PanelWidget, SectionHeader


class WorkbenchPage(QWidget):
    def __init__(self, title: str, subtitle: str = '', parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(12)
        self.header = SectionHeader(title, subtitle)
        self.layout.addWidget(self.header)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.layout.addWidget(self.splitter, 1)

        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.right_scroll = QScrollArea()
        self.right_scroll.setWidgetResizable(True)

        self.left_panel = PanelWidget()
        self.right_panel = PanelWidget()

        self.left_scroll.setWidget(self.left_panel)
        self.right_scroll.setWidget(self.right_panel)

        # 左栏由“可滚动内容 + 固定底栏”组成。未使用固定底栏的页面保持
        # 原有外观；需要固定操作按钮的页面可调用 add_left_footer_widget()。
        self.left_column = QWidget()
        self.left_column_layout = QVBoxLayout(self.left_column)
        self.left_column_layout.setContentsMargins(0, 0, 0, 0)
        self.left_column_layout.setSpacing(8)
        self.left_column_layout.addWidget(self.left_scroll, 1)
        self.left_footer = PanelWidget()
        self.left_footer.layout.setContentsMargins(12, 10, 12, 10)
        self.left_footer.hide()
        self.left_column_layout.addWidget(self.left_footer, 0)

        self.splitter.addWidget(self.left_column)
        self.splitter.addWidget(self.right_scroll)
        self.splitter.setSizes([420, 1020])

    def add_left_widget(self, widget: QWidget) -> None:
        self.left_panel.layout.addWidget(widget)

    def add_right_widget(self, widget: QWidget) -> None:
        self.right_panel.layout.addWidget(widget)

    def add_left_footer_widget(self, widget: QWidget) -> None:
        """把操作控件固定在左栏底部，使其不随参数区滚动。"""
        self.left_footer.layout.addWidget(widget)
        self.left_footer.show()

    def add_left_stretch(self) -> None:
        self.left_panel.layout.addStretch(1)

    def add_right_stretch(self) -> None:
        self.right_panel.layout.addStretch(1)
