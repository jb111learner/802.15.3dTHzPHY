from __future__ import annotations

from typing import Iterable, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


def combo(items: Sequence[str]) -> QComboBox:
    widget = QComboBox()
    widget.addItems(list(items))
    return widget


def spin(minimum: int = 0, maximum: int = 99999, value: int = 0, suffix: str = '') -> QSpinBox:
    widget = QSpinBox()
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    if suffix:
        widget.setSuffix(f' {suffix}')
    return widget


def dspin(minimum: float = 0, maximum: float = 99999, value: float = 0, decimals: int = 2, suffix: str = '') -> QDoubleSpinBox:
    widget = QDoubleSpinBox()
    widget.setRange(minimum, maximum)
    widget.setDecimals(decimals)
    widget.setValue(value)
    if suffix:
        widget.setSuffix(f' {suffix}')
    return widget


def line(text: str = '', placeholder: str = '') -> QLineEdit:
    widget = QLineEdit(text)
    if placeholder:
        widget.setPlaceholderText(placeholder)
    return widget


def make_form_group(title: str, rows: Iterable[tuple[str, QWidget]]) -> QGroupBox:
    box = QGroupBox(title)
    layout = QFormLayout(box)
    layout.setContentsMargins(14, 18, 14, 14)
    layout.setSpacing(10)
    for label, editor in rows:
        row_label = QLabel(label)
        row_label.setToolTip(f'{label}：接口已预留，后续可绑定真实参数模型。')
        layout.addRow(row_label, editor)
    return box


def make_checkbox_grid(title: str, items: Sequence[str], columns: int = 2) -> QGroupBox:
    box = QGroupBox(title)
    layout = QGridLayout(box)
    layout.setContentsMargins(14, 18, 14, 14)
    layout.setSpacing(10)
    for idx, item in enumerate(items):
        cb = QCheckBox(item)
        cb.setChecked(idx < 2)
        layout.addWidget(cb, idx // columns, idx % columns)
    return box


def make_radio_group(title: str, items: Sequence[str], checked: int = 0) -> QGroupBox:
    box = QGroupBox(title)
    layout = QVBoxLayout(box)
    layout.setContentsMargins(14, 18, 14, 14)
    layout.setSpacing(8)
    for idx, item in enumerate(items):
        rb = QRadioButton(item)
        if idx == checked:
            rb.setChecked(True)
        layout.addWidget(rb)
    return box


def make_toggle_list(title: str, items: Sequence[str]) -> QGroupBox:
    box = QGroupBox(title)
    layout = QVBoxLayout(box)
    layout.setContentsMargins(14, 18, 14, 14)
    layout.setSpacing(10)
    for idx, item in enumerate(items):
        row = QHBoxLayout()
        name = QLabel(item)
        name.setMinimumWidth(120)
        enable = QCheckBox('启用')
        enable.setChecked(idx < 4)
        mark = QCheckBox('批量对比')
        edit = QPushButton('编辑')
        row.addWidget(name)
        row.addWidget(enable)
        row.addWidget(mark)
        row.addStretch(1)
        row.addWidget(edit)
        layout.addLayout(row)
    return box


def text_area(title: str, default_text: str) -> QGroupBox:
    box = QGroupBox(title)
    layout = QVBoxLayout(box)
    layout.setContentsMargins(14, 18, 14, 14)
    editor = QTextEdit()
    editor.setPlainText(default_text)
    layout.addWidget(editor)
    return box
