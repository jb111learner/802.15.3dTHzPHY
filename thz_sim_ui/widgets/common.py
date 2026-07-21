from __future__ import annotations

from typing import Iterable, Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from thz_sim_ui.constants import STATUS_COLORS


class CardWidget(QFrame):
    def __init__(self, title: str = '', hint: str = '', parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName('Card')
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 16, 16, 16)
        self.layout.setSpacing(10)
        if title:
            title_label = QLabel(title)
            title_label.setObjectName('CardTitle')
            self.layout.addWidget(title_label)
        if hint:
            hint_label = QLabel(hint)
            hint_label.setObjectName('CardHint')
            hint_label.setWordWrap(True)
            self.layout.addWidget(hint_label)


class PanelWidget(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName('Panel')
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(12, 12, 12, 12)
        self.layout.setSpacing(12)


class MetricCard(CardWidget):
    def __init__(self, label: str, value: str, note: str = '', parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        value_label = QLabel(value)
        value_label.setObjectName('MetricValue')
        name_label = QLabel(label)
        name_label.setObjectName('MetricLabel')
        self.layout.addWidget(value_label)
        self.layout.addWidget(name_label)
        if note:
            note_label = QLabel(note)
            note_label.setObjectName('CardHint')
            note_label.setWordWrap(True)
            self.layout.addWidget(note_label)


class StatusBadge(QLabel):
    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        color = STATUS_COLORS.get(text, '#7C869B')
        self.setStyleSheet(
            f'padding: 4px 10px; border-radius: 12px; background: {color}22; color: {color}; font-weight: 700;'
        )


class SectionHeader(QWidget):
    def __init__(self, title: str, subtitle: str = '', action_text: str = '', parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        left = QVBoxLayout()
        left.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName('CardTitle')
        left.addWidget(title_label)
        if subtitle:
            hint = QLabel(subtitle)
            hint.setObjectName('CardHint')
            hint.setWordWrap(True)
            left.addWidget(hint)
        layout.addLayout(left)
        layout.addStretch(1)

        self.action_button: QPushButton | None = None
        if action_text:
            self.action_button = QPushButton(action_text)
            layout.addWidget(self.action_button)


class ChipList(QWidget):
    def __init__(self, items: Sequence[str], columns: int = 5, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        for idx, item in enumerate(items):
            chip = QLabel(item)
            chip.setStyleSheet(
                'padding: 6px 12px; border-radius: 12px; background: #EEF3FF; color: #2F6BFF; font-weight: 600;'
            )
            layout.addWidget(chip, idx // columns, idx % columns)


class ActionGrid(QWidget):
    def __init__(self, items: Iterable[tuple[str, str]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        for idx, (title, desc) in enumerate(items):
            card = CardWidget(title, desc)
            button = QPushButton('进入')
            button.setProperty('role', 'primary')
            card.layout.addStretch(1)
            card.layout.addWidget(button, alignment=Qt.AlignLeft)
            layout.addWidget(card, idx // 3, idx % 3)


class PlaceholderList(CardWidget):
    def __init__(self, title: str, items: Sequence[str], parent: QWidget | None = None) -> None:
        super().__init__(title=title, parent=parent)
        for item in items:
            label = QLabel(f'• {item}')
            label.setWordWrap(True)
            self.layout.addWidget(label)


class SimpleFlowWidget(QWidget):
    def __init__(self, steps: Sequence[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.steps = list(steps)
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(12, 18, -12, -18)
        count = max(len(self.steps), 1)
        step_w = max(92, rect.width() // count)
        box_w = min(122, step_w - 12)
        y = rect.center().y() - 20
        for index, step in enumerate(self.steps):
            x = rect.x() + index * step_w + (step_w - box_w) // 2
            painter.setPen(QPen(QColor('#CAD4E5'), 1.4))
            painter.setBrush(QColor('#FFFFFF'))
            painter.drawRoundedRect(x, y, box_w, 44, 12, 12)
            painter.setPen(QColor('#1B2430'))
            painter.drawText(x + 8, y, box_w - 16, 44, Qt.AlignCenter, step)
            if index < len(self.steps) - 1:
                line_y = y + 22
                start_x = x + box_w + 4
                next_x = rect.x() + (index + 1) * step_w + (step_w - box_w) // 2
                end_x = next_x - 4
                painter.setPen(QPen(QColor('#2F6BFF'), 2))
                painter.drawLine(start_x, line_y, end_x, line_y)
                arrow = QPainterPath()
                arrow.moveTo(end_x - 6, line_y - 5)
                arrow.lineTo(end_x, line_y)
                arrow.lineTo(end_x - 6, line_y + 5)
                painter.drawPath(arrow)
        painter.end()


class TwoColumnMetricGrid(QWidget):
    def __init__(self, metrics: Sequence[tuple[str, str]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        for idx, (label, value) in enumerate(metrics):
            layout.addWidget(MetricCard(label, value), idx // 2, idx % 2)
