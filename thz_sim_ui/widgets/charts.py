from __future__ import annotations

from math import sin

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from thz_sim_ui.widgets.common import CardWidget


class ChartPlaceholder(CardWidget):
    def __init__(self, title: str, subtitle: str = '', mode: str = 'line', parent: QWidget | None = None) -> None:
        super().__init__(title='', hint='', parent=parent)
        self.chart_title = title
        self.mode = mode
        self.canvas = _ChartCanvas(mode=mode, title=title)
        self.layout.addWidget(self.canvas)

    def set_data(self, x_data: list[float] | None, y_data: list[float] | None) -> None:
        self.canvas.set_data(x_data, y_data)

    def set_mode(self, mode: str) -> None:
        self.canvas.mode = mode
        self.canvas.update()

    def set_image(self, image_path: str) -> None:
        """设置显示的图片"""
        self.canvas.set_image(image_path)


class _ChartCanvas(QWidget):
    def __init__(self, mode: str = 'line', title: str = '', parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.mode = mode
        self.title = title
        self.x_data: list[float] | None = None
        self.y_data: list[float] | None = None
        self.image_pixmap: QPixmap | None = None  # 存储图片
        self.setMinimumHeight(220)

    def set_data(self, x_data: list[float] | None, y_data: list[float] | None) -> None:
        self.image_pixmap = None  # 清除图片
        if x_data is None or y_data is None or len(x_data) == 0 or len(y_data) == 0:
            self.x_data = None
            self.y_data = None
            self.update()
            return

        # 保证长度一致，若不一致按最小长度缩放
        n = min(len(x_data), len(y_data))
        x_arr = np.asarray(x_data[:n])
        y_arr = np.asarray(y_data[:n])

        # 复数转实部；NaN/inf过滤
        if np.iscomplexobj(x_arr):
            x_arr = np.real(x_arr)
        if np.iscomplexobj(y_arr):
            y_arr = np.real(y_arr)

        x_arr = x_arr.astype(float)
        y_arr = y_arr.astype(float)

        # 降采样到最大点数，避免 UI 卡顿
        max_points = 5000
        if len(x_arr) > max_points:
            step = int(np.ceil(len(x_arr) / max_points))
            x_arr = x_arr[::step]
            y_arr = y_arr[::step]

        self.x_data = x_arr.tolist()
        self.y_data = y_arr.tolist()
        self.update()

    def set_image(self, image_path: str) -> None:
        """加载并显示图片"""
        self.image_pixmap = QPixmap(image_path)
        if not self.image_pixmap.isNull():
            self.x_data = None  # 清除数据，显示图片
            self.y_data = None
            self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        full_rect = self.rect()
        chart_rect = full_rect.adjusted(48, 32, -18, -44)

        painter.fillRect(full_rect, QColor('#F5F7FB'))
        painter.fillRect(chart_rect, QColor('#FAFBFE'))

        # 如果有图片，直接绘制图片
        if self.image_pixmap is not None and not self.image_pixmap.isNull():
            scaled_pixmap = self.image_pixmap.scaled(chart_rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = chart_rect.left() + (chart_rect.width() - scaled_pixmap.width()) // 2
            y = chart_rect.top() + (chart_rect.height() - scaled_pixmap.height()) // 2
            painter.drawPixmap(x, y, scaled_pixmap)

            # 绘制标题
            title_font = painter.font()
            title_font.setPointSize(10)
            title_font.setBold(True)
            painter.setFont(title_font)
            painter.drawText(chart_rect.left(), full_rect.top() + 8, chart_rect.width(), 20, Qt.AlignHCenter | Qt.AlignTop, self.title)
            painter.end()
            return

        grid_pen = QPen(QColor('#E6ECF6'), 1)
        painter.setPen(grid_pen)
        for i in range(6):
            y = chart_rect.top() + i * chart_rect.height() / 5
            painter.drawLine(chart_rect.left(), int(y), chart_rect.right(), int(y))
        for i in range(6):
            x = chart_rect.left() + i * chart_rect.width() / 5
            painter.drawLine(int(x), chart_rect.top(), int(x), chart_rect.bottom())

        axis_pen = QPen(QColor('#A0A8B8'), 1.2)
        painter.setPen(axis_pen)
        painter.drawLine(chart_rect.left(), chart_rect.bottom(), chart_rect.right(), chart_rect.bottom())
        painter.drawLine(chart_rect.left(), chart_rect.top(), chart_rect.left(), chart_rect.bottom())

        def _AxisLabel():
            title = self.title or ''
            if '频谱' in title:
                return '频率 (GHz)', '幅度 (dB)'
            if '星座' in title:
                return '实部', '虚部'
            if '眼图' in title:
                return '时间 / 采样', '幅度'
            if '基带' in title:
                return '样本点', '幅度'
            return 'X', 'Y'

        x_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
        y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
        if self.x_data is not None and self.y_data is not None and len(self.x_data) > 1 and len(self.y_data) > 1:
            def _as_real_list(values):
                clean_values = []
                for v in values:
                    if v is None:
                        continue
                    if isinstance(v, complex):
                        clean_values.append(float(v.real))
                    else:
                        try:
                            clean_values.append(float(v))
                        except Exception:
                            continue
                return clean_values
            x_real = _as_real_list(self.x_data)
            y_real = _as_real_list(self.y_data)
            if len(x_real) > 1 and len(y_real) > 1:
                x_min, x_max = min(x_real), max(x_real)
                y_min, y_max = min(y_real), max(y_real)
                if x_max == x_min:
                    x_max = x_min + 1
                if y_max == y_min:
                    y_max = y_min + 1
                x_ticks = np.linspace(x_min, x_max, 5).tolist()
                y_ticks = np.linspace(y_min, y_max, 5).tolist()

        painter.setPen(axis_pen)
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        for tick in x_ticks:
            ratio = (tick - x_ticks[0]) / (x_ticks[-1] - x_ticks[0]) if x_ticks[-1] != x_ticks[0] else 0.0
            px = chart_rect.left() + chart_rect.width() * ratio
            painter.drawLine(int(px), chart_rect.bottom(), int(px), chart_rect.bottom() + 4)
            painter.drawText(int(px - 22), chart_rect.bottom() + 4, 44, 18, Qt.AlignHCenter | Qt.AlignTop, f'{tick:.2g}')
        for tick in y_ticks:
            ratio = (tick - y_ticks[0]) / (y_ticks[-1] - y_ticks[0]) if y_ticks[-1] != y_ticks[0] else 0.0
            py = chart_rect.bottom() - chart_rect.height() * ratio
            painter.drawLine(chart_rect.left() - 4, int(py), chart_rect.left(), int(py))
            painter.drawText(chart_rect.left() - 50, int(py - 8), 46, 16, Qt.AlignRight | Qt.AlignVCenter, f'{tick:.2g}')

        x_label, y_label = _AxisLabel()
        painter.drawText(chart_rect.left(), chart_rect.bottom() + 18, chart_rect.width(), 18, Qt.AlignHCenter | Qt.AlignTop, x_label)
        painter.save()
        painter.translate(chart_rect.left() - 24, chart_rect.top() + chart_rect.height() / 2)
        painter.rotate(-90)
        painter.drawText(0, 0, chart_rect.height(), 18, Qt.AlignCenter | Qt.AlignVCenter, y_label)
        painter.restore()

        title_font = painter.font()
        title_font.setPointSize(10)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.drawText(chart_rect.left(), full_rect.top() + 8, chart_rect.width(), 20, Qt.AlignHCenter | Qt.AlignTop, self.title)

        draw_placeholder_text = True
        if self.mode == 'bar':
            bar_w = chart_rect.width() / 9
            values = [0.48, 0.72, 0.65, 0.88, 0.57, 0.76]
            for idx, value in enumerate(values):
                x = chart_rect.left() + (idx + 1) * bar_w * 1.15
                h = chart_rect.height() * value
                painter.fillRect(int(x), int(chart_rect.bottom() - h), int(bar_w * 0.7), int(h), QColor('#7FA8FF'))
        elif self.mode == 'scatter':
            x = self.x_data
            y = self.y_data
            if x is not None and y is not None and len(x) > 0 and len(y) > 0:
                def _as_real_list(values):
                    clean_values = []
                    for v in values:
                        if v is None:
                            continue
                        if isinstance(v, complex):
                            clean_values.append(float(v.real))
                        else:
                            try:
                                clean_values.append(float(v))
                            except Exception:
                                continue
                    return clean_values

                x_real = _as_real_list(x)
                y_real = _as_real_list(y)
                if len(x_real) == len(y_real) and len(x_real) > 0:
                    x_min, x_max = min(x_real), max(x_real)
                    y_min, y_max = min(y_real), max(y_real)
                    if x_max == x_min:
                        x_max = x_min + 1
                    if y_max == y_min:
                        y_max = y_min + 1
                    painter.setPen(QPen(QColor('#0EA76B'), 1.8))
                    for xi, yi in zip(x_real, y_real):
                        px = chart_rect.left() + chart_rect.width() * ((xi - x_min) / (x_max - x_min))
                        py = chart_rect.bottom() - chart_rect.height() * ((yi - y_min) / (y_max - y_min))
                        painter.drawEllipse(int(px) - 3, int(py) - 3, 6, 6)
                    draw_placeholder_text = False
                else:
                    painter.setPen(QPen(QColor('#0EA76B'), 1.8))
                    points = [
                        (0.22, 0.26), (0.31, 0.35), (0.44, 0.48), (0.51, 0.45), (0.62, 0.64), (0.74, 0.72),
                        (0.28, 0.70), (0.37, 0.62), (0.57, 0.22), (0.68, 0.35),
                    ]
                    for px, py in points:
                        x = chart_rect.left() + chart_rect.width() * px
                        y = chart_rect.bottom() - chart_rect.height() * py
                        painter.drawEllipse(int(x), int(y), 8, 8)
        elif self.mode == 'heatmap':
            cols, rows = 6, 4
            cell_w = chart_rect.width() / cols
            cell_h = chart_rect.height() / rows
            for r in range(rows):
                for c in range(cols):
                    strength = ((r + 1) * (c + 2)) % 10 / 10
                    color = QColor.fromRgbF(0.18 + strength * 0.35, 0.45, 1.0 - strength * 0.55, 0.92)
                    painter.fillRect(
                        int(chart_rect.left() + c * cell_w),
                        int(chart_rect.top() + r * cell_h),
                        int(cell_w - 2),
                        int(cell_h - 2),
                        color,
                    )
        elif self.mode == 'line' and self.x_data is not None and self.y_data is not None and len(self.x_data) > 1 and len(self.y_data) > 1:
            x = self.x_data
            y = self.y_data
            if len(x) != len(y):
                painter.setPen(QColor('#7C869B'))
                painter.drawText(chart_rect.adjusted(8, 8, -8, -8), Qt.AlignBottom | Qt.AlignRight, '占位图表')
                painter.end()
                return

            def _as_real_list(values):
                clean_values = []
                for v in values:
                    if v is None:
                        continue
                    if isinstance(v, complex):
                        clean_values.append(float(v.real))
                    else:
                        try:
                            clean_values.append(float(v))
                        except Exception:
                            continue
                return clean_values

            x_real = _as_real_list(x)
            y_real = _as_real_list(y)
            if len(x_real) < 2 or len(y_real) < 2:
                painter.setPen(QColor('#7C869B'))
                painter.drawText(chart_rect.adjusted(8, 8, -8, -8), Qt.AlignBottom | Qt.AlignRight, '占位图表')
                painter.end()
                return

            x_min, x_max = min(x_real), max(x_real)
            y_min, y_max = min(y_real), max(y_real)
            if x_max == x_min:
                x_max = x_min + 1
            if y_max == y_min:
                y_max = y_min + 1

            points = []
            for xi, yi in zip(x_real, y_real):
                px = chart_rect.left() + chart_rect.width() * ((xi - x_min) / (x_max - x_min))
                py = chart_rect.bottom() - chart_rect.height() * ((yi - y_min) / (y_max - y_min))
                points.append((px, py))

            painter.setPen(QPen(QColor('#2F6BFF'), 2.2))
            for i in range(len(points) - 1):
                painter.drawLine(int(points[i][0]), int(points[i][1]), int(points[i + 1][0]), int(points[i + 1][1]))
            draw_placeholder_text = False
        else:
            points = []
            for i in range(36):
                ratio = i / 35
                x = chart_rect.left() + chart_rect.width() * ratio
                y = chart_rect.center().y() - chart_rect.height() * 0.28 * sin(ratio * 6.28 * 1.5) - chart_rect.height() * 0.1 * ratio
                points.append((x, y))
            for i in range(len(points) - 1):
                painter.drawLine(int(points[i][0]), int(points[i][1]), int(points[i + 1][0]), int(points[i + 1][1]))

        if draw_placeholder_text:
            painter.setPen(QColor('#7C869B'))
            painter.drawText(chart_rect.adjusted(8, 8, -8, -8), Qt.AlignBottom | Qt.AlignRight, '占位图表')
        painter.end()


class TextSummaryCard(CardWidget):
    def __init__(self, title: str, lines: list[str], parent: QWidget | None = None) -> None:
        super().__init__(title=title, parent=parent)
        body = QLabel('\n'.join(f'• {line}' for line in lines))
        body.setWordWrap(True)
        self.layout.addWidget(body)
