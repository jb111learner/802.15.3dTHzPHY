APP_QSS = """
QWidget {
    background: transparent;
    color: #1B2430;
    font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif;
    font-size: 13px;
}
QMainWindow {
    background: #F5F7FB;
}
QFrame#Card,
QFrame#Panel,
QFrame#TopBar,
QFrame#NavBar,
QFrame#ChartCard {
    background: #FFFFFF;
    border: 1px solid #E5EAF3;
    border-radius: 14px;
}
QFrame#TopBar {
    border-radius: 16px;
    border-color: #DFE6F2;
}
QFrame#NavBar {
    background: #FFFFFF;
    border-color: #DFE6F2;
    border-radius: 14px;
}
QFrame#SectionHeader {
    background: transparent;
    border: none;
}
QLabel#Title {
    font-size: 24px;
    font-weight: 700;
    color: #152033;
}
QLabel#Subtitle {
    color: #6C778A;
    font-size: 12px;
}
QLabel#CardTitle {
    font-size: 15px;
    font-weight: 700;
    color: #1B2430;
}
QLabel#CardHint {
    color: #7C869B;
    font-size: 12px;
}
QLabel#MetricValue {
    font-size: 22px;
    font-weight: 700;
    color: #1B2430;
}
QLabel#MetricLabel {
    color: #7C869B;
    font-size: 12px;
}
QPushButton {
    background: #FFFFFF;
    border: 1px solid #D5DDEA;
    border-radius: 10px;
    padding: 8px 14px;
}
QPushButton:hover {
    background: #F8FAFF;
    border-color: #B7C5E4;
}
QPushButton:pressed {
    background: #EEF3FF;
}
QPushButton[compact="true"] {
    min-width: 64px;
    padding: 6px 12px;
    border-radius: 8px;
    color: #43516A;
}
QPushButton[compact="true"]:hover {
    background: #EEF3FF;
    border-color: #AFC1EA;
    color: #2F6BFF;
}
QPushButton[role="primary"] {
    background: #2F6BFF;
    color: white;
    border: none;
}
QPushButton[role="success"] {
    background: #0EA76B;
    color: white;
    border: none;
}
QPushButton[role="warning"] {
    background: #E6901E;
    color: white;
    border: none;
}
QPushButton[role="danger"] {
    background: #FF4D4F;
    color: white;
    border: none;
}
QPushButton[nav="true"] {
    border: none;
    background: transparent;
    min-height: 22px;
    padding: 9px 16px;
    border-radius: 10px;
    font-weight: 600;
    color: #4B5870;
}
QPushButton[nav="true"]:hover {
    background: #F3F6FC;
    color: #2F6BFF;
}
QPushButton[nav="true"]:checked {
    background: #EAF0FF;
    color: #2F6BFF;
    font-weight: 700;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {
    background: #FCFDFF;
    border: 1px solid #D8E0EF;
    border-radius: 8px;
    padding: 6px 8px;
}
QSpinBox, QDoubleSpinBox {
    padding-right: 28px;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 24px;
    background: #F3F6FB;
    border-left: 1px solid #D8E0EF;
    border-bottom: 1px solid #D8E0EF;
    border-top-right-radius: 7px;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 24px;
    background: #F3F6FB;
    border-left: 1px solid #D8E0EF;
    border-bottom-right-radius: 7px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: #EAF0FF;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: url(thz_sim_ui/assets/spin_up.svg);
    width: 10px;
    height: 6px;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: url(thz_sim_ui/assets/spin_down.svg);
    width: 10px;
    height: 6px;
}
QComboBox QAbstractItemView {
    background: #FFFFFF;
    color: #1B2430;
    border: 1px solid #D8E0EF;
    border-radius: 6px;
    padding: 4px;
    outline: none;
    selection-background-color: #EAF0FF;
    selection-color: #2F6BFF;
}
QComboBoxPrivateContainer {
    background: #FFFFFF;
    border: 1px solid #D8E0EF;
    border-radius: 8px;
}
QComboBox QAbstractItemView::item {
    min-height: 28px;
    padding: 4px 8px;
}
QDialog {
    background: #F5F7FB;
}
QMessageBox {
    background: #FFFFFF;
}
QMessageBox QLabel {
    background: transparent;
    color: #1B2430;
}
QMessageBox QPushButton {
    min-width: 64px;
    background: #FFFFFF;
    color: #1B2430;
}
QMenu {
    background: #FFFFFF;
    color: #1B2430;
    border: 1px solid #D8E0EF;
}
QMenu::item:selected {
    background: #EAF0FF;
    color: #2F6BFF;
}
QTabWidget::pane {
    border: 1px solid #E5EAF3;
    border-radius: 12px;
    background: #FFFFFF;
    top: -1px;
}
QTabBar::tab {
    background: #F1F4FA;
    padding: 8px 14px;
    margin-right: 6px;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
}
QTabBar::tab:selected {
    background: #FFFFFF;
    color: #2F6BFF;
    font-weight: 700;
}
QGroupBox {
    border: 1px solid #E5EAF3;
    border-radius: 12px;
    margin-top: 12px;
    background: #FFFFFF;
    font-weight: 700;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 4px;
}
QTableWidget {
    background: #FFFFFF;
    border: 1px solid #E5EAF3;
    border-radius: 12px;
    gridline-color: #ECF0F7;
}
QHeaderView::section {
    background: #F3F6FB;
    border: none;
    border-bottom: 1px solid #E5EAF3;
    padding: 8px;
    font-weight: 700;
}
QScrollArea {
    border: none;
    background: transparent;
}
QListWidget {
    background: #FFFFFF;
    border: 1px solid #E5EAF3;
    border-radius: 12px;
    padding: 6px;
}
QListWidget::item {
    padding: 8px 10px;
    margin: 2px 0;
    border-radius: 8px;
}
QListWidget::item:selected {
    background: #EAF0FF;
    color: #2F6BFF;
}
QProgressBar {
    border: none;
    border-radius: 8px;
    background: #EAEFF8;
    text-align: center;
    min-height: 10px;
}
QProgressBar::chunk {
    background: #2F6BFF;
    border-radius: 8px;
}
QStatusBar {
    background: #FFFFFF;
    border-top: 1px solid #E5EAF3;
}
QSplitter::handle {
    background: #E9EEF7;
}
QCheckBox, QRadioButton {
    spacing: 8px;
}
QCheckBox[optionCard="true"],
QRadioButton[optionCard="true"] {
    background: #F8FAFD;
    border: 1px solid #E1E7F0;
    border-radius: 9px;
    padding: 8px 10px;
    color: #344158;
    font-weight: 500;
}
QCheckBox[optionCard="true"]:hover,
QRadioButton[optionCard="true"]:hover {
    background: #F1F5FF;
    border-color: #B8C8EA;
    color: #2F6BFF;
}
QCheckBox[optionCard="true"]:checked,
QRadioButton[optionCard="true"]:checked {
    background: #EAF0FF;
    border-color: #9CB4ED;
    color: #245DE0;
    font-weight: 700;
}
QToolTip {
    background: #1F2937;
    color: white;
    border: none;
    padding: 6px;
}
"""
