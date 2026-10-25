import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from thz_sim_ui.pages.parameter_config_page import ParameterConfigPage
from thz_sim_ui.services.backend import BackendService


def test_two_modes_and_project_numeric_roundtrip():
    app = QApplication.instance() or QApplication([])
    page = ParameterConfigPage()
    assert [page.numeric_mode.itemText(i) for i in range(page.numeric_mode.count())] == ["浮点", "定点"]
    page.set_all_parameters({"仿真数值模式": "定点", "定点字长": 18, "定点小数位": 14,
                             "范围统计": "启用", "_fixed_point": {"modules": {"ofdm.tx.ifft": {"arithmetic_level": "operation"}}}})
    exported = page.get_all_parameters()
    mapped = BackendService.map_ui_params_to_phy_params(exported)
    assert mapped["fixed_point"]["enable_fixed"]
    assert mapped["fixed_point"]["default_format"]["fractional_length"] == 14
    assert mapped["fixed_point"]["modules"]["ofdm.tx.ifft"]["arithmetic_level"] == "operation"
    assert mapped["range_analysis"]["enabled"]
    page.set_all_parameters({"链路模式": "SC"})
    assert page.numeric_mode.currentText() == "浮点"
    page.close()


def test_fractional_bits_follow_word_length():
    app = QApplication.instance() or QApplication([])
    page = ParameterConfigPage()
    page.fixed_word_length.setValue(8)
    assert page.fixed_fractional_length.value() <= 7
    page.close()
