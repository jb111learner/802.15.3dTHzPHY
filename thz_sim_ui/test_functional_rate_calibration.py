import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")


def test_single_link_actual_rate_exceeds_50_gbps():
    pytest.importorskip("PySide6")
    from thz_sim_ui.services.functional_test_service import run_single_link_rate_test

    result = run_single_link_rate_test(duration=2e-7)
    assert result["ok"]
    assert result["data"]["actual_rate_bps"] >= 50e9
    assert result["data"]["actual_rate_bps"] == pytest.approx(
        result["data"]["theoretical_rate_bps"], rel=1e-12
    )
    assert {item["label"] for item in result["summary"]} >= {"理论速率", "实际速率"}


def test_total_phy_rate_uses_required_configuration_and_exceeds_half_tbps():
    pytest.importorskip("PySide6")
    from thz_sim_ui.services.functional_test_service import run_total_phy_rate_test

    result = run_total_phy_rate_test()
    data = result["data"]
    assert result["ok"]
    assert data["configuration"] == "256QAM / 1024 子载波 / 2×2 MIMO / LDPC(11/15)"
    assert data["frame_information_bits"] == 270336
    assert data["ldpc_padding_bits"] == 0
    assert data["theoretical_rate_bps"] == pytest.approx(550956521739.1305)
    assert data["actual_rate_bps"] >= 0.5e12


def test_function_calibration_matches_theory_and_matlab_references():
    pytest.importorskip("PySide6")
    from thz_sim_ui.services.functional_test_service import run_function_calibration_test

    result = run_function_calibration_test(bits_per_point=10000, random_seed=2026)
    assert result["ok"]
    assert len(result["checks"]) == 5
    assert all(item["ok"] for item in result["checks"])
    assert result["plots"][0]["png"].startswith(b"\x89PNG")
    assert {curve["name"] for curve in result["data"]["curves"]} == {
        "QPSK", "16QAM", "64QAM"
    }


def test_new_ui_tests_follow_precision_in_required_order():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from thz_sim_ui.pages.functional_test_page import FunctionalTestPage, _TEST_KEYS

    app = QApplication.instance() or QApplication([])
    page = FunctionalTestPage()
    try:
        assert _TEST_KEYS[3:] == (
            "precision", "single_link_rate", "total_phy_rate", "function_calibration"
        )
        assert page.test_radios[4].text().startswith("单链路物理层速率测试")
        assert page.test_radios[5].text().startswith("总物理层速率测试")
        assert page.test_radios[6].text() == "功能校准测试"
        assert page.param_stack.count() == 7
        assert page.result_stack.count() == 7
        assert page.total_rate_configuration.currentText().startswith("256QAM / 1024")
        page.resize(1440, 900)
        page.show()
        page.test_radios[5].setChecked(True)
        app.processEvents()
        assert page.param_stack.height() == page.param_stack.currentWidget().sizeHint().height()
        assert page.left_scroll.verticalScrollBar().maximum() == 0
        assert page.left_footer.isVisible()
        assert page.run_button.mapTo(page, page.run_button.rect().bottomLeft()).y() >= 850
    finally:
        page.close()
        app.processEvents()
