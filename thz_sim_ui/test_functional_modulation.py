import os

import pytest


@pytest.mark.parametrize(
    ("modulation", "ncbps"),
    [("QPSK", 2), ("16QAM", 4), ("64QAM", 6)],
)
def test_tx_modulation_generates_expected_symbol_count(modulation, ncbps):
    pytest.importorskip("PySide6")
    from thz_sim_ui.services.functional_test_service import run_modulation_test

    result = run_modulation_test(
        modulation=modulation,
        num_symbols=256,
        show_points=128,
        random_seed=2026,
    )

    assert result["ok"]
    assert result["checks"] == []
    assert result["data"]["ncbps"] == ncbps
    assert result["data"]["input_bit_count"] == 256 * ncbps
    assert result["data"]["output_symbol_count"] == 256
    assert result["data"]["shown_symbol_count"] == 128
    assert result["plots"][0]["png"].startswith(b"\x89PNG")


def test_modulation_test_is_first_ui_option():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from thz_sim_ui.pages.functional_test_page import FunctionalTestPage

    app = QApplication.instance() or QApplication([])
    page = FunctionalTestPage()
    try:
        assert page.test_radios[0].text().startswith("调制方式测试")
        assert page._selected_test() == "modulation"
        assert [page.modulation_type.itemText(i)
                for i in range(page.modulation_type.count())] == [
                    "QPSK", "16QAM", "64QAM"]
    finally:
        page.close()
        app.processEvents()
