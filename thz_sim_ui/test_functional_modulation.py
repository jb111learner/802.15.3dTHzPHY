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
    # 星座点 I/Q 表格：每个唯一星座点一行，且不再显示调制耗时
    assert result["table"]["columns"] == ["星座点", "I 分量", "Q 分量"]
    assert len(result["table"]["rows"]) == 2 ** ncbps
    assert not any(item["label"] == "调制耗时" for item in result["summary"])


def test_modulation_test_disables_pi2_rotation_for_qpsk():
    pytest.importorskip("PySide6")
    from thz_sim_ui.services.functional_test_service import run_modulation_test

    result = run_modulation_test(
        modulation="QPSK",
        num_symbols=1024,
        show_points=1024,
        random_seed=2026,
    )

    # 关闭 pi/2 旋转与 pi/4 补偿后 QPSK 为方型四点 {(±1±1j)/√2}
    scale = 1.0 / (2 ** 0.5)
    pairs = sorted(
        (round(float(row[1]), 6), round(float(row[2]), 6))
        for row in result["table"]["rows"])
    expected = sorted([
        (round(scale, 6), round(scale, 6)),
        (round(scale, 6), round(-scale, 6)),
        (round(-scale, 6), round(scale, 6)),
        (round(-scale, 6), round(-scale, 6)),
    ])
    assert pairs == expected


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
