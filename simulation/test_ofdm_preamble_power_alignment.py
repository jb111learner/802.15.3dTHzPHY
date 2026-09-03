"""OFDM preamble/payload power-alignment regression tests."""

import numpy as np

from params.PHYParams import PHYParams
from transmitter.PreambleInsertor import PreambleInsertor


def _params(link_mode, oversampling=4):
    params = PHYParams()
    params.update(
        link_mode=link_mode,
        oversampling=oversampling,
        Preamble_type="short",
    )
    return params


def test_ofdm_preamble_matches_zero_padded_ifft_power():
    oversampling = 4
    insertor = PreambleInsertor(_params("ofdm", oversampling))

    preamble_power = np.mean(np.abs(insertor.preamble_upsampled) ** 2)

    assert np.isclose(preamble_power, 1.0 / oversampling, rtol=1e-12)
    assert 0.49 < insertor.preamble_power_scale < 0.52


def test_sc_fde_preamble_power_is_unchanged():
    insertor = PreambleInsertor(_params("sc-fde"))

    assert insertor.preamble_upsampled is insertor.preamble
    assert insertor.preamble_power_scale == 1.0
    assert np.isclose(
        np.mean(np.abs(insertor.preamble_upsampled) ** 2),
        np.mean(np.abs(insertor.preamble) ** 2),
    )


def test_alignment_removes_frame_power_inflation():
    oversampling = 4
    insertor = PreambleInsertor(_params("ofdm", oversampling))
    preamble = insertor.preamble_upsampled
    payload = np.full(48 * (512 + 32) * oversampling, 0.5, dtype=np.complex128)

    frame = np.concatenate([preamble, payload])
    frame_power = np.mean(np.abs(frame) ** 2)

    assert np.isclose(np.mean(np.abs(preamble) ** 2), 0.25, rtol=1e-12)
    assert np.isclose(np.mean(np.abs(payload) ** 2), 0.25, rtol=1e-12)
    assert np.isclose(frame_power, 0.25, rtol=1e-12)
