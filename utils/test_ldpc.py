import numpy as np

from params.PHYParams import PHYParams
from transmitter.DataProcesser import BitStreamProcessor
from transmitter.Encoder import Encoder
from utils.Coder import LDPCCoder
from utils.LDPCMatrix import (
    build_ieee802153d_1440_h,
    gf2_rank,
)


def _rate11_params():
    params = PHYParams()
    params.update(
        code_type="LDPC",
        ldpc_matrix_type="ieee802153d_1440",
        ldpc_standard_rate="11/15",
        ldpc_rate11_direction="minus_literal",
    )
    return params


def test_ieee802153d_rate11_matrix_and_parity_part_are_full_rank():
    h = build_ieee802153d_1440_h("11/15", "minus_literal")

    assert h.shape == (384, 1440)
    assert int(np.sum(h)) == 4320
    assert np.all(np.sum(h, axis=0) == 3)
    assert gf2_rank(h) == 384
    assert gf2_rank(h[:, 1056:1440]) == 384


def test_ieee802153d_rate11_systematic_encode_and_noiseless_decode():
    coder = LDPCCoder(_rate11_params())
    rng = np.random.default_rng(1530)
    info = rng.integers(0, 2, size=coder.k, dtype=np.uint8)

    codeword = coder.encode(info)
    decoded, debug = coder.decode(np.where(codeword == 0, 12.0, -12.0), original_bit_len=len(info))

    assert np.array_equal(codeword[:coder.k], info)
    assert np.all(coder.syndrome(codeword) == 0)
    assert np.array_equal(decoded, info)
    assert debug["decode_success"]


def test_rate11_wrappers_use_configured_ldpc_dimensions():
    params = _rate11_params()
    processor = BitStreamProcessor(params)
    encoder = Encoder(params)

    assert processor.code_rate == 11 / 15
    assert (encoder.packet_size, encoder.nsym, encoder.encoded_packet_size) == (1056, 384, 1440)

    info = np.zeros(1056, dtype=np.uint8)
    encoded = encoder.encode({
        "signal_stream": info,
        "sample_rate_Hz": 1056,
        "duration_seconds": 1.0,
        "signal_length": 1056,
        "padding_bit_num": 0,
        "frame_bit_num": 1056,
        "frame_num": 1,
    })

    assert encoded["signal_length"] == 1440
    assert len(encoded["signal_stream"]) == 1440
