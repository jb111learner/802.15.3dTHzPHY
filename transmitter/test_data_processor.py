import numpy as np
import pytest

from transmitter.DataProcesser import BitStreamProcessor


@pytest.mark.parametrize('frame_bit_num', [4, np.int64(4), 4.0, np.float64(4.0)])
def test_frame_process_accepts_integral_numeric_types(frame_bit_num):
    processor = BitStreamProcessor.__new__(BitStreamProcessor)
    processor.bit_stream = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
    processor.frame_bit_num = frame_bit_num
    processor.padding_bit_num = 0
    processor.sample_rate = 1.0
    processor.NCBPS = 1

    processor.frame_process()

    assert processor.frame_bit_num == 4
    assert processor.frame_num == 2
    assert processor.padding_bit_num == 3


@pytest.mark.parametrize('frame_bit_num', [4.5, 0, -2, np.nan, np.inf, None])
def test_frame_process_rejects_invalid_frame_size(frame_bit_num):
    processor = BitStreamProcessor.__new__(BitStreamProcessor)
    processor.bit_stream = np.array([1, 0, 1, 0], dtype=np.uint8)
    processor.frame_bit_num = frame_bit_num

    with pytest.raises(ValueError):
        processor.frame_process()
