import numpy as np

from receiver.sync.CoarseSync import CoarseSync


def test_fft_correlation_matches_numpy_valid_correlation():
    rng = np.random.default_rng(23)
    signal = rng.normal(size=257) + 1j * rng.normal(size=257)
    reference = rng.normal(size=31) + 1j * rng.normal(size=31)

    expected = np.correlate(signal, reference, mode="valid")
    actual = CoarseSync._valid_correlation(signal, reference)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert np.argmax(np.abs(actual)) == np.argmax(np.abs(expected))
