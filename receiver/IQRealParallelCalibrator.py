import warnings
import numpy as np


class IQRealParallelCalibrator:
    """
    Real-based parallel LS estimator and TX precompensator for IQ imbalance.

    The model is:
        xI = h1 * sI + h2 * sQ + bI
        xQ = h3 * sI + h4 * sQ + bQ

    All FIR convolutions are causal. Same-length signal outputs are produced by
    truncating the full causal convolution to the input sequence length.
    """

    _COND_WARNING_THRESHOLD = 1e10

    @staticmethod
    def _as_1d_array(values, dtype=float, name="array"):
        array = np.asarray(values, dtype=dtype)
        if array.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array")
        return array

    @staticmethod
    def _validate_filter_len(filter_len):
        try:
            filter_len = int(filter_len)
        except (TypeError, ValueError) as exc:
            raise ValueError("filter_len must be a positive integer") from exc

        if filter_len <= 0:
            raise ValueError("filter_len must be a positive integer")
        return filter_len

    @staticmethod
    def _causal_fir_same_length(signal, taps):
        signal = IQRealParallelCalibrator._as_1d_array(
            signal, dtype=float, name="signal"
        )
        taps = IQRealParallelCalibrator._as_1d_array(taps, dtype=float, name="taps")
        if taps.size == 0:
            raise ValueError("taps must not be empty")
        if signal.size == 0:
            return np.zeros(0, dtype=float)
        return np.convolve(signal, taps, mode="full")[: signal.size]

    @staticmethod
    def _build_full_convolution_matrix(taps, input_len, output_len=None):
        taps = IQRealParallelCalibrator._as_1d_array(taps, dtype=float, name="taps")
        if taps.size == 0:
            raise ValueError("taps must not be empty")

        input_len = IQRealParallelCalibrator._validate_filter_len(input_len)
        if output_len is None:
            output_len = taps.size + input_len - 1
        output_len = IQRealParallelCalibrator._validate_filter_len(output_len)

        matrix = np.zeros((output_len, input_len), dtype=float)
        for col in range(input_len):
            row_start = col
            row_stop = min(col + taps.size, output_len)
            if row_stop > row_start:
                matrix[row_start:row_stop, col] = taps[: row_stop - row_start]
        return matrix

    @staticmethod
    def build_toeplitz(signal, filter_len):
        """
        Build the same-length causal FIR convolution matrix.

        If T = build_toeplitz(x, L), then T @ h equals
        np.convolve(x, h, mode="full")[:len(x)] for len(h) == L.
        """
        filter_len = IQRealParallelCalibrator._validate_filter_len(filter_len)
        signal = IQRealParallelCalibrator._as_1d_array(
            signal, dtype=float, name="signal"
        )
        if signal.size == 0:
            raise ValueError("signal must not be empty")

        toeplitz = np.zeros((signal.size, filter_len), dtype=float)
        for tap_idx in range(min(filter_len, signal.size)):
            toeplitz[tap_idx:, tap_idx] = signal[: signal.size - tap_idx]
        return toeplitz

    @staticmethod
    def estimate_impairment_filters(s, x, filter_len, ridge_lambda=0.0):
        """
        Estimate h1, h2, h3, h4 and I/Q DC offsets by LS.
        """
        filter_len = IQRealParallelCalibrator._validate_filter_len(filter_len)
        ridge_lambda = float(ridge_lambda)
        if not np.isfinite(ridge_lambda):
            raise ValueError("ridge_lambda must be finite")

        s = IQRealParallelCalibrator._as_1d_array(s, dtype=complex, name="s")
        x = IQRealParallelCalibrator._as_1d_array(x, dtype=complex, name="x")

        if s.size != x.size:
            raise ValueError("s and x must have the same length")
        if s.size <= filter_len:
            raise ValueError("input length must be greater than filter_len")

        sI = np.real(s)
        sQ = np.imag(s)
        xI = np.real(x)
        xQ = np.imag(x)

        design_matrix = np.column_stack(
            (
                np.ones(s.size, dtype=float),
                IQRealParallelCalibrator.build_toeplitz(sI, filter_len),
                IQRealParallelCalibrator.build_toeplitz(sQ, filter_len),
            )
        )
        design_condition_number = float(np.linalg.cond(design_matrix))
        warning_message = None

        if ridge_lambda <= 0.0:
            estimation_method = "pinv"
            design_pinv = np.linalg.pinv(design_matrix)
            coeff_I = design_pinv @ xI
            coeff_Q = design_pinv @ xQ
        else:
            estimation_method = "ridge"
            regularizer = np.eye(design_matrix.shape[1], dtype=float)
            regularizer[0, 0] = 0.0
            lhs = (
                design_matrix.T @ design_matrix
                + ridge_lambda * regularizer
            )
            rhs_I = design_matrix.T @ xI
            rhs_Q = design_matrix.T @ xQ
            lhs_condition_number = float(np.linalg.cond(lhs))
            use_pinv = (
                not np.isfinite(lhs_condition_number)
                or lhs_condition_number > IQRealParallelCalibrator._COND_WARNING_THRESHOLD
            )

            if use_pinv:
                warning_message = (
                    "ridge normal-equation matrix is ill-conditioned; "
                    f"condition_number={lhs_condition_number}. "
                    "Using pseudoinverse fallback."
                )
                warnings.warn(warning_message, RuntimeWarning, stacklevel=2)
                lhs_pinv = np.linalg.pinv(lhs)
                coeff_I = lhs_pinv @ rhs_I
                coeff_Q = lhs_pinv @ rhs_Q
            else:
                try:
                    coeff_I = np.linalg.solve(lhs, rhs_I)
                    coeff_Q = np.linalg.solve(lhs, rhs_Q)
                except np.linalg.LinAlgError:
                    warning_message = (
                        "ridge normal-equation solve failed. "
                        "Using pseudoinverse fallback."
                    )
                    warnings.warn(warning_message, RuntimeWarning, stacklevel=2)
                    lhs_pinv = np.linalg.pinv(lhs)
                    coeff_I = lhs_pinv @ rhs_I
                    coeff_Q = lhs_pinv @ rhs_Q

        if not (np.all(np.isfinite(coeff_I)) and np.all(np.isfinite(coeff_Q))):
            raise np.linalg.LinAlgError(
                "impairment filter estimation produced non-finite coefficients"
            )

        bI = float(coeff_I[0])
        h1 = coeff_I[1 : 1 + filter_len].astype(float)
        h2 = coeff_I[1 + filter_len :].astype(float)

        bQ = float(coeff_Q[0])
        h3 = coeff_Q[1 : 1 + filter_len].astype(float)
        h4 = coeff_Q[1 + filter_len :].astype(float)

        residual_I = design_matrix @ coeff_I - xI
        residual_Q = design_matrix @ coeff_Q - xQ

        result = {
            "h1": h1,
            "h2": h2,
            "h3": h3,
            "h4": h4,
            "bI": bI,
            "bQ": bQ,
            "residual_norm_I": float(np.linalg.norm(residual_I)),
            "residual_norm_Q": float(np.linalg.norm(residual_Q)),
            "ridge_lambda": ridge_lambda,
            "design_condition_number": design_condition_number,
            "estimation_method": estimation_method,
        }
        if warning_message is not None:
            result["warning"] = warning_message
        return result

    @staticmethod
    def estimate_rx_impairment_filters(y, r, filter_len, ridge_lambda=0.0):
        """
        Estimate RX-side real-based impairment filters with RX-oriented names.
        """
        estimate = IQRealParallelCalibrator.estimate_impairment_filters(
            y, r, filter_len, ridge_lambda=ridge_lambda
        )
        rx_estimate = {
            "e1": estimate["h1"],
            "e2": estimate["h2"],
            "e3": estimate["h3"],
            "e4": estimate["h4"],
            "dI": estimate["bI"],
            "dQ": estimate["bQ"],
            "residual_norm_I": estimate["residual_norm_I"],
            "residual_norm_Q": estimate["residual_norm_Q"],
            "ridge_lambda": estimate["ridge_lambda"],
            "design_condition_number": estimate["design_condition_number"],
            "estimation_method": estimate["estimation_method"],
        }
        if "warning" in estimate:
            rx_estimate["warning"] = estimate["warning"]
        return rx_estimate

    @staticmethod
    def design_precompensation_filters(h1, h2, h3, h4, filter_len):
        """
        Design real FIR TX precompensation filters g1, g2, g3, g4.
        """
        filter_len = IQRealParallelCalibrator._validate_filter_len(filter_len)
        h1 = IQRealParallelCalibrator._as_1d_array(h1, dtype=float, name="h1")
        h2 = IQRealParallelCalibrator._as_1d_array(h2, dtype=float, name="h2")
        h3 = IQRealParallelCalibrator._as_1d_array(h3, dtype=float, name="h3")
        h4 = IQRealParallelCalibrator._as_1d_array(h4, dtype=float, name="h4")

        if min(h1.size, h2.size, h3.size, h4.size) == 0:
            raise ValueError("h1, h2, h3, and h4 must not be empty")

        output_len = max(h1.size, h2.size, h3.size, h4.size) + filter_len - 1
        H1 = IQRealParallelCalibrator._build_full_convolution_matrix(
            h1, filter_len, output_len=output_len
        )
        H2 = IQRealParallelCalibrator._build_full_convolution_matrix(
            h2, filter_len, output_len=output_len
        )
        H3 = IQRealParallelCalibrator._build_full_convolution_matrix(
            h3, filter_len, output_len=output_len
        )
        H4 = IQRealParallelCalibrator._build_full_convolution_matrix(
            h4, filter_len, output_len=output_len
        )

        block_matrix = np.block([[H1, H2], [H3, H4]])
        delta = np.zeros(output_len, dtype=float)
        delta[0] = 1.0
        zero = np.zeros(output_len, dtype=float)

        condition_number = float(np.linalg.cond(block_matrix))
        warning_message = None
        if (
            not np.isfinite(condition_number)
            or condition_number > IQRealParallelCalibrator._COND_WARNING_THRESHOLD
        ):
            warning_message = (
                "precompensation LS matrix is ill-conditioned; "
                f"condition_number={condition_number}"
            )
            warnings.warn(warning_message, RuntimeWarning, stacklevel=2)

        block_pinv = np.linalg.pinv(block_matrix)
        solution_col_1 = block_pinv @ np.concatenate((delta, zero))
        solution_col_2 = block_pinv @ np.concatenate((zero, delta))

        if not (
            np.all(np.isfinite(solution_col_1))
            and np.all(np.isfinite(solution_col_2))
        ):
            raise np.linalg.LinAlgError(
                "precompensation filter design produced non-finite coefficients"
            )

        result = {
            "g1": solution_col_1[:filter_len].astype(float),
            "g3": solution_col_1[filter_len:].astype(float),
            "g2": solution_col_2[:filter_len].astype(float),
            "g4": solution_col_2[filter_len:].astype(float),
            "condition_number": condition_number,
        }
        if warning_message is not None:
            result["warning"] = warning_message
        return result

    @staticmethod
    def design_postcompensation_filters(e1, e2, e3, e4, filter_len):
        """
        Design real FIR RX postcompensation filters f1, f2, f3, f4.
        """
        filter_len = IQRealParallelCalibrator._validate_filter_len(filter_len)
        e1 = IQRealParallelCalibrator._as_1d_array(e1, dtype=float, name="e1")
        e2 = IQRealParallelCalibrator._as_1d_array(e2, dtype=float, name="e2")
        e3 = IQRealParallelCalibrator._as_1d_array(e3, dtype=float, name="e3")
        e4 = IQRealParallelCalibrator._as_1d_array(e4, dtype=float, name="e4")

        if min(e1.size, e2.size, e3.size, e4.size) == 0:
            raise ValueError("e1, e2, e3, and e4 must not be empty")

        output_len = max(e1.size, e2.size, e3.size, e4.size) + filter_len - 1
        E1 = IQRealParallelCalibrator._build_full_convolution_matrix(
            e1, filter_len, output_len=output_len
        )
        E2 = IQRealParallelCalibrator._build_full_convolution_matrix(
            e2, filter_len, output_len=output_len
        )
        E3 = IQRealParallelCalibrator._build_full_convolution_matrix(
            e3, filter_len, output_len=output_len
        )
        E4 = IQRealParallelCalibrator._build_full_convolution_matrix(
            e4, filter_len, output_len=output_len
        )

        block_matrix = np.block([[E1, E3], [E2, E4]])
        delta = np.zeros(output_len, dtype=float)
        delta[0] = 1.0
        zero = np.zeros(output_len, dtype=float)

        condition_number = float(np.linalg.cond(block_matrix))
        warning_message = None
        if (
            not np.isfinite(condition_number)
            or condition_number > IQRealParallelCalibrator._COND_WARNING_THRESHOLD
        ):
            warning_message = (
                "postcompensation LS matrix is ill-conditioned; "
                f"condition_number={condition_number}"
            )
            warnings.warn(warning_message, RuntimeWarning, stacklevel=2)

        block_pinv = np.linalg.pinv(block_matrix)
        solution_row_1 = block_pinv @ np.concatenate((delta, zero))
        solution_row_2 = block_pinv @ np.concatenate((zero, delta))

        if not (
            np.all(np.isfinite(solution_row_1))
            and np.all(np.isfinite(solution_row_2))
        ):
            raise np.linalg.LinAlgError(
                "postcompensation filter design produced non-finite coefficients"
            )

        result = {
            "f1": solution_row_1[:filter_len].astype(float),
            "f2": solution_row_1[filter_len:].astype(float),
            "f3": solution_row_2[:filter_len].astype(float),
            "f4": solution_row_2[filter_len:].astype(float),
            "condition_number": condition_number,
        }
        if warning_message is not None:
            result["warning"] = warning_message
        return result

    @staticmethod
    def design_precompensation_dc_offset(h1, h2, h3, h4, bI, bQ):
        """
        Design constant I/Q TX offsets that cancel real-based RF DC offsets.
        """
        h1 = IQRealParallelCalibrator._as_1d_array(h1, dtype=float, name="h1")
        h2 = IQRealParallelCalibrator._as_1d_array(h2, dtype=float, name="h2")
        h3 = IQRealParallelCalibrator._as_1d_array(h3, dtype=float, name="h3")
        h4 = IQRealParallelCalibrator._as_1d_array(h4, dtype=float, name="h4")

        if min(h1.size, h2.size, h3.size, h4.size) == 0:
            raise ValueError("h1, h2, h3, and h4 must not be empty")

        bI = float(bI)
        bQ = float(bQ)
        dc_matrix = np.array(
            [
                [np.sum(h1), np.sum(h2)],
                [np.sum(h3), np.sum(h4)],
            ],
            dtype=float,
        )
        rhs = np.array([-bI, -bQ], dtype=float)

        condition_number = float(np.linalg.cond(dc_matrix))
        use_pinv = (
            not np.isfinite(condition_number)
            or condition_number > IQRealParallelCalibrator._COND_WARNING_THRESHOLD
        )

        if use_pinv:
            warnings.warn(
                "DC offset precompensation matrix is ill-conditioned; "
                f"condition_number={condition_number}. Using pseudoinverse LS.",
                RuntimeWarning,
                stacklevel=2,
            )
            solution = np.linalg.pinv(dc_matrix) @ rhs
        else:
            try:
                solution = np.linalg.solve(dc_matrix, rhs)
            except np.linalg.LinAlgError:
                warnings.warn(
                    "DC offset precompensation matrix is singular. "
                    "Using pseudoinverse LS.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                solution = np.linalg.pinv(dc_matrix) @ rhs

        if not np.all(np.isfinite(solution)):
            raise np.linalg.LinAlgError(
                "DC offset precompensation produced non-finite coefficients"
            )

        return {
            "aI": float(solution[0]),
            "aQ": float(solution[1]),
            "dc_matrix_condition_number": condition_number,
        }

    @staticmethod
    def design_postcompensation_dc_offset(f1, f2, f3, f4, dI, dQ):
        """
        Design RX postcompensation DC offsets after f1-f4 filtering.
        """
        f1 = IQRealParallelCalibrator._as_1d_array(f1, dtype=float, name="f1")
        f2 = IQRealParallelCalibrator._as_1d_array(f2, dtype=float, name="f2")
        f3 = IQRealParallelCalibrator._as_1d_array(f3, dtype=float, name="f3")
        f4 = IQRealParallelCalibrator._as_1d_array(f4, dtype=float, name="f4")

        if min(f1.size, f2.size, f3.size, f4.size) == 0:
            raise ValueError("f1, f2, f3, and f4 must not be empty")

        dI = float(dI)
        dQ = float(dQ)
        cI = -(dI * np.sum(f1) + dQ * np.sum(f2))
        cQ = -(dI * np.sum(f3) + dQ * np.sum(f4))

        if not (np.isfinite(cI) and np.isfinite(cQ)):
            raise np.linalg.LinAlgError(
                "RX postcompensation DC offset produced non-finite coefficients"
            )

        return {
            "cI": float(cI),
            "cQ": float(cQ),
        }

    @staticmethod
    def apply_precompensation(s, g1, g2, g3, g4, aI=0.0, aQ=0.0):
        """
        Apply real-based parallel TX precompensation to a complex signal.
        """
        s = IQRealParallelCalibrator._as_1d_array(s, dtype=complex, name="s")
        sI = np.real(s)
        sQ = np.imag(s)
        aI = float(aI)
        aQ = float(aQ)

        spI = (
            IQRealParallelCalibrator._causal_fir_same_length(sI, g1)
            + IQRealParallelCalibrator._causal_fir_same_length(sQ, g2)
            + aI
        )
        spQ = (
            IQRealParallelCalibrator._causal_fir_same_length(sI, g3)
            + IQRealParallelCalibrator._causal_fir_same_length(sQ, g4)
            + aQ
        )
        return spI + 1j * spQ

    @staticmethod
    def apply_postcompensation(r, f1, f2, f3, f4, cI=0.0, cQ=0.0):
        """
        Apply real-based parallel RX postcompensation to a complex signal.
        """
        r = IQRealParallelCalibrator._as_1d_array(r, dtype=complex, name="r")
        rI = np.real(r)
        rQ = np.imag(r)
        cI = float(cI)
        cQ = float(cQ)

        rcI = (
            IQRealParallelCalibrator._causal_fir_same_length(rI, f1)
            + IQRealParallelCalibrator._causal_fir_same_length(rQ, f2)
            + cI
        )
        rcQ = (
            IQRealParallelCalibrator._causal_fir_same_length(rI, f3)
            + IQRealParallelCalibrator._causal_fir_same_length(rQ, f4)
            + cQ
        )
        return rcI + 1j * rcQ
