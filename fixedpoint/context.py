"""模块定点执行与只读范围观察；普通运行不捕获或重放随机刺激。"""
from dataclasses import replace
from collections.abc import Mapping
import math
import numpy as np
from .config import FixedPointConfig, parameter_value
from .array import FixedPointArray
from .format import FixedPointFormat
from . import arithmetic as ar
from .range_analysis import RangeAnalyzer
from .kernels.fft import fixed_fft
from .kernels.fir import fixed_fir, fixed_correlate


class FixedPointContext:
    def __init__(self, config=None, range_config=None):
        # 运行快照与调用者嵌套字典隔离；frozen dataclass本身不冻结dict。
        snapshot = config.to_dict() if isinstance(config, FixedPointConfig) else config
        self.config = FixedPointConfig.from_dict(snapshot)
        range_config = {} if range_config is None else range_config
        if isinstance(range_config, bool):
            range_config = {"enabled": range_config}
        if not isinstance(range_config, Mapping):
            raise TypeError("range_analysis must be a mapping")
        if type(range_config.get("enabled", False)) is not bool:
            raise TypeError("range_analysis.enabled must be bool")
        self.range_analyzer = RangeAnalyzer(range_config.get("enabled", False),
                                            range_config.get("reservoir_size", 4096))
        self.coverage = {}
        self.diagnostics = {}
        self.stimulus = None

    @classmethod
    def from_params(cls, params):
        config = FixedPointConfig.from_params(params)
        range_config = parameter_value(params, "range_analysis", {})
        if params is not None:
            current = getattr(params, "_fixed_context", None)
            if isinstance(current, cls):
                settings = {"enabled": range_config} if isinstance(range_config, bool) else (range_config or {})
                same_config = current.config.to_dict() == config.to_dict()
                if (same_config and isinstance(settings, Mapping)
                        and current.range_enabled == settings.get("enabled", False)
                        and current.range_analyzer.reservoir_size == settings.get("reservoir_size", 4096)):
                    return current
        context = cls(config, range_config)
        if params is not None and not isinstance(params, Mapping):
            try:
                setattr(params, "_fixed_context", context)
            except (AttributeError, TypeError):
                pass
        return context

    @property
    def range_enabled(self):
        return self.range_analyzer.enabled

    def module_config(self, module_id):
        return self.config.module_config(module_id)

    def module_enabled(self, module_id):
        return self.config.enable_fixed and self.module_config(module_id).enabled

    def arithmetic_level(self, module_id):
        return self.module_config(module_id).arithmetic_level

    def format_for(self, module_id, role="output"):
        return self.module_config(module_id).format(role)

    def require_boundary(self, module_id, reason):
        if self.module_enabled(module_id) and self.arithmetic_level(module_id) == "operation":
            self.record_coverage(module_id, "unsupported", reason)
            raise NotImplementedError("operation fixed-point is not supported for " + module_id + ": " + reason)

    ensure_supported = require_boundary

    def record_coverage(self, module_id, level, detail=None):
        entry = self.coverage.setdefault(module_id, {"enabled": self.module_enabled(module_id),
                                                     "requested_level": self.arithmetic_level(module_id),
                                                     "levels": [], "details": []})
        cfg = self.module_config(module_id)
        entry["resolved_formats"] = {role: fmt.to_dict() for role, fmt in cfg.formats.items()}
        entry["options"] = cfg.options.copy()
        if level not in entry["levels"]:
            entry["levels"].append(level)
        if detail is not None and detail not in entry["details"]:
            entry["details"].append(str(detail))

    def observe(self, node, value):
        # 兼容早期接入者的 observe(value,node)，推荐 node 在前。
        if not isinstance(node, str) and isinstance(value, str):
            node, value = value, node
        if isinstance(value, FixedPointArray):
            value = value.dequantize()
        self.range_analyzer.observe(node, value)

    def _event(self, node):
        def collect(event):
            stats = self.diagnostics.setdefault(node, {"overflow_count": 0, "conversion_count": 0})
            stats["overflow_count"] += event.get("overflow_count", event.get("overflow", 0))
            stats["conversion_count"] += event.get("count", event.get("total", 0))
            if "intermediate_bits" in event:
                stats["max_intermediate_bits"] = max(stats.get("max_intermediate_bits", 0), event["intermediate_bits"])
                stats["declared_intermediate_bits"] = max(stats.get("declared_intermediate_bits", 0), event["intermediate_bound_bits"])
                stats["intermediate_backend"] = event["integer_backend"]
                stats["wide_division_count"] = stats.get("wide_division_count", 0) + event.get("operation_count", 0)
        return collect

    def _result(self, result, module_id, node, level):
        value = result.dequantize() if isinstance(result, FixedPointArray) else result
        self.observe(node, value)
        self.record_coverage(module_id, level)
        return value

    def quantize(self, value, module_id, role="output", node=None):
        node = node or module_id + "." + role
        self.observe(node + ".pre_quantization", value)
        if not self.module_enabled(module_id):
            self.record_coverage(module_id, "float")
            self.observe(node, value)
            return value.dequantize() if isinstance(value, FixedPointArray) else value
        result = ar.quantize(value, self.format_for(module_id, role), self._event(node))
        return self._result(result, module_id, node, "boundary")

    def _input(self, value, module_id, role="input", node=None):
        if isinstance(value, FixedPointArray):
            return value
        return ar.quantize(value, self.format_for(module_id, role), self._event(node or module_id + "." + role))

    def operand(self, value, module_id, role="input", node=None):
        """保留宽中间节点格式，供多步运算继续使用整数码值。"""
        node = node or module_id + "." + role
        self.observe(node + ".pre_quantization", value)
        result = ar.quantize(value, self.format_for(module_id, role), self._event(node))
        self.observe(node, result)
        return result

    def _binary(self, operation, a, b, module_id, role, node, a_role="input", b_role="input"):
        node = node or module_id + "." + operation
        self.observe(node + ".input_a", a)
        self.observe(node + ".input_b", b)
        if not self.module_enabled(module_id):
            av = a.dequantize() if isinstance(a, FixedPointArray) else a
            bv = b.dequantize() if isinstance(b, FixedPointArray) else b
            result = {"mul": np.multiply, "add": np.add, "div": np.divide}[operation](av, bv)
            return self._result(result, module_id, node, "float")
        if self.arithmetic_level(module_id) == "boundary":
            av = a.dequantize() if isinstance(a, FixedPointArray) else a
            bv = b.dequantize() if isinstance(b, FixedPointArray) else b
            result = {"mul": np.multiply, "add": np.add, "div": np.divide}[operation](av, bv)
            return self.quantize(result, module_id, role, node)
        result = getattr(ar, operation)(self._input(a, module_id, a_role, node=node + ".a"),
                                        self._input(b, module_id, b_role, node=node + ".b"),
                                        self.format_for(module_id, role), self._event(node))
        return self._result(result, module_id, node, "operation")

    def mul(self, a, b, module_id, role="output", node=None, a_role="input", b_role="input"):
        return self._binary("mul", a, b, module_id, role, node, a_role, b_role)

    def add(self, a, b, module_id, role="output", node=None, a_role="input", b_role="input"):
        return self._binary("add", a, b, module_id, role, node, a_role, b_role)

    def div(self, a, b, module_id, role="output", node=None, a_role="input", b_role="input"):
        return self._binary("div", a, b, module_id, role, node, a_role, b_role)

    def sum(self, value, module_id, axis=None, role="accumulator", node=None, input_role="input"):
        node = node or module_id + ".sum"
        self.observe(node + ".input", value)
        float_value = value.dequantize() if isinstance(value, FixedPointArray) else value
        if not self.module_enabled(module_id):
            return self._result(np.sum(float_value, axis=axis), module_id, node, "float")
        if self.arithmetic_level(module_id) == "boundary":
            return self.quantize(np.sum(float_value, axis=axis), module_id, role, node)
        result = ar.sum_values(self._input(value, module_id, input_role, node=node + ".input_quantized"),
                               self.format_for(module_id, role), axis, self._event(node))
        return self._result(result, module_id, node, "operation")

    def fft(self, x, module_id, inverse=False, axis=-1, norm=None, n=None,
            input_role="input", output_role="output"):
        node = module_id + (".ifft" if inverse else ".fft")
        self.observe(node + ".input", x)
        if not self.module_enabled(module_id):
            fn = np.fft.ifft if inverse else np.fft.fft
            value = x.dequantize() if isinstance(x, FixedPointArray) else x
            return self._result(fn(value, n=n, axis=axis, norm=norm), module_id, node, "float")
        cfg = self.module_config(module_id)
        if cfg.arithmetic_level == "boundary":
            value = self.quantize(x, module_id, input_role, node + ".input_quantized")
            fn = np.fft.ifft if inverse else np.fft.fft
            return self.quantize(fn(value, n=n, axis=axis, norm=norm), module_id, output_role, node)
        result = fixed_fft(self._input(x, module_id, input_role), cfg.format("state"), cfg.format("coefficient"),
                           cfg.format(output_role), inverse, axis, norm, n, cfg.get("stage_shifts", ()),
                           self._event(node), lambda stage, value: self.observe(node + "." + stage, value))
        result = self._result(result, module_id, node, "operation")
        transforms = self.coverage[module_id].setdefault("transforms", [])
        metadata = {"inverse": bool(inverse), "length": int(n or np.shape(x)[axis]), "norm": norm,
                    "stage_shifts": list(cfg.get("stage_shifts", ())), "external_gain_preserved": True,
                    "input_role": input_role, "output_role": output_role}
        if metadata not in transforms:
            transforms.append(metadata)
        return result

    def _filter(self, x, h, module_id, mode, correlation):
        node = module_id + (".correlate" if correlation else ".fir")
        self.observe(node + ".input", x)
        self.observe(node + ".coefficient", h)
        fn = np.correlate if correlation else np.convolve
        if not self.module_enabled(module_id):
            return self._result(fn(x, h, mode=mode), module_id, node, "float")
        cfg = self.module_config(module_id)
        if cfg.arithmetic_level == "boundary":
            x = self.quantize(x, module_id, "input", node + ".input_quantized")
            h = self.quantize(h, module_id, "coefficient", node + ".coefficient_quantized")
            return self.quantize(fn(x, h, mode=mode), module_id, "output", node)
        kernel = fixed_correlate if correlation else fixed_fir
        result = kernel(self._input(x, module_id), self._input(h, module_id, "coefficient"),
                        cfg.format("accumulator"), cfg.format("product"), mode, self._event(node))
        self.observe(node + ".accumulator", result)
        result = ar.requantize(result, cfg.format("output"), self._event(node + ".output"))
        return self._result(result, module_id, node, "operation")

    def fir(self, x, h, module_id, mode="full"):
        return self._filter(x, h, module_id, mode, False)

    def correlate(self, x, h, module_id, mode="valid"):
        return self._filter(x, h, module_id, mode, True)

    def angle(self, value, module_id, role="phase", node=None):
        node = node or module_id + ".angle"
        float_value = value.dequantize() if isinstance(value, FixedPointArray) else value
        if not self.module_enabled(module_id):
            return self._result(np.angle(float_value), module_id, node, "float")
        if self.arithmetic_level(module_id) == "boundary":
            return self.quantize(np.angle(float_value), module_id, role, node)
        from .kernels.phase import cordic_angle
        cfg = self.module_config(module_id)
        input_format = cfg.format("accumulator")
        state_format = cfg.formats.get("cordic_state") or FixedPointFormat(
            min(62, input_format.word_length + 3), input_format.fractional_length,
            rounding=input_format.rounding, saturation=input_format.saturation,
            overflow_policy=input_format.overflow_policy)
        result = cordic_angle(value, input_format, cfg.format(role),
                              iterations=cfg.get("cordic_iterations"), event=self._event(node), state_format=state_format)
        value = self._result(result, module_id, node, "operation")
        self.coverage[module_id].setdefault("derived_formats", {})["cordic_vector_state"] = state_format.to_dict()
        return value

    def phasor(self, theta, module_id, role="coefficient", node=None):
        node = node or module_id + ".phasor"
        float_theta = theta.dequantize() if isinstance(theta, FixedPointArray) else theta
        if not self.module_enabled(module_id):
            return self._result(np.exp(1j * np.asarray(float_theta)), module_id, node, "float")
        if self.arithmetic_level(module_id) == "boundary":
            return self.quantize(np.exp(1j * np.asarray(float_theta)), module_id, role, node)
        from .kernels.phase import cordic_phasor
        cfg = self.module_config(module_id)
        coefficient_format = cfg.format(role)
        state_format = cfg.formats.get("cordic_state") or FixedPointFormat(
            min(62, coefficient_format.word_length + 3), coefficient_format.fractional_length,
            rounding=coefficient_format.rounding, saturation=coefficient_format.saturation,
            overflow_policy=coefficient_format.overflow_policy)
        result = cordic_phasor(theta, coefficient_format, cfg.format("phase"),
                               iterations=cfg.get("cordic_iterations"), event=self._event(node), state_format=state_format)
        value = self._result(result, module_id, node, "operation")
        self.coverage[module_id].setdefault("derived_formats", {})["cordic_rotation_state"] = state_format.to_dict()
        return value

    def _integer_sqrt(self, codes, fractional_length, fmt, node):
        """完整Python整数中间精度；最终格式转换后才转回int64。"""
        exponent = 2 * fmt.fractional_length - fractional_length
        overflows = [0]
        def isqrt(code):
            numerator, denominator = int(code), 1
            if exponent >= 0:
                numerator <<= exponent
            else:
                denominator <<= -exponent
            root = math.isqrt(numerator // denominator)
            exact = root * root * denominator == numerator
            if fmt.rounding == "ceil":
                root += not exact
            elif fmt.rounding in {"nearest_even", "nearest_away"}:
                lhs = 4 * numerator
                rhs = denominator * (4 * root * root + 4 * root + 1)
                root += lhs > rhs or (lhs == rhs and (fmt.rounding == "nearest_away" or root % 2))
            if root < fmt.min_code or root > fmt.max_code:
                overflows[0] += 1
                if fmt.saturation:
                    root = min(max(root, fmt.min_code), fmt.max_code)
                elif fmt.overflow_policy == "error":
                    raise OverflowError("fixed-point sqrt overflow in " + fmt.q_format)
                else:
                    root %= 1 << fmt.word_length
                    if fmt.signed and root >= 1 << (fmt.word_length - 1):
                        root -= 1 << fmt.word_length
            return root
        result = np.frompyfunc(isqrt, 1, 1)(codes)
        result = np.asarray(result, dtype=np.int64)
        self._event(node)({"overflow_count": overflows[0], "count": int(result.size)})
        return FixedPointArray(result, fmt)

    def sqrt(self, value, module_id, role="output", node=None, input_role="input"):
        node = node or module_id + ".sqrt"
        float_value = value.dequantize() if isinstance(value, FixedPointArray) else value
        if not self.module_enabled(module_id):
            return self._result(np.sqrt(float_value), module_id, node, "float")
        if self.arithmetic_level(module_id) == "boundary":
            return self.quantize(np.sqrt(float_value), module_id, role, node)
        data = self._input(value, module_id, input_role)
        if data.imag is not None or np.any(data.real < 0):
            raise ValueError("operation sqrt requires nonnegative real inputs")
        result = self._integer_sqrt(data.real, data.format.fractional_length,
                                    self.format_for(module_id, role), node)
        return self._result(result, module_id, node, "operation")

    def magnitude(self, value, module_id, role="output", input_role="output", node=None):
        node = node or module_id + ".magnitude"
        self.observe(node + ".input", value)
        float_value = value.dequantize() if isinstance(value, FixedPointArray) else value
        if not self.module_enabled(module_id):
            return self._result(np.abs(float_value), module_id, node, "float")
        if self.arithmetic_level(module_id) == "boundary":
            return self.quantize(np.abs(float_value), module_id, role, node)
        data = self._input(value, module_id, input_role)
        real = data.real.astype(object)
        imag = 0 if data.imag is None else data.imag.astype(object)
        squared = real * real + imag * imag
        result = self._integer_sqrt(squared, 2 * data.format.fractional_length,
                                    self.format_for(module_id, role), node)
        output = self._result(result, module_id, node, "operation")
        self.coverage[module_id].setdefault("derived_formats", {})["magnitude_full_squared_sum"] = {
            "word_length": 2 * data.format.word_length + 1,
            "fractional_length": 2 * data.format.fractional_length,
            "signed": False, "implementation": "complete Python integer intermediate"}
        return output

    def matmul(self, a, b, module_id, role="matrix", node=None, a_role="input", b_role="input"):
        node = node or module_id + ".matmul"
        if not self.module_enabled(module_id) or self.arithmetic_level(module_id) == "boundary":
            av = a.dequantize() if isinstance(a, FixedPointArray) else a
            bv = b.dequantize() if isinstance(b, FixedPointArray) else b
            result = np.matmul(av, bv)
            return (self.quantize(result, module_id, role, node) if self.module_enabled(module_id)
                    else self._result(result, module_id, node, "float"))
        from .kernels.matrix import fixed_matmul
        result = fixed_matmul(self._input(a, module_id, a_role), self._input(b, module_id, b_role),
                              self.format_for(module_id, role), self._event(node))
        return self._result(result, module_id, node, "operation")

    def solve(self, a, b, module_id, role="matrix", node=None, a_role="matrix", b_role="matrix"):
        node = node or module_id + ".solve"
        if not self.module_enabled(module_id) or self.arithmetic_level(module_id) == "boundary":
            av = a.dequantize() if isinstance(a, FixedPointArray) else a
            bv = b.dequantize() if isinstance(b, FixedPointArray) else b
            result = np.linalg.solve(av, bv)
            return (self.quantize(result, module_id, role, node) if self.module_enabled(module_id)
                    else self._result(result, module_id, node, "float"))
        from .kernels.matrix import fixed_solve
        av, bv = self._input(a, module_id, a_role), self._input(b, module_id, b_role)
        vector_rhs = bv.real.ndim == 1
        if vector_rhs:
            bv = FixedPointArray(bv.real[:, None], bv.format, None if bv.imag is None else bv.imag[:, None])
        result = fixed_solve(av, bv,
                             self.format_for(module_id, role), self._event(node))
        if vector_rhs:
            result = FixedPointArray(result.real[:, 0], result.format,
                                     None if result.imag is None else result.imag[:, 0])
        return self._result(result, module_id, node, "operation")

    def pinv(self, a, module_id, role="matrix", node=None, input_role="input"):
        node = node or module_id + ".pinv"
        if not self.module_enabled(module_id) or self.arithmetic_level(module_id) == "boundary":
            av = a.dequantize() if isinstance(a, FixedPointArray) else a
            result = np.linalg.pinv(av)
            return (self.quantize(result, module_id, role, node) if self.module_enabled(module_id)
                    else self._result(result, module_id, node, "float"))
        from .kernels.matrix import fixed_pseudoinverse_full_rank
        result = fixed_pseudoinverse_full_rank(self._input(a, module_id, input_role),
                                              self.format_for(module_id, role), self._event(node))
        return self._result(result, module_id, node, "operation")

    def report(self):
        diagnostics = {node: {**stats, "overflow_rate": stats["overflow_count"] / stats["conversion_count"]
                                      if stats["conversion_count"] else 0.0}
                       for node, stats in self.diagnostics.items()}
        unvisited = [name for name in self.config.modules
                     if not any(module == name or module.startswith(name + ".") for module in self.coverage)]
        return {"config": self.config.to_dict(), "coverage": self.coverage,
                "unvisited_configuration": unvisited,
                "ranges": self.range_analyzer.report(), "diagnostics": diagnostics,
                "arithmetic_semantics": {"standard": "ISO/IEC TR 18037:2008 Clause 4",
                                         "claim": "Python arithmetic model; not compiler conformance",
                                         "signed_Q_integer_bits_include_sign": True,
                                         "native_overflow": "checked before int64 operations"}}
