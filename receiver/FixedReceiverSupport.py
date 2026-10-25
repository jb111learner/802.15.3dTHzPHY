"""接收端定点接口；浮点关闭时不改变原数值路径。"""

from fixedpoint.mixin import FixedPointMixin


class FixedReceiverMixin(FixedPointMixin):
    def _record_length_event(self, stage, actual_length, expected_length):
        context = getattr(self, "_fixed_context", None)
        if context is None:
            return
        analysis = context.config.enable_fixed or context.range_enabled or context.stimulus is not None
        if analysis and int(actual_length) != int(expected_length):
            if not hasattr(context, "length_events"):
                context.length_events = []
            context.length_events.append({
                "stage": stage,
                "actual_length": int(actual_length),
                "expected_length": int(expected_length),
                "padded_count": max(0, int(expected_length) - int(actual_length)),
                "truncated_count": max(0, int(actual_length) - int(expected_length)),
            })

    def _fixed_enabled(self, module_id=None):
        context = getattr(self, "fixed_context", None)
        return bool(context is not None and context.module_enabled(
            module_id or self.fixed_module_id
        ))

    def _boundary_only(self, reason, module_id=None):
        context = getattr(self, "fixed_context", None)
        module_id = module_id or self.fixed_module_id
        if self._fixed_enabled(module_id) and context.arithmetic_level(module_id) == "operation":
            context.record_coverage(module_id, "unsupported", reason)
            raise NotImplementedError(f"{module_id}: {reason}，当前仅支持 boundary")

    def _watch(self, value, node):
        context = getattr(self, "fixed_context", None)
        if context is not None:
            context.observe(node, value)
        return value

    def _boundary_dict_call(self, function, data_dict, *args, **kwargs):
        self._boundary_only("adaptive matrix/filter fitting has no internal fixed implementation")
        if self._fixed_enabled():
            data_dict = dict(data_dict)
            data_dict["signal_stream"] = self._fq(data_dict["signal_stream"], role="input")
        else:
            self._watch(data_dict["signal_stream"], self.fixed_module_id + ".input")
        result = function(data_dict, *args, **kwargs)
        if self._fixed_enabled():
            result = dict(result)
            result["signal_stream"] = self._fq(result["signal_stream"])
            self.fixed_context.record_coverage(self.fixed_module_id, "boundary", "float adaptive algorithm with input/output quantization")
        else:
            self._watch(result["signal_stream"], self.fixed_module_id + ".output")
        return result
