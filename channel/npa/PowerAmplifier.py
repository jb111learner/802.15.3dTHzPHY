import copy
import numpy as np


def dbm_to_watt(power_dbm: float) -> float:
    return 1e-3 * 10 ** (power_dbm / 10)


def watt_to_dbm(power_watt: float) -> float:
    power_watt = max(float(power_watt), 1e-30)
    return 10 * np.log10(power_watt / 1e-3)


def compute_average_power_watt(x: np.ndarray, load_ohm: float = 1.0) -> float:
    return float(np.mean(np.abs(x) ** 2) / load_ohm)


def compute_papr_db(x: np.ndarray) -> float:
    avg_power = np.mean(np.abs(x) ** 2)
    peak_power = np.max(np.abs(x) ** 2)

    if avg_power <= 0:
        return 0.0

    return float(10 * np.log10(peak_power / avg_power))


class ModifiedRappPA:
    """
    Modified Rapp PA model.

    AM-AM:
        rho_out = G * rho / (1 + |G*rho / Vsat|^(2p))^(1/(2p))

    AM-PM:
        phi = A * rho^q1 / (1 + |rho / B|^q2)

    输出:
        z = rho_out * exp(j * (theta + phi))
    """

    def __init__(self, pa_params: dict):
        self.G = float(pa_params["pa_G"])
        self.Vsat = float(pa_params["pa_Vsat"])
        self.p = float(pa_params["pa_p"])

        self.A = float(pa_params["pa_A"])
        self.B = float(pa_params["pa_B"])
        self.q1 = float(pa_params["pa_q1"])
        self.q2 = float(pa_params["pa_q2"])

        self.phase_unit = pa_params.get("pa_phase_unit", "deg")
        self.load_ohm = float(pa_params.get("pa_load_ohm", 1.0))

        self.auto_input_scaling = bool(pa_params.get("pa_auto_input_scaling", True))
        self.input_power_dbm = float(pa_params.get("pa_input_power_dbm", -13.2))

    def _scale_to_target_input_power(self, x: np.ndarray):
        if not self.auto_input_scaling:
            return x, 1.0

        target_power_watt = dbm_to_watt(self.input_power_dbm)
        current_power_watt = compute_average_power_watt(x, self.load_ohm)

        if current_power_watt <= 0:
            return x, 1.0

        scale = np.sqrt(target_power_watt / current_power_watt)
        return x * scale, float(scale)

    def _am_am(self, rho: np.ndarray) -> np.ndarray:
        numerator = self.G * rho
        denominator = (1.0 + np.abs((self.G * rho) / self.Vsat) ** (2.0 * self.p)) ** (1.0 / (2.0 * self.p))
        return numerator / denominator

    def _am_pm(self, rho: np.ndarray) -> np.ndarray:
        phi = self.A * (rho ** self.q1) / (1.0 + np.abs(rho / self.B) ** self.q2)

        if self.phase_unit.lower() in ["deg", "degree", "degrees"]:
            phi = np.deg2rad(phi)

        return phi

    def process_array(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.complex128)

        x_scaled, scale = self._scale_to_target_input_power(x)

        rho = np.abs(x_scaled)
        theta = np.angle(x_scaled)

        rho_out = self._am_am(rho)
        phi = self._am_pm(rho)

        y = rho_out * np.exp(1j * (theta + phi))

        diagnostics = {
            "pa_input_scaling": scale,
            "pa_target_input_power_dbm": self.input_power_dbm,
            "pa_input_power_watt": compute_average_power_watt(x_scaled, self.load_ohm),
            "pa_input_power_dbm": watt_to_dbm(compute_average_power_watt(x_scaled, self.load_ohm)),
            "pa_output_power_watt": compute_average_power_watt(y, self.load_ohm),
            "pa_output_power_dbm": watt_to_dbm(compute_average_power_watt(y, self.load_ohm)),
            "pa_actual_gain_db": watt_to_dbm(compute_average_power_watt(y, self.load_ohm))
                              - watt_to_dbm(compute_average_power_watt(x_scaled, self.load_ohm)),
            "pa_input_papr_db": compute_papr_db(x_scaled),
            "pa_output_papr_db": compute_papr_db(y),
            "pa_model": "modified_rapp",
        }

        return y, diagnostics

    def process_signal_dict(self, signal_dict: dict):
        out_dict = copy.deepcopy(signal_dict)

        x = out_dict["signal_stream"]
        y, diagnostics = self.process_array(x)

        out_dict["signal_stream"] = y
        out_dict["pa_diagnostics"] = diagnostics

        return out_dict