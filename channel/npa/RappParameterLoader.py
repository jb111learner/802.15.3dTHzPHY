import os
import numpy as np
import pandas as pd


class RappParameterLoader:
    """
    Load modified Rapp PA model parameters from the paper's open dataset.

    Expected files:
        AMAM_Rapp_model_parameters.csv
        AMPM_Rapp_model_parameters.csv
    """

    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir

    @staticmethod
    def _read_csv_auto(path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Rapp parameter file not found: {path}")

        # sep=None 可以自动兼容逗号、Tab 等分隔符
        df = pd.read_csv(path, sep=None, engine="python")
        df.columns = [str(c).strip() for c in df.columns]

        return df

    @staticmethod
    def _select_frequency_row(df: pd.DataFrame, fc_ghz: float, nearest: bool = False) -> pd.Series:
        if "fc" not in df.columns:
            raise KeyError(f"CSV must contain 'fc' column, got columns: {list(df.columns)}")

        fc_values = df["fc"].astype(float).to_numpy()

        exact_indices = np.where(np.isclose(fc_values, fc_ghz, atol=1e-9))[0]

        if len(exact_indices) > 0:
            return df.iloc[int(exact_indices[0])]

        if nearest:
            nearest_index = int(np.argmin(np.abs(fc_values - fc_ghz)))
            return df.iloc[nearest_index]

        raise ValueError(
            f"Frequency {fc_ghz} GHz not found. "
            f"Available frequencies: {fc_values.tolist()}"
        )

    def load_params_for_fc(self, fc_ghz: float, nearest: bool = False) -> dict:
        amam_path = os.path.join(self.dataset_dir, "AMAM_Rapp_model_parameters.csv")
        ampm_path = os.path.join(self.dataset_dir, "AMPM_Rapp_model_parameters.csv")

        amam = self._read_csv_auto(amam_path)
        ampm = self._read_csv_auto(ampm_path)

        amam_row = self._select_frequency_row(amam, fc_ghz, nearest=nearest)
        ampm_row = self._select_frequency_row(ampm, fc_ghz, nearest=nearest)

        required_amam = ["G", "V_sat", "p"]
        required_ampm = ["A", "q_1", "B", "q_2"]

        for col in required_amam:
            if col not in amam.columns:
                raise KeyError(f"AMAM CSV missing column '{col}', got {list(amam.columns)}")

        for col in required_ampm:
            if col not in ampm.columns:
                raise KeyError(f"AMPM CSV missing column '{col}', got {list(ampm.columns)}")

        return {
            "pa_fc_GHz_loaded": float(amam_row["fc"]),

            # AM-AM
            "pa_G": float(amam_row["G"]),
            "pa_Vsat": float(amam_row["V_sat"]),
            "pa_p": float(amam_row["p"]),

            # AM-PM
            "pa_A": float(ampm_row["A"]),
            "pa_q1": float(ampm_row["q_1"]),
            "pa_B": float(ampm_row["B"]),
            "pa_q2": float(ampm_row["q_2"]),
        }