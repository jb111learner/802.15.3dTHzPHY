import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import erfc
from params.PHYParams import PHYParams
from transmitter.Modulator import THzModulator
from receiver.DeModulator import THzDemodulator


# ===================== Matplotlib 设置 =====================
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
mpl.rcParams["axes.formatter.use_mathtext"] = False


# ===================== Q 函数 & 理论 BER（QAM/PSK） =====================
def qfunc(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * erfc(x / np.sqrt(2.0))


def ber_theory_bpsk(ebn0_db):
    ebn0 = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    return qfunc(np.sqrt(2.0 * ebn0))


def ber_theory_qpsk(ebn0_db):
    return ber_theory_bpsk(ebn0_db)


def ber_theory_mqam_gray_approx(ebn0_db, M: int):
    k = np.log2(M)
    ebn0 = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    a = np.sqrt(3.0 * k / (M - 1.0) * ebn0)
    Q = qfunc(a)
    term1 = (4.0 / k) * (1.0 - 1.0 / np.sqrt(M)) * Q
    term2 = (4.0 / k) * (1.0 - 1.0 / np.sqrt(M)) ** 2 * (Q ** 2)
    pb = term1 - term2
    return np.clip(pb, 0.0, 0.5)


def ber_theory_qam_psk(mod_name: str, ebn0_db):
    if mod_name == "BPSK":
        return ber_theory_bpsk(ebn0_db)
    if mod_name == "QPSK":
        return ber_theory_qpsk(ebn0_db)
    if mod_name == "16QAM":
        return ber_theory_mqam_gray_approx(ebn0_db, 16)
    if mod_name == "64QAM":
        return ber_theory_mqam_gray_approx(ebn0_db, 64)
    if mod_name == "256QAM":
        return ber_theory_mqam_gray_approx(ebn0_db, 256)
    raise ValueError(f"Unknown QAM/PSK modulation: {mod_name}")


# ===================== 噪声：Eb/N0 -> sigma =====================
def ebn0_to_sigma(ebn0_db: float, Eb: float) -> float:
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    N0 = Eb / ebn0
    return np.sqrt(N0 / 2.0)


def add_awgn(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    n = sigma * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    return x + n


# ===================== Mod 配置 =====================
QAM_PSK_MODS = {
    "BPSK":   {"MOD": "QAM", "MCS": 1, "NCBPS": 1, "k": 1, "M": 2,   "ebn0": list(range(0, 11)), "n_bits": 2_000_000},
    "QPSK":   {"MOD": "QAM", "MCS": 1, "NCBPS": 2, "k": 2, "M": 4,   "ebn0": list(range(0, 11)), "n_bits": 2_000_000},
    "16QAM":  {"MOD": "QAM", "MCS": 1, "NCBPS": 4, "k": 4, "M": 16,  "ebn0": list(range(0, 16)), "n_bits": 2_000_000},
    "64QAM":  {"MOD": "QAM", "MCS": 1, "NCBPS": 6, "k": 6, "M": 64,  "ebn0": list(range(5, 26)), "n_bits": 10_000_000},
    "256QAM": {"MOD": "QAM", "MCS": 1, "NCBPS": 8, "k": 8, "M": 256, "ebn0": list(range(5, 26)), "n_bits": 10_000_000},
}

APSK_MODS = {
    "16APSK": {
        "MOD": "APSK", "MCS": 1, "NCBPS": 4, "k": 4, "M": 16,
        "ebn0": list(range(0, 16)), "n_bits": 2_000_000,
        "APSK_M": 16,
        "APSK_RINGS": [(4, 1.0), (12, 2.85)],
        "APSK_PHASE_OFFSETS": [0.0, 0.0],
    },
    "32APSK": {
        "MOD": "APSK", "MCS": 1, "NCBPS": 5, "k": 5, "M": 32,
        "ebn0": list(range(2, 21)), "n_bits": 5_000_000,
        "APSK_M": 32,
        "APSK_RINGS": [(4, 1.0), (12, 2.84), (16, 5.27)],
        "APSK_PHASE_OFFSETS": [0.0, 0.0, 0.0],
    },
    "64APSK": {
        "MOD": "APSK", "MCS": 1, "NCBPS": 6, "k": 6, "M": 64,
        "ebn0": list(range(5, 26)), "n_bits": 10_000_000,
        "APSK_M": 64,
        "APSK_RINGS": [(4, 1.0), (12, 2.73), (20, 4.52), (28, 6.31)],
        "APSK_PHASE_OFFSETS": [0.0, 0.0, 0.0, 0.0],
    },
}


def apply_params(params: PHYParams, cfg: dict):
    params.update(MCS=cfg["MCS"], NCBPS=cfg["NCBPS"])
    if cfg.get("MOD") is not None:
        params.update(MOD=cfg["MOD"])

    if str(cfg.get("MOD", "")).upper().startswith("APSK"):
        params.update(
            APSK_M=int(cfg["APSK_M"]),
            APSK_RINGS=cfg["APSK_RINGS"],
            APSK_PHASE_OFFSETS=cfg["APSK_PHASE_OFFSETS"],
        )


def run_one(cfg: dict, mod_name: str, seed: int = 1, verbose: bool = True):
    k = int(cfg["k"])
    ebn0_db = np.array(cfg["ebn0"], dtype=float)
    n_bits = int(cfg["n_bits"])
    n_bits = (n_bits // k) * k

    params = PHYParams()
    apply_params(params, cfg)

    mod = THzModulator(params)
    demod = THzDemodulator(params)

    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)

    tx = mod.modulate(bits)
    Es = float(np.mean(np.abs(tx) ** 2))
    Eb = Es / k

    if verbose:
        print(f"\n=== {mod_name} ===")
        print(f"k={k}, Es(mean)={Es:.6f}, Eb={Eb:.6f}, bits={n_bits}")

    ber_sim = []
    for x in ebn0_db:
        sigma = ebn0_to_sigma(float(x), Eb)
        rx = add_awgn(tx, sigma, rng)
        llr = demod.demodulate(rx, sigma)
        hard = demod.llr_to_bits(llr)
        err = int(np.sum(bits != hard[:n_bits]))
        ber = err / n_bits
        ber_sim.append(ber)
        if verbose:
            print(f"Eb/N0={int(x):>3} dB  sigma={sigma:.3e}  BER={ber:.6e}  errors={err}")

    ber_sim = np.array(ber_sim, dtype=float)
    esn0_db = ebn0_db + 10.0 * np.log10(k)

    return {
        "mod": mod_name,
        "k": k,
        "Es": Es,
        "Eb": Eb,
        "ebn0_db": ebn0_db,
        "esn0_db": esn0_db,
        "ber_sim": ber_sim,
    }


def plot_qam_psk(results):
    # 图A：BER vs Eb/N0（QAM/PSK）
    plt.figure(figsize=(11, 7))
    for r in results:
        ebn0 = r["ebn0_db"]
        sim = r["ber_sim"]
        th = ber_theory_qam_psk(r["mod"], ebn0)
        plt.semilogy(ebn0, sim, marker="o", linestyle="-", label=f"{r['mod']} Sim")
        plt.semilogy(ebn0, th, linestyle="--", label=f"{r['mod']} Theory")

    plt.grid(True, which="both")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BER")
    plt.title("QAM/PSK: BER vs Eb/N0 (Simulation vs Theory)")
    plt.ylim(1e-6, 5e-1)
    plt.legend(ncol=2)
    plt.show()

    # 图B：BER vs Es/N0（QAM/PSK）
    plt.figure(figsize=(11, 7))
    for r in results:
        esn0 = r["esn0_db"]
        sim = r["ber_sim"]
        # 理论本质还是 Eb/N0 的函数，这里只是把横轴换成 Es/N0（点对点对应即可）
        th = ber_theory_qam_psk(r["mod"], r["ebn0_db"])
        plt.semilogy(esn0, sim, marker="o", linestyle="-", label=f"{r['mod']} Sim")
        plt.semilogy(esn0, th, linestyle="--", label=f"{r['mod']} Theory")

    plt.grid(True, which="both")
    plt.xlabel("Es/N0 (dB)")
    plt.ylabel("BER")
    plt.title("QAM/PSK: BER vs Es/N0 (Simulation vs Theory)")
    plt.ylim(1e-6, 5e-1)
    plt.legend(ncol=2)
    plt.show()


def plot_apsk(results):
    # 图C：BER vs Eb/N0（APSK）
    plt.figure(figsize=(11, 7))
    for r in results:
        plt.semilogy(r["ebn0_db"], r["ber_sim"], marker="o", linestyle="-", label=f"{r['mod']} Sim")

    plt.grid(True, which="both")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BER")
    plt.title("APSK: BER vs Eb/N0 (Simulation)")
    plt.ylim(1e-6, 5e-1)
    plt.legend(ncol=1)
    plt.show()

    # 图D：BER vs Es/N0（APSK）
    plt.figure(figsize=(11, 7))
    for r in results:
        plt.semilogy(r["esn0_db"], r["ber_sim"], marker="o", linestyle="-", label=f"{r['mod']} Sim")

    plt.grid(True, which="both")
    plt.xlabel("Es/N0 (dB)")
    plt.ylabel("BER")
    plt.title("APSK: BER vs Es/N0 (Simulation)")
    plt.ylim(1e-6, 5e-1)
    plt.legend(ncol=1)
    plt.show()


def main(seed: int = 1):
    # 先跑 QAM/PSK
    qam_psk_names = ["BPSK", "QPSK", "16QAM", "64QAM", "256QAM"]
    qam_psk_res = []
    for name in qam_psk_names:
        qam_psk_res.append(run_one(QAM_PSK_MODS[name], name, seed=seed, verbose=True))

    # 再跑 APSK
    apsk_names = ["16APSK", "32APSK", "64APSK"]
    apsk_res = []
    for name in apsk_names:
        apsk_res.append(run_one(APSK_MODS[name], name, seed=seed, verbose=True))

    # 分开画图
    plot_qam_psk(qam_psk_res)
    plot_apsk(apsk_res)


if __name__ == "__main__":
    main(seed=1)
