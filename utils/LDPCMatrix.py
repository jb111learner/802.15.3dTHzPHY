from __future__ import annotations

from functools import lru_cache

import numpy as np


IEEE802153D_1440_N = 1440
RATE11_MINUS_LITERAL = "minus_literal"
RATE11_PLUS_SYSTEMATIC_CANDIDATE = "plus_systematic_candidate"
RATE11_DIRECTIONS = (RATE11_MINUS_LITERAL, RATE11_PLUS_SYSTEMATIC_CANDIDATE)

_IEEE802153D_1440_TABLES = {
    "14/15": {
        "k": 1344,
        "m": 96,
        "base": (
            (0, 1, 4),
            (32, 34, 39),
            (64, 70, 78),
            (8, 18, 95),
            (31, 42, 54),
            (63, 76, 91),
            (14, 45, 94),
            (30, 47, 83),
            (17, 62, 80),
            (28, 48, 82),
            (22, 60, 81),
            (27, 49, 84),
            (7, 53, 77),
            (19, 44, 85),
            (6, 46, 75),
        ),
    },
    "11/15": {
        "k": 1056,
        "m": 384,
        "base": (
            (96, 193, 4),
            (34, 320, 135),
            (352, 70, 270),
            (104, 306, 287),
            (31, 234, 150),
            (159, 364, 91),
            (302, 45, 286),
            (126, 239, 71),
            (17, 158, 272),
            (28, 336, 178),
            (214, 60, 369),
            (219, 145, 372),
            (7, 245, 173),
            (19, 140, 373),
            (6, 238, 363),
        ),
    },
}


def normalize_ieee802153d_rate(rate: str) -> str:
    rate = str(rate)
    if rate not in _IEEE802153D_1440_TABLES:
        raise ValueError("ldpc_standard_rate must be '14/15' or '11/15'")
    return rate


def normalize_ieee802153d_rate11_direction(direction: str | None) -> str:
    if direction is None:
        return RATE11_MINUS_LITERAL

    direction = str(direction).lower()
    aliases = {
        "minus": RATE11_MINUS_LITERAL,
        "minus_literal": RATE11_MINUS_LITERAL,
        "literal": RATE11_MINUS_LITERAL,
        "plus": RATE11_PLUS_SYSTEMATIC_CANDIDATE,
        "plus_systematic_candidate": RATE11_PLUS_SYSTEMATIC_CANDIDATE,
        "systematic_candidate": RATE11_PLUS_SYSTEMATIC_CANDIDATE,
    }
    if direction not in aliases:
        raise ValueError(
            "ldpc_rate11_direction must be 'minus_literal' or "
            "'plus_systematic_candidate'"
        )
    return aliases[direction]


def ieee802153d_rate11_direction_note(direction: str) -> str:
    direction = normalize_ieee802153d_rate11_direction(direction)
    if direction == RATE11_MINUS_LITERAL:
        return (
            "minus_literal follows the literal Equation 13-1 derivation "
            "i = 96*g + ((r0 - s) % 96); H[:,1056:1440] has rank 383, "
            "so standard c=[i,p] systematic encoding is not available."
        )
    return (
        "plus_systematic_candidate uses i = 96*g + ((r0 + s) % 96); "
        "H[:,1056:1440] is full rank and supports c=[i,p] systematic "
        "encoding. This is a candidate mode pending advisor or IEEE errata "
        "confirmation, not a confirmed standard correction."
    )


def ieee802153d_standard_confirmation_status(rate: str, rate11_direction: str | None = None) -> str:
    rate = normalize_ieee802153d_rate(rate)
    if rate == "14/15":
        return "standard_systematic_encoding_confirmed_by_rank_checks"

    direction = normalize_ieee802153d_rate11_direction(rate11_direction)
    if direction == RATE11_MINUS_LITERAL:
        return "minus_literal_equation_conflicts_with_systematic_encoding"
    return "plus_systematic_candidate_pending_advisor_or_ieee_errata_confirmation"


def ieee802153d_1440_dimensions(rate: str) -> tuple[int, int, int]:
    rate = normalize_ieee802153d_rate(rate)
    spec = _IEEE802153D_1440_TABLES[rate]
    n = IEEE802153D_1440_N
    k = int(spec["k"])
    return n, k, n - k


@lru_cache(maxsize=None)
def build_ieee802153d_1440_h(
    rate: str,
    rate11_direction: str = RATE11_MINUS_LITERAL,
) -> np.ndarray:
    """
    Build IEEE 802.15.3d / IEEE 802.15.3-2023 THz-SC mandatory LDPC H.

    The entries are generated from Table 12-17 + Equation 12-6 for rate 14/15,
    and Table 13-10 + Equation 13-1 for rate 11/15.

    For rate 11/15, Table 13-10 has been manually checked. The
    minus_literal direction follows Equation 13-1 literally, but conflicts
    with standard c=[i,p] systematic encodability because Hp rank is 383.
    plus_systematic_candidate gives full-rank Hp and is kept as an explicit
    candidate mode pending advisor or IEEE errata confirmation.
    """
    rate = normalize_ieee802153d_rate(rate)
    if rate == "11/15":
        return build_ieee802153d_1440_h_rate11_direction(rate11_direction)

    spec = _IEEE802153D_1440_TABLES[rate]
    m = int(spec["m"])
    base = spec["base"]
    h = np.zeros((m, IEEE802153D_1440_N), dtype=np.uint8)

    for j in range(IEEE802153D_1440_N):
        b = j % 15
        s = j // 15
        for row in base[b]:
            i = (row - s) % 96
            h[i, j] = 1

    h.setflags(write=False)
    return h


@lru_cache(maxsize=None)
def build_ieee802153d_1440_h_rate11_direction(
    direction: str = RATE11_MINUS_LITERAL,
) -> np.ndarray:
    direction = normalize_ieee802153d_rate11_direction(direction)

    spec = _IEEE802153D_1440_TABLES["11/15"]
    base = spec["base"]
    h = np.zeros((int(spec["m"]), IEEE802153D_1440_N), dtype=np.uint8)

    for j in range(IEEE802153D_1440_N):
        b = j % 15
        s = j // 15
        for row in base[b]:
            g = row // 96
            r0 = row % 96
            shifted = (
                (r0 - s) % 96
                if direction == RATE11_MINUS_LITERAL
                else (r0 + s) % 96
            )
            h[96 * g + shifted, j] = 1

    h.setflags(write=False)
    return h


def gf2_rank(matrix: np.ndarray) -> int:
    a = (np.asarray(matrix, dtype=np.uint8) & 1).copy()
    rows, cols = a.shape
    rank = 0

    for col in range(cols):
        pivot_rows = np.flatnonzero(a[rank:, col])
        if pivot_rows.size == 0:
            continue

        pivot = rank + int(pivot_rows[0])
        if pivot != rank:
            a[[rank, pivot]] = a[[pivot, rank]]

        rows_to_clear = np.flatnonzero(a[:, col])
        rows_to_clear = rows_to_clear[rows_to_clear != rank]
        if rows_to_clear.size:
            a[rows_to_clear] ^= a[rank]

        rank += 1
        if rank == rows:
            break

    return int(rank)


def gf2_pivot_columns(matrix: np.ndarray, column_order: np.ndarray | None = None) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.uint8) & 1
    if column_order is None:
        column_order = np.arange(matrix.shape[1], dtype=np.int32)
    else:
        column_order = np.asarray(column_order, dtype=np.int32)

    a = matrix[:, column_order].copy()
    rows, cols = a.shape
    rank = 0
    pivots: list[int] = []

    for local_col in range(cols):
        pivot_rows = np.flatnonzero(a[rank:, local_col])
        if pivot_rows.size == 0:
            continue

        pivot = rank + int(pivot_rows[0])
        if pivot != rank:
            a[[rank, pivot]] = a[[pivot, rank]]

        rows_to_clear = np.flatnonzero(a[:, local_col])
        rows_to_clear = rows_to_clear[rows_to_clear != rank]
        if rows_to_clear.size:
            a[rows_to_clear] ^= a[rank]

        pivots.append(int(column_order[local_col]))
        rank += 1
        if rank == rows:
            break

    return np.asarray(pivots, dtype=np.int32)


def gf2_inverse(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.uint8) & 1
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("GF(2) inverse requires a square matrix")

    n = rows
    a = np.concatenate([matrix.copy(), np.eye(n, dtype=np.uint8)], axis=1)
    rank = 0

    for col in range(n):
        pivot_rows = np.flatnonzero(a[rank:, col])
        if pivot_rows.size == 0:
            raise ValueError("Matrix is singular over GF(2)")

        pivot = rank + int(pivot_rows[0])
        if pivot != rank:
            a[[rank, pivot]] = a[[pivot, rank]]

        rows_to_clear = np.flatnonzero(a[:, col])
        rows_to_clear = rows_to_clear[rows_to_clear != rank]
        if rows_to_clear.size:
            a[rows_to_clear] ^= a[rank]

        rank += 1

    return a[:, n:].astype(np.uint8, copy=False)


def validate_ldpc_h(
    h: np.ndarray,
    n: int,
    k: int,
    column_weight: int = 3,
) -> dict:
    h = np.asarray(h)
    expected_shape = (n - k, n)
    if h.shape != expected_shape:
        raise ValueError(f"LDPC H shape must be {expected_shape}, got {h.shape}")

    if not np.array_equal(h, h.astype(np.uint8) & 1):
        raise ValueError("LDPC H must contain only 0/1 elements")

    col_weights = np.sum(h, axis=0)
    if not np.all(col_weights == column_weight):
        raise ValueError(f"LDPC H column weight must be {column_weight}")

    rank = gf2_rank(h)
    if rank != n - k:
        raise ValueError(f"LDPC H rank must be {n - k}, got {rank}")

    return {
        "shape": h.shape,
        "column_weight": int(column_weight),
        "rank": int(rank),
        "ones": int(np.sum(h)),
    }


@lru_cache(maxsize=None)
def ieee802153d_1440_matrix_metadata(
    rate: str,
    rate11_direction: str = RATE11_MINUS_LITERAL,
) -> dict:
    rate = normalize_ieee802153d_rate(rate)
    n, k, _ = ieee802153d_1440_dimensions(rate)
    h = build_ieee802153d_1440_h(rate, rate11_direction)
    metadata = validate_ldpc_h(h, n, k)
    metadata["rate11_direction"] = (
        normalize_ieee802153d_rate11_direction(rate11_direction)
        if rate == "11/15"
        else None
    )
    metadata["rate11_direction_note"] = (
        ieee802153d_rate11_direction_note(rate11_direction)
        if rate == "11/15"
        else None
    )
    return metadata


@lru_cache(maxsize=None)
def ieee802153d_1440_encoder_matrices(
    rate: str,
    rate11_direction: str = RATE11_MINUS_LITERAL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    rate = normalize_ieee802153d_rate(rate)
    h = build_ieee802153d_1440_h(rate, rate11_direction)
    n, k, r = ieee802153d_1440_dimensions(rate)

    info_positions = np.arange(k, dtype=np.int32)
    parity_positions = np.arange(k, n, dtype=np.int32)
    hp_rank = gf2_rank(h[:, parity_positions])
    if hp_rank != r:
        direction_note = (
            " "
            + ieee802153d_rate11_direction_note(rate11_direction)
            if rate == "11/15"
            else ""
        )
        raise NotImplementedError(
            "IEEE 802.15.3d LDPC standard systematic encoding requires "
            f"H[:, {k}:{n}] to be full-rank over GF(2); got rank {hp_rank}/{r}. "
            "Automatic parity-column replacement is intentionally disabled."
            f"{direction_note}"
        )

    h_info = h[:, info_positions]
    h_parity = h[:, parity_positions]
    h_parity_inv = gf2_inverse(h_parity)
    parity_matrix = ((
        h_info.T.astype(np.uint16) @ h_parity_inv.T.astype(np.uint16)
    ) & 1).astype(np.uint8, copy=False)

    natural_prefix = bool(np.array_equal(info_positions, np.arange(k, dtype=np.int32)))
    for arr in (info_positions, parity_positions, h_parity_inv, parity_matrix):
        arr.setflags(write=False)

    return (
        info_positions,
        parity_positions,
        h_parity_inv,
        parity_matrix,
        natural_prefix,
    )


def _syndrome_roundtrip_possible(h: np.ndarray, k: int, r: int) -> bool:
    h_info = h[:, :k]
    h_parity = h[:, k : k + r]
    if gf2_rank(h_parity) != r:
        return False

    h_parity_inv = gf2_inverse(h_parity)
    parity_matrix = ((h_info.T.astype(np.uint16) @ h_parity_inv.T.astype(np.uint16)) & 1).astype(np.uint8)
    rng = np.random.default_rng(1530)
    info = rng.integers(0, 2, size=k, dtype=np.uint8)
    parity = ((info.astype(np.uint16) @ parity_matrix.astype(np.uint16)) & 1).astype(np.uint8)
    codeword = np.concatenate([info, parity])
    syndrome = (h.astype(np.uint16) @ codeword.astype(np.uint16)) & 1
    return bool(np.all(syndrome == 0))


def diagnose_ieee802153d_1440_rate11_direction(direction: str) -> dict:
    direction = normalize_ieee802153d_rate11_direction(direction)
    h = build_ieee802153d_1440_h_rate11_direction(direction)
    n, k, r = ieee802153d_1440_dimensions("11/15")
    col_weights = np.sum(h, axis=0).astype(np.int32)
    h_rank = gf2_rank(h)
    hp_rank = gf2_rank(h[:, k:n])

    return {
        "direction": direction,
        "direction_note": ieee802153d_rate11_direction_note(direction),
        "shape": h.shape,
        "rank_h": int(h_rank),
        "rank_hp": int(hp_rank),
        "column_weight_min": int(np.min(col_weights)),
        "column_weight_max": int(np.max(col_weights)),
        "column_weight_unique": [int(x) for x in np.unique(col_weights)],
        "first_k_systematic_possible": bool(hp_rank == r),
        "syndrome_roundtrip": _syndrome_roundtrip_possible(h, k, r),
    }


def diagnose_ieee802153d_1440_rate11_directions() -> dict:
    return {
        RATE11_MINUS_LITERAL: diagnose_ieee802153d_1440_rate11_direction(RATE11_MINUS_LITERAL),
        RATE11_PLUS_SYSTEMATIC_CANDIDATE: diagnose_ieee802153d_1440_rate11_direction(
            RATE11_PLUS_SYSTEMATIC_CANDIDATE
        ),
    }
