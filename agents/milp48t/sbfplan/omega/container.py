"""The omega container: named arrays, canonical hash, `.npz` storage (design §4.1 table; Q58 E21, Q86).

omega is stored as `.npz` of named arrays (no pickled objects). Its hash (and the seal's MAC, M6) is the SHA-256 of,
per array name in sorted order: the name, the dtype string, the shape and the C-order bytes. Weeks of the burn-in-
spanning arrays run from index 0 = week -B_burn; ``meta_burn_in`` holds B_burn (phase-3 addition to the §4.1 table, so
that week keys and the X/W/z arrays can be indexed without the generator config). NaN marks an absent value and has no
other meaning, so every NaN of a float array is stored, and hashed, as the one quiet NaN of ``np.nan``
(0x7ff8000000000000): a NaN's sign or payload never changes the hash (DET-P2-2); every other bit is kept.
"""

import hashlib
import math
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from sbfplan.instance.schema import FrozenDict
from sbfplan.omega.seeds import SPAWN_BOUND  # a key component is one uint32 word of (27)


EVENT_FIELDS: tuple[tuple[str, str], ...] = (
    ("type", "int16"),
    ("block", "int16"),
    ("region", "int16"),
    ("counterpart", "int16"),
    ("target_kind", "int8"),
    ("target", "int32"),
    ("commodity", "int16"),
    ("onset", "float64"),
    ("duration", "float64"),
    ("severity", "float64"),
    ("rate", "float64"),
    ("lead", "float64"),  # (events, 6), NaN where not announced on that channel
    ("restoration", "int8"),
    ("T0", "float64"),
    ("tau_rho", "float64"),
)
KEY_FIELDS: tuple[tuple[str, str], ...] = (("key_ptr", "int64"), ("key", "int64"))  # CSR genealogical keys
# the scalar float64 columns of EVENT_FIELDS (``lead`` holds one column per channel), in EVENT_FIELDS order
EVENT_FLOATS: tuple[str, ...] = ("onset", "duration", "severity", "rate", "T0", "tau_rho")

# name -> dtype for every array of schema_version 1; event groups ev_, sh_ (plus sh_channel) and op_ share EVENT_FIELDS
ARRAY_DTYPES: dict[str, str] = {
    "meta_schema_version": "<U64",
    "meta_instance_hash": "<U64",
    "meta_instance_digest": "<U64",  # content digest of the instance omega was built on (V3; phase 3)
    "meta_generator_id": "<U64",
    "meta_split": "<U64",
    "meta_episode": "<U64",
    "meta_burn_in": "int64",
    "meta_mark_params": "<U4096",  # canonical JSON of the MarkParams the stored marks were built under (phase 3)
    "z_c": "int8",
    "z_p": "int8",
    "z_dyad": "int8",
    "X": "float64",
    "W": "float64",
    "U_decoy": "float64",
    "V_lead": "float64",
    "d": "float64",
    "eps_parts": "float64",
    "R_f": "float64",
    "R_osat": "float64",  # (O, T) OSAT restoration (13) of §4.4, stored like R_f (Q91; milestone M2, phase-3 addition)
    "alpha_bar": "float64",
    "sigma_scr": "float64",
    "wr_class": "int8",
    "adv_family": "int16",
    "adv_member": "int16",
    "blk_start": "int32",
    "blk_end": "int32",
    "sh_channel": "int8",
    # ----- milestone M3, generated omegas only (added to schema 1 without a bump, as R_osat was; design §12 M3 rows) --
    "z_c_own": "int8",  # (R, B+T+1) each region's own conflict chain (29), the region-unit labels of §5.2; trusted
    "dyad_regions": "int16",  # (D, 2) region indices (a, b) of each z_dyad row (SIMP-M2R3-05): Static.dyads
    # canonical JSON of what the observation wrapper needs of the generator, which omega's arrays do not give back:
    # P^yr (Pi^c of (48)), phi-bar per decoy-bearing channel ((49)) and whether d carries Xi (False on omega^0, (57))
    "meta_information_params": "<U4096",
}
for _g in ("ev", "sh", "op"):
    for _name, _dt in KEY_FIELDS + EVENT_FIELDS:
        ARRAY_DTYPES[f"{_g}_{_name}"] = _dt
# arrays only a generated omega (and its event-free twin (57)) stores: an injected list writes none, so its hash, the
# M1 golden files and the frozen cents do not move (design §12 M3 rows)
GENERATED_ONLY: frozenset[str] = frozenset({"z_c_own", "dyad_regions", "meta_information_params"})


def canonical_nan(a: np.ndarray) -> np.ndarray:
    """``a`` with every NaN replaced by ``np.nan`` (0x7ff8000000000000) if it is a float array holding one; else ``a``.

    Only NaN entries change (sign and payload dropped); every other value keeps its bits, signed zeros included. The
    caller's array is never modified.
    """
    if a.dtype.kind == "f":
        nan = np.isnan(a)
        if nan.any():
            return np.where(nan, a.dtype.type(np.nan), a)
    return a


def event_float(i: int, name: str, value: object) -> float:
    """A numeric event field in its canonical form: every NaN the one quiet NaN of ``np.nan``, -0.0 as 0.0.

    The one rule for the float columns of an event group (``EVENT_FLOATS``), used by the injected builder
    (``omega.injected``) and the generator (``disruption.events``) alike, so both store an event the same way. NaN means
    absent (§12 rows) and a signed zero is the same instant or value, so each realisation has one form in the ``ev_*``
    arrays and in the generator config (one omega hash, one ``generator_id``). ``i`` and ``name`` name the event row and
    the field in an error message.

    Raises:
        TypeError: if the value is not a real number (a bool, a string or None included).
        ValueError: if it is infinite, which the design gives no reading, or a real number beyond the float range (an
            integer such as 10**400), which a float can only read as an infinity (DOC-4).

    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"event {i}: {name} must be a real number, got {value!r}")
    try:
        v = float(value)
    except OverflowError as err:
        raise ValueError(f"event {i}: {name} lies beyond the float range; an event field is finite, or NaN") from err
    if math.isnan(v):
        return float(np.nan)
    if math.isinf(v):
        raise ValueError(f"event {i}: {name} is {v}; an event field is finite, or NaN where it may be absent")
    return v + 0.0  # -0.0 + 0.0 = +0.0


# the integer columns of EVENT_FIELDS, in EVENT_FIELDS order (``lead`` is float, written by the announcement stage)
EVENT_INTS: tuple[str, ...] = tuple(n for n, dt in EVENT_FIELDS if dt.startswith("int"))


def event_group(rows: Sequence[Mapping[str, object]], prefix: str = "ev") -> dict[str, np.ndarray]:
    """One event group of omega (``ev_*``, ``sh_*`` or ``op_*``) with its CSR keys, one row per mapping, in order.

    The one writer of an event group (SIMP-M2R2-01): the injected builder (``omega.injected.event_arrays``) and the
    generator (``disruption.events.event_arrays``, the shadow group Y through ``prefix='sh'``) both resolve their
    events to rows and write them here. A row maps ``key`` (the genealogical key, a tuple of ints), every integer
    column of ``EVENT_INTS`` and every float column of ``EVENT_FLOATS``; the float columns are stored by
    ``event_float`` (every NaN the one quiet NaN, -0.0 as 0.0); ``lead`` is NaN on every channel (the announcement
    stage writes the leads). The codes' meaning is checked by the one decoder, ``marks.read_events``, which the
    generator runs on the written group; this writes what it is given, within the dtypes.

    Raises:
        ValueError: if a key is empty or holds a component outside [0, 2**32), an integer column does not fit its
            dtype, or a float field is infinite (``event_float``).
        TypeError: if a key component or an integer column is not an integer, or a float field not a real number (a
            bool included; ``event_float``).

    """
    n = len(rows)
    out = empty_events(prefix, n)
    ints = {name: np.iinfo(np.dtype(dt)) for name, dt in EVENT_FIELDS if name in EVENT_INTS}
    keys: list[int] = []
    ptr = [0]
    for i, row in enumerate(rows):
        key = tuple(row["key"])
        if not key or not all(
            isinstance(k, (int, np.integer)) and not isinstance(k, (bool, np.bool_)) and 0 <= k < SPAWN_BOUND
            for k in key
        ):
            raise ValueError(f"event {i}: key components must be integers in [0, 2**32), got {key}")
        keys += [int(k) for k in key]
        ptr.append(len(keys))
        for name, info in ints.items():
            v = row[name]
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
                raise TypeError(f"event {i}: {name} must be an integer code, got {v!r}")
            if not info.min <= v <= info.max:
                raise ValueError(f"event {i}: {name} {v} does not fit {info.dtype}")
            out[f"{prefix}_{name}"][i] = v
        for name in EVENT_FLOATS:
            out[f"{prefix}_{name}"][i] = event_float(i, name, row[name])
    out[f"{prefix}_key_ptr"] = np.array(ptr, dtype=np.int64)
    out[f"{prefix}_key"] = np.array(keys, dtype=np.int64)
    return out


def chokepoint_region(inst, node: int) -> int:
    """m_q of an event on chokepoint ``node``: its first adjacent region, else the node's own region (§12).

    The one rule for a chokepoint target's region (SIMP-M2R2-12), read by the injected builder's default region and
    by the generator's weather and accident closures (``inst`` an ``instance.schema.Instance``).
    """
    adjacent = inst.chokepoint_adjacency.get(node, ())
    return int(adjacent[0]) if adjacent else int(inst.nodes[node].region)


def omega_hash(arrays: Mapping[str, np.ndarray]) -> str:
    """Canonical SHA-256 of an omega: per name in sorted order, name, dtype string, shape, C-order bytes (§4.1).

    NaN entries are hashed as ``np.nan`` (``canonical_nan``), as the container stores them.
    """
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = canonical_nan(np.asarray(arrays[name]))  # own shape: a 0-d scalar and a 1-element vector hash differently
        h.update(name.encode("utf-8") + b"\0")
        h.update(a.dtype.str.encode("ascii") + b"\0")
        h.update(repr(tuple(int(n) for n in a.shape)).encode("ascii") + b"\0")
        h.update(np.ascontiguousarray(a).tobytes(order="C"))
    return h.hexdigest()


def empty_events(prefix: str = "ev", n: int = 0) -> dict[str, np.ndarray]:
    """Arrays of an event group with ``n`` rows, filled with the 'absent' codes (-1, NaN) and empty keys."""
    out: dict[str, np.ndarray] = {
        f"{prefix}_key_ptr": np.zeros(n + 1, dtype=np.int64),
        f"{prefix}_key": np.zeros(0, dtype=np.int64),
    }
    for name, dt in EVENT_FIELDS:
        shape = (n, 6) if name == "lead" else (n,)
        fill = np.nan if dt == "float64" else -1
        out[f"{prefix}_{name}"] = np.full(shape, fill, dtype=dt)
    return out


def immutable_copy(a: np.ndarray) -> np.ndarray:
    """A read-only C-order copy of ``a`` over an immutable ``bytes`` buffer: its WRITEABLE flag can never be set again.

    The caller's array stays writable and is not shared. Used for omega's arrays and for the ``WeeklyMarks`` arrays,
    so neither the omega hash nor the marks digest can describe content that has since changed (V3).
    """
    return np.frombuffer(a.tobytes(order="C"), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True, eq=False)
class Omega:
    """One pre-sampled exogenous scenario (26): an immutable, picklable mapping of named arrays.

    The arrays are private copies over immutable buffers (the caller's arrays stay writable and unshared), held in a
    ``FrozenDict``, every NaN stored as ``np.nan`` (``canonical_nan``); the hash is recomputed from them on every
    access, so no cached identity can go stale.
    """

    arrays: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        frozen = {}
        for name, a in self.arrays.items():
            if name not in ARRAY_DTYPES:
                raise ValueError(f"unknown omega array {name!r}")
            if a.dtype != np.dtype(ARRAY_DTYPES[name]):
                raise ValueError(f"omega array {name!r} has dtype {a.dtype}, expected {ARRAY_DTYPES[name]}")
            frozen[name] = immutable_copy(canonical_nan(a))
        object.__setattr__(self, "arrays", FrozenDict(frozen))

    def __reduce__(self):
        """Pickle as the plain arrays; unpickling rebuilds (and re-validates and re-freezes) through the constructor."""
        return (Omega, (dict(self.arrays),))

    def __eq__(self, other: object) -> bool:
        """Two omegas are equal when their canonical hashes are (§4.1)."""
        return isinstance(other, Omega) and self.hash == other.hash

    def __hash__(self) -> int:
        return hash(self.hash)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]

    def __contains__(self, name: str) -> bool:
        return name in self.arrays

    @property
    def hash(self) -> str:
        """Canonical SHA-256 of the arrays (§4.1), recomputed on every access (no cache that could go stale)."""
        return omega_hash(self.arrays)

    @property
    def instance_hash(self) -> str:
        return str(self.arrays["meta_instance_hash"])

    @property
    def burn_in(self) -> int:
        return int(self.arrays["meta_burn_in"])

    @property
    def generator_id(self) -> str:
        """``meta_generator_id``: the generator config that drew this omega (§4.1 rule table)."""
        return str(self.arrays["meta_generator_id"])

    @property
    def generated(self) -> bool:
        """Whether the disruption generator drew this omega: it stores regime paths (``z_c`` has week columns, §4.1).

        An injected event list stores none (``z_c`` of shape (R, 0)); the event-free twin (57) of a generated omega
        keeps them.
        """
        z_c = self.arrays.get("z_c")
        return z_c is not None and z_c.ndim == 2 and z_c.shape[1] > 0

    @property
    def n_events(self) -> int:
        return int(self.arrays["ev_type"].shape[0])

    def event_key(self, i: int, prefix: str = "ev") -> tuple[int, ...]:
        """Genealogical key of event ``i`` (CSR rows of ``<prefix>_key``)."""
        ptr = self.arrays[f"{prefix}_key_ptr"]
        return tuple(int(v) for v in self.arrays[f"{prefix}_key"][ptr[i] : ptr[i + 1]])

    def replace(self, **arrays: np.ndarray) -> "Omega":
        """A new omega with some arrays replaced or added."""
        return Omega({**dict(self.arrays), **arrays})


def save_omega(omega: Omega, path: str | Path) -> None:
    """Store as `.npz` of named arrays (no pickles)."""
    np.savez(Path(path), **{k: np.asarray(v) for k, v in omega.arrays.items()})


def load_omega(path: str | Path) -> Omega:
    """Load an omega `.npz`; pickled objects are refused."""
    with np.load(Path(path), allow_pickle=False) as z:
        return Omega({k: z[k].copy() for k in z.files})
