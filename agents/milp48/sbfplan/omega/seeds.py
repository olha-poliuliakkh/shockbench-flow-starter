"""Seed rule (27): one 128-bit root per split and phase, keyed streams, the HMAC policy seed (design §4.1; Q65, Q87).

``u_{n,id,j} = SeedSequence(E_split, spawn_key=(n, id, *j))``: the episode index n first, then the stream id of
``omega.codes``, then the stream's key tuple (fixed arity per stream, except genealogical event keys). A key yields a
``Generator(PCG64DXSM(...))`` for a vector of draws, or ``SeedSequence(...).generate_state(k, uint64)`` mapped to (0, 1)
for a few scalars. Time keys use the calendar week plus the fixed burn-in offset B_burn, so keys stay non-negative.
Every spawn-key component (n, id and each j) must be below 2**32, one uint32 word each, or distinct keys could alias;
``check_entropy`` and ``check_episode`` apply the root's and the episode's bounds even where no key is drawn. The policy
seed's HMAC message is the canonical JSON array of its fields, so distinct fields never share a message. Nothing here
touches NumPy's global random state (B2).

Construction: NumPy assembles a spawned SeedSequence's entropy as the uint32 words of E_split (little-endian),
zero-padded to the pool size of 4 words when a spawn key follows, then the spawn key's words, and mixes that array into
its pool. E_split < 2**128 is at most 4 words, so ``SeedSequence(E_split, spawn_key=(n, id, *j))`` and the SeedSequence
of the flat uint32 array [E_split's 4 words, n, id, *j] with no spawn key mix the same words into the same pool and give
the same ``generate_state`` (bit-identity tested over thousands of keys). Every keyed draw (``generator``,
``uniforms``, ``KeyedStream``) is built the flat way, in ``keyed_sequence`` (the one constructor: a test that watches
the keys drawn watches it), which skips NumPy's per-component conversion of the spawn key; ``seed_sequence`` returns
the spawned form itself, the canonical object of (27). ``KeyedStream`` holds one stream of one (E_split, n), validated
once, for the samplers' many keys.
"""

import hashlib
import hmac
import json
import operator
import os
import secrets
from collections.abc import Sequence

import numpy as np


_ENTROPY_BITS = 128
SPAWN_BOUND = 1 << 32  # spawn-key components are single uint32 words (27)
_POOL_WORDS = 4  # NumPy's SeedSequence pool size: E_split is zero-padded to it before the spawn words
_WORD_MASK = SPAWN_BOUND - 1
_MASK_63 = (1 << 63) - 1
_BELOW_ONE = float(np.nextafter(1.0, 0.0))  # 1 - 2**-53, the largest double below 1
# Q95: route KeyedStream's uniforms (and stream 13's latent normals, ``disruption.regime._latent_normals``) through the
# bit-identical numba kernels of ``omega._kernels``, imported lazily on the first routed draw; without numba (or if the
# kernels fail their self-check) the draws take the reference path below. ``keys_seen`` and the V2 draw audit set it
# False, since they watch ``keyed_sequence``, which the routed draws skip. The environment variable ``SBF_KERNELS=0``
# switches it off in every process of a run (joblib's workers read it when they import this module): the draws are the
# same either way, so it only moves time (the before/after timings of the kernels).
KERNELS_VAR = "SBF_KERNELS"
KERNELS = os.environ.get(KERNELS_VAR, "1") != "0"


def new_entropy() -> int:
    """A fresh 128-bit root E_split (``secrets.randbits(128)``), held outside ``configs/`` (§4.1)."""
    return secrets.randbits(_ENTROPY_BITS)


def _component(value: object, what: str) -> int:
    """A key component as a Python int; rejects non-integers and bools (``TypeError``) and negatives (``ValueError``).

    A bool is refused although Python counts it as an int: ``episode=True`` would state the episode 1 under another
    name, one realisation with two configs (DC-7).
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"seed rule (27): {what} must be an integer, got the bool {value!r}")
    try:
        v = operator.index(value)  # int or NumPy integer; floats and strings raise TypeError
    except TypeError as err:
        raise TypeError(f"seed rule (27): {what} must be an integer, got {value!r}") from err
    if v < 0:
        raise ValueError(f"seed rule (27): {what} must be >= 0, got {v}")
    return v


def _spawn_component(value: object, what: str) -> int:
    """A spawn-key component of (27): an integer in [0, 2**32).

    NumPy's SeedSequence splits a wider int into several uint32 words, so ``(..., 2**32)`` and ``(..., 0, 1)`` would
    give the same stream; one word per component keeps distinct keys distinct.
    """
    v = _component(value, what)
    if v >= SPAWN_BOUND:
        raise ValueError(f"seed rule (27): {what} must be < 2**32 (one uint32 word per spawn-key component), got {v}")
    return v


def check_entropy(entropy: object) -> int:
    """E_split of (27) as a Python int, checked whether or not a key is drawn from it.

    Raises:
        ValueError: if the entropy is not in [0, 2**128).
        TypeError: if it is not an integer (a bool included).

    """
    ent = _component(entropy, "entropy")
    if ent >= 1 << _ENTROPY_BITS:  # (27): 128-bit roots; a wider root would alias keys of other (episode, stream) pairs
        raise ValueError("seed rule (27): entropy must be a 128-bit integer, 0 <= E_split < 2**128")
    return ent


def check_episode(episode: object) -> int:
    """The episode index n of (27) as a Python int: a spawn-key component, so in [0, 2**32).

    Raises:
        ValueError: if the episode is negative or >= 2**32.
        TypeError: if it is not an integer (a bool included).

    """
    return _spawn_component(episode, "episode")


def check_key(key: Sequence[int]) -> tuple[int, ...]:
    """A key tuple j of (27) as Python ints, each a spawn-key component in [0, 2**32) (named ``key[i]`` on error).

    Raises:
        ValueError: if a component is negative or >= 2**32.
        TypeError: if a component is not an integer (a bool included).

    """
    return tuple(_spawn_component(j, f"key[{i}]") for i, j in enumerate(key))


def _head(entropy: int, episode: int) -> tuple[int, ...]:
    """The flat entropy's first words: E_split as its 4 little-endian uint32 words (zero-padded), then n."""
    return tuple((entropy >> (32 * i)) & _WORD_MASK for i in range(_POOL_WORDS)) + (episode,)


def keyed_sequence(head: tuple[int, ...], stream: int, key: Sequence[int]) -> np.random.SeedSequence:
    """``SeedSequence(E_split, spawn_key=(n, stream, *key))`` of (27), built from its flat uint32 entropy.

    ``head`` is E_split's 4 little-endian words and n (``_head``); the SeedSequence is that of the one array
    [*head, stream, *key] with no spawn key, the same pool as the spawned form (module docstring). The one constructor
    of every keyed draw; nothing is validated here (the callers validate: ``generator``, ``uniforms``, ``KeyedStream``).
    """
    return np.random.SeedSequence(np.array((*head, stream, *key), dtype=np.uint32))


def _checked(entropy: object, episode: object, stream: object, key: Sequence[int]) -> tuple:
    """(E_split, n, stream id, key) of (27) as Python ints, checked in that order (the errors of ``seed_sequence``)."""
    return check_entropy(entropy), check_episode(episode), _spawn_component(stream, "stream id"), check_key(key)


def seed_sequence(entropy: int, episode: int, stream: int, key: tuple[int, ...]) -> np.random.SeedSequence:
    """SeedSequence(E_split, spawn_key=(episode, stream, *key)) of (27); every key component lies in [0, 2**32).

    The spawned form itself (``entropy`` and ``spawn_key`` as (27) states them); a draw from the same key
    (``generator``, ``uniforms``) is built the flat way with the same state.

    Raises:
        ValueError: if the entropy is not in [0, 2**128) or any spawn-key component (episode, stream id, key) is
            negative or >= 2**32.
        TypeError: if a component is not an integer (a bool included).

    """
    ent, n, sid, j = _checked(entropy, episode, stream, key)
    return np.random.SeedSequence(ent, spawn_key=(n, sid, *j))


def generator(entropy: int, episode: int, stream: int, key: tuple[int, ...]) -> np.random.Generator:
    """``Generator(PCG64DXSM(...))`` of one key, for a vector of draws; checked as ``seed_sequence``."""
    ent, n, sid, j = _checked(entropy, episode, stream, key)
    return np.random.Generator(np.random.PCG64DXSM(keyed_sequence(_head(ent, n), sid, j)))


def to_open_unit(words: np.ndarray) -> np.ndarray:
    """Map uint64 words to doubles in the open interval (0, 1): ``u = ((x >> 11) + 0.5) * 2**-53``, capped below 1.

    The top 53 bits of each word pick one of 2**53 equal bins and ``u`` is the bin's midpoint, so ``u`` is never 0 and
    is non-decreasing in the word (the rule of ``scripts/python/evidence/fix_crn_streams.py``). Above 0.5 a midpoint is
    not a double and rounds half-to-even; for the single top bin (all 53 bits set) that rounding gives exactly 1.0, so
    the result is capped at ``1 - 2**-53``, the largest double below 1. The cap changes no other word.
    """
    w = np.asarray(words, dtype=np.uint64)
    u = ((w >> np.uint64(11)).astype(np.float64) + 0.5) * 2.0**-53
    return np.minimum(u, _BELOW_ONE)


def open_unit(word: int) -> float:
    """``to_open_unit`` of one uint64 word in Python arithmetic: ((w >> 11) + 0.5) 2**-53, capped below 1.

    The same IEEE double operations (an exact int-to-float conversion below 2**53, one rounded addition, an exact
    scaling by a power of two), so the same value bit for bit (unit-tested on edge words).
    """
    return min((float(word >> 11) + 0.5) * 2.0**-53, _BELOW_ONE)


def uniforms(entropy: int, episode: int, stream: int, key: tuple[int, ...], n: int = 1) -> np.ndarray:
    """``n`` uniforms in the open interval (0, 1) from ``generate_state(n, uint64)`` of one key (inversion draws).

    The words are mapped by ``to_open_unit``; the key is checked as ``seed_sequence``.
    """
    ent, ep, sid, j = _checked(entropy, episode, stream, key)
    return to_open_unit(keyed_sequence(_head(ent, ep), sid, j).generate_state(n, np.uint64))


class KeyedStream:
    """The keyed draws of one stream of (27) for one (E_split, n): ``SeedSequence(E_split, (n, stream, *key))``.

    E_split, n and the stream id are validated once, when the stream is built (``check_entropy``, ``check_episode``,
    the spawn-key bound); keys are not re-validated on every draw: a sampler builds its keys from codes, week keys and
    ranks, or validates one with ``check_key`` where it comes from elsewhere. The draws are those of ``uniforms`` and
    ``generator`` for the same key, bit for bit.

    Raises:
        ValueError: if the entropy is not in [0, 2**128), or the episode or stream id not in [0, 2**32).
        TypeError: if one of them is not an integer (a bool included).

    """

    __slots__ = ("_head", "stream")

    def __init__(self, entropy: int, episode: int, stream: int) -> None:
        ent = check_entropy(entropy)
        self._head = _head(ent, check_episode(episode))
        self.stream = _spawn_component(stream, "stream id")

    def sequence(self, key: Sequence[int]) -> np.random.SeedSequence:
        """The SeedSequence of ``key``."""
        return keyed_sequence(self._head, self.stream, key)

    @property
    def head(self) -> tuple[int, ...]:
        """E_split's 4 little-endian uint32 words and n: the first words of every key's flat entropy (27)."""
        return self._head

    def _words(self, key: Sequence[int], n: int) -> np.ndarray:
        """``generate_state(n, uint64)`` of ``key``, the same words bit for bit on either path (``omega._kernels``).

        The numba kernel when ``KERNELS`` is on and the kernels are built (Q95), else ``keyed_sequence``'s SeedSequence.
        """
        if KERNELS:
            from sbfplan.omega import _kernels

            words = _kernels.stream_words(self._head, self.stream, key, n)
            if words is not None:
                return words
        return self.sequence(key).generate_state(n, np.uint64)

    def uniforms(self, key: Sequence[int], n: int = 1) -> np.ndarray:
        """``n`` open uniforms of ``key``: ``to_open_unit`` of ``generate_state(n, uint64)``."""
        return to_open_unit(self._words(key, n))

    def uniform(self, key: Sequence[int]) -> float:
        """The first open uniform of ``key`` as a Python float (``open_unit`` of the first word)."""
        return open_unit(int(self._words(key, 1)[0]))

    def generator(self, key: Sequence[int]) -> np.random.Generator:
        """``Generator(PCG64DXSM(...))`` of ``key``, for a vector of draws."""
        return np.random.Generator(np.random.PCG64DXSM(self.sequence(key)))


def policy_seed(entropy: int, phase: str, episode: int, submission_id: str) -> int:
    """HMAC-SHA256(E_split, "policy" || phase || n || submission id) truncated to 63 bits (27).

    One-way, so a policy cannot recover E_split or omega (D6), and salted by the submission (Q72). The message is the
    canonical JSON array ``["policy", phase, n, submission id]`` (no spaces, non-ASCII escaped, so pure ASCII): the
    strings are quoted and escaped, so distinct (phase, n, submission id) triples never share a message, whatever
    separators the ids contain (DET-3). The key is E_split as 16 big-endian bytes. The seed is the first 8 bytes of the
    digest read big-endian, masked to their low 63 bits, so ``0 <= seed < 2**63``.

    Raises:
        ValueError: if the entropy is not in [0, 2**128) or the episode not in [0, 2**32).
        TypeError: if the entropy or the episode is not an integer (a bool included), or the phase or submission id
            not a string.

    """
    ent = check_entropy(entropy)
    n = check_episode(episode)  # an int, so 1 and np.int64(1) give one message and 1.0 is refused
    if not isinstance(phase, str) or not isinstance(submission_id, str):
        raise TypeError("policy seed (27): the phase and the submission id must be strings")
    msg = json.dumps(["policy", phase, n, submission_id], ensure_ascii=True, separators=(",", ":")).encode("ascii")
    digest = hmac.new(ent.to_bytes(_ENTROPY_BITS // 8, "big"), msg, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") & _MASK_63
