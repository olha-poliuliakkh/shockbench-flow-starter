"""The local runner: one episode over the §9.2 wire, and the policy side that answers it (design §9.2-9.4; Q71).

``play_wire_episode`` drives an ``Env`` through a ``Transport``: reset, the Reset line (no reply), then for each week
t = 1..T drain and discard the pipe, send Request(t) with the episode's one nonce (Q86: one episode tag and one nonce
per (submission, episode)), read until a reply, a failure or the deadline (discarding stale lines, those whose action
names an earlier week, Q71 X5), and step the Env with the reply's action, or with ``Env.step(None,
wire_failure=<code>)`` on a whole-week wire failure (``wire.WIRE_FAILURES``). After a kill no Request is sent and every
remaining week steps with ``wire_failure='killed'`` (naive plays the rest, §9.3). Its trajectory equals the in-process
``dynamics.env.rollout`` of the same policy bit for bit (the wire moves nothing). The cgroup freeze, CPU metering and
the overage bank are M6 (§9.4): locally ``remaining_bank`` is sent from the limits and never metered; the hosted
runner of Q113 passes a ``WeekMeter`` (``hosting.metering.WeekCpu``), whose week over its CPU budget steps with
``wire_failure='cpu'``, and may end a week early through ``SubprocessTransport.interrupt``. With
``hide_regime`` the runner sends ``Static.regime`` null (several regimes scored in one run; design §12 M3 rows). Its
settings (the transport, ``WireLimits``, the deadline, ``hide_regime``) are read by
``scripts/python/run_wire_episode.py`` alone; ``run_episode.py`` refuses them.

``serve`` is the policy side, which a child process runs under ``sys.executable -c`` (``child_argv``, the policy's
pickle handed over a private temporary file; ``serve_stdio``): it loops ``serve_line`` over the lines it reads, and
``LoopbackTransport`` calls ``serve_line`` once per line sent, so both answer a line by one rule. Both reset the policy
by ``policies.base.reset_policy`` (the keyword ``omega`` exactly when ``wants_omega``), as ``rollout`` does.

Readings of design §12 (M3 wire rows):

- Week read loop: the lines drained before Request(t), and those completed while it is written, are discarded and
  counted stale for week t (counted, never stored); after Request(t) the first line that is not stale decides the
  week; the deadline is one per week, measured from the send of Request(t), so stale lines never extend it.
- Closed channel: a ``Receipt`` of kind 'closed' makes that week and every later week 'killed', with no line sent
  after it (the transport is killed).
- Subprocess transport: non-blocking reads through a selector, a line buffered to at most ``max_reply_bytes`` + 1
  bytes (an overlong line returned as those bytes, its remainder discarded up to its newline); at a deadline a partial
  line is discarded the same way; a send not written within the deadline kills the child; the first line (the
  Reset) is written, and read by the child, within the start-up budget ``startup_s`` instead: the transport holds the
  read end of the child's stdin until the pipe is empty (FIONREAD), so the child's start-up counts against no week,
  whatever the Reset's size.
- Loopback transport: in process, synchronous; an exception of the policy propagates, as in ``rollout``.
- Resume: a Reset with fresh tokens carrying the restored week's observation and the reset info that the caller kept
  beside the snapshot, then Requests to T (the runner reads nothing private of the Env).
"""

import atexit
import collections
import fcntl
import json
import os
import pickle
import selectors
import struct
import subprocess
import sys
import tempfile
import termios
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import shockbench_flow
from shockbench_flow.dynamics.env import Env, received_copy, took_fallback  # received_copy re-exported
from shockbench_flow.dynamics.state import Trajectory
from shockbench_flow.information.theta import REGIME_NAMES, Theta, regime_label, resolve_regime
from shockbench_flow.information.wire import (
    WireLimits,
    decode_reply,
    encode,
    new_token,
    reply_message,
    request_message,
    reset_message,
)
from shockbench_flow.instance.schema import Instance
from shockbench_flow.marks import WeeklyMarks
from shockbench_flow.omega.container import Omega
from shockbench_flow.policies.base import Policy, reset_policy


READ_CHUNK = 1 << 16  # bytes per read of a child's stdout (a buffer size, not a model value)
CLOSE_GRACE_S = 5.0  # seconds a child is given to exit after its stdin closes before it is killed (not a model value)
DRAIN_READS = 1024  # reads per drain at most (64 MiB of stale output), so a flooding child cannot hold the runner
STARTUP_POLL_S = 0.002  # seconds between two looks at the child's unread stdin during start-up (not a model value)
INTERRUPT_POLL_S = (
    0.02  # seconds between two calls of a transport's ``interrupt`` while a week waits (not a model value)
)


@dataclass(frozen=True)
class Receipt:
    """What one ``Transport.receive`` got: a line, the deadline, or a closed channel (EOF, or the process gone).

    The runner tells them apart: a deadline is the week's 'timeout'; a closed channel means no later line can come,
    and that week and every later week are 'killed' (design §12 "Closed channel (M3 wire)").
    """

    kind: Literal["line", "deadline", "closed"]
    line: bytes | None = None  # the line (with its newline) when kind is 'line'; an overlong line comes without one


class Transport(Protocol):
    """A line channel to one policy process: bytes lines ending in exactly one newline.

    ``send`` returns how many complete lines of the policy arrived, and were discarded unread, while the line was
    being written; ``drain`` discards every complete line waiting now, without blocking, and returns how many. Such
    lines answer an earlier request (stale), so they are counted, never stored (threats row 19).
    """

    def send(self, line: bytes) -> int: ...

    def drain(self) -> int: ...

    def receive(self) -> Receipt: ...  # the next line, the deadline, or a closed channel

    def kill(self) -> None: ...

    def close(self) -> None: ...


class WeekMeter(Protocol):
    """A per-week CPU meter of the policy process, read outside it (the hosted runner's; Q113, design §12 rows).

    ``start`` is called once the Reset is sent (the process has read it), ``end_week(t)`` once week t's read is over
    (a reply, a failure or the deadline; never after a closed channel). ``end_week`` returns True when the week went
    over its CPU budget: the runner then steps it with ``wire_failure='cpu'`` (D9) whatever the reply held.
    """

    def start(self) -> None: ...

    def end_week(self, week: int) -> bool: ...


# ----- the policy side -----------------------------------------------------------------------------------------------
@dataclass
class ServeSession:
    """The policy side's state across lines: the policy, and whether a Reset has come.

    A Request's tags are echoed from the Request itself (the runner checks them, ``wire.decode_reply``), so the session
    keeps none.
    """

    policy: Policy
    reset_seen: bool = False


def encode_reply(reply: dict) -> bytes:
    """A Reply line as the policy side writes it: ``wire.encode`` of the reply with its action's ``received_copy``.

    ``received_copy`` is ``dynamics.env.received_copy``, the copy ``Env.step`` stores and validates.

    The action goes as the Env's own copy, so the Env's copy of the decoded line is that copy again and the wire moves
    nothing (W2): a key or value ``json.dumps`` would write otherwise (a bool key as 'true') or refuse (a NumPy key, an
    int beyond ``int_max_str_digits``, a set) reaches the Env as it does in process, and the line never exceeds the
    wire's nesting or digit limits (``wire.MAX_REPLY_DEPTH``, ``wire.MAX_INT_DIGITS``; design §12 "Wire line encoding
    (M3 wire)"; M3 attempt 1's gate, INT-M3-03 and DET-M3-4). A NaN goes as null, which the Env reads as it reads a NaN.
    """
    return encode({**reply, "action": received_copy(reply.get("action"))})


def serve_line(session: ServeSession, line: bytes) -> bytes | None:
    """Answer one line on the policy side: a Reset resets the policy (no reply), a Request returns the Reply line.

    A Reset becomes ``policies.base.reset_policy(policy, static, obs, policy_seed, omega)`` (``omega`` its
    ``Reset.omega``, absent outside clairvoyant); a Request becomes ``encode_reply(reply_message(episode=, nonce=,
    action=policy.act(obs)))``, the action unchanged and the Request's tags echoed.

    Raises:
        ValueError: on a line that is not a Reset or a Request of the §9.2 schema (the runner never sends one), or a
            Request before any Reset.

    """
    msg = json.loads(line)
    kind = msg.get("type") if isinstance(msg, dict) else None
    if kind == "reset":
        session.reset_seen = True
        reset_policy(session.policy, msg["static"], msg["obs"], msg["policy_seed"], msg.get("omega"))
        return None
    if kind == "step":
        if not session.reset_seen:
            raise ValueError("a Request before any Reset (the runner never sends one)")
        action = session.policy.act(msg["obs"])
        return encode_reply(reply_message(episode=msg["episode"], nonce=msg["nonce"], action=action))
    raise ValueError(f"not a Reset or a Request of §9.2: type {kind!r}")


def serve(policy: Policy, read_line: Callable[[], bytes | None], write_line: Callable[[bytes], None]) -> None:
    """The policy side of the wire: ``serve_line`` on every line read, until EOF (``read_line`` None).

    Each reply line ``serve_line`` returns is written with ``write_line``; a Reset gets none.
    """
    session = ServeSession(policy)
    while (line := read_line()) is not None:
        reply = serve_line(session, line)
        if reply is not None:
            write_line(reply)


def serve_stdio(policy: Policy) -> None:
    """``serve`` over this process's stdin and stdout: the child side of ``SubprocessTransport``."""
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer

    def write(line: bytes) -> None:
        stdout.write(line)
        stdout.flush()

    serve(policy, lambda: stdin.readline() or None, write)


_CHILD = (
    "import os, pickle, sys; sys.path.insert(0, sys.argv[1]); "
    "from shockbench_flow.information.runner import serve_stdio; "
    "f = open(sys.argv[2], 'rb'); os.unlink(sys.argv[2]); policy = pickle.load(f); f.close(); serve_stdio(policy)"
)


def _remove(path: str) -> None:
    Path(path).unlink(missing_ok=True)


def child_argv(policy: Policy) -> list[str]:
    """The argv of a child process that serves ``policy`` over its stdin and stdout (``serve_stdio``).

    ``sys.executable -c`` with this package's directory first on the child's path, so the child runs the code the
    parent runs (a local runner; a hosted policy brings its own process, §9.4), and the policy's pickle in a private
    temporary file (``tempfile.mkstemp``: mode 0600), never in the argv: one argv string is capped (Linux's
    MAX_ARG_STRLEN, 128 KiB; macOS caps the whole list at 1 MiB), which a policy's pickle may pass once it carries
    state (the M5 gate's fixes found it on the worker VM). The child opens the file, removes it and unpickles it before
    it reads its stdin (its start-up, ``SubprocessTransport``'s ``startup_s``); a file left by a child that never
    started is removed when this process exits.
    """
    src = str(Path(shockbench_flow.__path__[0]).resolve().parent)
    fd, path = tempfile.mkstemp(prefix="sbf-policy-", suffix=".pickle")
    with os.fdopen(fd, "wb") as fh:
        pickle.dump(policy, fh, protocol=pickle.HIGHEST_PROTOCOL)
    atexit.register(_remove, path)
    return [sys.executable, "-c", _CHILD, src, path]


# ----- transports ----------------------------------------------------------------------------------------------------
class LoopbackTransport:
    """An in-process transport: each line sent is answered synchronously by ``serve_line`` on ``policy``.

    ``receive`` returns the first queued reply, else the deadline (a Reset has no reply); an exception of the policy
    propagates to the caller, as in ``rollout`` (a dev and test tool, not a sandbox).
    """

    def __init__(self, policy: Policy) -> None:
        self.session = ServeSession(policy)
        self._queue: collections.deque[bytes] = collections.deque()
        self._closed = False

    def send(self, line: bytes) -> int:
        if self._closed:
            return 0  # nothing reads the line any more
        reply = serve_line(self.session, line)
        if reply is not None:
            self._queue.append(reply)  # the answer to this very line: queued, never discarded
        return 0

    def drain(self) -> int:
        n = len(self._queue)
        self._queue.clear()
        return n

    def receive(self) -> Receipt:
        if self._queue:
            return Receipt("line", self._queue.popleft())
        return Receipt("closed") if self._closed else Receipt("deadline")

    def kill(self) -> None:
        self._closed = True
        self._queue.clear()

    def close(self) -> None:
        self._closed = True


class SubprocessTransport:
    """A child process spoken to over its real stdin and stdout, under one deadline per line sent.

    The first line sent (the Reset, which gets no reply) is written and read by the child within ``startup_s`` seconds,
    the child's start-up (the interpreter, its imports, the policy's unpickling and its read of that line), and every
    later line is written within ``deadline_s``, so the start-up is charged to no week (M3 attempt 1's gate, DET-M3-2;
    the policy's own reset work after it has read the Reset still counts against week 1, since the Reset gets no
    reply). The child's read is seen on the pipe itself: the transport keeps its own copy of the read end of the
    child's stdin until the first line is read, and ``send`` returns once no byte of it is left unread (FIONREAD on
    that end, Linux and macOS alike), so the allowance does not depend on the line's size (the wire skeptic of M3's
    gate-1 fixes: a Reset that fit in the OS pipe buffer was written at once, and the child's start-up counted against
    week 1 again). A child that has not read the whole first line by then is killed, as is one that stops reading its
    stdin later; one that exits or closes its stdout first is a closed channel. Every ``receive`` after a
    ``send`` waits until that send's deadline (so stale lines never extend a week); reads are non-blocking through a
    selector and a line is buffered to at most ``max_reply_bytes`` + 1 bytes: an overlong line is returned as those
    bytes, without a newline, and its remainder discarded up to its newline; at a deadline a partial line is discarded
    the same way, whenever its newline comes, so its bytes never prefix a later line; a send not written by its
    deadline (a child that stops reading its stdin) kills the child. The lines read by ``drain`` or while a send is
    blocked (read so that a child writing while it reads never deadlocks) are counted and discarded, never queued, so
    a flooding child cannot grow the runner's memory; only ``receive`` queues lines, one read at a time. ``peak`` is
    the largest line buffer held (design §12 "Subprocess transport (M3 wire)"; threats row 19).

    ``interrupt`` (None by default: nothing changes) is the hosted runner's hook (Q113): while ``receive`` waits for a
    reply it is called every ``interrupt_poll_s`` seconds, and True ends the wait as the deadline does (a partial line
    abandoned), so a week whose CPU went over its budget ends at once instead of at the wall-clock deadline. Lines
    already read are returned first; the start-up's ready line is read before the runner sets it.
    """

    interrupt: Callable[[], bool] | None = None
    interrupt_poll_s: float = INTERRUPT_POLL_S

    def __init__(
        self,
        argv: Sequence[str],
        *,
        deadline_s: float,
        startup_s: float,
        max_reply_bytes: int,
        env: dict[str, str] | None = None,
        stderr: int | None = None,
    ) -> None:
        # stderr: the child's standard error as ``subprocess.Popen`` takes it (None inherits the runner's; the hosted
        # runner passes DEVNULL or a file descriptor, so no policy output reaches the runner's own log, §9.5 'Logs')
        for name, v in (("deadline_s", deadline_s), ("startup_s", startup_s)):
            if isinstance(v, bool) or not (isinstance(v, (int, float)) and v > 0):
                raise ValueError(f"{name} must be a positive number of seconds, got {v!r}")
        if isinstance(max_reply_bytes, bool) or not isinstance(max_reply_bytes, int) or max_reply_bytes < 1:
            raise ValueError(f"max_reply_bytes must be an integer >= 1, got {max_reply_bytes!r}")
        self.deadline_s, self.startup_s, self.max_reply_bytes = float(deadline_s), float(startup_s), max_reply_bytes
        self._started = False  # whether the first line (the Reset) was sent
        read_end, self._in = os.pipe()  # the child's stdin; both ends close on exec, Popen dups the read end to fd 0
        try:
            self.proc = subprocess.Popen(
                list(argv), stdin=read_end, stdout=subprocess.PIPE, stderr=stderr, bufsize=0, env=env
            )
        except BaseException:
            os.close(read_end)
            os.close(self._in)
            raise
        self._probe: int | None = read_end  # our copy of the read end, until the child has read its first line
        self._out = self.proc.stdout.fileno()
        os.set_blocking(self._in, False)
        os.set_blocking(self._out, False)
        self._sel = selectors.DefaultSelector()
        self._sel.register(self._out, selectors.EVENT_READ)
        self._buf = bytearray()  # the line being read, at most max_reply_bytes + 1 bytes
        self._lines: collections.deque[bytes] = collections.deque()
        self._skip = False  # discarding up to the next newline (an overlong or abandoned partial line)
        self._discarding = False  # count complete lines instead of queuing them (drain, and a send in progress)
        self._discarded = 0
        self._eof = self._gone = False
        self._deadline = time.monotonic() + self.deadline_s
        self.peak = 0

    # reading
    def _emit(self, line: bytes) -> None:
        if self._discarding:
            self._discarded += 1
        else:
            self._lines.append(line)

    def _feed(self, data: bytes) -> None:
        if self._discarding and b"\n" in data:  # every line the chunk completes is only counted, at C speed
            head, _nl, data = data.rpartition(b"\n")  # data: the partial line after the chunk's last newline
            self._discarded += head.count(b"\n") + (0 if self._skip else 1)  # a skipped remainder is no line
            self._buf.clear()
            self._skip = False
        cap, pos, n = self.max_reply_bytes + 1, 0, len(data)
        while pos < n:
            nl = data.find(b"\n", pos)
            stop = n if nl < 0 else nl
            if self._skip:
                if nl < 0:
                    return
                self._skip, pos = False, nl + 1
                continue
            take = min(stop - pos, cap - len(self._buf))
            self._buf += data[pos : pos + take]
            self.peak = max(self.peak, len(self._buf))
            pos += take
            if len(self._buf) >= cap and (nl < 0 or pos < stop):  # cap bytes and no newline among them: too long
                self._emit(bytes(self._buf))
                self._buf.clear()
                self._skip = True
                continue
            if nl < 0:
                return
            self._emit(bytes(self._buf) + b"\n")
            self._buf.clear()
            pos = nl + 1

    def _read_once(self) -> None:
        try:
            data = os.read(self._out, READ_CHUNK)
        except BlockingIOError:
            return
        except OSError:
            data = b""
        if not data:
            self._eof = True
            self._buf.clear()  # a last line without its newline is incomplete: dropped
            self._sel.unregister(self._out)
            return
        self._feed(data)

    def _pump(self, timeout: float) -> None:
        if self._eof:
            return
        for _key, _mask in self._sel.select(max(0.0, timeout)):
            self._read_once()

    def _abandon_partial(self) -> None:
        if self._buf:
            self._buf.clear()
            self._skip = True

    def _unread(self) -> int:
        """Bytes written to the child's stdin that it has not read yet (FIONREAD on our copy of the read end)."""
        return struct.unpack("i", fcntl.ioctl(self._probe, termios.FIONREAD, b"\0\0\0\0"))[0]

    def _close_probe(self) -> None:
        if self._probe is not None:
            os.close(self._probe)
            self._probe = None

    def _await_read(self) -> None:
        """Wait, by the send's deadline, until the child has read every byte written (its stdin pipe empty).

        What the child writes meanwhile is read in discarding mode, so a child writing while it starts never blocks. A
        child still holding unread bytes at the deadline is killed (it did not read its first line in time); one whose
        stdout closes, or that exits, first is left to ``receive``, which reports the closed channel.
        """
        while not self._gone and self._unread() > 0:
            if self._eof or self.proc.poll() is not None:
                return
            remaining = self._deadline - time.monotonic()
            if remaining <= 0:
                self.kill()
                return
            self._pump(min(remaining, STARTUP_POLL_S))  # waits the poll interval unless the child writes

    # the Transport protocol
    def _write(self, line: bytes) -> None:
        """Write ``line`` by the deadline, reading (in discarding mode) whatever the child writes meanwhile."""
        view = memoryview(line)
        with selectors.DefaultSelector() as sel:
            sel.register(self._in, selectors.EVENT_WRITE)
            if not self._eof:
                sel.register(self._out, selectors.EVENT_READ)
            while view:
                remaining = self._deadline - time.monotonic()
                if remaining <= 0:
                    self.kill()  # the child stopped reading its stdin: a closed channel
                    return
                for key, _mask in sel.select(remaining):
                    if key.fd == self._out:
                        self._read_once()  # keep reading, so a child writing while it reads never blocks
                        if self._eof:
                            sel.unregister(self._out)
                            if self._probe is not None:  # start-up: our read end keeps the pipe open, no EPIPE
                                return  # the child's stdout is closed, so it can never answer: receive says closed
                        continue
                    try:
                        view = view[os.write(self._in, view) :]
                    except BlockingIOError:
                        pass
                    except OSError:  # the child is gone (BrokenPipeError)
                        self._gone = True
                        return

    def send(self, line: bytes) -> int:
        first = not self._started
        self._deadline = time.monotonic() + (self.startup_s if first else self.deadline_s)
        self._started = True
        try:
            if self._gone:
                return 0
            self._discarding, self._discarded = True, 0  # a line completed before this one is written answers none
            try:
                self._write(line)
                if first and not self._gone:
                    self._await_read()  # the first line counts as sent once the child has read all of it
            finally:
                self._discarding = False
            return self._discarded
        finally:
            if first:
                self._close_probe()

    def drain(self) -> int:
        self._discarding, self._discarded = True, len(self._lines)
        self._lines.clear()
        try:
            for _ in range(DRAIN_READS):  # bounded, so a child that writes without end cannot hold the runner here
                if self._eof or not self._sel.select(0):
                    break
                self._read_once()
        finally:
            self._discarding = False
        return self._discarded

    def receive(self) -> Receipt:
        while True:
            if self._lines:
                return Receipt("line", self._lines.popleft())
            if self._eof or self._gone:
                return Receipt("closed")
            remaining = self._deadline - time.monotonic()
            if remaining <= 0 or (self.interrupt is not None and self.interrupt()):
                self._abandon_partial()
                return Receipt("deadline")
            self._pump(remaining if self.interrupt is None else min(remaining, self.interrupt_poll_s))

    def kill(self) -> None:
        self._gone = True
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()

    def close(self) -> None:
        self._close_probe()
        if self._in >= 0:
            try:
                os.close(self._in)
            except OSError:
                pass
            self._in = -1
        try:
            self.proc.wait(timeout=CLOSE_GRACE_S)
        except subprocess.TimeoutExpired:
            self.kill()
        self._gone = True
        self._sel.close()
        self.proc.stdout.close()


# ----- the runner ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class WireEpisode:
    """One episode over the wire, with the runner's own log.

    One episode over the wire: the Env's trajectory and the runner's own log (never inside the trajectory hash
    beyond ``Trajectory.wire_failures``).
    """

    trajectory: Trajectory
    substitutions: tuple[tuple[int, str], ...]  # (week, WIRE_FAILURES code, or 'action' for a failed validation)
    stale: tuple[tuple[int, int], ...]  # (week, stale lines discarded before its reply)


def _read_week(transport: Transport, *, episode: str, nonce: str, week: int, limits: WireLimits) -> tuple:
    """(action, code, stale lines read, closed) of week ``week`` after its Request (design §12 "Week read loop")."""
    stale = 0
    while True:
        r = transport.receive()
        if r.kind == "deadline":
            return None, "timeout", stale, False
        if r.kind == "closed":
            return None, "killed", stale, True
        out = decode_reply(r.line, episode=episode, nonce=nonce, week=week, limits=limits)
        if out.kind == "stale":
            stale += 1
            continue
        if out.kind == "failure":
            return None, out.failure, stale, False
        return out.action, None, stale, False


def play_wire_episode(
    env: Env,
    instance: Instance,
    transport: Transport,
    *,
    limits: WireLimits,
    regime: str | Theta = "prediction_free",
    omega: Omega | str | Path | None = None,
    policy_seed: int = 0,
    marks: WeeklyMarks | None = None,
    token: Callable[[], str] = new_token,
    policy_name: str = "",
    hide_regime: bool = False,
    resume: tuple[dict, dict] | None = None,
    meter: WeekMeter | None = None,
) -> WireEpisode:
    """Play one episode of the policy behind ``transport`` (module docstring); ``token`` draws its tag and nonce.

    ``resume`` = (obs, info) continues ``env`` as it stands, an Env restored from a snapshot mid-episode (E4, B2): no
    reset; ``obs`` is the observation of the restored week and ``info`` the episode's reset info (``static``, and
    ``omega`` under clairvoyant), which the caller keeps beside the snapshot (the runner reads nothing private of the
    Env); the runner sends a Reset carrying them with fresh tokens, then Requests from that week to T; the trajectory
    is the env's, so its hash equals the uninterrupted run's when the policy's replies do. The transport is closed
    when the episode ends. ``meter`` (``WeekMeter``, the hosted runner's) is started after the Reset and asked after
    every week's read whether the week went over its CPU budget; such a week is the whole-week failure 'cpu', in
    ``Trajectory.wire_failures`` like every wire code, so ``replay`` re-simulates it.

    Raises:
        ValueError: if ``resume``'s observation is not of the restored week (the env's next week).
        Exception: whatever ``env.reset`` or ``env.step`` raises (a bug: a hostile line never makes the runner raise).

    """
    if resume is not None:
        obs, info = resume
        restored_week = len(env.trajectory.actions) + 1
        if obs["week"] != restored_week:
            raise ValueError(f"resume: an observation of week {obs['week']!r}, not the restored week {restored_week}")
    else:
        obs, info = env.reset(instance, regime, omega, policy_seed, marks=marks, policy_name=policy_name)
    static, payload = info["static"], info.get("omega")
    if hide_regime:
        static = {**static, "regime": None}
    episode, nonce = token(), token()  # one tag and one nonce per (submission, episode) (Q86)
    bank = limits.bank_seconds  # sent, never metered locally (M6)
    transport.send(
        encode(
            reset_message(
                episode=episode,
                nonce=nonce,
                policy_seed=policy_seed,
                static=static,
                obs=obs,
                remaining_bank=bank,
                omega=payload,
            )
        )
    )
    if meter is not None:
        meter.start()
    substitutions, stale = [], []
    killed, done = False, obs["week"] > instance.T
    while not done:
        week = obs["week"]
        action, code, n_stale = None, "killed", 0
        if not killed:
            n_stale = transport.drain()  # every line waiting now answers an earlier request: stale
            n_stale += transport.send(  # and so does every line completed while Request(t) is written
                encode(request_message(episode=episode, week=week, nonce=nonce, obs=obs, remaining_bank=bank))
            )
            action, code, read_stale, killed = _read_week(
                transport, episode=episode, nonce=nonce, week=week, limits=limits
            )
            n_stale += read_stale
            if killed:
                transport.kill()
            elif meter is not None and meter.end_week(week):
                action, code = None, "cpu"  # over the week's CPU budget: D9 whatever the reply held (Q113)
        obs, _reward, done, _truncated, _info = env.step(action, wire_failure=code)
        if code is not None:
            substitutions.append((week, code))
        elif took_fallback(env.trajectory.records[-1]):
            substitutions.append((week, "action"))
        if n_stale:
            stale.append((week, n_stale))
    transport.close()
    return WireEpisode(env.trajectory, tuple(substitutions), tuple(stale))


def replay(
    env: Env,
    instance: Instance,
    trajectory: Trajectory,
    *,
    regime: str | Theta | None = None,
    omega: Omega | str | Path | None = None,
    policy_seed: int = 0,
    marks: WeeklyMarks | None = None,
) -> Trajectory:
    """Re-simulate a trajectory from its stored actions and wire failures, under its own regime.

    Re-simulate ``trajectory`` from its stored actions and ``wire_failures`` in ``env`` (V24, B2): each week steps
    with its stored action, or ``Env.step(action, wire_failure=code)`` where a wire failure is stored; the result's
    ``sha256`` equals the original's. The regime is the trajectory's own (design §12 "Stored-action replay (M3)"):
    ``regime`` None reads the registry name ``Trajectory.regime``; a custom theta, recorded as its label, must be
    passed; a ``regime`` whose label (``theta.regime_label``) is not ``Trajectory.regime`` is refused, since the D9
    fallback reads the regime's observation and another regime replays to other cents (ORC-M3-2). An instance, omega
    or marks other than the trajectory's (its hashes and digests) are refused too (V3).

    Raises:
        ValueError: if ``regime`` is None and ``Trajectory.regime`` is no registry name, if ``regime``'s label is not
            ``Trajectory.regime``, or if the reset episode's instance or omega hash, or instance or marks digest, is
            not the trajectory's.

    """
    if regime is None:
        if trajectory.regime not in REGIME_NAMES:
            raise ValueError(
                f"replay: the trajectory's regime {trajectory.regime!r} is a custom theta's label: pass that theta"
            )
        regime = trajectory.regime
    label = regime_label(resolve_regime(regime))
    if label != trajectory.regime:
        raise ValueError(f"replay: regime {label!r} is not the trajectory's regime {trajectory.regime!r}")
    env.reset(instance, regime, omega, policy_seed, marks=marks, policy_name=trajectory.policy)
    now, fields = env.trajectory, ("instance_hash", "omega_hash", "instance_digest", "marks_digest")
    if any(getattr(now, f) != getattr(trajectory, f) for f in fields):
        raise ValueError("replay: the instance, omega or marks are not the trajectory's (V3)")
    failures = dict(trajectory.wire_failures)
    for week, action in enumerate(trajectory.actions, start=1):
        env.step(action, wire_failure=failures.get(week))
    return env.trajectory
