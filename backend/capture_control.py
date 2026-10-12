"""Start, stop and batch the one-minute captures from the web, so the operator at the wired laptop needs no shell.

The capture itself is ``scripts/capture_mtk_hourly.sh``, unchanged: it wakes and arms the board over the cable and the
serial console, streams the CSI into ``captures/<stamp>.bin``, grabs webcam frames beside it, and ships both to the
collection host when it can reach it. This module only *runs* that script and reads its log:

* one job at a time -- ``count`` captures of ``duration_s`` each, a ``delay_s`` countdown before the first (the
  operator has to leave the room) and ``gap_s`` between them;
* each run in its own process group, so one signal stops every local process of a run at once (wrapper, webcam
  grabber, live shipper, both ssh) -- the lesson of 2026-09-17, when killing the wrapper alone left three running;
* after a stop, the board is cleaned explicitly: the stream there outlives its ssh, so the PIDs holding
  ``/var/csi/capture.lock`` are killed and the driver disarmed, and the partial ``.bin`` is removed locally and from
  the collection host's ``live/`` directory;
* ``RETAIN_DAYS=0`` for web-started runs: the script's default prunes local captures older than seven days, which is
  only safe while every capture has been uploaded, and offline sessions are exactly when it has not.

Where the files go is the script's business and is not changed here: ``captures/`` beside the shared checkout, then
``lg:/home/lg_csi/lg_csi_captures/<day>/``. Everything the UI shows comes from the script's own log
(``captures/mtk_capture.log``: START / WAKE / DONE / FAIL / SKIP / ABORT / UPLOAD / LABEL lines).
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SCRIPT = REPO / "scripts" / "capture_mtk_hourly.sh"
# The script's own default; web-started runs write where cron and the operators' shells always have.
SHARED_CAPTURE_DIR = Path("/home/cyphy/feitcsi_parser/captures")
DEFAULT_BOARD = "192.168.50.80"
DEFAULT_REMOTE = "lg"
DEFAULT_REMOTE_DIR = "/home/lg_csi/lg_csi_captures"
WEBCAM_DEV = "/dev/video2"
LOCAL_LOCK = "/tmp/mtk-capture.lock"          # the script's flock; busy = a capture is running, ours or not
FRAME_TMP = "/tmp/csi-frames"
BOARD_LOCK = "/var/csi/capture.lock"
BOARD_IW = "/var/iwtools/iw-priv"

LIMITS = {"duration_s": (10, 3300), "delay_s": (0, 3600), "count": (1, 200), "gap_s": (0, 3600)}
# Stimulus ping interval on the board, FIXED (user, 2026-10-12: "약 43 Hz로 고정"). The script's own default is 0.05 s
# (~19 Hz of CSI); every capture in the dataset of record was taken at 0.02 s (~42-44 Hz), set through the environment
# by whoever ran it, and the AF8 + C3 pipeline's dimensions assume that rate. Not exposed on the tab on purpose.
PING_INTERVAL = 0.02
TEXT_MAX = 200
LOG_TAIL = 80
POLL_S = 0.5
STOP_GRACE_S = 5.0
# A run may legitimately outlive its duration by a lot: serial wake (~165 s), six ssh attempts, upload, labelling on
# the collection host. Past this it is declared hung and killed.
RUN_OVERHEAD_S = 900.0
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "LogLevel=ERROR"]


@dataclass
class Job:
    duration_s: int
    delay_s: int
    count: int
    gap_s: int
    room: str | None
    scenario: str | None
    note: str | None
    upload: bool
    label: bool
    created_at: float


@dataclass
class Run:
    index: int
    started_at: float
    pid: int | None = None
    stamp: str | None = None
    capture_started_at: float | None = None     # when the script logged START
    ended_at: float | None = None
    rc: int | None = None
    outcome: str = "running"                   # running | done | failed | skipped | aborted | cancelled | hung
    summary: str = ""
    sidecar: str | None = None


@dataclass
class Preflight:
    checked_at: float | None = None
    board_reachable: bool | None = None
    webcam_busy: bool | None = None
    webcam_holder: str | None = None
    remote_reachable: bool | None = None
    disk_free_mb: int | None = None
    capture_lock_busy: bool | None = None
    checking: bool = False


def _clean_text(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    value = " ".join(str(value).split())
    if not value:
        return None
    if len(value) > TEXT_MAX:
        raise ValueError(f"{name} is longer than {TEXT_MAX} characters")
    return value


class CaptureManager:
    """One capture job at a time; everything observable through :meth:`status`."""

    def __init__(self, script: Path | str = DEFAULT_SCRIPT, capture_dir: Path | str | None = None,
                 board: str | None = DEFAULT_BOARD, remote: str | None = DEFAULT_REMOTE,
                 remote_dir: str = DEFAULT_REMOTE_DIR, webcam_dev: str = WEBCAM_DEV,
                 local_lock: str = LOCAL_LOCK, frame_tmp: str = FRAME_TMP) -> None:
        self.script = Path(script)
        if capture_dir is None:
            capture_dir = SHARED_CAPTURE_DIR if SHARED_CAPTURE_DIR.is_dir() else REPO / "captures"
        self.capture_dir = Path(capture_dir)
        self.board = board or None
        self.remote = remote or None
        self.remote_dir = remote_dir
        self.webcam_dev = webcam_dev
        self.local_lock = local_lock
        self.frame_tmp = Path(frame_tmp)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None
        self._log_offset = 0
        self.state = "idle"            # idle | countdown | starting | waking | running | finishing | between | stopping
        self.job: Job | None = None
        self.runs: list[Run] = []
        self.current: Run | None = None
        self.countdown_until: float | None = None
        self.error: str | None = None
        self.log: list[str] = []
        self.preflight = Preflight()
        self._preflight_thread: threading.Thread | None = None

    # ------------------------------------------------------------------ public
    def start(self, *, duration_s: int = 60, delay_s: int = 10, count: int = 1, gap_s: int = 0,
              room: str | None = None, scenario: str | None = None, note: str | None = None,
              upload: bool = True, label: bool = True) -> None:
        values = {"duration_s": int(duration_s), "delay_s": int(delay_s), "count": int(count), "gap_s": int(gap_s)}
        for key, (lo, hi) in LIMITS.items():
            if not lo <= values[key] <= hi:
                raise ValueError(f"{key} must be between {lo} and {hi}, got {values[key]}")
        job = Job(**values, room=_clean_text(room, "room"), scenario=_clean_text(scenario, "scenario"),
                  note=_clean_text(note, "note"), upload=bool(upload), label=bool(label),
                  created_at=time.time())
        if not self.script.is_file():
            raise RuntimeError(f"capture script not found: {self.script}")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("a capture job is already running; stop it first")
            if self._lock_busy():
                raise RuntimeError("a capture is already running outside this page (cron or a shell); wait for it or stop it there")
            self._stop.clear()
            self.job = job
            self.runs = []
            self.current = None
            self.error = None
            self.log = []
            self.state = "countdown" if job.delay_s > 0 else "starting"
            # Published here, not only by the worker: a status read in the few
            # milliseconds before the thread runs must already show the countdown.
            self.countdown_until = time.time() + job.delay_s if job.delay_s > 0 else None
            self._note(f"job: {job.count} x {job.duration_s} s, delay {job.delay_s} s, gap {job.gap_s} s, "
                       f"stimulus every {PING_INTERVAL:g} s"
                       + (f", scenario '{job.scenario}'" if job.scenario else "")
                       + ("" if job.upload else ", upload off") + ("" if job.label else ", labelling off"))
            self._thread = threading.Thread(target=self._worker, args=(job,), name="capture-job", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop the running capture and the rest of the job; returns when the worker has cleaned up."""
        with self._lock:
            thread = self._thread
            if thread is None or not thread.is_alive():
                return
            self._note("stop requested")
            self._stop.set()
            if self.state in ("countdown", "between"):
                self.state = "stopping"
        thread.join(timeout=STOP_GRACE_S + 40.0)

    def status(self) -> dict[str, Any]:
        self._refresh_preflight()
        with self._lock:
            now = time.time()
            cur = self.current
            current = None
            if cur is not None:
                base = cur.capture_started_at or cur.started_at
                current = {**asdict(cur), "elapsed_s": round(now - base, 1),
                           "duration_s": self.job.duration_s if self.job else None}
            remaining = max(0.0, self.countdown_until - now) if self.countdown_until else None
            return {
                "state": self.state,
                "busy": self._thread is not None and self._thread.is_alive(),
                "job": asdict(self.job) if self.job else None,
                "runs": [asdict(r) for r in self.runs],
                "current": current,
                "countdown_remaining_s": None if remaining is None else round(remaining, 1),
                "error": self.error,
                "preflight": asdict(self.preflight),
                "log": self.log[-LOG_TAIL:],
                "script": str(self.script),
                "capture_dir": str(self.capture_dir),
                "board": self.board,
                "remote": self.remote,
                "limits": LIMITS,
                "ping_interval": PING_INTERVAL,
            }

    # ------------------------------------------------------------------ worker
    def _worker(self, job: Job) -> None:
        try:
            for i in range(job.count):
                if self._stop.is_set():
                    break
                wait = job.delay_s if i == 0 else job.gap_s
                if wait > 0 and not self._countdown(wait, "countdown" if i == 0 else "between"):
                    break
                if self._stop.is_set():
                    break
                run = self._launch(job, i)
                self._follow(run, job)
                if self._stop.is_set():
                    break
                if run.outcome != "done":
                    with self._lock:
                        self.error = run.summary or f"capture {i + 1} ended with rc={run.rc}"
                    break
        except Exception as exc:  # the worker must never die silently
            with self._lock:
                self.error = f"capture worker failed: {exc!r}"
                self._note(self.error)
        finally:
            with self._lock:
                if self._stop.is_set() and self.error is None:
                    self._note("job stopped")
                self.state = "idle"
                self.current = None
                self._proc = None
                self.countdown_until = None
                self._thread = None

    def _countdown(self, seconds: float, state: str) -> bool:
        deadline = time.time() + seconds
        with self._lock:
            self.state = state
            self.countdown_until = deadline
        while time.time() < deadline:
            if self._stop.is_set():
                with self._lock:
                    self.countdown_until = None
                return False
            time.sleep(min(0.2, max(0.0, deadline - time.time())))
        with self._lock:
            self.countdown_until = None
        return True

    def _launch(self, job: Job, index: int) -> Run:
        env = os.environ.copy()
        env.update({
            "CAPTURE_DIR": str(self.capture_dir),
            "UPLOAD": "1" if job.upload else "0",
            "LIVE": "1" if job.upload else "0",
            "CV": "1" if job.label else "0",
            "RETAIN_DAYS": "0",
            "PING_INTERVAL": f"{PING_INTERVAL:g}",
        })
        if self.board:
            env["BOARD"] = self.board
        if self.remote:
            env["REMOTE"] = self.remote
            env["REMOTE_DIR"] = self.remote_dir
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._script_log()
        self._log_offset = log_path.stat().st_size if log_path.exists() else 0
        stderr_path = self.capture_dir / "web_capture.stderr"
        stderr = open(stderr_path, "ab")
        try:
            proc = subprocess.Popen(
                [str(self.script), "--duration", str(job.duration_s)],
                cwd=str(self.script.parent.parent), env=env,
                stdin=subprocess.DEVNULL, stdout=stderr, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            stderr.close()
        run = Run(index=index, started_at=time.time(), pid=proc.pid)
        with self._lock:
            self._proc = proc
            self.current = run
            self.runs.append(run)
            self.state = "starting"
            self._note(f"capture {index + 1}/{job.count}: launched {self.script.name} --duration {job.duration_s} (pid {proc.pid})")
        return run

    def _follow(self, run: Run, job: Job) -> None:
        proc = self._proc
        assert proc is not None
        hard_deadline = run.started_at + job.duration_s + RUN_OVERHEAD_S
        while True:
            self._read_script_log(run)
            rc = proc.poll()
            if rc is not None:
                self._read_script_log(run)
                self._finish(run, rc, job)
                return
            if self._stop.is_set():
                self._terminate(run, proc, "cancelled")
                return
            if time.time() > hard_deadline:
                with self._lock:
                    self._note(f"capture {run.index + 1} exceeded {job.duration_s + RUN_OVERHEAD_S:.0f} s, killing it")
                self._terminate(run, proc, "hung")
                return
            time.sleep(POLL_S)

    def _finish(self, run: Run, rc: int, job: Job) -> None:
        with self._lock:
            run.rc = rc
            run.ended_at = time.time()
            if run.outcome == "running":
                head = run.summary.split()[0] if run.summary else ""
                if head == "DONE" and rc == 0:
                    run.outcome = "done"
                elif head == "SKIP":
                    run.outcome = "skipped"
                elif head == "ABORT":
                    run.outcome = "aborted"
                else:
                    run.outcome = "failed" if (rc != 0 or not run.stamp) else "done"
                if not run.summary:
                    run.summary = f"script exited rc={rc} without a DONE line" if run.outcome != "done" else ""
            self.state = "finishing"
            self._note(f"capture {run.index + 1}: {run.outcome}" + (f" -- {run.summary}" if run.summary else ""))
        if run.outcome == "done" and run.stamp and (job.room or job.scenario or job.note):
            self._write_sidecar(run, job)

    # ------------------------------------------------------------------ stop path
    def _terminate(self, run: Run, proc: subprocess.Popen, outcome: str) -> None:
        with self._lock:
            self.state = "stopping"
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            pgid = None
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.time() + STOP_GRACE_S
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        if proc.poll() is None and pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)
        self._read_script_log(run)
        with self._lock:
            run.rc = proc.returncode
            run.ended_at = time.time()
            run.outcome = outcome
            self._note(f"capture {run.index + 1}: local processes stopped (pgid {pgid})")
        self._cleanup_after_stop(run)

    def _cleanup_after_stop(self, run: Run) -> None:
        """The three things a local kill does not do: disarm the board, drop the partial file here and on lg."""
        if self.board:
            cmd = (f"pids=$(fuser {BOARD_LOCK} 2>/dev/null); [ -n \"$pids\" ] && kill $pids 2>/dev/null; "
                   f"pkill -x ping 2>/dev/null; {BOARD_IW} wlan0 driver 'set_csi 0' >/dev/null 2>&1; sleep 1; "
                   f"left=$(fuser {BOARD_LOCK} 2>/dev/null); [ -z \"$left\" ] && echo clean || echo \"still held: $left\"")
            out = self._ssh(f"root@{self.board}", cmd, timeout=25)
            with self._lock:
                self._note(f"board cleanup: {out}")
        if run.stamp:
            partial = self.capture_dir / f"{run.stamp}.bin"
            removed = []
            if partial.exists():
                partial.unlink()
                removed.append(partial.name)
            frames = self.frame_tmp / run.stamp
            if frames.is_dir():
                shutil.rmtree(frames, ignore_errors=True)
                removed.append(f"{frames.name}/ frames")
            archive = self.capture_dir / f"{run.stamp}_frames.tar"
            if archive.exists():
                archive.unlink()
                removed.append(archive.name)
            if self.remote:
                out = self._ssh(self.remote, f"rm -f '{self.remote_dir}/live/{run.stamp}.bin' && echo removed", timeout=12)
                removed.append(f"lg live copy: {out}")
            with self._lock:
                self._note("partial capture removed: " + (", ".join(removed) if removed else "nothing to remove"))

    def _ssh(self, host: str, command: str, timeout: float) -> str:
        try:
            res = subprocess.run(["ssh", *SSH_OPTS, "-o", f"ConnectTimeout={int(min(10, timeout))}", host, command],
                                 capture_output=True, text=True, timeout=timeout)
            text = (res.stdout.strip() or res.stderr.strip() or f"rc={res.returncode}")
            return text.splitlines()[-1] if text else f"rc={res.returncode}"
        except subprocess.TimeoutExpired:
            return "timed out"
        except OSError as exc:
            return f"ssh failed: {exc}"

    # ------------------------------------------------------------------ log reading
    def _script_log(self) -> Path:
        return self.capture_dir / "mtk_capture.log"

    def _read_script_log(self, run: Run) -> None:
        path = self._script_log()
        if not path.exists():
            return
        with open(path, "rb") as fh:
            fh.seek(self._log_offset)
            chunk = fh.read()
        if not chunk:
            return
        self._log_offset += len(chunk)
        for raw in chunk.decode("utf-8", "replace").splitlines():
            line = raw.rstrip()
            if not line:
                continue
            with self._lock:
                self.log.append(line)
                if len(self.log) > 4 * LOG_TAIL:
                    del self.log[: len(self.log) - 2 * LOG_TAIL]
                # "YYYY-mm-dd HH:MM:SS  TAG ..." -- 19 characters of clock, two spaces, then the line proper.
                body = line[21:] if len(line) > 21 and line[19:21] == "  " else line
                tag = body.split()[0] if body.split() else ""
                if tag == "START":
                    parts = body.split()
                    if len(parts) > 1 and parts[1].endswith(".bin"):
                        run.stamp = parts[1][:-4]
                    run.capture_started_at = time.time()
                    self.state = "running"
                elif tag == "WAKE":
                    self.state = "waking"
                elif tag in ("DONE", "FAIL", "SKIP", "ABORT"):
                    run.summary = body
                    if tag == "DONE":
                        self.state = "finishing"
                elif tag in ("UPLOAD", "ARCHIVE", "FRAMES") or tag.startswith("LABEL"):
                    self.state = "finishing"

    def _note(self, text: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log.append(f"{stamp}  web: {text}")

    # ------------------------------------------------------------------ sidecar
    def _write_sidecar(self, run: Run, job: Job) -> None:
        stamp = run.stamp
        assert stamp is not None
        try:
            local = datetime.strptime(stamp, "%Y%m%d_%H%M%S").astimezone()
            epoch = int(local.timestamp())
        except ValueError:
            epoch = None
        meta: dict[str, Any] = {"stamp": stamp, "capture_start_utc_epoch": epoch, "requested_duration_s": job.duration_s,
                                "stimulus_interval_s": PING_INTERVAL,
                                "scenario_source": "operator_reported_at_capture", "recorded_by": "web capture tab"}
        for key in ("room", "scenario", "note"):
            if getattr(job, key):
                meta[key] = getattr(job, key)
        path = self.capture_dir / f"{stamp}_meta.json"
        path.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        run.sidecar = path.name
        msg = f"sidecar {path.name} written"
        if job.upload and self.remote:
            day = stamp[:8]
            try:
                res = subprocess.run(["scp", *SSH_OPTS, "-o", "ConnectTimeout=5", "-q", str(path),
                                      f"{self.remote}:{self.remote_dir}/{day}/"], capture_output=True, text=True, timeout=30)
                msg += ", uploaded" if res.returncode == 0 else f", upload failed ({(res.stderr or '').strip()[:80] or res.returncode})"
            except (subprocess.TimeoutExpired, OSError) as exc:
                msg += f", upload failed ({exc})"
        with self._lock:
            self._note(msg)

    # ------------------------------------------------------------------ preflight
    def _lock_busy(self) -> bool:
        try:
            fd = os.open(self.local_lock, os.O_WRONLY | os.O_CREAT, 0o666)
        except OSError:
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        finally:
            os.close(fd)
        return False

    def _refresh_preflight(self, max_age_s: float = 10.0) -> None:
        with self._lock:
            p = self.preflight
            fresh = p.checked_at is not None and time.time() - p.checked_at < max_age_s
            if fresh or p.checking:
                return
            p.checking = True
            self._preflight_thread = threading.Thread(target=self._check_preflight, name="capture-preflight", daemon=True)
            self._preflight_thread.start()

    def _check_preflight(self) -> None:
        board = None
        if self.board:
            try:
                board = subprocess.run(["ping", "-c", "1", "-W", "1", self.board], capture_output=True, timeout=5).returncode == 0
            except (subprocess.TimeoutExpired, OSError):
                board = False
        busy = holder = None
        if os.path.exists(self.webcam_dev):
            try:
                res = subprocess.run(["fuser", "-v", self.webcam_dev], capture_output=True, text=True, timeout=5)
                busy = res.returncode == 0
                if busy:
                    tail = (res.stderr or res.stdout).strip().splitlines()
                    holder = tail[-1].split()[-1] if tail else None
            except (subprocess.TimeoutExpired, OSError):
                busy = None
        remote = None
        if self.remote:
            try:
                remote = subprocess.run(["ssh", *SSH_OPTS, "-o", "ConnectTimeout=3", self.remote, "true"],
                                        capture_output=True, timeout=8).returncode == 0
            except (subprocess.TimeoutExpired, OSError):
                remote = False
        try:
            free = int(shutil.disk_usage(self.capture_dir if self.capture_dir.exists() else self.capture_dir.parent).free // 1_000_000)
        except OSError:
            free = None
        with self._lock:
            self.preflight = Preflight(checked_at=time.time(), board_reachable=board, webcam_busy=busy, webcam_holder=holder,
                                       remote_reachable=remote, disk_free_mb=free, capture_lock_busy=self._lock_busy(), checking=False)


# ---------------------------------------------------------------------- singleton
_manager: CaptureManager | None = None
_manager_lock = threading.Lock()


def manager() -> CaptureManager:
    """The process-wide manager, configured from the environment on first use.

    ``CAPTURE_SCRIPT``, ``CAPTURE_DIR``, ``CAPTURE_BOARD`` (empty disables the board step), ``CAPTURE_REMOTE`` (empty
    disables the collection host), ``CAPTURE_REMOTE_DIR``, ``CAPTURE_LOCK`` (the script's flock file).
    """
    global _manager
    with _manager_lock:
        if _manager is None:
            env = os.environ
            _manager = CaptureManager(
                script=env.get("CAPTURE_SCRIPT", str(DEFAULT_SCRIPT)),
                capture_dir=env.get("CAPTURE_DIR") or None,
                board=env.get("CAPTURE_BOARD", DEFAULT_BOARD) or None,
                remote=env.get("CAPTURE_REMOTE", DEFAULT_REMOTE) or None,
                remote_dir=env.get("CAPTURE_REMOTE_DIR", DEFAULT_REMOTE_DIR),
                local_lock=env.get("CAPTURE_LOCK", LOCAL_LOCK),
            )
        return _manager


def reset_manager() -> None:
    """Forget the singleton (tests)."""
    global _manager
    with _manager_lock:
        _manager = None
