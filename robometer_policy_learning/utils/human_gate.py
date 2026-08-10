"""HG-DAgger baseline: the HUMAN decides when to intervene; pi0 still performs the correction.

Either way a person decides the ONE step at which the expert takes over. The backends differ in
when they ask, which is a real trade-off in reaction-time realism:

    live    The operator watches the rollout as it happens and hits SPACE to take over. Genuine
            HG-DAgger, with a genuine reaction delay, but it needs a display and a human sitting
            there, so it cannot run on a headless compute node.

    replay  The episode is rolled out student-solo first, then the operator watches the recording
            -- with the step index burned into every frame -- and answers "intervene? y/n" and, if
            yes, at which step. The SAME episode is then re-run with the expert taking over there.
            Headless, and the operator judges without time pressure, which biases this arm's
            LATENCY favourably; that is the right direction for a human upper bound.

Both ask about the CURRENT policy's rollouts, so both work inside a DAgger loop where the student
changes every iteration. See ``collect_interactive_episode`` for the replay driver.

Determinism requirement for `replay`: the re-run must unfold identically to the recording the
operator judged, but ONLY up to the takeover step -- after that the trajectory legitimately
diverges because the expert has control. Re-seeding per episode is sufficient; see ``seed_episode``.

Usage: 
On local computer: 
    ssh -L 8420:localhost:8420 liryan@snoopy1

Then, in the same terminal: 
    cd /scr/liryan/robometer_policy_learning
    srun --gres=shard:8 --mem=48G --time=12:00:00 --pty \
        uv run python scripts/train_reward_dagger.py \
        --config-name libero_hgdagger_task1_live_config \
        rdagger.num_iterations=3 rdagger.rollouts_per_iter=5
Open http://localhost:8420 to collect data.
"""

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np


def render_operator_view(frame, step: int, scale: int = 3, note: str = "",
                         already_bgr: bool = False, label: Optional[str] = None):
    """The exact image the operator sees, in BGR for cv2.

    The step index is burned into every frame because BOTH backends depend on a person reading a
    number off the footage and that number matching `gate_fires` exactly -- scrubbing an ordinary
    mp4 would make it guesswork.

    ``label`` ("STUDENT" / "EXPERT") draws the who-is-driving banner across the top. It lives here
    rather than in the caller so the live stream and the recorded video cannot disagree about it --
    they previously did, because the worker drew the banner itself and the live path never went
    through that code.
    """
    import cv2

    img = np.asarray(frame)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))              # CHW -> HWC
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if not already_bgr:
        img = img[..., ::-1]                            # RGB -> BGR
    img = np.ascontiguousarray(img)
    if scale != 1:
        img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale),
                         interpolation=cv2.INTER_NEAREST)
    # Step index anchored to the BOTTOM so it never collides with the top banner.
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 26), (w, h), (0, 0, 0), -1)
    cv2.putText(img, f"step {step}", (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2, cv2.LINE_AA)
    if note:
        cv2.putText(img, note, (116, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)
    if label:
        bar = max(18, int(9 * scale))
        colour = (0, 0, 255) if label == "EXPERT" else (0, 180, 0)   # BGR: red / green
        cv2.rectangle(img, (0, 0), (w, bar), colour, -1)
        cv2.putText(img, label, (5, bar - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45 * max(1, scale * 0.6), (255, 255, 255), 1, cv2.LINE_AA)
    return img


OPERATOR_PAGE = b"""<!doctype html><meta charset=utf-8><title>HG-DAgger operator</title>
<style>
 body{background:#111;color:#eee;font:14px system-ui;margin:0;display:flex;
      flex-direction:column;align-items:center;gap:12px;padding:16px}
 img{image-rendering:pixelated;border:1px solid #333;max-width:96vw}
 .row{display:flex;gap:12px}
 button{font:600 20px system-ui;padding:14px 40px;border:0;border-radius:8px;
      color:#fff;cursor:pointer}
 #btn{background:#c62828}
 #nxt{background:#2e7d32}
 button:disabled{background:#444;color:#888;cursor:default}
 #s{color:#9e9e9e;font-variant-numeric:tabular-nums}
</style>
<img src="/stream">
<div class=row>
  <button id=btn>TAKE OVER &nbsp;(space)</button>
  <button id=nxt>START EPISODE &nbsp;(r)</button>
</div>
<div id=s>connecting...</div>
<script>
const btn=document.getElementById('btn'), nxt=document.getElementById('nxt'),
      s=document.getElementById('s');
async function post(p,b){
  if(b.disabled) return;
  b.disabled=true;
  await fetch(p,{method:'POST'});
  setTimeout(()=>{b.disabled=false;},1000);
}
btn.onclick=()=>post('/takeover',btn);
nxt.onclick=()=>post('/next',nxt);
addEventListener('keydown',e=>{
  if(e.code==='Space'){e.preventDefault();post('/takeover',btn);}
  if(e.code==='KeyR'){e.preventDefault();post('/next',nxt);}
});
setInterval(async()=>{
  try{ s.textContent=await (await fetch('/status')).text(); }catch(e){ s.textContent='disconnected'; }
},500);
</script>"""


class OperatorServer:
    """Serves the operator view over HTTP so a human can drive 'live' mode with no X display.

    Why not cv2.imshow: this env has opencv-python-headless installed (GUI: NONE) alongside two
    other opencv builds, and the compute box has no DISPLAY. Rather than untangle that in an env
    MuJoCo also renders through, the frames go out as MJPEG and the takeover comes back as a POST.
    Reach it with `ssh -L <port>:localhost:<port> <host>`, the same tunnel pattern as
    scripts/robometer_http_server.py.

    Renders through ``render_operator_view`` so the operator's pixels cannot drift from the
    annotation video's.
    """

    def __init__(self, port: int = 8420, quality: int = 80):
        self.port = int(port)
        self.quality = int(quality)
        self._jpeg: Optional[bytes] = None
        self._status = "waiting for first frame"
        self._takeover = False
        self._start = False
        self._lock = __import__("threading").Lock()
        self._httpd = None
        self._thread = None

    # ---------------------------------------------------------------- lifecycle
    def start(self):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):        # keep the rollout log readable
                pass

            def _send(self, code, ctype, body: bytes):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.startswith("/stream"):
                    return self._stream()
                if self.path.startswith("/status"):
                    return self._send(200, "text/plain; charset=utf-8",
                                      outer.status.encode())
                self._send(200, "text/html; charset=utf-8", OPERATOR_PAGE)

            def do_POST(self):
                if self.path.startswith("/takeover"):
                    outer.request_takeover()
                    return self._send(200, "text/plain", b"ok")
                if self.path.startswith("/next"):
                    outer.request_start()
                    return self._send(200, "text/plain", b"ok")
                self._send(404, "text/plain", b"no")

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=f")
                self.end_headers()
                try:
                    while True:
                        frame = outer.frame
                        if frame is not None:
                            self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n"
                                             b"Content-Length: " + str(len(frame)).encode()
                                             + b"\r\n\r\n" + frame + b"\r\n")
                        time.sleep(0.05)      # ~20 fps ceiling
                except (BrokenPipeError, ConnectionResetError):
                    pass                       # operator closed the tab

        self._httpd = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        print(f"\n  operator UI on http://localhost:{self.port}"
              f"   (ssh -L {self.port}:localhost:{self.port} {os.uname().nodename})\n")
        return self

    def stop(self):
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    # ---------------------------------------------------------------- state
    @property
    def frame(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    def publish(self, bgr, status: str = ""):
        """Push one already-rendered BGR frame to the stream."""
        import cv2

        ok, buf = cv2.imencode(".jpg", bgr,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if ok:
            with self._lock:
                self._jpeg = buf.tobytes()
                if status:
                    self._status = status

    def request_takeover(self):
        with self._lock:
            self._takeover = True

    def take_takeover(self) -> bool:
        """Read-and-clear: True at most once per press."""
        with self._lock:
            fired, self._takeover = self._takeover, False
        return fired

    def request_start(self):
        with self._lock:
            self._start = True

    def wait_for_start(self, label: str):
        """Block until the operator presses START EPISODE / r.

        Held between episodes so the operator is never surprised by a rollout that began while
        they were looking away -- a takeover decision needs the whole episode watched from step 0.
        The HTTP server runs on its own thread, so the stream and buttons stay live while this
        blocks.
        """
        with self._lock:
            self._start = False
            # Drop any takeover pressed while we were NOT polling -- during the gap between
            # episodes nothing consumes the flag, so a stray space would survive into the next
            # episode and fire at step 0, handing the expert the whole rollout.
            self._takeover = False
            self._status = f"{label} -- press r / START EPISODE to begin"
        print(f"  [operator] waiting for START ({label}) -> http://localhost:{self.port}")
        while True:
            with self._lock:
                if self._start:
                    self._start = False
                    # Cleared on RELEASE, not just on entry: a space pressed at any point during
                    # the wait would otherwise survive into step 0 of the episode about to start.
                    self._takeover = False
                    return
            time.sleep(0.05)


class HumanScorer:
    """Returns 1.0 to intervene, 0.0 otherwise."""

    def __init__(self, backend: str = "replay", frame_key: str = "agentview_image",
                 window: str = "HG-DAgger", scale: int = 3,
                 port: int = 8420, pace_hz: float = 20.0):
        if backend not in ("live", "replay"):
            raise ValueError(f"backend must be 'live' or 'replay', got {backend!r}")
        self.backend = backend
        self.frame_key = frame_key
        self.window = window
        self.scale = int(scale)
        # 'live' only. The rollout is otherwise as fast as the sim + policy allow, so an operator
        # watching it would be reacting to a fast-forward -- and the measured reaction time would
        # be an artefact of machine speed. Pace the loop to a fixed wall-clock rate instead, and
        # report that rate alongside any latency number.
        self.pace_hz = float(pace_hz)
        self.server = OperatorServer(port=port) if backend == "live" else None
        self._next_frame_at = 0.0
        self._took_over = False      # 'live': has the operator already handed over this episode?

        self.task = ""
        self.episode_id = 0          # ordinal, for logging
        self.step = -1               # aligned to the worker's `steps` -- see observe()
        self._frame = None           # 'live': latest frame to show the operator
        self._seen_reset = False
        # 'replay': the step this rollout hands over on, set by the driver between the solo pass
        # and the takeover pass. None = run student-solo to the end.
        self.takeover_step: Optional[int] = None

    def set_takeover(self, step: Optional[int]):
        """'replay': fix the takeover step for the NEXT rollout. None = no intervention."""
        self.takeover_step = None if step is None else int(step)

    def reset(self, task: str = ""):
        self.task = str(task)
        self.episode_id += 1
        self.step = -1
        self._frame = None
        self._seen_reset = False
        self._next_frame_at = 0.0
        self._took_over = False
        if self.backend == "live":
            # Always wait for the operator to press r. The alternative -- starting the next
            # rollout the instant the previous one ends -- means an operator who looked away has
            # already missed the steps their takeover decision depends on, so there is no sane
            # setting other than "on".
            if self.server._httpd is None:
                self.server.start()
            self.server.wait_for_start(f"episode {self.episode_id}")

    def observe(self, obs, frame_key: Optional[str] = None):
        key = frame_key or self.frame_key
        if self.backend == "live" and key in obs:
            self._frame = np.asarray(obs[key])
        # Skip the initial reset observation so step 0 matches the first real env step.
        if not self._seen_reset:
            self._seen_reset = True
            return
        self.step += 1

    def score(self):
        if self.backend == "live":
            return (1.0 if self._live_fires() else 0.0), 0.0
        fires = self.takeover_step is not None and self.step == self.takeover_step
        return (1.0 if fires else 0.0), 0.0

    def _live_fires(self) -> bool:
        """Publish the current frame to the operator UI; True once they hit TAKE OVER / space."""
        if self._frame is None:
            return False
        if self.server._httpd is None:
            self.server.start()

        # Hold the loop to pace_hz so the operator sees real time, not a fast-forward.
        now = time.monotonic()
        if self._next_frame_at and now < self._next_frame_at:
            time.sleep(self._next_frame_at - now)
        self._next_frame_at = time.monotonic() + (1.0 / self.pace_hz if self.pace_hz > 0 else 0.0)

        # score() keeps being called after the takeover, so the stream stays live through the
        # expert's segment -- show who is actually driving.
        fired = self.server.take_takeover()
        if fired:
            self._took_over = True
        who = "EXPERT" if self._took_over else "STUDENT"
        self.server.publish(
            render_operator_view(self._frame, self.step, scale=self.scale,
                                 note="" if self._took_over else "SPACE=take over",
                                 already_bgr=False, label=who),
            status=f"ep {self.episode_id}  step {self.step}  [{who}]  ({self.pace_hz:g} Hz)")
        return fired


class HumanGate:
    def __init__(self, **_ignored: Any):
        from collections import deque
        self.history = deque(maxlen=1)
        self.short_window = 1
        self.last_trigger = None

    def update(self, value) -> bool:
        v = float(value)
        self.history.append(v)
        fired = v > 0.5
        if fired:
            self.last_trigger = "human"
        return fired

    def reset(self):
        self.history.clear()

    def describe(self) -> Dict[str, Any]:
        return dict(gate="hgdagger")


def _ask(prompt_fn, question: str, validate):
    """Re-ask until `validate` accepts the answer. Returns the validated value."""
    while True:
        try:
            raw = prompt_fn(question).strip()
        except EOFError:
            # Almost always a missing TTY: `srun` without --pty gives the task no stdin, so the
            # first prompt hits EOF. Say so, rather than dumping a bare traceback.
            raise SystemExit(
                "\nHG-DAgger 'replay' needs an interactive terminal to ask the operator, but "
                "stdin is closed.\nRe-run with a TTY, e.g. `srun --pty ...`, or drive it "
                "programmatically via collect_interactive_episode(prompt_fn=...)."
            ) from None
        ok, value = validate(raw)
        if ok:
            return value
        print(f"    ? '{raw}' is not valid -- try again.")


def collect_interactive_episode(worker, scorer, tag: str, seed: int, *,
                                store: bool = True, prompt_fn=input) -> Dict[str, Any]:
    """'replay' mode in HG-DAgger: two passes over each episode.

    Pass 1 rolls the student out solo and writes a video with the step index burned into every
    frame. The operator watches it and answers "intervene? y/n", then names the step. Pass 2
    re-runs that identical episode with pi0 taking over at that step. The episode is kept ONLY if pi0's rescue succeeds.

    Returns a record dict; ``stats`` is None when the operator declines to intervene.
    """
    def _run(pass_tag, takeover, store_this):
        seed_episode(seed)                 # identical episode in both passes, up to the takeover
        scorer.set_takeover(takeover)
        return worker.rollout_episode(pass_tag, store=store_this,
                                      require_success=True, require_intervention=True)

    # Student solo rollout
    solo = _run(f"{tag}_solo", None, False)
    n = int(solo["steps"])
    video = os.path.join(worker.video_dir, f"gated_{tag}_solo.mp4") if worker.video_dir else None
    print(f"\n  [{tag}] student-solo: {n} steps, success={solo['success']}")
    if video:
        print(f"  watch: {video}   ")

    yes = _ask(prompt_fn, "  intervene? [y/n] ",
               lambda r: (True, True) if r.lower() in ("y", "yes")
               else ((True, False) if r.lower() in ("n", "no") else (False, None)))
    if not yes:
        print("  -> declined; no human data from this episode")
        return dict(tag=tag, takeover_step=None, declined=True, stats=None, kept=False,
                    solo_steps=n, solo_success=bool(solo["success"]))
    
    lo = 0

    def _valid_step(r):
        if not r.lstrip("-").isdigit():
            return False, None
        v = int(r)
        return (lo <= v < n), v          # must be a step that actually happened, and be fireable
    step = _ask(prompt_fn, f"  takeover step [{lo}..{n - 1}]: ", _valid_step)

    # Expert takeover rollout
    stats = _run(tag, step, store)
    kept = bool(stats["success"]) and (not store or stats["stored"] > 0)
    print(f"  -> takeover@{step}: success={stats['success']}, expert_steps={stats['expert_steps']}, "
          f"stored={stats['stored']}  [{'KEPT' if kept else 'DISCARDED'}]")
    return dict(tag=tag, takeover_step=step, declined=False, stats=stats, kept=kept,
                solo_steps=n, solo_success=bool(solo["success"]))


def seed_episode(seed: int):
    """Make the student's action sampling reproducible up to the takeover step, so a replayed
    episode matches the footage the annotator watched."""
    import random

    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def human_effort_steps(episode_stats: List[dict]) -> Dict[str, float]:
    """HG-DAgger's effort is monitoring the WHOLE episode, not just the interventions.

    Robot-gated arms should be charged only their expert steps (plus context-switch latency);
    reporting both on the same axis is the point of the money plot, so keep the distinction
    explicit rather than burying it in a constant.
    """
    total_steps = float(sum(s["steps"] for s in episode_stats))
    expert_steps = float(sum(s.get("expert_steps", 0) for s in episode_stats))
    return dict(
        monitored_steps=total_steps,          # what HG-DAgger costs a human
        expert_steps=expert_steps,            # what a robot-gated arm costs a human
        episodes=float(len(episode_stats)),
    )
