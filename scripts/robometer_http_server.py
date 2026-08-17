#!/usr/bin/env python3
"""Robometer scoring HTTP server -- runs on the LAB SERVER (where the GPU + weights live).

The real-world reward-DAgger control loop runs on the WORKSTATION next to the robot and calls
THIS over the network for progress scores -- exactly like it already calls the pi0 policy server
on port 8000. Keeping Robometer here means it never competes with pi0 for the workstation GPU.

Protocol (stateless): POST / with a pickled dict
    {"frames": (N, H, W, 3) uint8, "prompt": str}
-> pickled reply {"progress": float, "success_prob": float}.
The client subsamples its causal prefix to <= max_frames BEFORE sending, so payloads stay small
and scoring matches the offline scripts/label_real_world.py pipeline (same RobometerLabeler).

Run (lab server):
    srun --gres=shard:8 --mem=32G uv run python scripts/robometer_http_server.py \
        --model robometer/Robometer-4B --port 8900
Then make it reachable from the workstation (same pattern as the DROID Pinggy tunnels), e.g.
from the workstation:  ssh -N -L 8900:localhost:8900 <lab-server>   (loop uses --reward-host localhost).
Sanity check:  curl http://localhost:8900/   -> "ok".
"""

import argparse
import os
import pickle
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_real_world import RobometerLabeler  # noqa: E402  (same scorer as the offline labels)


def make_handler(labeler: RobometerLabeler, lock: threading.Lock):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence per-request logging
            pass

        def do_GET(self):  # health check
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = pickle.loads(self.rfile.read(n))
                frames = np.asarray(req["frames"])
                prompt = str(req.get("prompt", ""))
                # Single GPU model -> serialize calls (the client already scores off its control thread).
                with lock:
                    progress, success = labeler.score_prefixes(
                        frames=frames, task=prompt, end_indices=[len(frames) - 1], batch_size=1
                    )
                body = pickle.dumps({"progress": float(progress[0]), "success_prob": float(success[0])})
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[robometer-server] scoring error: {e}")
                self.send_response(500)
                self.end_headers()

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="robometer/Robometer-4B")
    ap.add_argument("--port", type=int, default=8900)
    ap.add_argument("--max-frames", type=int, default=None, help="override Robometer context frames")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    labeler = RobometerLabeler(model_path=args.model, device=device, max_frames=args.max_frames)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(labeler, threading.Lock()))
    logger.info(f"[robometer-server] listening on 0.0.0.0:{args.port} (model={args.model}, device={device})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("[robometer-server] shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
