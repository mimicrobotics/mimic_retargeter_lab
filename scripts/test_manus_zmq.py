"""Smoke test: connect to the MANUS ZMQ bridge and print skeleton data.

Prerequisites:
  1. ManusZmqBridge.out is running

Usage:
  python scripts/test_manus_zmq.py              # show both hands
  python scripts/test_manus_zmq.py RIGHT         # show right hand only
  python scripts/test_manus_zmq.py LEFT           # show left hand only
"""

import json
import sys

import zmq

HOST = "127.0.0.1"
PORT = 8000

# Subscribe to a specific hand topic, or "" for all
topic = sys.argv[1].upper() if len(sys.argv) > 1 else ""

ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.setsockopt(zmq.CONFLATE, True)
sock.setsockopt_string(zmq.SUBSCRIBE, topic)
sock.connect(f"tcp://{HOST}:{PORT}")

filter_label = topic if topic else "ALL"
print(
    f"Connected to tcp://{HOST}:{PORT}. Filter: {filter_label}. Waiting for data...\n"
)

try:
    while True:
        msg = sock.recv_string()

        # Strip topic prefix
        space_idx = msg.find(" ")
        if space_idx == -1:
            print(f"  Malformed message: {msg[:80]}")
            continue

        side = msg[:space_idx]
        frame = json.loads(msg[space_idx + 1 :])

        glove_id = frame.get("glove_id", "?")
        nodes = frame.get("nodes", [])
        print(f"[{side}] Glove: {glove_id}  |  {len(nodes)} nodes")

        for node in nodes:
            nid = node["id"]
            px, py, pz = node["pos"]
            qw, qx, qy, qz = node["quat"]
            print(
                f"  node {nid:2d}  pos=({px:+.4f}, {py:+.4f}, {pz:+.4f})  "
                f"quat(wxyz)=({qw:+.4f}, {qx:+.4f}, {qy:+.4f}, {qz:+.4f})"
            )

        print()

except KeyboardInterrupt:
    print("\nDone.")
finally:
    sock.close()
    ctx.term()
