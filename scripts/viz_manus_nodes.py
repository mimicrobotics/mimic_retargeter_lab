"""Real-time 3D visualization of MANUS skeleton nodes with ID labels.

Connects to the ManusZmqBridge and plots all 25 nodes as a live 3D scatter
plot with node ID labels. Move one finger at a time to identify which IDs
belong to which finger.

Usage:
    python scripts/viz_manus_nodes.py          # right hand (default)
    python scripts/viz_manus_nodes.py LEFT     # left hand
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import zmq

HOST = "127.0.0.1"
PORT = 8000
NUM_NODES = 25

topic = sys.argv[1].upper() if len(sys.argv) > 1 else "RIGHT"

# --- ZMQ setup ---
ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.setsockopt(zmq.CONFLATE, True)
sock.setsockopt_string(zmq.SUBSCRIBE, topic)
sock.connect(f"tcp://{HOST}:{PORT}")

print(f"Connected. Subscribing to '{topic}'. Waiting for first frame...")

# --- Matplotlib setup (non-blocking) ---
plt.ion()
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_title(f"MANUS Skeleton Nodes — {topic} hand")

# Pre-create scatter and text artists for speed
scatter = ax.scatter([], [], [], s=60, c="blue", depthshade=True)
labels = [
    ax.text(0, 0, 0, str(i), fontsize=8, color="red", fontweight="bold")
    for i in range(NUM_NODES)
]

# Axis limits (will auto-adjust after first frame)
limits_set = False

try:
    while plt.fignum_exists(fig.number):
        # Non-blocking recv — skip if no new data
        try:
            msg = sock.recv_string(flags=zmq.NOBLOCK)
        except zmq.Again:
            plt.pause(0.01)
            continue

        # Parse
        space_idx = msg.find(" ")
        if space_idx == -1:
            continue
        frame = json.loads(msg[space_idx + 1 :])
        nodes = frame.get("nodes", [])
        if len(nodes) < NUM_NODES:
            continue

        # Extract positions indexed by node ID
        positions = np.zeros((NUM_NODES, 3), dtype=np.float32)
        for n in nodes:
            nid = n["id"]
            if nid < NUM_NODES:
                positions[nid] = n["pos"]

        # Update scatter
        scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])

        # Update labels
        for i in range(NUM_NODES):
            labels[i].set_position_3d(positions[i])
            labels[i].set_text(str(i))

        # Auto-fit axis limits on first frame
        if not limits_set:
            center = positions.mean(axis=0)
            span = max(np.ptp(positions, axis=0)) * 0.6
            ax.set_xlim(center[0] - span, center[0] + span)
            ax.set_ylim(center[1] - span, center[1] + span)
            ax.set_zlim(center[2] - span, center[2] + span)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            limits_set = True

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

except KeyboardInterrupt:
    print("\nDone.")
finally:
    sock.close()
    ctx.term()
    plt.close("all")
