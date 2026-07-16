# Regenerating the hand preview animations

The animated previews in the [supported hands table](../README.md#supported-hands)
are WebP files under `media/hands/`, rendered offscreen from each hand's MJCF by
[`scripts/generate_robot_hand_webp.py`](../scripts/generate_robot_hand_webp.py).
They are committed to the repo, so you only need to regenerate them when a hand's
model or assets change — or when you add a new hand.

## Regenerating all previews

```bash
MUJOCO_GL=osmesa python scripts/generate_robot_hand_webp.py --all --frames 480 --fps 24
```

That renders every hand in the registry as a 20-second loop orbiting at roughly
18°/s, and writes `media/hands/<hand_name>.webp`.

For a single hand:

```bash
MUJOCO_GL=osmesa python scripts/generate_robot_hand_webp.py --hand shadow_hand
```

`MUJOCO_GL` selects MuJoCo's offscreen rendering backend. On a headless machine
set it to `osmesa` (software, always available) or `egl` (GPU-accelerated). On a
desktop with a display you can omit it.

## Options

Run with `--help` for the full list. The ones you are most likely to reach for:

| Flag | Default | Purpose |
|------|---------|---------|
| `--hand` / `--all` | — | One hand, or every registered hand (mutually exclusive, one required) |
| `--frames` | 360 | Frames rendered per loop |
| `--fps` | 30 | Playback rate; with `--frames`, sets both loop length and orbit speed |
| `--size` | 480 | Output resolution in pixels (square) |
| `--quality` | 80 | WebP quality 0-100; 60-70 still looks fine and is noticeably smaller |
| `--sweep` | 360 | Angular range; 360 is a full orbit, smaller values swing sinusoidally |
| `--start-azimuth` | 90 | Center azimuth of the rotation, in degrees |
| `--elevation` | -10 | Camera elevation, in degrees |
| `--distance-scale` | 0.8 | Camera distance as a multiple of the just-fits distance; <1 zooms in |
| `--no-spotlight` | off | Drop the injected overhead spotlight, keeping only the scene's own lights |
| `--out-dir` | `media/hands/` | Where the `.webp` files are written |

Loop length is `frames / fps` seconds and orbit speed is `360 / (frames / fps)`
degrees per second — so the command above (480 frames at 24 fps) yields a 20 s
loop at 18°/s. Raising `--frames` makes the motion smoother and the file larger.

## Adding a preview for a new hand

The script renders whatever is in its hand registry, so a hand integrated per
[integrating_new_hand.md](integrating_new_hand.md) is picked up automatically by
`--all`. After generating, add a row to the table in the README pointing at the
new `media/hands/<hand_name>.webp`.
