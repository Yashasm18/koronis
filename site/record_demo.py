"""Record the README's screen capture of the demo console, and encode it.

Drives the built docs/index.html in Chromium, films the replay and a tour of the
evaluation and method tabs, then writes the three assets the README links:

    docs/assets/koronis-demo.gif   the inline hero image
    docs/assets/koronis-demo.mp4   the full-resolution recording

Finally it refreshes docs/assets/koronis-demo.stamp, which
tests/test_demo_recording_is_current.py checks so a re-run of the experiments
cannot leave a recording of the old numbers sitting at the top of the README.

Needs Playwright's Chromium (requirements-dev.txt) and ffmpeg on PATH.

    python site/build.py && python site/record_demo.py
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile

from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
INDEX = ROOT / "docs" / "index.html"

# The replay is paced against the wall clock: one pass is 42 s at 1x. An earlier
# version of this script throttled requestAnimationFrame to slow the replay down.
# That worked only while the replay advanced a fixed amount per frame; once the
# pacing moved onto the wall clock, throttling cut the frame rate and nothing
# else, so the recording came out juddery and cut away mid-replay. Use the page's
# own speed control instead - the same one a viewer has. 42 s / 4 = ~11 s.
SPEED = 4
W, H = 1280, 800

# 64 colours is ample for a near-monochrome console, and 8 fps keeps the size
# down. The previous recording compressed better only because the broken frame
# throttle produced near-duplicate frames; a recording that actually animates
# needs the palette and frame rate turned down instead.
# 640 px rather than 760: measured at 2.1 MB against 2.9 MB for the same 20 s,
# and the headline figures, feed rows and threshold label all stay legible - the
# GIF is a teaser, the MP4 and the live site carry the detail.
GIF_FPS, GIF_COLORS, GIF_W = 8, 64, 640

# The GIF is trimmed to the product story - load, replay, the finished board,
# the incident panel and the dossier - and stops before the documentation tour.
# That tour is worth watching and is why the MP4 exists; it is not worth making
# a reader wait for. At 6.5 MB the full-length GIF left the top of the README
# blank for three to four seconds, because a browser will not paint an animated
# GIF until a large part of it has arrived: the first frame here is decodable
# after 3,471 bytes, so the delay was transfer, not decode.
GIF_SECONDS = 20
#: Mean-luma gap that counts as "something has been drawn", on a 0-255 scale.
BLANK_LUMA_TOLERANCE = 1.5
MP4_W, MP4_H = 1200, 750


def capture(out_dir: pathlib.Path) -> pathlib.Path:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": W, "height": H},
            record_video_dir=str(out_dir),
            record_video_size={"width": W, "height": H},
        )
        pg = ctx.new_page()
        pg.goto(INDEX.as_uri())
        pg.wait_for_timeout(1800)

        # --- replay, watched from the top (stats + chart + live feed) ---
        pg.click(f'.speed button[data-speed="{SPEED}"]')
        pg.wait_for_timeout(700)
        pg.click("#play")
        pg.wait_for_timeout(3800)
        # slide down to the evidence graph + action ladder while it is still running
        pg.mouse.move(640, 400)
        pg.mouse.wheel(0, 520)
        pg.wait_for_timeout(3400)
        pg.mouse.wheel(0, 460)

        # Wait for the stream to actually drain rather than guessing a duration -
        # a fixed sleep is what let an earlier recording cut away early.
        pg.wait_for_function(
            "() => document.getElementById('play').textContent === 'Replay incident'",
            timeout=60_000,
        )
        pg.wait_for_timeout(2600)

        # --- the audit dossier, opened ---
        pg.eval_on_selector("#dossier-wrap", "e => e.open = true")
        pg.eval_on_selector("#dossier-wrap", "e => e.scrollIntoView({block:'center'})")
        pg.wait_for_timeout(3400)

        # --- tour: evaluation, including the frontier chart ---
        pg.mouse.wheel(0, -2600)
        pg.wait_for_timeout(700)
        pg.click('.tabs button[data-tab="eval"]')
        pg.wait_for_timeout(1800)
        for _ in range(7):
            pg.mouse.wheel(0, 520)
            pg.wait_for_timeout(1400)

        # --- tour: method & limitations ---
        pg.mouse.wheel(0, -6000)
        pg.wait_for_timeout(400)
        pg.click('.tabs button[data-tab="method"]')
        pg.wait_for_timeout(1600)
        for _ in range(3):
            pg.mouse.wheel(0, 520)
            pg.wait_for_timeout(1500)
        pg.wait_for_timeout(900)

        ctx.close()
        browser.close()

    return next(iter(out_dir.glob("*.webm")))


def run(*args) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *map(str, args)], check=True)


def first_painted(src: pathlib.Path) -> float:
    """Seconds of blank page at the head of the clip, measured rather than guessed.

    Playwright starts recording when the context is created, which is before
    `goto` -- so every clip opens on an unpainted page. Four white frames at the
    head of a README GIF are the first thing a reader sees, and a hard-coded
    offset would rot the moment any wait above it changes. So the blank run is
    found by comparing mean luma against the opening frame: the first frame that
    differs is the first painted one.

    Returns 0.0 if the probe fails for any reason -- a slightly long lead-in is
    a much better failure than no recording at all -- but says so on the way out.
    The first version of this asked ffprobe for `pkt_pts_time`, which newer builds
    have removed and answer with an empty column rather than an error: the probe
    "succeeded", the row list came back empty, and the silent fallback shipped the
    white frames it was written to remove. A fallback that cannot be seen is
    indistinguishable from a fix.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-f", "lavfi", f"movie={src},signalstats",
             "-show_entries", "frame=pts_time:frame_tags=lavfi.signalstats.YAVG",
             "-of", "csv=p=0"],
            capture_output=True, text=True, timeout=120, check=True).stdout
        rows = [r.split(",") for r in out.splitlines() if "," in r]
        frames = [(float(t), float(y)) for t, y in rows]
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        print(f"  lead-in probe failed ({exc.__class__.__name__}); trimming nothing")
        return 0.0
    if not frames:
        print("  lead-in probe returned no frames; trimming nothing")
        return 0.0
    blank = frames[0][1]
    for t, y in frames:
        if abs(y - blank) > BLANK_LUMA_TOLERANCE:
            print(f"  trimming {t:.2f}s of unpainted lead-in ({len(frames)} frames probed)")
            return t
    print("  no painted frame found; trimming nothing")
    return 0.0


def encode(src: pathlib.Path, tmp: pathlib.Path) -> None:
    run("-i", src, "-vf", f"scale={MP4_W}:{MP4_H}:flags=lanczos",
        "-c:v", "libx264", "-preset", "slow", "-crf", "24",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        ASSETS / "koronis-demo.mp4")

    # Two passes: build a palette from the whole clip, then map onto it. A single
    # pass would quantise per frame and make flat panels crawl.
    pal = tmp / "palette.png"
    scale = f"scale={GIF_W}:-1:flags=lanczos"
    # Seek past the unpainted head so the GIF opens on the console, not on white.
    ss = ("-ss", f"{first_painted(src):.3f}")
    run(*ss, "-t", GIF_SECONDS, "-i", src, "-vf",
        f"fps={GIF_FPS},{scale},palettegen=stats_mode=diff:max_colors={GIF_COLORS}", pal)
    run(*ss, "-t", GIF_SECONDS, "-i", src, "-i", pal, "-lavfi",
        f"fps={GIF_FPS},{scale}[x];"
        "[x][1:v]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
        ASSETS / "koronis-demo.gif")


def main() -> int:
    if not INDEX.exists():
        print("docs/index.html is missing - run  python site/build.py  first.")
        return 1
    if shutil.which("ffmpeg") is None:
        print("ffmpeg is not on PATH; it is needed to encode the recording.")
        return 1

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        src = capture(tmp / "video")
        encode(src, tmp)

    sys.path.insert(0, str(ROOT))
    from tests.test_demo_recording_is_current import write_stamp
    write_stamp()

    for name in ("koronis-demo.gif", "koronis-demo.mp4"):
        p = ASSETS / name
        print(f"wrote {p}  ({p.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
