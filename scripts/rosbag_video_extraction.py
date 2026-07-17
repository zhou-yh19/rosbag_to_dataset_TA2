#!/usr/bin/env python3
# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""Shared HEVC packet buffering and frame-extraction machinery for the rosbag
converters (HDF5 / RLDS).

The robot cameras publish H.265 with inter-coded GOPs (an IDR keyframe only
every ~1s), so per-episode image extraction must:

  (a) feed the decoder from the last IDR *before* the episode start — starting
      mid-GOP leaves the first up-to-1s of every episode undecodable;
  (b) decode every packet of the episode (P-frames depend on their
      predecessors) even though only fps-grid samples are kept.

This module buffers raw packets with their real bag timestamps, muxes them
into a temporary MPEG-TS (PyAV, no re-encode), and extracts exactly the
fps-grid frames with one ffmpeg pass per camera: NVDEC-accelerated when an
NVIDIA GPU is available, software decode otherwise. Grid frame k is the
latest camera frame at (first_tick_time + k/fps) — floor semantics, matching
the LeRobot converter.
"""

import logging
import os
import subprocess
import tempfile
from fractions import Fraction

import numpy as np

try:
    import av
    av.logging.set_level(av.logging.ERROR)
except ImportError as e:
    raise ImportError(
        "PyAV dependency not found. Install it in the current env, e.g. "
        "`pip install av`."
    ) from e


# =============================================================================
# HEVC NAL unit utilities (for IDR detection)
# =============================================================================

_NAL_START3 = b'\x00\x00\x01'


def iter_nal_types(data: bytes):
    """Yield NAL unit types in annex-B HEVC data (3- and 4-byte start codes)."""
    pos = data.find(_NAL_START3)
    n = len(data)
    while pos != -1 and pos + 3 < n:
        yield (data[pos + 3] >> 1) & 0x3F
        pos = data.find(_NAL_START3, pos + 3)


def has_idr(data: bytes) -> bool:
    """Check if HEVC data contains an IDR NAL unit (type 19 or 20).

    Stops at the first VCL NAL (type < 32): in a single access unit the first
    VCL NAL determines the picture type, so P-frame packets bail immediately.
    """
    for nal_type in iter_nal_types(data):
        if nal_type < 32:
            return nal_type in (19, 20)
    return False


def packets_from_last_idr(window) -> list:
    """Return the tail of a packet window starting at its last IDR packet.

    `window` is an iterable of dicts with at least a 'data' key. Returns []
    when the window contains no IDR (the caller should then buffer from the
    next incoming IDR instead).
    """
    packets = list(window)
    for i in range(len(packets) - 1, -1, -1):
        if has_idr(packets[i]['data']):
            return packets[i:]
    return []


def detect_gpu_count() -> int:
    """Count NVIDIA GPUs visible to CUDA (respects CUDA_VISIBLE_DEVICES)."""
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        pass
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return 0
        return sum(1 for line in result.stdout.splitlines() if line.startswith("GPU "))
    except Exception:
        return 0


# =============================================================================
# Packet muxing and grid-frame extraction
# =============================================================================

def mux_packets_to_ts(packets: list, codec: str = 'hevc',
                      width: int = 0, height: int = 0) -> str:
    """Mux raw annex-B packets into a temp MPEG-TS with PTS from bag timestamps.

    `packets` is a list of dicts with 'data' (bytes) and 'ts' (bag time in
    seconds). MPEG-TS stores H.265 in annex-B natively, so packet bytes go in
    unchanged; only the real-timing PTS index is added. Returns the temp file
    path (caller must unlink).
    """
    # Prefer RAM-backed /dev/shm: the .ts is a pure intermediate that ffmpeg
    # reads back immediately, no reason to round-trip the disk. Fall back to
    # the default temp dir when shm can't hold the episode (small-RAM boxes,
    # whole-bag-sized episodes).
    total_bytes = sum(len(p['data']) for p in packets)
    tmp_dir = "/dev/shm" if os.path.isdir("/dev/shm") else None
    if tmp_dir is not None:
        try:
            st = os.statvfs(tmp_dir)
            if st.f_bavail * st.f_frsize < total_bytes * 2:
                tmp_dir = None
        except OSError:
            tmp_dir = None
    fd, ts_path = tempfile.mkstemp(suffix=".ts", prefix="rosbag_ep_", dir=tmp_dir)
    os.close(fd)
    container = av.open(ts_path, "w", format="mpegts")
    stream = container.add_stream(codec)
    if width and height:
        stream.width = width
        stream.height = height
    if codec == 'hevc':
        # We never encode through this context (packets are muxed as-is),
        # but PyAV still opens it — keep libx265's banner out of the logs
        stream.codec_context.options = {'x265-params': 'log-level=none'}
    time_base = Fraction(1, 90000)
    base_ts = packets[0]['ts']
    last_pts = -1
    for p in packets:
        pkt = av.Packet(p['data'])
        pts = int(round((p['ts'] - base_ts) * 90000))
        if pts <= last_pts:  # guard against duplicate/rounded-equal timestamps
            pts = last_pts + 1
        last_pts = pts
        pkt.pts = pts
        pkt.dts = pts
        pkt.time_base = time_base
        pkt.stream = stream
        container.mux(pkt)
    # Sentinel: re-mux the last packet 1s past the end. The fps filter only
    # emits a grid slot once it sees a frame beyond it, so without this the
    # final tick can go unemitted when the last real frame lands just before
    # it. The sentinel's own (re-decoded) picture is never output — it lies
    # beyond the requested grid.
    sentinel = av.Packet(packets[-1]['data'])
    sentinel.pts = sentinel.dts = last_pts + 90000
    sentinel.time_base = time_base
    sentinel.stream = stream
    container.mux(sentinel)
    container.close()
    return ts_path


def _build_extract_cmd(ts_path: str, first_tick_offset: float, fps: int,
                       n_grid: int, vf_post: str, use_gpu: bool,
                       gpu_index: int) -> list:
    """ffmpeg command decoding the TS and emitting grid frames as raw BGR24."""
    resample = (
        f"fps={fps}:start_time={first_tick_offset:.6f}:round=up,"
        f"setpts=PTS-STARTPTS"
    )
    if use_gpu:
        # fps/setpts are timestamp-only filters and pass CUDA hw frames
        # through untouched; hwdownload brings only the ~fps sampled frames
        # back to system memory.
        vf = resample + ",hwdownload,format=nv12"
        head = [
            "ffmpeg", "-y", "-v", "error",
            "-hwaccel", "cuda",
            "-hwaccel_device", str(gpu_index),
            "-hwaccel_output_format", "cuda",
            "-c:v", "hevc_cuvid",
            "-i", ts_path,
        ]
    else:
        vf = resample
        head = ["ffmpeg", "-y", "-v", "error", "-i", ts_path]
    if vf_post:
        vf += "," + vf_post
    return head + [
        "-vf", vf,
        # Pass filter output through 1:1 — without this, ffmpeg 7.x CFR-syncs
        # the rawvideo pipe to its default 25fps and silently drops ~44% of
        # the 45fps grid frames (4.4's default sync was already passthrough).
        "-vsync", "passthrough",
        "-frames:v", str(n_grid),
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]


def extract_grid_frames(
    packets: list,
    first_tick_time: float,
    n_grid: int,
    fps: int,
    out_width: int,
    out_height: int,
    vf_post: str = "",
    gpu_index: int = 0,
    use_gpu: bool = True,
    logger: logging.Logger = None,
):
    """Yield the episode's fps-grid frames as BGR24 ndarrays, in grid order.

    Grid frame k (k = 0..n_grid-1) is the latest camera frame at wall time
    first_tick_time + k/fps. If the source ends early, the last decoded frame
    is repeated (it IS the floor-correct content for the remaining ticks).

    Args:
        packets: list of {'data': bytes, 'ts': float} starting at an IDR
        first_tick_time: bag time of grid tick 0
        n_grid: number of grid frames to yield
        fps: grid rate
        out_width/out_height: frame size AFTER vf_post (pipe framing)
        vf_post: extra CPU-side filters after decode+sampling (e.g. crop/scale)
        gpu_index: CUDA device for NVDEC
        use_gpu: try NVDEC first (falls back to software decode on failure)
        logger: optional logger for fallback warnings
    """
    log = logger or logging.getLogger(__name__)
    frame_bytes = out_width * out_height * 3
    ts_path = mux_packets_to_ts(packets)
    first_tick_offset = max(first_tick_time - packets[0]['ts'], 0.0)

    try:
        attempts = ([True, False] if use_gpu else [False])
        for attempt_gpu in attempts:
            cmd = _build_extract_cmd(
                ts_path, first_tick_offset, fps, n_grid, vf_post,
                attempt_gpu, gpu_index,
            )
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=frame_bytes * 4,
            )
            emitted = 0
            last_frame = None
            try:
                while emitted < n_grid:
                    buf = proc.stdout.read(frame_bytes)
                    if buf is None or len(buf) < frame_bytes:
                        break  # EOF (or short read at EOF)
                    last_frame = np.frombuffer(buf, dtype=np.uint8).reshape(
                        out_height, out_width, 3
                    )
                    yield last_frame
                    emitted += 1
            finally:
                proc.stdout.close()
                err = proc.stderr.read().decode(errors="replace")
                proc.stderr.close()
                proc.wait()

            if emitted == 0:
                if attempt_gpu:
                    log.warning(
                        f"GPU decode produced no frames "
                        f"({err.strip()[:200] or 'unknown'}), falling back to CPU..."
                    )
                    continue
                raise IOError(
                    f"Frame extraction produced no frames: {err.strip()[:300]}"
                )

            # Source ended before the last ticks: the last decoded frame is
            # the floor-correct content for every remaining tick.
            while emitted < n_grid:
                yield last_frame
                emitted += 1
            return
    finally:
        if os.path.exists(ts_path):
            os.unlink(ts_path)
