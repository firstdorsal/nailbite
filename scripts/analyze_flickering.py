#!/usr/bin/env python3
"""Analyze hand detection flickering from FRAME_TELEMETRY debug logs.

Usage:
    RUST_LOG=debug pnpm tauri dev 2>&1 | python3 scripts/analyze_flickering.py --duration 30

The script reads tracing output from stdin and reports:
- Hand count changes between consecutive frames (flickering rate)
- Average hand lifetime (consecutive frames visible)
- Confidence distribution
- Grace period and confirmation delay statistics
"""

import argparse
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class FrameTelemetry:
    frame_ts: int = 0
    raw_hand_count: int = 0
    tracked_hand_count: int = 0
    face_detected: bool = False
    pose_detected: bool = False
    left_miss_count: int = 255
    right_miss_count: int = 255
    left_consecutive: int = 0
    right_consecutive: int = 0


@dataclass
class Stats:
    total_frames: int = 0
    hand_count_changes: int = 0
    hand_lifetimes: list = field(default_factory=list)
    raw_hand_counts: list = field(default_factory=list)
    tracked_hand_counts: list = field(default_factory=list)
    grace_usage: int = 0  # frames where miss_count > 0 but hand still visible
    confirmation_blocks: int = 0  # frames where consecutive < threshold
    prev_tracked: int = -1
    current_lifetime: int = 0


TELEMETRY_PATTERN = re.compile(
    r'message="?FRAME_TELEMETRY"?'
    r'.*?frame_ts=(\d+)'
    r'.*?raw_hand_count=(\d+)'
    r'.*?tracked_hand_count=(\d+)'
    r'.*?face_detected=(true|false)'
    r'.*?pose_detected=(true|false)'
    r'.*?left_miss_count=(\d+)'
    r'.*?right_miss_count=(\d+)'
    r'.*?left_consecutive=(\d+)'
    r'.*?right_consecutive=(\d+)'
)

# Also try key=value format (tracing fmt layer)
KV_PATTERN = re.compile(r'(\w+)=(\w+)')


def parse_line(line: str) -> FrameTelemetry | None:
    if "FRAME_TELEMETRY" not in line:
        return None

    m = TELEMETRY_PATTERN.search(line)
    if m:
        return FrameTelemetry(
            frame_ts=int(m.group(1)),
            raw_hand_count=int(m.group(2)),
            tracked_hand_count=int(m.group(3)),
            face_detected=m.group(4) == "true",
            pose_detected=m.group(5) == "true",
            left_miss_count=int(m.group(6)),
            right_miss_count=int(m.group(7)),
            left_consecutive=int(m.group(8)),
            right_consecutive=int(m.group(9)),
        )

    # Fallback: parse key=value pairs
    kvs = dict(KV_PATTERN.findall(line))
    if "frame_ts" in kvs:
        return FrameTelemetry(
            frame_ts=int(kvs.get("frame_ts", 0)),
            raw_hand_count=int(kvs.get("raw_hand_count", 0)),
            tracked_hand_count=int(kvs.get("tracked_hand_count", 0)),
            face_detected=kvs.get("face_detected", "false") == "true",
            pose_detected=kvs.get("pose_detected", "false") == "true",
            left_miss_count=int(kvs.get("left_miss_count", 255)),
            right_miss_count=int(kvs.get("right_miss_count", 255)),
            left_consecutive=int(kvs.get("left_consecutive", 0)),
            right_consecutive=int(kvs.get("right_consecutive", 0)),
        )

    return None


def analyze(duration_seconds: int):
    stats = Stats()
    start_time = time.time()

    print(f"Collecting telemetry for {duration_seconds}s... (Ctrl+C to stop early)")
    print()

    try:
        for line in sys.stdin:
            if time.time() - start_time > duration_seconds:
                break

            frame = parse_line(line.strip())
            if frame is None:
                continue

            stats.total_frames += 1
            stats.raw_hand_counts.append(frame.raw_hand_count)
            stats.tracked_hand_counts.append(frame.tracked_hand_count)

            # Track hand count changes (flickering)
            if stats.prev_tracked >= 0:
                if frame.tracked_hand_count != stats.prev_tracked:
                    stats.hand_count_changes += 1
                    # End current lifetime
                    if stats.current_lifetime > 0:
                        stats.hand_lifetimes.append(stats.current_lifetime)
                    stats.current_lifetime = 0
                else:
                    if frame.tracked_hand_count > 0:
                        stats.current_lifetime += 1

            stats.prev_tracked = frame.tracked_hand_count

            # Grace period usage
            for miss in [frame.left_miss_count, frame.right_miss_count]:
                if 0 < miss < 255:
                    stats.grace_usage += 1

            # Confirmation blocks
            for cons in [frame.left_consecutive, frame.right_consecutive]:
                if 0 < cons < 2:  # Below default confirmation threshold
                    stats.confirmation_blocks += 1

    except KeyboardInterrupt:
        pass

    # Final lifetime
    if stats.current_lifetime > 0:
        stats.hand_lifetimes.append(stats.current_lifetime)

    print_report(stats)


def percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return float(sorted_data[f])
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def print_report(stats: Stats):
    print("=" * 60)
    print("HAND DETECTION FLICKERING ANALYSIS")
    print("=" * 60)
    print()

    if stats.total_frames == 0:
        print("No FRAME_TELEMETRY events found.")
        print("Make sure RUST_LOG=debug is set.")
        return

    print(f"Total frames analyzed: {stats.total_frames}")
    print()

    # Flickering rate
    flicker_rate = (
        stats.hand_count_changes / (stats.total_frames - 1) * 100
        if stats.total_frames > 1
        else 0
    )
    print(f"--- Flickering ---")
    print(f"Hand count changes: {stats.hand_count_changes}")
    print(f"Flickering rate: {flicker_rate:.1f}% of frames")
    if flicker_rate < 10:
        print("  -> GOOD: Below 10% target")
    else:
        print("  -> WARNING: Above 10% target, consider tuning")
    print()

    # Hand lifetime
    if stats.hand_lifetimes:
        avg_life = sum(stats.hand_lifetimes) / len(stats.hand_lifetimes)
        print(f"--- Hand Lifetime ---")
        print(f"Average: {avg_life:.1f} frames")
        print(f"Median: {percentile(stats.hand_lifetimes, 50):.0f} frames")
        print(f"Min: {min(stats.hand_lifetimes)} frames")
        print(f"Max: {max(stats.hand_lifetimes)} frames")
        print()

    # Raw vs tracked hand counts
    if stats.raw_hand_counts:
        avg_raw = sum(stats.raw_hand_counts) / len(stats.raw_hand_counts)
        avg_tracked = sum(stats.tracked_hand_counts) / len(stats.tracked_hand_counts)
        print(f"--- Hand Counts ---")
        print(f"Average raw detections/frame: {avg_raw:.2f}")
        print(f"Average tracked hands/frame: {avg_tracked:.2f}")
        print(f"Filtering ratio: {avg_tracked/avg_raw*100:.0f}%" if avg_raw > 0 else "N/A")
        print()

    # Grace period and confirmation stats
    print(f"--- Stability Mechanisms ---")
    print(f"Grace period active frames: {stats.grace_usage}")
    print(f"Confirmation blocking frames: {stats.confirmation_blocks}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze hand detection flickering from FRAME_TELEMETRY logs"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Duration in seconds to collect data (default: 30)",
    )
    args = parser.parse_args()
    analyze(args.duration)


if __name__ == "__main__":
    main()
