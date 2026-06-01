from __future__ import annotations

import re
import subprocess
import time

_SPEED_RE = re.compile(r"CPU_Speed_Limit\s*=\s*(\d+)")


def cpu_speed_limit() -> int | None:
    """Read the macOS CPU speed limit via `pmset -g therm` (no sudo).

    Returns the percentage (100 = unthrottled) or None if macOS hasn't recorded
    a thermal/performance limit (i.e. the system is cool). On non-macOS or if
    pmset is unavailable, returns None.
    """
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True,
                             text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = _SPEED_RE.search(out)
    return int(m.group(1)) if m else None


class ThermalGuard:
    """Pauses training when the CPU is thermally throttling, so a long run
    doesn't keep the Mac pinned hot. Uses the OS throttle signal as a proxy
    (we can't read the actual die temperature without sudo)."""

    def __init__(self, enabled: bool = True, check_every: int = 25,
                 cooldown_s: int = 30, pause_limit: int = 90):
        self.enabled = enabled
        self.check_every = max(1, check_every)
        self.cooldown_s = cooldown_s
        self.pause_limit = pause_limit

    def check(self, it: int, log) -> bool:
        """Call once per iteration. Returns True if a cooldown happened.
        `log` is a function (e.g. tqdm.write) for messages."""
        if not self.enabled or (it % self.check_every) != 0:
            return False

        limit = cpu_speed_limit()
        if limit is None or limit >= self.pause_limit:
            return False

        # Hot: pause and re-check until cool (or until the signal clears).
        cooled = False
        while True:
            log(f"  [thermal] CPU throttled to {limit}% — pausing "
                f"{self.cooldown_s}s to cool down")
            time.sleep(self.cooldown_s)
            cooled = True
            limit = cpu_speed_limit()
            if limit is None or limit >= self.pause_limit:
                log(f"  [thermal] cooled (limit "
                    f"{'100' if limit is None else limit}%) — resuming")
                break
        return cooled
