from __future__ import annotations

import sys


class ConsoleProgressReporter:
    def __init__(self) -> None:
        self._is_tty = sys.stdout.isatty()
        self._active_line = False
        self._last_line_width = 0

    def stage(self, index: int, total: int, title: str, detail: str | None = None) -> None:
        self._flush_active_line()
        suffix = f" | {detail}" if detail else ""
        print(f"[{index}/{total}] {title}{suffix}", flush=True)

    def info(self, message: str) -> None:
        self._flush_active_line()
        print(f"    {message}", flush=True)

    def update(self, label: str, current: int, total: int, detail: str | None = None) -> None:
        total = max(total, 1)
        current = min(max(current, 0), total)
        percent = int((current / total) * 100)
        message = f"    {label}: {current}/{total} ({percent:3d}%)"
        if detail:
            message += f" | {detail}"

        if self._is_tty:
            padded = message.ljust(self._last_line_width)
            self._last_line_width = max(self._last_line_width, len(message))
            print(f"\r{padded}", end="", flush=True)
            self._active_line = current < total
            if current >= total:
                print(flush=True)
                self._active_line = False
                self._last_line_width = 0
            return

        step = max(1, total // 10)
        should_print = current in (1, total) or current % step == 0
        if should_print:
            self._flush_active_line()
            print(message, flush=True)

    def _flush_active_line(self) -> None:
        if self._active_line:
            print(flush=True)
            self._active_line = False
            self._last_line_width = 0
