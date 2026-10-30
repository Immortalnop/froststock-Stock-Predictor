"""Minimal asyncio job scheduler: fixed intervals, per-job error isolation, status for the API."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

log = logging.getLogger(__name__)


@dataclass
class Job:
    name: str
    fn: Callable[[], Awaitable[object]]
    every_s: float
    run_at_start: bool = True
    last_run: float | None = None
    last_ok: bool | None = None
    last_error: str = ""
    runs: int = 0
    next_due: float = field(default=0.0)


class Scheduler:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self._stop = asyncio.Event()

    def add(self, name: str, fn: Callable[[], Awaitable[object]], every_s: float, run_at_start: bool = True) -> None:
        j = Job(name, fn, every_s, run_at_start)
        j.next_due = time.time() + (0 if run_at_start else every_s)
        self.jobs[name] = j

    def status(self) -> list[dict]:
        now = time.time()
        return [{"name": j.name, "every_s": j.every_s, "runs": j.runs, "last_ok": j.last_ok,
                 "last_error": j.last_error, "seconds_since_run": None if j.last_run is None else now - j.last_run}
                for j in self.jobs.values()]

    async def _run(self, j: Job) -> None:
        j.last_run = time.time()
        try:
            await j.fn()
            j.last_ok, j.last_error = True, ""
        except Exception as exc:  # one failing job must not stop the others
            log.exception("job %s failed", j.name)
            j.last_ok, j.last_error = False, f"{type(exc).__name__}: {exc}"[:300]
        j.runs += 1
        j.next_due = time.time() + j.every_s

    async def run_forever(self, tick_s: float = 1.0) -> None:
        running: dict[str, asyncio.Task] = {}
        while not self._stop.is_set():
            now = time.time()
            for j in self.jobs.values():
                t = running.get(j.name)
                if (t is None or t.done()) and now >= j.next_due:  # never overlap a job with itself
                    running[j.name] = asyncio.create_task(self._run(j))
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=tick_s)
            except asyncio.TimeoutError:
                pass
        for t in running.values():
            t.cancel()

    def stop(self) -> None:
        self._stop.set()
