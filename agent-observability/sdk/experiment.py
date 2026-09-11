"""Dataset-based experiment runner.

Run your agent over a fixed dataset, once per variant, and compare the results.
This is what separates an evaluation platform from a trace viewer: without it
you can see what your agent did, but you cannot tell whether a change made it
better.

    from openweave import OpenWeave
    from experiment import Experiment

    ow = OpenWeave()
    exp = Experiment(ow, api_base="http://localhost:8000")

    exp.run("support-qa", "prompt-v7", my_agent, metadata={"prompt_version": 7})
    exp.run("support-qa", "prompt-v8", my_agent_v8, metadata={"prompt_version": 8})

    print(exp.compare("support-qa", ["prompt-v7", "prompt-v8"]))

Two ordering details that matter and are easy to get wrong:

1. THE RUN ITEM IS LINKED BEFORE THE TRACE IS EMITTED. The judge resolves
   `dataset_item.expected_output` by joining dataset_run_items on trace_id, so
   if the link lands after the evaluation starts the judge silently grades
   against a missing expected output. We mint the trace id ourselves, link
   first, then trace.

2. FLUSH BEFORE COMPARING. Spans sit in the SDK's buffer for up to a second,
   then the eval worker waits for the trace to go quiet. `wait_for_scores`
   polls rather than guessing a sleep.
"""

from __future__ import annotations

import base64
import logging
import time
import uuid
from typing import Any, Callable

import httpx

logger = logging.getLogger("openweave.experiment")


class Experiment:
    def __init__(self, client, api_base: str = "http://localhost:8000",
                 public_key: str | None = None, secret_key: str | None = None,
                 timeout: float = 30.0):
        self.ow = client
        self.api_base = api_base.rstrip("/")
        pk = public_key or client.public_key
        sk = secret_key or client.secret_key
        self._headers = {
            "authorization": "Basic " + base64.b64encode(f"{pk}:{sk}".encode()).decode()
        }
        self._http = httpx.Client(timeout=timeout, headers=self._headers)

    # --- dataset management ---------------------------------------------- #
    def create_dataset(self, name: str, description: str | None = None) -> dict:
        return self._post("/datasets", {"name": name, "description": description})

    def add_item(self, dataset: str, input: Any, expected_output: Any = None,
                 metadata: dict | None = None) -> str:
        return self._post(f"/datasets/{dataset}/items", {
            "input": input, "expected_output": expected_output, "metadata": metadata,
        })["id"]

    def add_item_from_trace(self, dataset: str, trace_id: str) -> str:
        """Promote a production trace to a regression test."""
        return self._post(
            f"/datasets/{dataset}/items/from-trace/{trace_id}", None)["id"]

    def items(self, dataset: str) -> list[dict]:
        return self._get(f"/datasets/{dataset}/items")["data"]

    # --- running ---------------------------------------------------------- #
    def run(self, dataset: str, run_name: str, task: Callable[[Any], Any],
            metadata: dict | None = None, description: str | None = None) -> dict:
        """Execute `task` over every item in `dataset`, recording one trace each.

        `task` takes the item's input and returns the agent's output. Anything it
        does with the SDK's @observe decorators nests under that item's trace, so
        an experiment run produces exactly the same shape of data as production
        traffic — which is what lets one judge serve both.
        """
        self._post(f"/datasets/{dataset}/runs", {
            "name": run_name, "description": description, "metadata": metadata,
        })
        items = self.items(dataset)
        if not items:
            raise ValueError(f"dataset {dataset!r} has no items")

        ok = failed = 0
        for item in items:
            trace_id = str(uuid.uuid4())
            # Link FIRST — see the module docstring, point 1.
            self._post(f"/datasets/{dataset}/runs/{run_name}/items", {
                "dataset_item_id": item["id"], "trace_id": trace_id,
            })
            with self.ow.trace(
                name=f"{dataset}/{run_name}", trace_id=trace_id,
                input=item["input"], is_experiment=True,
                tags=["experiment", run_name],
                metadata={"dataset": dataset, "run": run_name,
                          "dataset_item_id": item["id"], **(metadata or {})},
            ) as trace:
                try:
                    output = task(item["input"])
                    trace.update(output=output)
                    ok += 1
                except Exception as exc:  # noqa: BLE001
                    # One failing item must not abandon the run — a run that
                    # dies halfway is not comparable to anything.
                    logger.warning("item %s failed: %s", item["id"], exc)
                    trace.update(output=f"ERROR: {type(exc).__name__}: {exc}")
                    failed += 1

        self.ow.flush()
        logger.info("run %r: %d ok, %d failed", run_name, ok, failed)
        return {"run": run_name, "items": len(items), "ok": ok, "failed": failed}

    # --- results ----------------------------------------------------------- #
    def wait_for_scores(self, dataset: str, run_name: str, expected: int,
                        timeout: float = 120.0, interval: float = 2.0) -> int:
        """Poll until the eval worker has scored the run (or we give up).

        Polling beats sleeping: the settle window, judge latency and queue depth
        all vary, and a fixed sleep is either flaky or slow.
        """
        deadline = time.monotonic() + timeout
        seen = 0
        while time.monotonic() < deadline:
            runs = self._get(f"/datasets/{dataset}/runs")["data"]
            if not any(r["name"] == run_name for r in runs):
                time.sleep(interval)
                continue
            try:
                report = self.compare(dataset, [run_name, run_name])
            except httpx.HTTPStatusError:
                time.sleep(interval)
                continue
            entry = next((r for r in report["runs"] if r["run"] == run_name), None)
            seen = max((s["n"] for s in (entry or {}).get("scores", {}).values()),
                       default=0)
            if seen >= expected:
                return seen
            time.sleep(interval)
        logger.warning("only %d/%d scores after %.0fs", seen, expected, timeout)
        return seen

    def compare(self, dataset: str, runs: list[str]) -> dict:
        return self._get(f"/datasets/{dataset}/compare",
                         params={"runs": ",".join(runs)})

    def format_comparison(self, report: dict) -> str:
        """The table you put in your README."""
        lines = []
        score_names = sorted({n for r in report["runs"] for n in r["scores"]})
        header = f"{'run':<18}{'items':>7}{'avg cost':>12}{'avg ms':>10}"
        header += "".join(f"{n:>12}" for n in score_names)
        lines += [header, "-" * len(header)]
        for run in report["runs"]:
            cost = f"${run['avg_cost']:.5f}" if run["avg_cost"] else "-"
            latency = f"{run['avg_latency_ms']:.0f}" if run["avg_latency_ms"] else "-"
            row = f"{run['run']:<18}{run['items']:>7}{cost:>12}{latency:>10}"
            for n in score_names:
                mean = run["scores"].get(n, {}).get("mean")
                row += f"{mean:>12.3f}" if mean is not None else f"{'-':>12}"
            lines.append(row)
        for label, diff in report.get("deltas", {}).items():
            parts = [f"{k} {v:+.3f}" if isinstance(v, float) and "pct" not in k
                     else f"{k} {v:+.1f}%" for k, v in diff.items()]
            if parts:
                lines.append(f"\n{label}:  " + "   ".join(parts))
        return "\n".join(lines)

    # --- http -------------------------------------------------------------- #
    def _post(self, path: str, body):
        response = self._http.post(self.api_base + path, json=body)
        response.raise_for_status()
        return response.json()

    def _get(self, path: str, params=None):
        response = self._http.get(self.api_base + path, params=params)
        response.raise_for_status()
        return response.json()

    def close(self):
        self._http.close()
