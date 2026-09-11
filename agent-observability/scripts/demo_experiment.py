"""End-to-end demo: two prompt versions, one dataset, one comparison table.

    python scripts/demo_experiment.py

Needs the ingestion server, worker, eval worker and API server running, plus a
judge (Ollama, or JUDGE_PROVIDER=openai).

The "agent" here is simulated so the demo runs anywhere. Swap `agent_v1` /
`agent_v2` for your real one — the only contract is that it takes the item's
input and returns the output; anything it does with @ow.observe nests under the
run's trace automatically.

The output of this script is the thing to put in your README. A table showing
relevance up and cost down between two prompt versions is a better argument for
your project than any paragraph about it.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdk"))

from experiment import Experiment  # noqa: E402
from openweave import OpenWeave  # noqa: E402

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
DATASET = os.getenv("DATASET", "support-qa")

CASES = [
    ("How do I reset my password?",
     "Use the 'Forgot password' link on the sign-in page."),
    ("Where can I download my invoices?",
     "Billing > Invoices, then use the download button on any row."),
    ("How do I add a teammate?",
     "Settings > Members > Invite, then enter their email address."),
    ("Can I export my data?",
     "Yes — Settings > Data > Export produces a JSON archive."),
    ("How do I cancel my subscription?",
     "Billing > Plan > Cancel. Access continues to the end of the period."),
]

# Two "prompt versions". v1 is terse and skips the specifics; v2 answers
# properly but spends more tokens doing it — the classic quality/cost trade
# this whole project exists to measure.
KB = {q: a for q, a in CASES}

ow = OpenWeave()


@ow.observe(type="RETRIEVER")
def retrieve(question: str) -> list[str]:
    time.sleep(0.01)
    return [a for q, a in CASES if q == question] or ["(no match)"]


@ow.observe(type="GENERATION", name="answer")
def generate(question: str, docs: list[str], *, verbose: bool,
             prompt_version: int) -> str:
    time.sleep(0.02)
    span = ow.current()
    span.first_token()
    answer = docs[0] if verbose else docs[0].split(".")[0][:24]
    # Realistic-looking usage so the cost engine has something to price.
    in_tokens = 180 + len(question) * 2 + (240 if verbose else 0)
    out_tokens = max(8, len(answer) // 3)
    span.update(
        model="gpt-4o-mini",
        usage={"input": in_tokens, "output": out_tokens},
        prompt_name="support-answer", prompt_version=prompt_version,
    )
    return answer


def make_agent(verbose: bool, prompt_version: int):
    @ow.observe(type="AGENT", name="support-agent")
    def agent(question: str) -> str:
        return generate(question, retrieve(question),
                        verbose=verbose, prompt_version=prompt_version)
    return agent


def main() -> None:
    if not ow.enabled:
        sys.exit("set OPENWEAVE_PUBLIC_KEY and OPENWEAVE_SECRET_KEY first "
                 "(scripts/bootstrap.py prints them)")

    exp = Experiment(ow, api_base=API_BASE)
    exp.create_dataset(DATASET, "Support questions with known-good answers")
    existing = {str(i["input"]) for i in exp.items(DATASET)}
    for question, answer in CASES:
        if f'"{question}"' not in existing and question not in existing:
            exp.add_item(DATASET, question, answer)
    n = len(exp.items(DATASET))
    print(f"dataset {DATASET!r}: {n} items\n")

    for run_name, verbose, version in (("prompt-v1", False, 1),
                                       ("prompt-v2", True, 2)):
        result = exp.run(DATASET, run_name, make_agent(verbose, version),
                         metadata={"prompt_version": version})
        print(f"  ran {run_name}: {result['ok']} ok, {result['failed']} failed")

    print("\nwaiting for the judge…")
    for run_name in ("prompt-v1", "prompt-v2"):
        seen = exp.wait_for_scores(DATASET, run_name, expected=n, timeout=180)
        print(f"  {run_name}: {seen}/{n} scored")

    print()
    print(exp.format_comparison(exp.compare(DATASET, ["prompt-v1", "prompt-v2"])))
    exp.close()
    ow.shutdown()


if __name__ == "__main__":
    main()
