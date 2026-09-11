"""Create a relevance judge and an evaluation rule for a project.

    python scripts/seed_evaluator.py <project-name> [--sample 0.1]

Re-running creates a NEW VERSION rather than editing the old one, which is the
whole point: existing scores stay attributable to the prompt that produced them.
Edit PROMPT below, re-run, and you can compare v1 against v2 honestly.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))

from sqlalchemy import create_engine, func, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from models import (  # noqa: E402
    EvalTarget, EvaluationRule, Evaluator, EvaluatorVersion, Project,
    ScoreDataType,
)

DATABASE_URL = (
    os.getenv("DATABASE_URL", "postgresql://openweave:openweave@localhost:5432/openweave")
    .replace("+asyncpg", "")
)

EVALUATOR_NAME = "relevance-judge"

# One criterion, not three. v1's prompt asked for relevance, correctness AND
# conciseness and collapsed them into a single float, which is unusable: you
# cannot tell a correct-but-rambling answer from a concise wrong one. One
# evaluator per criterion, each producing its own named score.
PROMPT = """You are grading an AI assistant's answer.

Judge ONLY relevance: does the answer address what was actually asked?
Ignore style, length, and factual accuracy — other graders cover those.

Return JSON exactly like:
{"score": <number between 0 and 1>, "reason": "<one sentence>"}

Where 0.0 = ignores the question entirely, 0.5 = partially on topic,
1.0 = directly and completely addresses it.

QUESTION:
{{input}}

ANSWER:
{{output}}
"""

# Where the prompt's {{variables}} come from. Using root.* rather than trace.*
# for the answer means this works even when the trace-level output was never
# set — which is common, because many agents have no single "the" output.
VARIABLE_MAPPING = {
    "input": "trace.input",
    "output": "root.output",
}

# The judge's verdict is checked against this before it becomes a Score. A
# verdict that fails becomes an ERROR job, never a silent zero.
OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["score"],
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project", nargs="?", default="default")
    parser.add_argument("--sample", type=float, default=1.0,
                        help="fraction of traces to evaluate (0-1)")
    parser.add_argument("--model", default=os.getenv("JUDGE_MODEL", "llama3"))
    parser.add_argument("--provider", default=os.getenv("JUDGE_PROVIDER", "ollama"))
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)
    with Session(engine) as session:
        project = session.scalar(select(Project).where(Project.name == args.project))
        if project is None:
            sys.exit(f"no project named {args.project!r} — run bootstrap.py first")

        evaluator = session.scalar(
            select(Evaluator).where(
                Evaluator.project_id == project.id, Evaluator.name == EVALUATOR_NAME
            )
        )
        if evaluator is None:
            evaluator = Evaluator(
                project_id=project.id, name=EVALUATOR_NAME,
                description="Grades answer relevance on a 0-1 scale.",
            )
            session.add(evaluator)
            session.flush()

        next_version = (session.scalar(
            select(func.max(EvaluatorVersion.version)).where(
                EvaluatorVersion.evaluator_id == evaluator.id
            )
        ) or 0) + 1

        version = EvaluatorVersion(
            project_id=project.id, evaluator_id=evaluator.id, version=next_version,
            prompt=PROMPT, model=args.model, provider=args.provider,
            model_params={"temperature": 0},
            variable_mapping=VARIABLE_MAPPING, output_schema=OUTPUT_SCHEMA,
            score_name="relevance", score_data_type=ScoreDataType.NUMERIC,
        )
        session.add(version)

        rule = session.scalar(
            select(EvaluationRule).where(
                EvaluationRule.project_id == project.id,
                EvaluationRule.evaluator_id == evaluator.id,
            )
        )
        if rule is None:
            session.add(EvaluationRule(
                project_id=project.id, evaluator_id=evaluator.id,
                target=EvalTarget.TRACE, filter=None,
                sampling_rate=args.sample, is_active=True,
            ))
        else:
            rule.sampling_rate = args.sample
            rule.is_active = True

        session.commit()
        print(f"{EVALUATOR_NAME} v{next_version} on project {args.project!r} "
              f"({args.provider}/{args.model}, sampling {args.sample:.0%})")
        print("The rule always uses the LATEST version; older scores keep "
              "pointing at the version that produced them.")


if __name__ == "__main__":
    main()
