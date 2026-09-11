"""OpenWeave data model — v2.

Replaces the flat `Trace` / `SpanEvent` / `Evaluation` model with a nested
observation tree, project-scoped tenancy, a real score model, and cost
attribution.

Design notes live in docs/M1-DESIGN-NOTES.md. The five decisions worth
knowing before you read further:

1.  Every tenant-owned row has a COMPOSITE primary key of (project_id, id).
    You cannot fetch a row without knowing its project, so tenant isolation is
    enforced by the schema rather than by remembering a WHERE clause.

2.  There is deliberately NO foreign key from observations -> traces.
    Events arrive over a queue and can be reordered; a child span often lands
    before its parent trace. A hard FK would either reject valid data or force
    the stream to be serialised. The worker upserts a stub trace instead.

3.  Usage and cost are Maps (JSONB), not columns. New usage types
    (cache_read, cache_write, reasoning) appear constantly; a schema migration
    per token type does not scale as a design.

4.  `provided_*` vs resolved columns are kept separately. The SDK may send a
    cost; we may also compute one from the price table. Keeping both means an
    incorrect price table can be re-run without losing what the client sent.

5.  Trace-level totals (cost, tokens, latency) are COMPUTED ON READ by
    aggregating observations, not denormalised onto `traces`. See the design
    notes for when to change that and what it would cost.

6.  Every tenant table declares a many-to-one `project` relationship. It is
    there for ORDERING, not navigation: without it SQLAlchemy's unit of work
    has no mapper-level dependency graph and will try to INSERT observations
    before their project, which fails on the FK. (Found the hard way — see
    the design notes.) DO NOT lazy-load through it in a request path.

    The ingestion worker should NOT use the ORM at all. Use Core bulk upserts:

        from sqlalchemy.dialects.postgresql import insert
        stmt = insert(Observation).values(rows)
        stmt.on_conflict_do_update(index_elements=["project_id", "id"], set_=...)

    The ORM is for the read API and for tests.

Verified against PostgreSQL 16: 17 tables, 7 enum types, both generated
columns emit as GENERATED ALWAYS ... STORED, and every CHECK/UNIQUE/FK
constraint below rejects the case it is written for.

SQLAlchemy 2.0 declarative style.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class ObservationType(str, enum.Enum):
    """What kind of step this observation represents.

    GENERATION is the only type that carries model/usage/cost. Everything else
    is structural. Keeping these as one enum on one table (rather than separate
    tables per type) is what makes a single recursive query able to return the
    whole tree.
    """

    SPAN = "SPAN"
    GENERATION = "GENERATION"
    EVENT = "EVENT"
    AGENT = "AGENT"
    TOOL = "TOOL"
    CHAIN = "CHAIN"
    RETRIEVER = "RETRIEVER"
    EMBEDDING = "EMBEDDING"
    GUARDRAIL = "GUARDRAIL"


class ObservationLevel(str, enum.Enum):
    DEBUG = "DEBUG"
    DEFAULT = "DEFAULT"
    WARNING = "WARNING"
    ERROR = "ERROR"


class ScoreDataType(str, enum.Enum):
    NUMERIC = "NUMERIC"
    CATEGORICAL = "CATEGORICAL"
    BOOLEAN = "BOOLEAN"


class ScoreSource(str, enum.Enum):
    """Where a score came from.

    This is the single most important field on the scores table. Having judge
    output (EVAL) and human labels (ANNOTATION) in the SAME table with the same
    shape is what lets you ask "how well does my LLM judge agree with a human?"
    — which is the standard way to validate a judge.
    """

    API = "API"
    EVAL = "EVAL"
    ANNOTATION = "ANNOTATION"


class DatasetItemStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"


class JobStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


class EvalTarget(str, enum.Enum):
    TRACE = "TRACE"
    DATASET_RUN_ITEM = "DATASET_RUN_ITEM"


# --------------------------------------------------------------------------- #
# Tenancy
# --------------------------------------------------------------------------- #
class Project(Base):
    """The tenant boundary.

    Deliberately one level, not Organization -> Project. Adding an org layer is
    a table plus one nullable FK; it buys nothing for a portfolio project and
    costs you an RBAC matrix. Note the path in your README instead.
    """

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String, nullable=False)
    retention_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    __table_args__ = (UniqueConstraint("name", name="uq_projects_name"),)


class ApiKey(Base):
    """Project-scoped API credential.

    The secret is stored as SHA-256, NOT bcrypt. That is a deliberate choice,
    not a shortcut, and you should be able to defend it:

    API keys are high-entropy random values generated by us (256 bits). Password
    hashes need bcrypt/argon2 because human passwords are low-entropy and
    brute-forceable; a 256-bit random key is not. Bcrypt on the ingestion hot
    path would add ~100ms to every request. Langfuse hits the same wall and
    solves it by storing BOTH a bcrypt hash and a fast SHA-256 hash, using the
    fast one for the cached lookup path.

    Verification path you will implement in M1:
        Authorization: Basic base64(public_key:secret_key)
        -> sha256(secret_key + salt) -> Redis cache lookup -> Postgres on miss.
    """

    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    project: Mapped["Project"] = relationship()
    public_key: Mapped[str] = mapped_column(String, nullable=False)
    hashed_secret_key: Mapped[str] = mapped_column(String, nullable=False)
    # Last 4 chars only, e.g. "sk-ow-...a3f9", so the UI can identify a key.
    display_secret_key: Mapped[str] = mapped_column(String, nullable=False)
    note: Mapped[str | None] = mapped_column(String, nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("public_key", name="uq_api_keys_public_key"),
        UniqueConstraint("hashed_secret_key", name="uq_api_keys_hashed_secret"),
        Index("ix_api_keys_project", "project_id"),
    )


# --------------------------------------------------------------------------- #
# Tracing
# --------------------------------------------------------------------------- #
class Trace(Base):
    """A container for one agent run. Carries almost no data itself.

    Everything that used to live here (input, output, latency, tokens, model)
    now lives on observations, because an agent run has many of each. What
    remains on the trace is the stuff that is genuinely per-run: who ran it,
    which session it belongs to, which code version produced it.

    `release` and `version` are cheap and high-value: they let you ask
    "did quality drop after we shipped v1.4?" which is the question this whole
    project exists to answer.
    """

    __tablename__ = "traces"

    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    project: Mapped["Project"] = relationship()
    id: Mapped[str] = mapped_column(String, primary_key=True)

    name: Mapped[str | None] = mapped_column(String, nullable=True)
    user_id: Mapped[str | None] = mapped_column(String, nullable=True)
    session_id: Mapped[str | None] = mapped_column(String, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Trace-level input/output are optional. Most SDKs set them from the
    # outermost observation; some agents have no single "the" input.
    input: Mapped[str | None] = mapped_column(Text, nullable=True)
    output: Mapped[str | None] = mapped_column(Text, nullable=True)

    trace_metadata: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    release: Mapped[str | None] = mapped_column(String, nullable=True)
    version: Mapped[str | None] = mapped_column(String, nullable=True)

    # True when this trace was produced by an experiment run rather than
    # production traffic. Lets you exclude eval traffic from cost dashboards.
    is_experiment: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    __table_args__ = (
        # The workhorse index: "latest traces for this project".
        Index("ix_traces_project_timestamp", "project_id", timestamp.desc()),
        Index("ix_traces_project_session", "project_id", "session_id"),
        Index("ix_traces_project_user", "project_id", "user_id"),
        # GIN index so metadata filters ("environment=prod") do not table-scan.
        Index("ix_traces_metadata", "metadata", postgresql_using="gin"),
        Index("ix_traces_tags", "tags", postgresql_using="gin"),
    )


class Observation(Base):
    """One step inside a trace. Self-referencing, so traces form a tree.

    THIS TABLE IS THE POINT OF M1. `parent_observation_id` + `type` +
    (start_time, end_time) is what turns "one row per agent run" into a
    waterfall you can actually read.

    Fetch a whole tree with a recursive CTE:

        WITH RECURSIVE tree AS (
          SELECT * FROM observations
           WHERE project_id = :pid AND trace_id = :tid
             AND parent_observation_id IS NULL
          UNION ALL
          SELECT o.* FROM observations o
            JOIN tree t ON o.parent_observation_id = t.id
                       AND o.project_id = t.project_id
        )
        SELECT * FROM tree ORDER BY start_time;

    Note there is NO ForeignKey to traces. See module docstring, point 2.
    """

    __tablename__ = "observations"

    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    project: Mapped["Project"] = relationship()
    id: Mapped[str] = mapped_column(String, primary_key=True)

    trace_id: Mapped[str] = mapped_column(String, nullable=False)
    parent_observation_id: Mapped[str | None] = mapped_column(String, nullable=True)

    type: Mapped[ObservationType] = mapped_column(
        SAEnum(ObservationType, name="observation_type"), nullable=False
    )
    name: Mapped[str | None] = mapped_column(String, nullable=True)

    # end_time is nullable because a span is created when it STARTS. The
    # observation-update event fills this in. This create/update split is why
    # the wire protocol needs two event types per observation.
    start_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    end_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Time to first token. The single most requested LLM latency metric and
    # you get it almost free — just record when the first chunk arrives.
    completion_start_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Generated column: Postgres computes this on write, you never set it.
    # Saves an EXTRACT() in every query and every dashboard aggregation.
    latency_ms: Mapped[float | None] = mapped_column(
        Float,
        Computed(
            "EXTRACT(EPOCH FROM (end_time - start_time)) * 1000", persisted=True
        ),
        nullable=True,
    )
    time_to_first_token_ms: Mapped[float | None] = mapped_column(
        Float,
        Computed(
            "EXTRACT(EPOCH FROM (completion_start_time - start_time)) * 1000",
            persisted=True,
        ),
        nullable=True,
    )

    level: Mapped[ObservationLevel] = mapped_column(
        SAEnum(ObservationLevel, name="observation_level"),
        nullable=False,
        default=ObservationLevel.DEFAULT,
    )
    # Populated when the SDK catches an exception. The old SDK dropped failed
    # calls entirely; this column is where those now land.
    status_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    input: Mapped[str | None] = mapped_column(Text, nullable=True)
    output: Mapped[str | None] = mapped_column(Text, nullable=True)
    observation_metadata: Mapped[dict | None] = mapped_column(
        "metadata", JSONB, nullable=True
    )

    # --- model + cost (GENERATION observations only) ---------------------- #
    provided_model_name: Mapped[str | None] = mapped_column(String, nullable=True)
    internal_model_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("models.id", ondelete="SET NULL"), nullable=True
    )
    model_parameters: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # {"input": 1200, "output": 340, "cache_read": 800, "reasoning": 512}
    provided_usage_details: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    usage_details: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # {"input": 0.0036, "output": 0.0051}
    provided_cost_details: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    cost_details: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    total_cost: Mapped[float | None] = mapped_column(Numeric(18, 12), nullable=True)

    # --- prompt linkage --------------------------------------------------- #
    # No prompts table in M1 (see design notes) but the columns exist now so
    # you never have to backfill. Set them from the SDK; group experiment runs
    # by prompt_version to get "did v7 beat v6" for free.
    prompt_name: Mapped[str | None] = mapped_column(String, nullable=True)
    prompt_version: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    __table_args__ = (
        # Fetching one trace's tree — by far the hottest read.
        Index("ix_obs_project_trace", "project_id", "trace_id"),
        # Walking down the tree.
        Index("ix_obs_project_parent", "project_id", "parent_observation_id"),
        # Dashboards: "p95 latency of GENERATION spans this week".
        Index("ix_obs_project_type_start", "project_id", "type", start_time.desc()),
        Index("ix_obs_project_model", "project_id", "provided_model_name"),
        Index("ix_obs_metadata", "metadata", postgresql_using="gin"),
        CheckConstraint(
            "end_time IS NULL OR end_time >= start_time",
            name="ck_obs_end_after_start",
        ),
    )


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
class ScoreConfig(Base):
    """Defines what a valid score of a given name looks like for a project.

    Small table, big payoff: it is what stops "relevance" meaning 0-1 in one
    place and 1-5 in another, and it is what a human-annotation UI reads to
    render the right widget (slider vs dropdown vs toggle).
    """

    __tablename__ = "score_configs"

    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    project: Mapped["Project"] = relationship()
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)

    name: Mapped[str] = mapped_column(String, nullable=False)
    data_type: Mapped[ScoreDataType] = mapped_column(
        SAEnum(ScoreDataType, name="score_data_type"), nullable=False
    )
    min_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    # [{"label": "helpful", "value": 1}, {"label": "harmful", "value": 0}]
    categories: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_score_config_project_name"),
    )


class Score(Base):
    """One quality measurement.

    Replaces the old single-float `evaluations` table. Four things changed and
    each one earns its place:

      name          many scores per trace, not one ("relevance", "toxicity")
      value / string_value / data_type
                    numeric, categorical and boolean scores in one table
      source        API | EVAL | ANNOTATION — judge and human side by side
      evaluator_version_id
                    WHICH version of WHICH judge produced this, so a judge
                    prompt change is attributable

    A score attaches to a trace, a specific observation inside it, or a dataset
    run item. Exactly one anchor must be set — enforced by the check constraint.
    """

    __tablename__ = "scores"

    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    project: Mapped["Project"] = relationship()
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)

    # --- anchors (exactly one) -------------------------------------------- #
    trace_id: Mapped[str | None] = mapped_column(String, nullable=True)
    observation_id: Mapped[str | None] = mapped_column(String, nullable=True)
    dataset_run_item_id: Mapped[str | None] = mapped_column(String, nullable=True)

    name: Mapped[str] = mapped_column(String, nullable=False)
    data_type: Mapped[ScoreDataType] = mapped_column(
        SAEnum(ScoreDataType, name="score_data_type", create_type=False),
        nullable=False,
        default=ScoreDataType.NUMERIC,
    )
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    string_value: Mapped[str | None] = mapped_column(String, nullable=True)

    source: Mapped[ScoreSource] = mapped_column(
        SAEnum(ScoreSource, name="score_source"), nullable=False
    )
    # The judge's reasoning, or the human annotator's note.
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)

    config_id: Mapped[str | None] = mapped_column(String, nullable=True)
    evaluator_version_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("evaluator_versions.id", ondelete="SET NULL"), nullable=True
    )
    author_user_id: Mapped[str | None] = mapped_column(String, nullable=True)

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_scores_project_trace", "project_id", "trace_id"),
        Index("ix_scores_project_observation", "project_id", "observation_id"),
        Index("ix_scores_project_run_item", "project_id", "dataset_run_item_id"),
        # "average relevance over time for this project" — the dashboard query.
        Index("ix_scores_project_name_ts", "project_id", "name", timestamp.desc()),
        CheckConstraint(
            "(trace_id IS NOT NULL)::int "
            "+ (dataset_run_item_id IS NOT NULL)::int = 1",
            name="ck_scores_exactly_one_anchor",
        ),
        CheckConstraint(
            "(data_type = 'CATEGORICAL' AND string_value IS NOT NULL) "
            "OR (data_type <> 'CATEGORICAL' AND value IS NOT NULL)",
            name="ck_scores_value_matches_type",
        ),
    )


# --------------------------------------------------------------------------- #
# Model pricing
# --------------------------------------------------------------------------- #
class Model(Base):
    """A model definition used to resolve cost from token usage.

    `match_pattern` is a regex, not an exact name, because providers ship
    variants faster than you can seed rows: one pattern
    `(?i)^(gpt-4o)(-\\d{4}-\\d{2}-\\d{2})?$` covers every dated snapshot.

    `start_date` exists because prices change. Resolve by picking the row with
    the greatest start_date <= observation.start_time, so re-costing historical
    traces gives the price that applied THEN, not today's.
    """

    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    # NULL project_id = built-in model available to every project.
    project_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=True
    )
    project: Mapped["Project | None"] = relationship()
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    match_pattern: Mapped[str] = mapped_column(String, nullable=False)
    start_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    tokenizer_id: Mapped[str | None] = mapped_column(String, nullable=True)
    tokenizer_config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_models_project_name", "project_id", "model_name"),
    )


class Price(Base):
    """Price per unit for one usage type of one model.

    One row per usage type rather than input_price/output_price columns, for
    the same reason usage_details is a map: cache_read, cache_write, audio and
    reasoning tokens are all priced differently and the list keeps growing.
    """

    __tablename__ = "prices"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    model_id: Mapped[str] = mapped_column(
        String, ForeignKey("models.id", ondelete="CASCADE"), nullable=False
    )
    usage_type: Mapped[str] = mapped_column(String, nullable=False)
    # Price for ONE token. Numeric, never float — floating point money is a bug.
    price: Mapped[float] = mapped_column(Numeric(24, 18), nullable=False)

    __table_args__ = (
        UniqueConstraint("model_id", "usage_type", name="uq_price_model_usage"),
    )


# --------------------------------------------------------------------------- #
# Datasets and experiments
# --------------------------------------------------------------------------- #
class Dataset(Base):
    """A fixed set of inputs (with optional expected outputs) to run against.

    This is what turns the project from an observability tool into an
    evaluation platform, and it is the part of the story that matters most for
    an AI infra role: change a prompt, re-run the dataset, compare the scores.
    """

    __tablename__ = "datasets"

    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    project: Mapped["Project"] = relationship()
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)

    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    dataset_metadata: Mapped[dict | None] = mapped_column(
        "metadata", JSONB, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_dataset_project_name"),
    )


class DatasetItem(Base):
    """One test case.

    `source_trace_id` is the feature that makes datasets actually get built:
    you see a bad trace in production, click "add to dataset", and it becomes a
    regression test. Datasets that must be authored from scratch stay empty.

    NOTE: no temporal versioning (valid_from / valid_to) in M1. Langfuse has it
    so that a historical run still shows the item as it was at run time. That is
    correct and it is also a whole extra dimension on every query. Scoped out —
    see design notes for how to add it later without a backfill.
    """

    __tablename__ = "dataset_items"

    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    project: Mapped["Project"] = relationship()
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)

    dataset_id: Mapped[str] = mapped_column(String, nullable=False)
    input: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    expected_output: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    item_metadata: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)

    source_trace_id: Mapped[str | None] = mapped_column(String, nullable=True)
    source_observation_id: Mapped[str | None] = mapped_column(String, nullable=True)

    status: Mapped[DatasetItemStatus] = mapped_column(
        SAEnum(DatasetItemStatus, name="dataset_item_status"),
        nullable=False,
        default=DatasetItemStatus.ACTIVE,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    __table_args__ = (
        Index("ix_dataset_items_project_dataset", "project_id", "dataset_id"),
        Index("ix_dataset_items_source_trace", "source_trace_id"),
    )


class DatasetRun(Base):
    """One execution of a whole dataset — i.e. one experiment.

    Name your runs after what changed ("prompt-v7", "gpt-4o-mini", "no-rag")
    and the comparison query writes itself.
    """

    __tablename__ = "dataset_runs"

    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    project: Mapped["Project"] = relationship()
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)

    dataset_id: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Put the thing you changed in here: {"prompt_version": 7, "model": "..."}
    run_metadata: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "project_id", "dataset_id", "name", name="uq_dataset_run_name"
        ),
        Index("ix_dataset_runs_project_dataset", "project_id", "dataset_id"),
    )


class DatasetRunItem(Base):
    """The join that makes experiments work: item x run -> the trace produced.

    Read it as: "when we ran dataset item X during run Y, the agent produced
    trace Z." Scores attach to this row, so a run's aggregate score is:

        SELECT s.name, AVG(s.value)
          FROM dataset_run_items ri
          JOIN scores s ON s.dataset_run_item_id = ri.id
                       AND s.project_id = ri.project_id
         WHERE ri.project_id = :pid AND ri.dataset_run_id = :run
         GROUP BY s.name;

    Run that for two runs and you have a regression report.
    """

    __tablename__ = "dataset_run_items"

    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    project: Mapped["Project"] = relationship()
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)

    dataset_run_id: Mapped[str] = mapped_column(String, nullable=False)
    dataset_item_id: Mapped[str] = mapped_column(String, nullable=False)
    trace_id: Mapped[str] = mapped_column(String, nullable=False)
    observation_id: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "project_id", "dataset_run_id", "dataset_item_id",
            name="uq_run_item_once_per_run",
        ),
        Index("ix_run_items_project_run", "project_id", "dataset_run_id"),
        Index("ix_run_items_trace", "project_id", "trace_id"),
    )


# --------------------------------------------------------------------------- #
# Evaluators
# --------------------------------------------------------------------------- #
class Evaluator(Base):
    """A named judge. The prompt itself lives on EvaluatorVersion."""

    __tablename__ = "evaluators"

    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    project: Mapped["Project"] = relationship()
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)

    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_evaluator_project_name"),
    )


class EvaluatorVersion(Base):
    """An immutable version of a judge.

    Versioning the judge is not optional. If the judge prompt can change
    silently, then a score drop is ambiguous — did the agent get worse, or did
    the judge get stricter? Every Score points at the exact version that
    produced it, so that question always has an answer.

    `variable_mapping` says where the prompt's variables come from, e.g.
        {"input": "trace.input", "output": "trace.output",
         "expected": "dataset_item.expected_output"}
    so one judge template works for both live traces and dataset runs.

    `output_schema` is the JSON schema the judge must return. Constrained
    decoding against it beats "please respond with only JSON" and a try/except.
    """

    __tablename__ = "evaluator_versions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    project: Mapped["Project"] = relationship()
    evaluator_id: Mapped[str] = mapped_column(String, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)

    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    provider: Mapped[str | None] = mapped_column(String, nullable=True)
    model_params: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    variable_mapping: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    output_schema: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # What the produced Score is called and what shape it has.
    score_name: Mapped[str] = mapped_column(String, nullable=False)
    score_data_type: Mapped[ScoreDataType] = mapped_column(
        SAEnum(ScoreDataType, name="score_data_type", create_type=False),
        nullable=False,
        default=ScoreDataType.NUMERIC,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("evaluator_id", "version", name="uq_evaluator_version"),
        Index("ix_evaluator_versions_project", "project_id", "evaluator_id"),
    )


class EvaluationRule(Base):
    """Which things get evaluated, and how often.

    Without sampling, every production trace costs you a judge call. A 5%
    sample of production plus 100% of dataset runs is the normal shape, and
    "I made evaluation cost configurable" is a better interview answer than
    "I evaluated everything".
    """

    __tablename__ = "evaluation_rules"

    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    project: Mapped["Project"] = relationship()
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)

    evaluator_id: Mapped[str] = mapped_column(String, nullable=False)
    target: Mapped[EvalTarget] = mapped_column(
        SAEnum(EvalTarget, name="eval_target"), nullable=False
    )
    # e.g. [{"column": "name", "op": "=", "value": "checkout-agent"}]
    filter: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    sampling_rate: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_eval_rules_project_active", "project_id", "is_active"),
        CheckConstraint(
            "sampling_rate >= 0 AND sampling_rate <= 1",
            name="ck_eval_rule_sampling_range",
        ),
    )


class JobExecution(Base):
    """Audit trail for one evaluation attempt.

    The old design had no equivalent: if the judge failed, the trace simply had
    no score and nothing recorded why. This table is the difference between
    "we have no score for this" and "we know exactly why we have no score for
    this", which is the whole job of a platform.
    """

    __tablename__ = "job_executions"

    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    project: Mapped["Project"] = relationship()
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)

    evaluator_version_id: Mapped[str] = mapped_column(String, nullable=False)
    rule_id: Mapped[str | None] = mapped_column(String, nullable=True)

    trace_id: Mapped[str | None] = mapped_column(String, nullable=True)
    observation_id: Mapped[str | None] = mapped_column(String, nullable=True)
    dataset_run_item_id: Mapped[str | None] = mapped_column(String, nullable=True)

    status: Mapped[JobStatus] = mapped_column(
        SAEnum(JobStatus, name="job_status"), nullable=False, default=JobStatus.PENDING
    )
    score_id: Mapped[str | None] = mapped_column(String, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Useful on its own: judge latency and judge cost are real operating costs.
    judge_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    judge_cost: Mapped[float | None] = mapped_column(Numeric(18, 12), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("ix_jobs_project_status", "project_id", "status"),
        Index("ix_jobs_project_trace", "project_id", "trace_id"),
        Index("ix_jobs_project_run_item", "project_id", "dataset_run_item_id"),
    )


# --------------------------------------------------------------------------- #
# Dead letter queue
# --------------------------------------------------------------------------- #
class IngestionDeadLetter(Base):
    """Events the worker could not process after N retries.

    The old worker acked malformed messages and logged them, which means the
    data was gone. Parking them here costs one table and turns an unrecoverable
    drop into a replayable backlog. Expose a `POST /admin/dlq/{id}/replay`
    endpoint and you have a genuinely good demo moment.
    """

    __tablename__ = "ingestion_dead_letters"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str | None] = mapped_column(String, nullable=True)
    event_id: Mapped[str | None] = mapped_column(String, nullable=True)
    stream_message_id: Mapped[str | None] = mapped_column(String, nullable=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    error: Mapped[str] = mapped_column(Text, nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    __table_args__ = (Index("ix_dlq_project_created", "project_id", created_at.desc()),)


# NOTE: there is intentionally no init_db()/create_all() helper here.
# The old code called create_all() at worker startup AND shipped an Alembic
# migration, which is two sources of schema truth that will silently diverge.
# Alembic is the only one now. See docs/M1-DESIGN-NOTES.md.
