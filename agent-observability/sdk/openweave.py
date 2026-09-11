"""OpenWeave Python SDK — v2.

Replaces the v1 tracer, which had four problems that made it unusable for real
agents. Each is fixed here and each is worth being able to explain:

  v1                                          v2
  ------------------------------------------  --------------------------------
  a fresh trace_id AND session_id per call,    contextvars span stack; nesting
  so nesting was impossible by construction    is automatic and correct
  blocked the caller on a gRPC round trip      in-memory buffer + background
  for every single trace                       flush thread, batched
  hardcoded model="unknown", token_count=0     read from the provider response
  no try/except around the wrapped call, so    exceptions are captured as
  a raising function emitted NOTHING           level=ERROR + status_message,
                                               then re-raised unchanged

That last one is the important one. An observability SDK that drops the failed
calls is worse than no SDK, because it makes the system look healthier than it
is. Errors are the traces you most want.

Two hard rules this SDK follows:

  1. IT MUST NEVER BREAK THE APP IT OBSERVES. Every failure path — transport
     down, queue full, serialisation error — degrades to a dropped span and a
     warning. Nothing raises out of the SDK into user code.
  2. IT MUST NEVER CHANGE THE APP'S BEHAVIOUR. Decorated functions return and
     raise exactly what they would have without the decorator.

Usage:

    ow = OpenWeave(public_key=..., secret_key=..., host="localhost")

    @ow.observe(type="AGENT")
    def handle(question):
        docs = search(question)          # nested automatically
        return answer(question, docs)

    @ow.observe(type="GENERATION")
    def answer(question, docs):
        r = client.chat.completions.create(...)
        ow.current().update(
            model=r.model,
            usage={"input": r.usage.prompt_tokens,
                   "output": r.usage.completion_tokens},
        )
        return r.choices[0].message.content

    ow.flush()   # also runs at exit
"""

from __future__ import annotations

import atexit
import base64
import contextvars
import functools
import inspect
import json
import logging
import os
import queue
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

import grpc
from google.protobuf.timestamp_pb2 import Timestamp

import openweave_pb2 as pb
import openweave_pb2_grpc as rpc

logger = logging.getLogger("openweave")

SDK_VERSION = "2.0.0"

# The current trace id, and the stack of open observation ids. ContextVars are
# what make nesting work under threads AND asyncio tasks without the caller
# having to thread a parent id through every function signature.
_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "openweave_trace_id", default=None
)
_span_stack: contextvars.ContextVar[tuple] = contextvars.ContextVar(
    "openweave_span_stack", default=()
)

TYPES = {
    "SPAN": pb.OBSERVATION_TYPE_SPAN,
    "GENERATION": pb.OBSERVATION_TYPE_GENERATION,
    "EVENT": pb.OBSERVATION_TYPE_EVENT,
    "AGENT": pb.OBSERVATION_TYPE_AGENT,
    "TOOL": pb.OBSERVATION_TYPE_TOOL,
    "CHAIN": pb.OBSERVATION_TYPE_CHAIN,
    "RETRIEVER": pb.OBSERVATION_TYPE_RETRIEVER,
    "EMBEDDING": pb.OBSERVATION_TYPE_EMBEDDING,
    "GUARDRAIL": pb.OBSERVATION_TYPE_GUARDRAIL,
}
LEVELS = {
    "DEBUG": pb.OBSERVATION_LEVEL_DEBUG,
    "DEFAULT": pb.OBSERVATION_LEVEL_DEFAULT,
    "WARNING": pb.OBSERVATION_LEVEL_WARNING,
    "ERROR": pb.OBSERVATION_LEVEL_ERROR,
}


def _now() -> Timestamp:
    ts = Timestamp()
    ts.FromDatetime(datetime.now(timezone.utc))
    return ts


def _stringify(value, limit: int = 20_000) -> str:
    """Best-effort text for an arbitrary Python value.

    Truncated because a single 2MB embedding argument should not become a 2MB
    span. Losing the tail of one payload beats losing the trace.
    """
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            text = repr(value)
    return text if len(text) <= limit else text[:limit] + f"...[{len(text)} chars]"


def _struct(value: dict | None):
    """dict -> protobuf Struct, dropping anything that will not serialise."""
    if not value:
        return None
    from google.protobuf.struct_pb2 import Struct

    s = Struct()
    try:
        s.update(json.loads(json.dumps(value, default=str)))
    except (TypeError, ValueError) as exc:
        logger.debug("dropping unserialisable metadata: %s", exc)
        return None
    return s


class Span:
    """Handle to an open observation. Returned by `ow.current()`."""

    def __init__(self, client: "OpenWeave", span_id: str, trace_id: str):
        self.id = span_id
        self.trace_id = trace_id
        self._client = client

    def update(self, *, output=None, model=None, usage=None, cost=None,
               metadata=None, level=None, status_message=None,
               model_parameters=None, prompt_name=None, prompt_version=None,
               completion_start=False, end=False) -> "Span":
        body = pb.ObservationBody(id=self.id, trace_id=self.trace_id)
        if output is not None:
            body.output = _stringify(output)
        if model is not None:
            body.model = model
        if usage:
            for k, v in usage.items():
                body.usage_details[k] = int(v)
        if cost:
            for k, v in cost.items():
                body.cost_details[k] = float(v)
        if metadata:
            struct = _struct(metadata)
            if struct is not None:
                body.metadata.CopyFrom(struct)
        if model_parameters:
            struct = _struct(model_parameters)
            if struct is not None:
                body.model_parameters.CopyFrom(struct)
        if level is not None:
            body.level = LEVELS.get(level.upper(), pb.OBSERVATION_LEVEL_DEFAULT)
        if status_message is not None:
            body.status_message = status_message[:4000]
        if prompt_name is not None:
            body.prompt_name = prompt_name
        if prompt_version is not None:
            body.prompt_version = int(prompt_version)
        if completion_start:
            body.completion_start_time.CopyFrom(_now())
        if end:
            body.end_time.CopyFrom(_now())

        self._client._enqueue(pb.Event(
            event_id=str(uuid.uuid4()), timestamp=_now(), observation_update=body
        ))
        return self

    def first_token(self) -> "Span":
        """Call when the first streamed token arrives; the server derives TTFT."""
        return self.update(completion_start=True)


class _NullSpan(Span):
    """Returned by current() outside any span, so user code never sees None."""

    def __init__(self):
        super().__init__(None, "", "")

    def update(self, **_kw):
        return self

    def first_token(self):
        return self


NULL_SPAN = _NullSpan()


class OpenWeave:
    def __init__(self, public_key: str | None = None, secret_key: str | None = None,
                 host: str = "localhost", port: int = 50051,
                 flush_interval: float = 1.0, max_batch: int = 200,
                 max_queue: int = 10_000, enabled: bool = True):
        self.public_key = public_key or os.getenv("OPENWEAVE_PUBLIC_KEY", "")
        self.secret_key = secret_key or os.getenv("OPENWEAVE_SECRET_KEY", "")
        self.host = os.getenv("OPENWEAVE_HOST", host)
        self.port = int(os.getenv("OPENWEAVE_PORT", port))
        self.enabled = enabled and bool(self.public_key and self.secret_key)

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._flush_interval = flush_interval
        self._max_batch = max_batch
        self._stop = threading.Event()
        self._flushed = threading.Event()
        self._dropped = 0
        self._sent = 0

        if not self.enabled:
            logger.warning("OpenWeave disabled: no credentials; tracing is a no-op")
            return

        self._metadata = (
            ("authorization", "Basic " + base64.b64encode(
                f"{self.public_key}:{self.secret_key}".encode()).decode()),
        )
        self._channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        self._stub = rpc.IngestionServiceStub(self._channel)

        # Daemon thread: a hung flush must never stop the host process exiting.
        # The atexit hook below is what gives buffered spans a chance to land.
        self._worker = threading.Thread(
            target=self._run, name="openweave-flush", daemon=True
        )
        self._worker.start()
        atexit.register(self.shutdown)

    # --- transport -------------------------------------------------------- #
    def _enqueue(self, event) -> None:
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Drop rather than block. Back-pressuring the caller's request path
            # to protect telemetry is the wrong trade every time.
            self._dropped += 1
            if self._dropped % 100 == 1:
                logger.warning("openweave queue full; dropped %d event(s)",
                               self._dropped)

    def _drain(self) -> list:
        batch = []
        while len(batch) < self._max_batch:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _send(self, batch: list) -> None:
        if not batch:
            return
        try:
            response = self._stub.IngestBatch(
                pb.IngestBatchRequest(
                    events=batch, sdk=f"openweave-python/{SDK_VERSION}"
                ),
                metadata=self._metadata,
                timeout=10,
            )
            self._sent += response.accepted
            for err in response.errors:
                logger.warning("event %s rejected (%d): %s",
                               err.event_id, err.code, err.message)
        except grpc.RpcError as exc:
            # Rule 1: never raise into the host application.
            logger.warning("openweave flush failed, dropping %d event(s): %s",
                           len(batch), exc.code() if hasattr(exc, "code") else exc)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self._flush_interval)
            batch = self._drain()
            if batch:
                self._send(batch)
            self._flushed.set()

    def flush(self, timeout: float = 10.0) -> None:
        """Block until the buffer is empty. Call before a short process exits."""
        if not self.enabled:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            batch = self._drain()
            if not batch:
                if self._queue.empty():
                    return
                continue
            self._send(batch)

    def shutdown(self) -> None:
        if not self.enabled or self._stop.is_set():
            return
        self.flush()
        self._stop.set()
        try:
            self._channel.close()
        except Exception:  # noqa: BLE001
            pass

    @property
    def stats(self) -> dict:
        return {"sent": self._sent, "dropped": self._dropped,
                "queued": self._queue.qsize()}

    # --- tracing ---------------------------------------------------------- #
    def current(self) -> Span:
        stack = _span_stack.get()
        if not stack:
            return NULL_SPAN
        return Span(self, stack[-1], _trace_id.get() or "")

    @contextmanager
    def trace(self, name: str | None = None, *, input=None, user_id=None,
              session_id=None, tags=None, metadata=None, release=None,
              version=None, is_experiment=False, trace_id=None):
        """Open a trace. Everything observed inside it becomes part of its tree."""
        tid = trace_id or str(uuid.uuid4())
        body = pb.TraceBody(id=tid, timestamp=_now())
        if name:
            body.name = name
        if input is not None:
            body.input = _stringify(input)
        if user_id:
            body.user_id = user_id
        if session_id:
            body.session_id = session_id
        if tags:
            body.tags.extend(tags)
        if release:
            body.release = release
        if version:
            body.version = version
        if is_experiment:
            body.is_experiment = True
        if metadata:
            struct = _struct(metadata)
            if struct is not None:
                body.metadata.CopyFrom(struct)

        self._enqueue(pb.Event(event_id=str(uuid.uuid4()), timestamp=_now(),
                               trace_create=body))

        token_t, token_s = _trace_id.set(tid), _span_stack.set(())
        handle = _TraceHandle(self, tid)
        try:
            yield handle
        finally:
            _trace_id.reset(token_t)
            _span_stack.reset(token_s)

    def _start_span(self, name: str, type: str = "SPAN", input=None,
                    metadata=None) -> tuple[str, object, object]:
        tid = _trace_id.get()
        if tid is None:
            # An observation with no enclosing trace still belongs somewhere.
            # Minting one here means a decorator works standalone, which is how
            # people actually try the SDK first.
            tid = str(uuid.uuid4())
            self._enqueue(pb.Event(
                event_id=str(uuid.uuid4()), timestamp=_now(),
                trace_create=pb.TraceBody(id=tid, name=name, timestamp=_now()),
            ))
        token_t = _trace_id.set(tid)

        stack = _span_stack.get()
        span_id = str(uuid.uuid4())
        body = pb.ObservationBody(
            id=span_id, trace_id=tid, type=TYPES.get(type.upper(), pb.OBSERVATION_TYPE_SPAN),
            name=name, start_time=_now(), level=pb.OBSERVATION_LEVEL_DEFAULT,
        )
        if stack:
            body.parent_observation_id = stack[-1]
        if input is not None:
            body.input = _stringify(input)
        if metadata:
            struct = _struct(metadata)
            if struct is not None:
                body.metadata.CopyFrom(struct)

        self._enqueue(pb.Event(event_id=str(uuid.uuid4()), timestamp=_now(),
                               observation_create=body))
        token_s = _span_stack.set(stack + (span_id,))
        return span_id, token_t, token_s

    def _end_span(self, span_id, tid, output=None, error: BaseException | None = None):
        body = pb.ObservationBody(id=span_id, trace_id=tid, end_time=_now())
        if error is not None:
            body.level = pb.OBSERVATION_LEVEL_ERROR
            body.status_message = f"{type(error).__name__}: {error}"[:4000]
        elif output is not None:
            body.output = _stringify(output)
        self._enqueue(pb.Event(event_id=str(uuid.uuid4()), timestamp=_now(),
                               observation_update=body))

    @contextmanager
    def span(self, name: str, type: str = "SPAN", input=None, metadata=None):
        """Context-manager form, for code you cannot decorate."""
        span_id, token_t, token_s = self._start_span(name, type, input, metadata)
        tid = _trace_id.get()
        error = None
        try:
            yield Span(self, span_id, tid)
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._end_span(span_id, tid, error=error)
            _span_stack.reset(token_s)
            _trace_id.reset(token_t)

    def observe(self, _func=None, *, name: str | None = None, type: str = "SPAN",
                capture_input: bool = True, capture_output: bool = True):
        """Decorator. Works on both sync and async functions."""

        def decorate(func):
            span_name = name or func.__name__

            if inspect.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    if not self.enabled:
                        return await func(*args, **kwargs)
                    payload = {"args": args, "kwargs": kwargs} if capture_input else None
                    span_id, token_t, token_s = self._start_span(
                        span_name, type, payload)
                    tid = _trace_id.get()
                    try:
                        result = await func(*args, **kwargs)
                    except BaseException as exc:
                        # Rule 2: record, then re-raise unchanged.
                        self._end_span(span_id, tid, error=exc)
                        raise
                    else:
                        self._end_span(span_id, tid,
                                       output=result if capture_output else None)
                        return result
                    finally:
                        _span_stack.reset(token_s)
                        _trace_id.reset(token_t)
                return async_wrapper

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                payload = {"args": args, "kwargs": kwargs} if capture_input else None
                span_id, token_t, token_s = self._start_span(span_name, type, payload)
                tid = _trace_id.get()
                try:
                    result = func(*args, **kwargs)
                except BaseException as exc:
                    self._end_span(span_id, tid, error=exc)
                    raise
                else:
                    self._end_span(span_id, tid,
                                   output=result if capture_output else None)
                    return result
                finally:
                    _span_stack.reset(token_s)
                    _trace_id.reset(token_t)
            return wrapper

        return decorate(_func) if _func is not None else decorate

    def score(self, name: str, value=None, *, trace_id=None, comment=None,
              string_value=None, data_type: str = "NUMERIC",
              dataset_run_item_id=None) -> None:
        """Attach a score from application code (source=API server-side)."""
        body = pb.ScoreBody(id=str(uuid.uuid4()), name=name)
        tid = trace_id or _trace_id.get()
        if dataset_run_item_id:
            body.dataset_run_item_id = dataset_run_item_id
        elif tid:
            body.trace_id = tid
        else:
            logger.warning("score %r has no trace to attach to; dropped", name)
            return
        body.data_type = {
            "NUMERIC": pb.SCORE_DATA_TYPE_NUMERIC,
            "CATEGORICAL": pb.SCORE_DATA_TYPE_CATEGORICAL,
            "BOOLEAN": pb.SCORE_DATA_TYPE_BOOLEAN,
        }.get(data_type.upper(), pb.SCORE_DATA_TYPE_NUMERIC)
        if value is not None:
            body.value = float(value)
        if string_value is not None:
            body.string_value = string_value
        if comment:
            body.comment = comment
        self._enqueue(pb.Event(event_id=str(uuid.uuid4()), timestamp=_now(),
                               score_create=body))


class _TraceHandle:
    """What `with ow.trace(...) as t` yields."""

    def __init__(self, client: OpenWeave, trace_id: str):
        self.id = trace_id
        self._client = client

    def update(self, *, output=None, metadata=None) -> "_TraceHandle":
        body = pb.TraceBody(id=self.id)
        if output is not None:
            body.output = _stringify(output)
        if metadata:
            struct = _struct(metadata)
            if struct is not None:
                body.metadata.CopyFrom(struct)
        self._client._enqueue(pb.Event(
            event_id=str(uuid.uuid4()), timestamp=_now(), trace_create=body))
        return self
