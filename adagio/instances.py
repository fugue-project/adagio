import logging
from enum import Enum
from threading import Event, RLock
from traceback import StackSummary, extract_stack
from typing import Any, Iterable, List, Optional, Tuple, Type, TypeVar, Union
from uuid import uuid4

from adagio.cache import NO_OP_CACHE, WorkflowResultCache
from adagio.exceptions import AbortedError, SkippedError, WorkflowBug
from adagio.specs import (
    ConfigSpec,
    InputSpec,
    OutputSpec,
    TaskSpec,
    WorkflowSpec,
    _WorkflowSpecNode,
)
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw as aot
from triad.utils.hash import to_uuid


class WorkflowContext(object):
    def __init__(
        self,
        cache: WorkflowResultCache = NO_OP_CACHE,
        logger: Optional[logging.Logger] = None,
    ):
        self._cache = cache
        self._logger = logging.getLogger() if logger is None else logger

    @property
    def log(self) -> logging.Logger:
        return self._logger

    @property
    def cache(self) -> WorkflowResultCache:
        return self._cache


class TaskContext(object):
    def __init__(self, task: "_Task"):
        self._task = task
        self._configs = _DependencyDict(self._task.configs)
        self._inputs = _DependencyDict(self._task.inputs)
        self._outputs = _OutputDict(self._task.outputs)

    def ensure_all_ready(self):
        """This is a blocking call to wait for all config values
        and input dependencies to be set by the upstreams.
        """
        for c in self._task.configs.values():
            c.get()
        for i in self._task.inputs.values():
            i.get()

    @property
    def configs(self) -> ParamDict:
        """Dictionary of config values. It's lazy, only when a certain key
        is requested, it will block the requesting thread until that
        key's value is ready.
        """
        return self._configs

    @property
    def inputs(self) -> ParamDict:
        """Dictionary of input dependencies. It's lazy, only when a certain key
        is requested, it will block the requesting thread until that
        key's value is ready.
        """
        return self._inputs

    @property
    def outputs(self) -> ParamDict:
        """Dictionary of outputs. For outputs in the spec but you don't set:
        if the caller finished with no exception the framework will set them
        to `skipped`, otherwise, will set them to `failed` with the caller's
        exception
        """
        return self._outputs

    @property
    def metadata(self) -> ParamDict:
        """Metadata of the task
        """
        return self.spec.metadata

    @property
    def spec(self) -> TaskSpec:
        """Spec of the task
        """
        return self._task.spec

    @property
    def abort_requested(self) -> bool:
        return self._task._abort_requested.is_set()

    @property
    def log(self) -> logging.Logger:
        return self._task.ctx.log


class _Dependency(object):
    def __init__(self):
        self.dependency: Optional["_Dependency"] = None
        self.children: List["_Dependency"] = []

    def set_dependency(self, other: "_Dependency") -> "_Dependency":
        other = other if other.dependency is None else other.dependency
        self.validate_dependency(other)
        self.dependency = other
        other.children.append(self)
        return self

    def validate_dependency(self, other: "_Dependency") -> None:
        pass


class _Output(_Dependency):
    def __init__(self, task: "_Task", spec: OutputSpec):
        super().__init__()
        self.task = task
        self.spec = spec

        self.exception: Optional[Exception] = None
        self.trace: Optional[StackSummary] = None
        self.value_set = Event()
        self.value: Any = None
        self.skipped = False

        self._lock = RLock()

    def __repr__(self) -> str:
        return f"{self.task}->{self.spec})"

    def __uuid__(self) -> str:
        return to_uuid(self.task, self.spec)

    def set(self, value: Any, from_cache: bool = False) -> "_Output":
        with self._lock:
            if not self.value_set.is_set():
                try:
                    self.value = self.spec.validate_value(value)
                    if self.task.spec.deterministic and not from_cache:
                        self.task.ctx.cache.set(self.__uuid__(), self.value)
                    self.value_set.set()
                except Exception as e:
                    e = ValueError(str(e))
                    self.fail(e)
            return self

    def fail(
        self,
        exception: Exception,
        trace: Optional[StackSummary] = None,
        throw: bool = True,
    ) -> None:
        with self._lock:
            if not self.value_set.is_set():
                self.exception = exception
                self.trace = trace or extract_stack()
                self.value_set.set()
                if throw:
                    raise exception

    def skip(self, from_cache: bool = False) -> None:
        with self._lock:
            if not self.value_set.is_set():
                self.skipped = True
                if self.task.spec.deterministic and not from_cache:
                    self.task.ctx.cache.skip(self.__uuid__())
                self.value_set.set()

    @property
    def is_set(self) -> bool:
        return self.value_set.is_set()

    @property
    def is_successful(self) -> bool:
        return self.value_set.is_set() and not self.skipped and self.exception is None

    @property
    def is_failed(self) -> bool:
        return self.value_set.is_set() and self.exception is not None

    @property
    def is_skipped(self) -> bool:
        return self.value_set.is_set() and self.skipped

    def validate_dependency(self, other: "_Dependency") -> None:
        aot(
            isinstance(other, (_Input, _Output)),
            TypeError(f"{other} is not Input or Output"),
        )
        self.spec.validate_spec(other.spec)  # type:ignore


class _Input(_Dependency):
    def __init__(self, task: "_Task", spec: InputSpec):
        super().__init__()
        self.task = task
        self.spec = spec
        self._cached = False
        self._cached_value: Any = None

    def __repr__(self) -> str:
        return f"{self.dependency}->{self.spec})"

    def __uuid__(self) -> str:
        return to_uuid(self.dependency, self.spec)

    def get(self) -> Any:
        if self._cached:
            return self._cached_value
        # the furthest dependency must be Output by definition
        assert isinstance(self.dependency, _Output)
        if not self.dependency.value_set.wait(self.spec.timeout):
            if self.spec.default_on_timeout and not self.spec.required:
                return self.spec.default_value
            raise TimeoutError(
                f"Unable to get value in {self.spec.timeout} seconds from {self}"
            )
        if self.dependency.exception is not None:
            raise self.dependency.exception
        elif self.dependency.is_skipped:
            if not self.spec.required:
                return self.spec.default_value
            raise SkippedError(f"{self.dependency} was skipped")
        else:
            return self.dependency.value

    def validate_dependency(self, other: "_Dependency") -> None:
        aot(
            isinstance(other, (_Input, _Output)),
            TypeError(f"{other} is not Input or Output"),
        )
        self.spec.validate_spec(other.spec)  # type:ignore

    def _cache(self) -> Any:
        """Get value and cache so following `get` calls will just return
        the cached value. This function should not be called by users. It
        must be called on a single thread.

        :return: cached value
        """
        if self._cached:
            return
        self._cached_value = self.get()
        self._cached = True
        return self._cached_value


class _ConfigVar(_Dependency):
    def __init__(self, task: "_Task", spec: ConfigSpec):
        super().__init__()
        self.task = task
        self.is_set = False
        self.value: Any = None
        self.spec = spec

    def __repr__(self) -> str:
        return f"{self.spec}: {self.value}"

    def __uuid__(self) -> str:
        return to_uuid(self.get(), self.spec)

    def set(self, value: Any):
        self.value = self.spec.validate_value(value)
        self.is_set = True

    def get(self) -> Any:
        if self.dependency is not None:
            return self.dependency.get()  # type:ignore
        if not self.is_set:
            aot(not self.spec.required, f"{self} is required but not set")
            return self.spec.default_value
        return self.value

    def validate_dependency(self, other: "_Dependency") -> None:
        aot(
            isinstance(other, (_ConfigVar)),
            TypeError(f"{other} is not Input or Output"),
        )
        self.spec.validate_spec(other.spec)  # type:ignore


class _DependencyDict(ParamDict):
    def __init__(self, data: IndexedOrderedDict[str, _Dependency]):
        super().__init__()
        for k, v in data.items():
            super().__setitem__(k, v)
        self.set_readonly()

    def __getitem__(self, key: str) -> Any:
        return super().__getitem__(key).get()

    def items(self) -> Iterable[Tuple[str, Any]]:
        for k in self.keys():
            yield k, self[k]


class _OutputDict(ParamDict):
    def __init__(self, data: IndexedOrderedDict[str, _Output]):
        super().__init__()
        for k, v in data.items():
            super().__setitem__(k, v)
        self.set_readonly()

    def __setitem__(self, key: str, value: Any) -> None:
        super().__getitem__(key).set(value)

    def __getitem__(self, key: str) -> Any:
        raise InvalidOperationError("Can't get items from outputs")

    def items(self) -> Iterable[Tuple[str, Any]]:
        raise InvalidOperationError("Can't get items from outputs")


class _State(Enum):
    CREATED = 0
    RUNNING = 1
    FINISHED = 2
    ABORTED = 3
    FAILED = 4
    SKIPPED = 5

    @staticmethod
    def transit(state_from: "_State", state_to: "_State") -> "_State":
        if state_from == _State.CREATED:
            if state_to in [
                _State.RUNNING,
                _State.ABORTED,
                _State.SKIPPED,
                _State.FINISHED,
            ]:
                return state_to
        elif state_from == _State.RUNNING:
            if state_to in [_State.FINISHED, _State.ABORTED, _State.FAILED]:
                return state_to
        raise InvalidOperationError(
            f"Unable to transit from {state_from} to {state_to}"
        )


T = TypeVar("T", bound=Union[_ConfigVar, _Input, _Output])


class _Task(object):
    def __init__(
        self,
        name: str,
        spec: TaskSpec,
        ctx: WorkflowContext,
        parent: Optional["_Workflow"] = None,
    ):
        self._lock = RLock()
        self._abort_requested = Event()
        self._exception: Optional[Exception] = None
        self._trace: Optional[StackSummary] = None

        self.parent = parent
        self.name = name
        self.ctx = ctx
        self.spec = spec
        self.configs = self._make_dict(spec.configs.values(), _ConfigVar)
        self.inputs = self._make_dict(spec.inputs.values(), _Input)
        self.outputs = self._make_dict(spec.outputs.values(), _Output)
        self._rand_id = str(uuid4())
        self.state = _State.CREATED

    def __repr__(self) -> str:
        return self.name

    @property
    def deterministic(self) -> bool:
        return self.spec.deterministic

    @property
    def lazy(self) -> bool:
        return self.spec.lazy

    @property
    def log(self) -> logging.Logger:
        return self.ctx.log

    def __uuid__(self) -> str:
        if self.deterministic:
            return to_uuid(self.spec, self.configs, self.inputs)
        return self._rand_id

    def skip(self) -> None:
        with self._lock:
            self._transit(_State.SKIPPED)
            self._skip_remaining()

    def request_abort(self) -> None:
        self._abort_requested.set()

    def run(self) -> None:
        with self._lock:
            if self.state == _State.SKIPPED:
                return
            self._transit(_State.RUNNING)
            try:
                if self._abort_requested.is_set():
                    raise AbortedError()
                self.spec.func(TaskContext(self))
            except Exception as e:
                self._fail_remaining(e)
                self._transit(
                    _State.ABORTED if isinstance(e, AbortedError) else _State.FAILED, e
                )
                return
            self._skip_remaining()
            self._transit(_State.FINISHED)

    def update_by_cache(self) -> None:
        if not self.spec.deterministic:
            return
        self.ensure_fully_connected()
        d = IndexedOrderedDict()
        for k, o in self.outputs.items():
            hasvalue, skipped, value = self.ctx.cache.get(o.__uuid__())
            if not hasvalue:
                return
            d[k] = (skipped, value)
        for k, v in d.items():
            if v[0]:
                self.outputs[k].skip(from_cache=True)
            else:
                self.outputs[k].set(v[1], from_cache=True)
        self._transit(_State.FINISHED)

    def ensure_fully_connected(self) -> None:  # pragma: no cover
        """By design, this should be called always when fully connected,
        but if this failed, it means there is a bug in the framework itself.
        """
        for k, v in self.configs.items():
            try:
                v.get()
            except Exception:
                raise WorkflowBug(f"BUG: config {k}'s value or dependency is not set")
        for k, vv in self.inputs.items():
            aot(
                vv.dependency is not None,
                WorkflowBug(f"BUG: input {k}'s dependency is not set"),
            )

    def _make_dict(
        self, data: Iterable[Any], out_type: Type[T]
    ) -> IndexedOrderedDict[str, T]:
        res = IndexedOrderedDict((v.name, out_type(self, v)) for v in data)
        res.set_readonly()
        return res

    def _skip_remaining(self):
        with self._lock:
            for o in self.outputs.values():
                if not o.is_set:
                    o.skip()

    def _fail_remaining(self, e: Exception):
        with self._lock:
            for o in self.outputs.values():
                if not o.is_set:
                    o.fail(e, throw=False)

    def _transit(self, new_state: _State, e: Optional[Exception] = None):
        with self._lock:
            old = self.state
            self.state = _State.transit(self.state, new_state)
            if e is not None:
                self._exception = e
                self._trace = extract_stack()
                self.log.error(f"{self} {old} -> {self.state}", e)
            else:
                self.log.debug(f"{self} {old} -> {self.state}")


class _Workflow(_Task):
    def __init__(
        self,
        name: str,
        spec: WorkflowSpec,
        ctx: WorkflowContext,
        parent: Optional["_Workflow"] = None,
    ):
        super().__init__(name, spec, ctx, parent)
        self.tasks = IndexedOrderedDict()
        for k, v in spec.nodes.items():
            self.tasks[k] = self._build_task(v)
        self._set_outputs()

    def __uuid__(self) -> str:
        return to_uuid(self.spec, self.configs, self.inputs, self.tasks)

    def _build_task(self, spec: _WorkflowSpecNode) -> _Task:
        if isinstance(spec.task, WorkflowSpec):
            task: _Task = _Workflow(spec.name, spec.task, self.ctx, self)
        else:
            task = _Task(spec.name, spec.task, self.ctx, self)
        self._set_configs(task, spec)
        self._set_inputs(task, spec)
        return task

    def _set_inputs(self, task: _Task, spec: _WorkflowSpecNode) -> None:
        for f, to_expr in spec.dependency.items():
            t = to_expr.split(".", 1)
            if len(t) == 1:
                task.inputs[f].set_dependency(self.inputs[t[0]])
            else:
                task.inputs[f].set_dependency(self.tasks[t[0]].outputs[t[1]])

    def _set_configs(self, task: _Task, spec: _WorkflowSpecNode) -> None:
        for f, v in spec.config.items():
            task.configs[f].set(v)
        for f, t in spec.config_dependency.items():
            task.configs[f].set_dependency(self.configs[t])

    def _set_outputs(self) -> None:
        assert isinstance(self.spec, WorkflowSpec)
        for f, to_expr in self.spec.internal_dependency.items():
            t = to_expr.split(".", 1)
            if len(t) == 1:
                self.outputs[f].set_dependency(self.inputs[t[0]])
            else:
                self.outputs[f].set_dependency(self.tasks[t[0]].task.outputs[t[1]])

    def update_by_cache(self) -> None:
        self.ensure_fully_connected()
        for n in self.tasks.values():
            n.task.update_by_cache()
