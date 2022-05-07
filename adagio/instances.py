import concurrent.futures as cf
import logging
import sys
from abc import ABC, abstractmethod
from enum import Enum
from threading import Event, RLock
from traceback import StackSummary, extract_stack
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    no_type_check,
)
from uuid import uuid4

from adagio.exceptions import AbortedError, SkippedError, WorkflowBug
from adagio.specs import ConfigSpec, InputSpec, OutputSpec, TaskSpec, WorkflowSpec
from six import reraise  # type: ignore
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw as aot
from triad.utils.convert import to_instance
from triad.utils.hash import to_uuid


class WorkflowContextMember(object):
    """Base class for components of func:`~adagio.instances.WorkflowContext`

    :param wf_ctx: parent workflow context
    """

    def __init__(self, wf_ctx: "WorkflowContext"):
        self._wf_ctx = wf_ctx

    @property
    def context(self) -> "WorkflowContext":
        """parent workflow context"""
        return self._wf_ctx

    @property
    def conf(self) -> ParamDict:
        """config of parent workflow context"""
        return self.context.conf


class WorkflowResultCache(WorkflowContextMember, ABC):
    """Interface for cachine workflow task outputs. This cache is
    normally for cross execution retrieval.

    The implementation should be thread safe, and all methods should catch all
    exceptions and not raise.
    """

    def __init__(self, wf_ctx: "WorkflowContext"):
        super().__init__(wf_ctx)

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set `key` with `value`

        :param key: uuid string
        :param value: any value
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def skip(self, key: str) -> None:
        """Skip `key`

        :param key: uuid string
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def get(self, key: str) -> Tuple[bool, bool, Any]:
        """Try to get value for `key`

        :param key: uuid string
        :return: <hasvalue>, <skipped>, <value>
        """
        raise NotImplementedError  # pragma: no cover


class NoOpCache(WorkflowResultCache):
    """Dummy WorkflowResultCache doing nothing"""

    def __init__(self, wf_ctx: "WorkflowContext"):
        super().__init__(wf_ctx)

    def set(self, key: str, value: Any) -> None:
        """Set `key` with `value`

        :param key: uuid string
        :param value: any value
        """
        return

    def skip(self, key: str) -> None:
        """Skip `key`

        :param key: uuid string
        """
        return

    def get(self, key: str) -> Tuple[bool, bool, Any]:
        """Try to get value for `key`

        :param key: uuid string
        :return: <hasvalue>, <skipped>, <value>
        """
        return False, False, None


class WorkflowExecutionEngine(WorkflowContextMember, ABC):
    def __init__(self, wf_ctx: "WorkflowContext"):
        super().__init__(wf_ctx)

    def run(self, spec: WorkflowSpec, configs: Dict[str, Any]) -> None:
        wf = _make_top_level_workflow(spec, self.context, configs)
        tasks_to_run = self.preprocess(wf)
        self.run_tasks(tasks_to_run)

    @abstractmethod
    def preprocess(self, wf: "_Workflow") -> List["_Task"]:  # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def run_tasks(self, tasks: List["_Task"]) -> None:  # pragma: no cover
        raise NotImplementedError


class ParallelExecutionEngine(WorkflowExecutionEngine):
    def __init__(self, concurrency: int, wf_ctx: "WorkflowContext"):
        super().__init__(wf_ctx)
        self._concurrency = concurrency

    def preprocess(self, wf: "_Workflow") -> List["_Task"]:
        temp: List["_Task"] = []
        wf._register(temp)
        if self._concurrency <= 1:
            return temp
        return [t for t in temp if len(t.upstream) == 0]

    def run_tasks(self, tasks: List["_Task"]) -> None:
        if self._concurrency <= 1:
            for t in tasks:
                self.run_single(t)
            return
        with cf.ThreadPoolExecutor(max_workers=self._concurrency) as e:
            jobs = [e.submit(self.run_single, task) for task in tasks]
            while jobs:
                for f in cf.as_completed(jobs):
                    jobs.remove(f)
                    try:
                        for task in f.result().downstream:
                            jobs.append(e.submit(self.run_single, task))
                    except Exception:
                        self.context.abort()
                        raise

    def run_single(self, task: "_Task") -> "_Task":
        task.update_by_cache()
        task.run()
        task.reraise()
        return task


class SequentialExecutionEngine(ParallelExecutionEngine):
    def __init__(self, wf_ctx: "WorkflowContext"):
        super().__init__(1, wf_ctx)


class WorkflowHooks(WorkflowContextMember):
    def __init__(self, wf_ctx: "WorkflowContext"):
        super().__init__(wf_ctx)

    def on_task_change(
        self,
        task: "_Task",
        old_state: "_State",
        new_state: "_State",
        e: Optional[Exception] = None,
    ):
        pass  # pragma: no cover


WFMT = TypeVar("WFMT")


class WorkflowContext(object):
    """Context of the workflow instance

    :param cache: cache type, instance or string representation,
      defaults to NoOpCache
    :param engine: engine type, instance or string representation,
      defaults to SequentialExecutionEngine
    :param hooks: hooks type, instance or string representation,
      defaults to WorkflowHooks
    :param logger: hooks type, instance or string representation,
      defaults to None (logging.getLogger())
    :param config: dict like configurations
    """

    @no_type_check
    def __init__(
        self,
        cache: Any = NoOpCache,
        engine: Any = SequentialExecutionEngine,
        hooks: Any = WorkflowHooks,
        logger: Any = None,
        config: Any = None,
    ):
        self._conf: ParamDict = ParamDict(config)
        self._abort_requested: Event = Event()

        self._cache: WorkflowResultCache = self._parse_config(
            cache, WorkflowResultCache, [self]
        )
        self._engine: WorkflowExecutionEngine = self._parse_config(
            engine, WorkflowExecutionEngine, [self]
        )
        self._hooks: WorkflowHooks = self._parse_config(hooks, WorkflowHooks, [self])
        if logger is None:
            logger = logging.getLogger()
        self._logger: logging.Logger = self._parse_config(logger, logging.Logger, [])

    @property
    def log(self) -> logging.Logger:
        """Logger for the workflow"""
        return self._logger

    @property
    def cache(self) -> WorkflowResultCache:
        """Cacher for the workflow"""
        return self._cache

    @property
    def conf(self) -> ParamDict:
        """Configs for the workflow"""
        return self._conf

    @property
    def hooks(self) -> WorkflowHooks:
        """Hooks for the workflow"""
        return self._hooks

    def abort(self) -> None:
        """Call this function to abort a running workflow"""
        self._abort_requested.set()

    @property
    def abort_requested(self) -> bool:
        """Abort requested"""
        return self._abort_requested.is_set()

    def run(self, spec: WorkflowSpec, conf: Dict[str, Any]) -> None:
        """Instantiate and run a workflow spec

        :param spec: workflow spec
        :param conf: configs to initialize the workflow
        """
        self._engine.run(spec, conf)

    def _parse_config(self, data: Any, tp: Type[WFMT], args: List[Any]) -> WFMT:
        if isinstance(data, tp):
            return data
        return cast(WFMT, to_instance(data, expected_base_type=tp, args=args))


class TaskContext(object):
    """Context for a task instance"""

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
    def workflow_context(self) -> WorkflowContext:
        """Workflow context of the task"""
        return self._task.ctx

    @property
    def metadata(self) -> ParamDict:
        """Metadata of the task"""
        return self.spec.metadata

    @property
    def spec(self) -> TaskSpec:
        """Spec of the task"""
        return self._task.spec

    @property
    def abort_requested(self) -> bool:
        """Abort requested"""
        return self._task.abort_requested

    @property
    def log(self) -> logging.Logger:
        """Logger for the task"""
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
            lambda: TypeError(f"{other} is not Input or Output"),
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
        if not self.dependency.value_set.wait(
            self.spec.timeout if self.spec.timeout > 0 else None
        ):
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
            lambda: TypeError(f"{other} is not Input or Output"),
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
            aot(not self.spec.required, lambda: f"{self} is required but not set")
            return self.spec.default_value
        return self.value

    def validate_dependency(self, other: "_Dependency") -> None:
        aot(
            isinstance(other, (_ConfigVar)),
            lambda: TypeError(f"{other} is not Input or Output"),
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

    def values(self) -> List[Any]:
        return [self[k] for k in self.keys()]


class _OutputDict(ParamDict):
    def __init__(self, data: IndexedOrderedDict[str, _Output]):
        super().__init__()
        for k, v in data.items():
            super().__setitem__(k, v)
        self.set_readonly()

    def __setitem__(self, key: str, value: Any) -> None:
        super().__getitem__(key).set(value)

    def __getitem__(self, key: str) -> Any:  # pragma: no cover
        raise InvalidOperationError("Can't get items from outputs")

    def items(self) -> Iterable[Tuple[str, Any]]:  # pragma: no cover
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
        spec: TaskSpec,
        ctx: WorkflowContext,
        parent_workflow: Optional["_Workflow"] = None,
    ):
        self._lock = RLock()
        self._exception: Optional[Exception] = None
        self._exec_info: Any = None

        self.parent_workflow = parent_workflow
        self.ctx = ctx
        self.spec = spec
        self.configs = self._make_dict(spec.configs.values(), _ConfigVar)
        self.inputs = self._make_dict(spec.inputs.values(), _Input)
        self.outputs = self._make_dict(spec.outputs.values(), _Output)
        self._id = ""
        self.execution_id = str(uuid4())
        self.state = _State.CREATED

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: Any) -> bool:
        return self is other

    def __hash__(self) -> int:
        return hash(self.__uuid__())

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def deterministic(self) -> bool:
        return self.spec.deterministic

    @property
    def lazy(self) -> bool:
        return self.spec.lazy

    @property
    def log(self) -> logging.Logger:
        return self.ctx.log

    @property
    def upstream(self) -> Set["_Task"]:
        return {i.dependency.task for i in self.inputs.values()}

    @property
    def downstream(self) -> Set["_Task"]:
        r: Set["_Task"] = set()
        for o in self.outputs.values():
            for c in o.children:
                if not isinstance(c.task, _Workflow):
                    r.add(c.task)
        return r

    @property
    def abort_requested(self) -> bool:
        return self.ctx.abort_requested

    def __uuid__(self) -> str:
        if self._id == "":
            self._ensure_fully_connected()
            if self.deterministic:
                self._id = to_uuid(self.spec, self.configs, self.inputs)
            else:
                self._id = str(uuid4())
        return self._id

    def skip(self) -> None:
        with self._lock:
            self._transit(_State.SKIPPED)
            self._skip_remaining()

    def run(self) -> None:
        with self._lock:
            if self.state in [_State.SKIPPED, _State.FINISHED]:
                return
            elif self.abort_requested:
                self.skip()
            else:
                self._transit(_State.RUNNING)
                try:
                    self.spec.func(TaskContext(self))
                except Exception as e:
                    self._fail_remaining(e)
                    self._transit(
                        _State.ABORTED
                        if isinstance(e, AbortedError)
                        else _State.FAILED,
                        e,
                    )
                    return
                self._skip_remaining()
                self._transit(_State.FINISHED)

    def update_by_cache(self) -> None:
        if not self.spec.deterministic:
            return
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

    def reraise(self):
        if self.state == _State.FAILED:
            reraise(
                type(self._exception),
                self._exception,
                self._exec_info[2],  # type: ignore
            )

    def _register(self, temp: List["_Task"]) -> None:
        temp.append(self)

    def _ensure_fully_connected(self) -> None:  # pragma: no cover
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
                lambda: WorkflowBug(f"BUG: input {k}'s dependency is not set"),
            )

    def _make_dict(
        self, data: Iterable[Any], out_type: Type[T]
    ) -> IndexedOrderedDict[str, T]:
        res: IndexedOrderedDict[str, T] = IndexedOrderedDict()
        for v in data:
            res[v.name] = out_type(self, v)
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
                self._exec_info = sys.exc_info()
                self.log.error(f"{self} {old} -> {self.state}  {e}")
                self.ctx.hooks.on_task_change(self, old, self.state, e)
            else:
                self.log.debug(f"{self} {old} -> {self.state}")
                self.ctx.hooks.on_task_change(self, old, self.state)


class _Workflow(_Task):
    def __init__(
        self,
        spec: WorkflowSpec,
        ctx: WorkflowContext,
        parent_workflow: Optional["_Workflow"] = None,
    ):
        super().__init__(spec, ctx, parent_workflow)
        self.tasks = IndexedOrderedDict()

    def _init_tasks(self):
        for k, v in self.spec.tasks.items():
            self.tasks[k] = self._build_task(v)
        self._set_outputs()

    def _build_task(self, spec: TaskSpec) -> _Task:
        if isinstance(spec, WorkflowSpec):
            task: _Task = _Workflow(spec, self.ctx, self)
        else:
            task = _Task(spec, self.ctx, self)
        self._set_configs(task, spec)
        self._set_inputs(task, spec)
        if isinstance(task, _Workflow):
            # internal initialization must be after external initialization
            task._init_tasks()
        return task

    def _set_inputs(self, task: _Task, spec: TaskSpec) -> None:
        for f, to_expr in spec.node_spec.dependency.items():
            t = to_expr.split(".", 1)
            if len(t) == 1:
                task.inputs[f].set_dependency(self.inputs[t[0]])
            else:
                task.inputs[f].set_dependency(self.tasks[t[0]].outputs[t[1]])

    def _set_configs(self, task: _Task, spec: TaskSpec) -> None:
        for f, v in spec.node_spec.config.items():
            task.configs[f].set(v)
        for f, t in spec.node_spec.config_dependency.items():
            task.configs[f].set_dependency(self.configs[t])

    def _set_outputs(self) -> None:
        assert isinstance(self.spec, WorkflowSpec)
        for f, to_expr in self.spec.internal_dependency.items():
            t = to_expr.split(".", 1)
            if len(t) == 1:
                self.outputs[f].set_dependency(self.inputs[t[0]])
            else:
                self.outputs[f].set_dependency(self.tasks[t[0]].outputs[t[1]])

    def _register(self, temp: List[_Task]) -> None:
        for n in self.tasks.values():
            n._register(temp)

    def update_by_cache(self) -> None:
        self._ensure_fully_connected()
        for n in self.tasks.values():
            n.task.update_by_cache()


def _make_top_level_workflow(
    spec: WorkflowSpec, ctx: WorkflowContext, configs: Dict[str, Any]
) -> _Workflow:
    aot(
        len(spec.inputs) == 0,
        InvalidOperationError("Can't have inputs for top level workflow"),
    )
    wf = _Workflow(spec, ctx)
    for k, vv in configs.items():
        wf.configs[k].set(vv)
    for k, v in wf.configs.items():
        try:
            v.get()
        except Exception:
            raise InvalidOperationError(
                f"config {k}'s value is not set for top level workflow"
            )
    wf._init_tasks()
    return wf
