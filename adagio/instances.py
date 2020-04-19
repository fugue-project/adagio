from threading import Event, RLock
from traceback import StackSummary, extract_stack
from typing import Any, Iterable, Optional, Tuple
from uuid import uuid4

from adagio.cache import NO_OP_CACHE, WorkflowResultCache
from adagio.exceptions import SkippedError
from adagio.specs import (
    ConfigSpec,
    InputSpec,
    OutputSpec,
    TaskSpec,
    WorkflowSpec,
    _WorkflowSpecNode,
)
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid


class TaskContext(object):
    def __init__(self, task: "_Task"):
        self._task = task
        self._configs = _DependencyDict(self._task.configs)
        self._inputs = _DependencyDict(self._task.inputs)

    def ensure_all_ready(self):
        """This is a blocking call to wait for all config values
        and input dependencies are to be set by upstreams.
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
    def outputs(self) -> IndexedOrderedDict[str, "_Output"]:
        """Dictionary of outputs. For outputs in the spec but you don't set,
        the framework will set them to `skipped`
        """
        return self._task.outputs

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


class _Dependency(object):
    def __init__(self):
        self.dependency: Optional["_Dependency"] = None

    def set_dependency(self, other: "_Dependency") -> "_Dependency":
        other = other if other.dependency is None else other.dependency
        self.validate_dependency(other)
        self.dependency = other
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
        self.skipped = False

        self._lock = RLock()

    def __repr__(self) -> str:
        return f"{self.task}->{self.spec})"

    def __uuid__(self) -> str:
        return to_uuid(self.task, self.spec)

    def set(self, value: Any) -> "_Output":
        with self._lock:
            if not self.value_set.is_set():
                try:
                    self.value = self.spec.validate_value(value)
                    if self.task.spec.deterministic:
                        self.task.cache.set(self.__uuid__(), self.value)
                    self.value_set.set()
                except Exception as e:
                    e = ValueError(str(e))
                    self.fail(e)
            return self

    def fail(self, exception: Exception, trace: Optional[StackSummary] = None) -> None:
        with self._lock:
            if not self.value_set.is_set():
                self.exception = exception
                self.trace = trace or extract_stack()
                self.value_set.set()
                raise exception

    def skip(self) -> None:
        with self._lock:
            if not self.value_set.is_set():
                self.skipped = True
                if self.task.spec.deterministic:
                    self.task.cache.skip(self.__uuid__())
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
        assert_or_throw(
            isinstance(other, (_Input, _Output)),
            TypeError(f"{other} is not Input or Output"),
        )
        self.spec.validate_spec(other.spec)  # type:ignore


class _Input(_Dependency):
    def __init__(self, spec: InputSpec):
        super().__init__()
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
        assert_or_throw(
            isinstance(other, (_Input, _Output)),
            TypeError(f"{other} is not Input or Output"),
        )
        self.spec.validate_spec(other.spec)  # type:ignore

    def _cache(self) -> Any:
        """Get value and cache so following `get` calls will just return
        the cached value. This function should not be called by user. It
        must be called on a single thread.

        :return: cached value
        """
        if self._cached:
            return
        self._cached_value = self.get()
        self._cached = True
        return self._cached_value


class _ConfigVar(_Dependency):
    def __init__(self, spec: ConfigSpec):
        super().__init__()
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
            assert_or_throw(not self.spec.required, f"{self} is required but not set")
            return self.spec.default_value
        return self.value

    def validate_dependency(self, other: "_Dependency") -> None:
        assert_or_throw(
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


class _Task(object):
    def __init__(self, spec: TaskSpec, cache: WorkflowResultCache):
        self.cache = cache
        self.spec = spec
        self.configs = IndexedOrderedDict(
            (v.name, _ConfigVar(v)) for v in spec.configs.values()
        )
        self.configs.set_readonly()
        self.inputs = IndexedOrderedDict(
            (v.name, _Input(v)) for v in spec.inputs.values()
        )
        self.inputs.set_readonly()
        self.outputs = IndexedOrderedDict(
            (v.name, _Output(self, v)) for v in spec.outputs.values()
        )
        self.outputs.set_readonly()
        self._id = str(uuid4())

    def __uuid__(self) -> str:
        if self.spec.deterministic:
            return to_uuid(self.spec, self.configs, self.inputs)
        return self._id

    def run(self) -> None:
        if self._update_by_cache():
            return
        self.spec.func(TaskContext(self))
        for o in self.outputs.values():
            if not o.is_set:
                o.skip()

    def _update_by_cache(self) -> bool:
        if not self.spec.deterministic:
            return False
        d = IndexedOrderedDict()
        for k, o in self.outputs.items():
            hasvalue, skipped, value = self.cache.get(o.__uuid__())
            if not hasvalue:
                return False
            d[k] = (skipped, value)
        for k, v in d.items():
            if v[0]:
                self.outputs[k].skip()
            else:
                self.outputs[k].set(v[1])
        return True


class _WorkflowNode(object):
    def __init__(self, spec: _WorkflowSpecNode, workflow: "_Workflow"):
        self.spec = spec
        self.workflow = workflow
        if isinstance(spec.task, WorkflowSpec):
            self.task: "_Task" = _Workflow(spec.task, workflow.cache)
        else:
            self.task = _Task(spec.task, workflow.cache)

    def __uuid__(self) -> str:
        return self.task.__uuid__()

    def link(self, expr: str) -> None:
        from_expr, to_expr = expr.split(",", 1)
        f = from_expr.split(".", 1)
        t = to_expr.split(".", 1)
        if f[0] == "input":
            if len(t) == 1:
                self.task.inputs[f[1]].set_dependency(self.workflow.inputs[t[0]])
            else:
                self.task.inputs[f[1]].set_dependency(
                    self.workflow.nodes[t[0]].task.outputs[t[1]]
                )
        else:  # config.
            self.task.configs[f[1]].set_dependency(self.workflow.configs[t[0]])


class _Workflow(_Task):
    def __init__(self, spec: WorkflowSpec, cache: WorkflowResultCache = NO_OP_CACHE):
        super().__init__(spec, cache)
        self.nodes = IndexedOrderedDict()
        for k, v in spec.nodes.items():
            self.nodes[k] = _WorkflowNode(v, self)

    def __uuid__(self) -> str:
        return to_uuid(self.spec, self.configs, self.inputs, self.nodes)

    def link(self, expr: str) -> None:
        f, to_expr = expr.split(",", 1)
        t = to_expr.split(".", 1)
        if len(t) == 1:
            self.outputs[f].set_dependency(self.inputs[t[0]])
        else:
            self.outputs[f].set_dependency(self.nodes[t[0]].task.outputs[t[1]])
