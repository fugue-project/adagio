import json
from threading import Event
from traceback import StackSummary, extract_stack
from typing import Any, Callable, Generic, Optional, Tuple, Type, TypeVar

from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.utils.convert import to_timedelta, to_type
from triad.utils.hash import to_uuid
from triad.utils.string import assert_triad_var_name
from triad.utils.assertion import assert_or_throw


class OutputSpec(object):
    def __init__(self, name: str, data_type: Any, nullable: bool, metadata: Any = None):
        self.name = assert_triad_var_name(name)
        self.data_type = to_type(data_type)
        self.nullable = nullable
        self.metadata = ParamDict(metadata, deep=True)

    @property
    def id(self) -> str:
        return to_uuid(self._tuple)

    def __repr__(self) -> str:
        return str(self._tuple)

    def validate_value(self, obj: Any) -> Any:
        if obj is not None:
            assert isinstance(
                obj, self.data_type), f"{obj} mismatches type {self}"
            return obj
        assert self.nullable, f"Can't set None to {self}"
        return obj

    @property
    def _tuple(self) -> Tuple:
        return (self.name, self.data_type, self.nullable)


class ConfigSpec(OutputSpec):
    def __init__(self, name: str, data_type: Any, nullable: bool,
                 required: bool, default_value: Any,
                 metadata: Any = None):
        super().__init__(name, data_type, nullable, metadata)
        self.required = required
        self.default_value = default_value
        if required:
            assert default_value is None, "required var can't have default_value"
        elif default_value is None:
            assert nullable, "default_value can't be None because it's not nullable"
        else:
            assert isinstance(
                default_value,
                self.data_type), f"{default_value} is not of type {data_type}"

    def validate_value(self, obj: Any) -> Any:
        if obj is not None:
            return super().validate_value(obj)
        assert self.nullable, f"Can't set None to {self}"
        return obj

    def validate_spec(self, spec: OutputSpec) -> OutputSpec:
        if not self.nullable:
            assert not spec.nullable, f"{self} - {spec} are not compatible on nullable"
        assert issubclass(spec.data_type, self.data_type), \
            f"{self} - {spec} are not compatible on data_type"
        return spec

    @property
    def _tuple(self) -> Tuple:
        return (self.name, self.data_type, self.nullable,
                self.required, self.default_value)


class InputSpec(ConfigSpec):
    def __init__(self, name: str, data_type: Any, nullable: bool,
                 required: bool, default_value: Any,
                 timeout: Any = 0, default_on_timeout: bool = True,
                 metadata: Any = None):
        super().__init__(name, data_type, nullable, required, default_value, metadata)
        self.timeout_sec = to_timedelta(timeout).total_seconds()
        self.default_on_timeout = default_on_timeout
        assert self.timeout_sec >= 0, "timeout can't be negative"
        if required:
            assert not default_on_timeout, "default is not allowed for required input"

    @property
    def _tuple(self) -> Tuple:
        return (self.name, self.data_type, self.nullable,
                self.required, self.default_value,
                self.timeout_sec, self.default_on_timeout)


T = TypeVar('T', bound='OutputSpec')


def _parse_spec(obj: Any, to_type: Generic[T]) -> T:
    if isinstance(obj, to_type):
        return obj
    if isinstance(obj, str):
        obj = json.loads(obj)
    assert isinstance(obj, dict)
    return to_type(**obj)


def _parse_spec_collection(
    obj: Any,
    to_type: Generic[T]
) -> IndexedOrderedDict[str, T]:
    res: IndexedOrderedDict[str, T] = IndexedOrderedDict()
    for k, v in IndexedOrderedDict(obj):
        k = str(k)
        assert_or_throw(k not in res, KeyError(f"{k} already exists"))
        res[str(k)] = _parse_spec(v)
    return res


class TaskSpec(object):
    def __init__(
        self,
        configs: IndexedOrderedDict[str, ConfigSpec],
        inputs: IndexedOrderedDict[str, InputSpec],
        outputs: IndexedOrderedDict[str, OutputSpec],
        func: Callable[["ConfigCollection",
                        "InputCollection",
                        "OutputCollection"],
                       None],
        metadata: Any = None
    ):
        self.configs = configs
        self.inputs = inputs
        self.outputs = outputs
        self.func = func
        self.metadata = ParamDict(metadata, deep=True)


class Output(object):
    def __init__(self, task: "Task", spec: OutputSpec):
        self._task = task
        self._spec = spec

        self._exception: Optional[Exception] = None
        self._trace: Optional[StackSummary] = None
        self._value_set = Event()

    def __repr__(self) -> str:
        return f"{self._task}->{self._spec})"

    @property
    def id(self) -> str:
        return to_uuid(self._task.id, self._spec.id)

    def set(self, value: Any) -> None:
        if not self._value_set.is_set():
            try:
                self._value = self._spec.validate_value(value)
                self._value_set.set()
            except Exception as e:
                e = ValueError(str(e))
                self.fail(e)

    def fail(self, exception: Exception, trace: Optional[StackSummary] = None) -> None:
        if not self._value_set.is_set():
            self._exception = exception
            self._trace = trace or extract_stack()
            self._value_set.set()
            raise exception

    @property
    def is_set(self) -> bool:
        return self._value_set.is_set()

    @property
    def is_successful(self) -> bool:
        return self._exception is None and self._value_set.is_set()

    @property
    def is_failed(self) -> bool:
        return self._exception is not None and self._value_set.is_set()


class Input(object):
    def __init__(self, spec: InputSpec, output: Output):
        self._output = output
        self._spec = spec

    def __repr__(self) -> str:
        return f"{self._output}->{self._spec})"

    @property
    def id(self) -> str:
        return to_uuid(self._output.id, self._spec.id)

    def get(self) -> Any:
        if not self._output._value_set.wait(self._spec.timeout_sec):
            if self._spec.default_on_timeout and not self._spec.required:
                return self._spec.default_value
            raise TimeoutError(
                f"Unable to get value in {self._spec.timeout_sec} seconds from {self}")
        if self._output._exception is not None:
            raise self._output._exception
        else:
            return self._output._value

    @property
    def is_set(self) -> bool:
        return self._output.is_set

    @property
    def is_successful(self) -> bool:
        return self._output.is_successful

    @property
    def is_failed(self) -> bool:
        return self._output.is_failed


class ConfigVar(object):
    def __init__(self, spec: ConfigSpec):
        self._is_set = False
        self._value = None
        self._spec = spec

    def __repr__(self) -> str:
        return f"{self._spec}: {self._value}"

    @property
    def id(self) -> str:
        return to_uuid(self._value, self._spec.id)

    def set(self, value: Any):
        if isinstance(value, ConfigVar):
            self._spec.validate_spec(value._spec)
            self._value = value
        else:
            self._value = self._spec.validate_value(value)
        self._is_set = True

    def get(self) -> Any:
        if not self._is_set:
            assert not self._spec.required, f"{self} is required but not set"
            return self._spec.default_value
        if isinstance(self._value, ConfigVar):
            return self._value.get()
        return self._value


class ConfigCollection(object):
    pass


class InputCollection(object):
    pass


class OutputCollection(object):
    def set_value(self, key: str, value: object) -> None:
        pass

    def get_value(self, key: str, timeout: Any) -> Any:
        # delta = to_timedelta(timeout)
        pass


class Task(object):
    @property
    def id(self) -> str:
        raise NotImplementedError


class Workflow(object):
    def add_task(
            self,
            config: ParamDict,
            input: ParamDict,
            output: ParamDict,
            func: Callable[[ConfigCollection, InputCollection, OutputCollection], None]
    ) -> Task:
        pass
