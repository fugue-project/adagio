import json
from threading import Event
from traceback import StackSummary, extract_stack
from typing import Any, List, Optional, Type, TypeVar

from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import (
    get_full_type_path,
    to_function,
    to_timedelta,
    to_type,
    as_type,
)
from triad.utils.hash import to_uuid
from triad.utils.string import assert_triad_var_name


class OutputSpec(object):
    def __init__(self, name: str, data_type: Any, nullable: bool, metadata: Any = None):
        self.name = assert_triad_var_name(name)
        self.data_type = to_type(data_type)
        self.nullable = nullable
        self.metadata = ParamDict(metadata, deep=True)

    def __uuid__(self) -> str:
        return to_uuid([self.__dict__[x] for x in self.attributes])

    def __repr__(self) -> str:
        return self.name

    def validate_value(self, obj: Any) -> Any:
        if obj is not None:
            assert_or_throw(
                isinstance(obj, self.data_type),
                f"{obj} mismatches type {self.paramdict}",
            )
            return obj
        assert_or_throw(self.nullable, f"Can't set None to {self}")
        return obj

    @property
    def attributes(self) -> List[str]:
        return ["name", "data_type", "nullable", "metadata"]

    @property
    def paramdict(self) -> ParamDict:
        return ParamDict((x, self.__dict__[x]) for x in self.attributes)

    @property
    def jsondict(self) -> ParamDict:
        res = ParamDict()
        for k, v in self.paramdict.items():
            if isinstance(v, type):
                v = get_full_type_path(v)
            res[k] = v
        return res


class ConfigSpec(OutputSpec):
    def __init__(
        self,
        name: str,
        data_type: Any,
        nullable: bool,
        required: bool = True,
        default_value: Any = None,
        metadata: Any = None,
    ):
        super().__init__(name, data_type, nullable, metadata)
        self.required = required
        self.default_value = default_value
        if required:
            assert_or_throw(
                default_value is None, "required var can't have default_value"
            )
        elif default_value is None:
            assert_or_throw(
                nullable, "default_value can't be None because it's not nullable"
            )
        else:
            self.default_value = as_type(self.default_value, self.data_type)

    def validate_value(self, obj: Any) -> Any:
        if obj is not None:
            return super().validate_value(obj)
        assert_or_throw(self.nullable, f"Can't set None to {self.paramdict}")
        return obj

    def validate_spec(self, spec: OutputSpec) -> OutputSpec:
        if not self.nullable:
            assert_or_throw(
                not spec.nullable, f"{self} - {spec} are not compatible on nullable"
            )
        assert_or_throw(
            issubclass(spec.data_type, self.data_type),
            f"{self} - {spec} are not compatible on data_type",
        )
        return spec

    @property
    def attributes(self) -> List[str]:
        return [
            "name",
            "data_type",
            "nullable",
            "required",
            "default_value",
            "metadata",
        ]


class InputSpec(ConfigSpec):
    def __init__(
        self,
        name: str,
        data_type: Any,
        nullable: bool,
        required: bool = True,
        default_value: Any = None,
        timeout: Any = 0,
        default_on_timeout: bool = False,
        metadata: Any = None,
    ):
        super().__init__(name, data_type, nullable, required, default_value, metadata)
        self.timeout = to_timedelta(timeout).total_seconds()
        self.default_on_timeout = default_on_timeout
        assert_or_throw(self.timeout >= 0, "timeout can't be negative")
        if required:
            assert_or_throw(
                not default_on_timeout, "default is not allowed for required input"
            )

    @property
    def attributes(self) -> List[str]:
        return [
            "name",
            "data_type",
            "nullable",
            "required",
            "default_value",
            "timeout",
            "default_on_timeout",
            "metadata",
        ]


T = TypeVar("T", bound="OutputSpec")


class TaskSpec(object):
    def __init__(
        self, configs: Any, inputs: Any, outputs: Any, func: Any, metadata: Any = None
    ):
        self.configs = self._parse_spec_collection(configs, ConfigSpec)
        self.inputs = self._parse_spec_collection(inputs, InputSpec)
        self.outputs = self._parse_spec_collection(outputs, OutputSpec)
        self.metadata = ParamDict(metadata, deep=True)
        self.func = to_function(func)

    def __uuid__(self) -> str:
        return to_uuid(
            self.configs,
            self.inputs,
            self.outputs,
            get_full_type_path(self.func),
            self.metadata,
        )

    def to_json(self, indent: bool = False) -> str:
        o = dict(
            configs=[c.jsondict for c in self.configs.values()],
            inputs=[c.jsondict for c in self.inputs.values()],
            outputs=[c.jsondict for c in self.outputs.values()],
            func=get_full_type_path(self.func),
            metadata=self.metadata,
        )
        if not indent:
            return json.dumps(o, separators=(",", ":"))
        else:
            return json.dumps(o, indent=4)

    def _parse_spec(self, obj: Any, to_type: Type[T]) -> T:
        if isinstance(obj, to_type):
            return obj
        if isinstance(obj, str):
            obj = json.loads(obj)
        assert_or_throw(isinstance(obj, dict), f"{obj} is not dict")
        return to_type(**obj)

    def _parse_spec_collection(
        self, obj: Any, to_type: Type[T]
    ) -> IndexedOrderedDict[str, T]:
        res: IndexedOrderedDict[str, T] = IndexedOrderedDict()
        assert_or_throw(isinstance(obj, List), "Spec collection must be a list")
        for v in obj:
            s = self._parse_spec(v, to_type)
            assert_or_throw(s.name not in res, KeyError(f"Duplicated key {s.name}"))
            res[s.name] = s
        return res


class Output(object):
    def __init__(self, task: "Task", spec: OutputSpec):
        self._task = task
        self._spec = spec

        self._exception: Optional[Exception] = None
        self._trace: Optional[StackSummary] = None
        self._value_set = Event()

    def __repr__(self) -> str:
        return f"{self._task}->{self._spec})"

    def __uuid__(self) -> str:
        return to_uuid(self._task.__uuid__(), self._spec.__uuid__())

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

    def __uuid__(self) -> str:
        return to_uuid(self._output.__uuid__(), self._spec.__uuid__())

    def get(self) -> Any:
        if not self._output._value_set.wait(self._spec.timeout):
            if self._spec.default_on_timeout and not self._spec.required:
                return self._spec.default_value
            raise TimeoutError(
                f"Unable to get value in {self._spec.timeout} seconds from {self}"
            )
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
        self._value: Any = None
        self._spec = spec

    def __repr__(self) -> str:
        return f"{self._spec}: {self._value}"

    def __uuid__(self) -> str:
        return to_uuid(self._value, self._spec.__uuid__())

    def set(self, value: Any):
        if isinstance(value, ConfigVar):
            self._spec.validate_spec(value._spec)
            self._value = value
        else:
            self._value = self._spec.validate_value(value)
        self._is_set = True

    def get(self) -> Any:
        if not self._is_set:
            assert_or_throw(not self._spec.required, f"{self} is required but not set")
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
    def __uuid__(self) -> str:
        raise NotImplementedError
