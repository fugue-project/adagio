from threading import Event
from traceback import StackSummary, extract_stack
from typing import Any, Optional

from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from adagio.specs import InputSpec, OutputSpec, ConfigSpec, TaskSpec
from adagio.exceptions import SkippedError


class Dependency(object):
    def __init__(self):
        self.dependency: Optional["Dependency"] = None

    def set_dependency(self, other: "Dependency") -> "Dependency":
        other = other if other.dependency is None else other.dependency
        self.validate_dependency(other)
        self.dependency = other
        return self

    def validate_dependency(self, other: "Dependency") -> None:
        pass


class Output(Dependency):
    def __init__(self, task: "Task", spec: OutputSpec):
        super().__init__()
        self.task = task
        self.spec = spec

        self.exception: Optional[Exception] = None
        self.trace: Optional[StackSummary] = None
        self.value_set = Event()
        self.skipped = False

    def __repr__(self) -> str:
        return f"{self.task}->{self.spec})"

    def __uuid__(self) -> str:
        return to_uuid(self.task, self.spec)

    def set(self, value: Any) -> "Output":
        if not self.value_set.is_set():
            try:
                self.value = self.spec.validate_value(value)
                self.value_set.set()
            except Exception as e:
                e = ValueError(str(e))
                self.fail(e)
        return self

    def fail(self, exception: Exception, trace: Optional[StackSummary] = None) -> None:
        if not self.value_set.is_set():
            self.exception = exception
            self.trace = trace or extract_stack()
            self.value_set.set()
            raise exception

    def skip(self) -> None:
        if not self.value_set.is_set():
            self.skipped = True
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

    def validate_dependency(self, other: "Dependency") -> None:
        assert_or_throw(
            isinstance(other, (Input, Output)),
            TypeError(f"{other} is not Input or Output"),
        )
        self.spec.validate_spec(other.spec)  # type:ignore


class Input(Dependency):
    def __init__(self, spec: InputSpec):
        super().__init__()
        self.spec = spec

    def __repr__(self) -> str:
        return f"{self.dependency}->{self.spec})"

    def __uuid__(self) -> str:
        return to_uuid(self.dependency, self.spec)

    def get(self) -> Any:
        # the furthest dependency must be Output by definition
        assert isinstance(self.dependency, Output)
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

    def validate_dependency(self, other: "Dependency") -> None:
        assert_or_throw(
            isinstance(other, (Input, Output)),
            TypeError(f"{other} is not Input or Output"),
        )
        self.spec.validate_spec(other.spec)  # type:ignore


class ConfigVar(Dependency):
    def __init__(self, spec: ConfigSpec):
        super().__init__()
        self.is_set = False
        self.value: Any = None
        self.spec = spec

    def __repr__(self) -> str:
        return f"{self.spec}: {self.value}"

    def __uuid__(self) -> str:
        return to_uuid(self.value, self.spec)

    def set(self, value: Any):
        if isinstance(value, ConfigVar):
            self.spec.validate_spec(value.spec)
            self.value = value
        else:
            self.value = self.spec.validate_value(value)
        self.is_set = True

    def get(self) -> Any:
        if not self.is_set:
            assert_or_throw(not self.spec.required, f"{self} is required but not set")
            return self.spec.default_value
        if isinstance(self.value, ConfigVar):
            return self.value.get()
        return self.value


class Task(object):
    def __init__(self, spec: TaskSpec):
        self.spec = spec
        self.configs = {v.name: ConfigVar(v) for v in spec.configs.values()}
        self.inputs = {v.name: Input(v) for v in spec.inputs.values()}
        self.outputs = {v.name: Output(self, v) for v in spec.outputs.values()}

    def __uuid__(self) -> str:
        return to_uuid(self.spec, self.configs, self.inputs, self.outputs)
