from threading import Event
from traceback import StackSummary, extract_stack
from typing import Any, Optional, Union

from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from adagio.specs import InputSpec, OutputSpec, ConfigSpec, TaskSpec
from adagio.exceptions import SkippedError


class Output(object):
    def __init__(self, task: "Task", spec: OutputSpec):
        self.task = task
        self.spec = spec

        self.exception: Optional[Exception] = None
        self.trace: Optional[StackSummary] = None
        self.value_set = Event()
        self.skipped = False

    def __repr__(self) -> str:
        return f"{self.task}->{self.spec})"

    def __uuid__(self) -> str:
        return to_uuid(self.task.__uuid__(), self.spec.__uuid__())

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


class Input(object):
    def __init__(self, spec: InputSpec):
        self.spec = spec

    def __repr__(self) -> str:
        return f"{self.output}->{self.spec})"

    def __uuid__(self) -> str:
        return to_uuid(self.output.__uuid__(), self.spec.__uuid__())

    def set_dependency(self, output: Union["Input", Output]) -> "Input":
        self.spec.validate_spec(output.spec)
        self.output = output
        return self

    def get(self) -> Any:
        if isinstance(self.output, Input):
            return self.output.get()
        if not self.output.value_set.wait(self.spec.timeout):
            if self.spec.default_on_timeout and not self.spec.required:
                return self.spec.default_value
            raise TimeoutError(
                f"Unable to get value in {self.spec.timeout} seconds from {self}"
            )
        if self.output.exception is not None:
            raise self.output.exception
        elif self.output.is_skipped:
            if not self.spec.required:
                return self.spec.default_value
            raise SkippedError(f"{self.output} was skipped")
        else:
            return self.output.value

    @property
    def is_set(self) -> bool:
        return self.output.is_set

    @property
    def is_successful(self) -> bool:
        return self.output.is_successful

    @property
    def is_failed(self) -> bool:
        return self.output.is_failed

    @property
    def is_skipped(self) -> bool:
        return self.output.is_skipped


class ConfigVar(object):
    def __init__(self, spec: ConfigSpec):
        self.is_set = False
        self.value: Any = None
        self.spec = spec

    def __repr__(self) -> str:
        return f"{self.spec}: {self.value}"

    def __uuid__(self) -> str:
        return to_uuid(self.value, self.spec.__uuid__())

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
