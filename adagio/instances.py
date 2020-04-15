from threading import Event
from traceback import StackSummary, extract_stack
from typing import Any, Optional

from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from adagio.specs import InputSpec, OutputSpec, ConfigSpec


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
