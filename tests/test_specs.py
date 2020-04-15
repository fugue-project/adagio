import json
from typing import Callable, cast

from adagio.instances import ConfigVar, Input, Output, Task
from adagio.specs import ConfigSpec, InputSpec, OutputSpec, TaskSpec
from pytest import raises
from triad.collections.dict import ParamDict
from triad.utils.hash import to_uuid


def test_outputspec():
    # bad name
    raises(AssertionError, lambda: OutputSpec("", int, True))
    # unknown type
    raises(TypeError, lambda: OutputSpec("a", "xyz", True))

    o = OutputSpec("a", Task, True)
    x = MockTaskForVar()
    assert x is o.validate_value(x)
    assert o.validate_value(None) is None
    o = OutputSpec("a", Task, False)
    x = MockTaskForVar()
    assert x is o.validate_value(x)
    raises(AssertionError, lambda: o.validate_value(None))


def test_inputspec():
    # bad name
    raises(AssertionError, lambda: InputSpec("", int, True, False, None))
    # unknown type
    raises(TypeError, lambda: InputSpec("a", "xyz", True, False, None))
    # default is None but not nullable
    raises(AssertionError, lambda: InputSpec("a", "int", False, False, None))
    # when required, default_value must be None
    raises(AssertionError, lambda: InputSpec(
        "a", "int", True, True, 1, default_on_timeout=False))
    # default_value must be an instance of data_type
    raises(ValueError, lambda: InputSpec("a", "int", True, False, "abc"))
    assert 10 == InputSpec("a", "int", True, False, "10").default_value

    InputSpec("a", int, False, True, None, default_on_timeout=False)
    InputSpec("a", int, True, True, None, default_on_timeout=False)
    s = InputSpec("a", Task, True, False, None)
    raises(AssertionError, lambda: s.validate_value(123))
    assert s.validate_value(None) is None
    t = MockTaskForVar()
    assert s.validate_value(t) is t

    s = InputSpec("a", Task, True, True, None, default_on_timeout=False)
    assert s.validate_value(None) is None
    t = MockTaskForVar()
    assert s.validate_value(t) is t

    s = InputSpec("a", Task, False, True, None, default_on_timeout=False)
    raises(AssertionError, lambda: s.validate_value(None))
    t = MockTaskForVar()
    assert s.validate_value(t) is t

    s = InputSpec("a", Task, False, True, None, timeout=3, default_on_timeout=False)
    assert 3 == s.timeout
    assert not s.default_on_timeout
    o = OutputSpec("x", Task, True)
    raises(AssertionError, lambda: s.validate_spec(o))  # nullable issue
    o = OutputSpec("x", Task, False)
    assert o is s.validate_spec(o)

    s = InputSpec("a", Task, True, True, None, timeout=3, default_on_timeout=False)
    o = OutputSpec("x", Task, True)
    assert o is s.validate_spec(o)
    o = OutputSpec("x", Task, False)
    assert o is s.validate_spec(o)


def test_taskspec():
    configs = [
        dict(
            name="ca",
            data_type=int,
            nullable=False,
            required=False,
            default_value=2
        )
    ]
    inputs = [
        dict(
            name="ia",
            data_type=str,
            nullable=True,
            required=True,
            timeout="1s"
        ),
        dict(
            name="ib",
            data_type=str,
            nullable=True,
            required=True,
            timeout="1s"
        )
    ]
    outputs = [
        dict(
            name="oa",
            data_type=float,
            nullable=False
        )
    ]
    func = _mock_task_func
    metadata = dict(x=1, y="b")
    ts = TaskSpec(configs, inputs, outputs, func, metadata)
    j = ts.to_json(True)
    j2 = TaskSpec(**json.loads(j)).to_json(True)
    assert j == j2
    j = ts.to_json(False)
    j2 = TaskSpec(**json.loads(j)).to_json(False)
    assert j == j2

    configs = [ConfigSpec(**configs[0])]
    outputs = [json.dumps(OutputSpec(**outputs[0]).jsondict)]
    ts = TaskSpec(configs, inputs, outputs, func, metadata)
    j2 = ts.to_json(False)
    assert j == j2



def _dummy(a: int, b: str) -> float:
    return 0.0


def test_cast():
    x = cast(Callable[[int], float], _dummy)
    print(x)


def _mock_task_func():
    pass


class MockTaskForVar(Task):
    def __uuid__(self) -> str:
        return "id"
