import json
from typing import Callable, Tuple, cast

from adagio.exceptions import (DependencyDefinitionError,
                               DependencyNotDefinedError)
from adagio.instances import _ConfigVar, _Input, _Output, _Task
from adagio.shells.interfaceless import function_to_taskspec
from adagio.specs import (ConfigSpec, InputSpec, OutputSpec, TaskSpec,
                          WorkflowSpec, to_taskspec)
from pytest import raises
from triad.collections.dict import ParamDict
from triad.utils.hash import to_uuid


def test_outputspec():
    # bad name
    raises(AssertionError, lambda: OutputSpec("", int, True))
    # unknown type
    raises(TypeError, lambda: OutputSpec("a", "xyz", True))

    o = OutputSpec("a", _Task, True)
    x = MockTaskForVar()
    assert x is o.validate_value(x)
    assert o.validate_value(None) is None
    o = OutputSpec("a", _Task, False)
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
    s = InputSpec("a", _Task, True, False, None)
    raises(TypeError, lambda: s.validate_value(123))
    assert s.validate_value(None) is None
    t = MockTaskForVar()
    assert s.validate_value(t) is t

    s = InputSpec("a", _Task, True, True, None, default_on_timeout=False)
    assert s.validate_value(None) is None
    t = MockTaskForVar()
    assert s.validate_value(t) is t

    s = InputSpec("a", _Task, False, True, None, default_on_timeout=False)
    raises(AssertionError, lambda: s.validate_value(None))
    t = MockTaskForVar()
    assert s.validate_value(t) is t

    s = InputSpec("a", _Task, False, True, None, timeout=3, default_on_timeout=False)
    assert 3 == s.timeout
    assert not s.default_on_timeout
    o = OutputSpec("x", _Task, True)
    raises(TypeError, lambda: s.validate_spec(o))  # nullable issue
    o = OutputSpec("x", _Task, False)
    assert o is s.validate_spec(o)

    s = InputSpec("a", _Task, True, True, None, timeout=3, default_on_timeout=False)
    o = OutputSpec("x", _Task, True)
    assert o is s.validate_spec(o)
    o = OutputSpec("x", _Task, False)
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
    assert to_taskspec(j2).__uuid__() == ts.__uuid__()


def test_workflowspec():
    configs = [
        dict(
            name="cx",
            data_type=str,
            nullable=False,
            required=False,
            default_value="c"
        )
    ]
    inputs = [
        dict(
            name="ia",
            data_type=int,
            nullable=False
        ),
        dict(
            name="ib",
            data_type=str,
            nullable=True
        )
    ]
    outputs = [
        dict(
            name="oa",
            data_type=str,
            nullable=False
        ),
        dict(
            name="ob",
            data_type=str,
            nullable=True
        )
    ]
    def is_config(ds): return [d["data_type"] is not int for d in ds]
    def is_int(ds): return [d["data_type"] is int for d in ds]
    f = WorkflowSpec(configs, inputs, outputs, {})
    f.add_task("a", function_to_taskspec(f0, is_config), [])
    f.add_task("b", function_to_taskspec(f1, is_config),
               dependency=dict(a="a._0"), config=dict(b="bb"))
    f.add_task("c", function_to_taskspec(f1, is_config),
               dependency=dict(a="a._1"), config=dict(b="bc"))
    # duplicated key
    raises(KeyError,
           lambda: f.add_task("b", function_to_taskspec(f1, is_config),
                              dependency=dict(a="a._0"), config=dict(b="bb")))
    # configs not fully defined
    raises(DependencyDefinitionError,
           lambda: f.add_task("d", function_to_taskspec(f1, is_config),
                              dependency=dict(a="a._0")))
    # duplicated config and config_dependency
    raises(DependencyDefinitionError,
           lambda: f.add_task("d", function_to_taskspec(f1, is_config),
                              dependency=dict(a="a._0"), config=dict(b="bb"),
                              config_dependency=dict(b="cx")))
    # dependency not existed
    raises(DependencyNotDefinedError,
           lambda: f.add_task("d", function_to_taskspec(f1, is_config),
                              dependency={"a.b": "a._0"}, config=dict(b="bb")))
    # invalid link to syntax
    raises(DependencyDefinitionError,
           lambda: f.add_task("d", function_to_taskspec(f1, is_config),
                              dependency={"a": "a.b._0"}, config=dict(b="bb")))
    # type mismatch
    raises(DependencyDefinitionError,
           lambda: f.add_task("d", function_to_taskspec(f1, is_int),
                              dependency={"b": "a._0"}, config=dict(a="bb")))
    f.add_task("d", function_to_taskspec(f2, is_config),
               dependency={"a": "b._0", "b": "c._0"}, config_dependency=dict(c="cx"))
    f.add_task("e", function_to_taskspec(f1, is_config),
               dependency=dict(a="ia"), config=dict(b="bb"))
    # wf output are not fully connected
    raises(DependencyNotDefinedError, lambda: f.validate())
    # type mismatch
    raises(DependencyDefinitionError, lambda: f.link("oa", "b._0"))
    f.link("oa", "d._0")
    raises(DependencyNotDefinedError, lambda: f.validate())
    f.link("ob", "ib")
    f.validate()
    # f.add_task("f", function_to_taskspec(f1, is_config),
    #           ["input.a,ia"])
    j1 = f.to_json(False)
    f_ = to_taskspec(j1)
    j2 = f_.to_json(False)
    assert j1 == j2
    assert f.__uuid__() == f_.__uuid__()


def _dummy(a: int, b: str) -> float:
    return 0.0


def test_cast():
    x = cast(Callable[[int], float], _dummy)
    print(x)


def _mock_task_func():
    pass


def f0() -> Tuple[int, int]:
    return 1, 2


def f1(a: int, b: str) -> int:
    return a * 10 + len(b)


def f2(a: int, b: int, c: str, d: str = "xyz") -> str:
    return f"{a} {b} {c}"


class MockSpec(object):
    @property
    def deterministic(self):
        return False


class MockTaskForVar(_Task):
    def __init__(self):
        self.spec = MockSpec()

    @property
    def name(self):
        return "taskname"

    def __uuid__(self) -> str:
        return "id"
