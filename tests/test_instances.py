import json
from typing import Callable, cast

from adagio.exceptions import SkippedError
from adagio.instances import _ConfigVar, _Dependency, _Input, _Output, _Task
from adagio.specs import ConfigSpec, InputSpec, OutputSpec, TaskSpec
from pytest import raises
from triad.collections.dict import ParamDict
from triad.utils.hash import to_uuid


def test_dependency():
    a = _Dependency()
    b = _Dependency().set_dependency(a)
    c = _Dependency().set_dependency(b)
    d = _Dependency().set_dependency(c)
    assert d.dependency is a


def test_output():
    t = MockTaskForVar()
    s = OutputSpec("o", dict, False)
    o = _Output(t, s)
    assert to_uuid(t, s) == o.__uuid__()
    assert not o.is_set
    assert not o.is_skipped
    assert not o.is_successful
    assert not o.is_failed
    raises(ValueError, lambda: o.set(1))
    assert o.is_set
    assert not o.is_skipped
    assert not o.is_successful
    assert o.is_failed
    assert isinstance(o.exception, ValueError)
    assert o.trace is not None
    o.set(dict())  # when is_set, setting again will do nothing
    assert o.is_set
    assert not o.is_skipped
    assert not o.is_successful
    assert o.is_failed
    assert isinstance(o.exception, ValueError)

    o = _Output(t, s)
    # setting a bad value will cause exception on both setters and getters
    raises(ValueError, lambda: o.set(None))
    assert o.is_set
    assert not o.is_skipped
    assert not o.is_successful
    assert o.is_failed
    assert isinstance(o.exception, ValueError)

    s2 = OutputSpec("o", dict, True)
    o = _Output(t, s2)
    o.set(None)
    assert o.is_set
    assert not o.is_skipped
    assert o.is_successful
    assert not o.is_failed
    assert o.exception is None

    s2 = OutputSpec("o", dict, True)
    o = _Output(t, s2)
    o.skip()
    assert o.is_set
    assert o.is_skipped
    assert not o.is_successful
    assert not o.is_failed
    assert o.exception is None


def test_input():
    t = MockTaskForVar()
    s = OutputSpec("o", dict, False)
    o = _Output(t, s)
    p = ParamDict()
    ii = InputSpec("x", dict, False, False, default_value=p, default_on_timeout=True)
    i = _Input(ii)
    i.set_dependency(o)
    raises(ValueError, lambda: o.set(None))
    raises(ValueError, lambda: i.get())

    t = MockTaskForVar()
    s = OutputSpec("o", ParamDict, False)
    o = _Output(t, s)
    raises(AssertionError, lambda: InputSpec("x", dict, False, False,
                                             timeout="0.1s",
                                             default_value=None,
                                             default_on_timeout=True))

    # Input linked with Output
    t = MockTaskForVar()
    s = OutputSpec("o", ParamDict, False)
    o = _Output(t, s)
    p = ParamDict()
    p2 = ParamDict()
    ii = InputSpec("x", dict, False, False, timeout="0.1s",
                   default_value=p, default_on_timeout=True)
    i = _Input(ii).set_dependency(o)
    assert p is i.get()
    o.set(p2)
    assert p is not i.get()
    assert p2 is i.get()
    # Input linked with Input
    i2 = _Input(ii).set_dependency(i)
    assert p is not i2.get()
    assert p2 is i2.get()

    t = MockTaskForVar()
    s = OutputSpec("o", ParamDict, False)
    o = _Output(t, s)
    p = ParamDict()
    p2 = ParamDict()
    ii = InputSpec("x", dict, False, False, timeout="0.1s",
                   default_value=p, default_on_timeout=False)
    i = _Input(ii).set_dependency(o)
    raises(TimeoutError, lambda: i.get())

    # Output skipped, input without default will raise error
    t = MockTaskForVar()
    s = OutputSpec("o", ParamDict, False)
    o = _Output(t, s)
    p = ParamDict()
    ii = InputSpec("x", dict, False)
    i = _Input(ii).set_dependency(o)
    o.skip()
    raises(SkippedError, lambda: i.get())

    # Output skipped, input with default will return default
    t = MockTaskForVar()
    s = OutputSpec("o", ParamDict, False)
    o = _Output(t, s)
    p = ParamDict()
    ii = InputSpec("x", dict, False, False, p)
    i = _Input(ii).set_dependency(o)
    o.skip()
    assert p is i.get()

    # Output -> workflow output -> Input
    t = MockTaskForVar()
    s = OutputSpec("o", ParamDict, False)
    oo = _Output(t, s)  # task output
    o = _Output(t, s)  # workflow output
    o.set_dependency(oo)
    p = ParamDict()
    ii = InputSpec("x", dict, False)
    i = _Input(ii).set_dependency(o)
    oo.set(p)
    assert p is i.get()


def test_configvar():
    s = ConfigSpec("a", dict, True, True, None)
    c = _ConfigVar(s)
    raises(AssertionError, lambda: c.get())  # required not set

    p = ParamDict()
    s = ConfigSpec("a", dict, True, False, p)
    c = _ConfigVar(s)
    assert p is c.get()
    c.set(None)
    assert c.get() is None

    p = ParamDict()
    s = ConfigSpec("a", ParamDict, False, False, p)
    c = _ConfigVar(s)
    assert p is c.get()
    raises(AssertionError, lambda: c.set(None))
    assert p is c.get()

    p2 = ParamDict()
    s2 = ConfigSpec("x", dict, False, False, p2)
    c2 = _ConfigVar(s2)
    assert p2 is c2.get()  # not set, use the defaut
    c2.set_dependency(c)  # set parent
    assert p is c2.get()  # get parent value
    p3 = ParamDict()
    c.set(p3)  # set on parent will change child get
    assert p3 is c.get()
    assert p3 is c2.get()


def _dummy(a: int, b: str) -> float:
    return 0.0


def test_cast():
    x = cast(Callable[[int], float], _dummy)
    print(x)


def _mock_task_func():
    pass


class MockTaskForVar(_Task):
    def __init__(self):
        pass

    def __uuid__(self) -> str:
        return "id"
