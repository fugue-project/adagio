import threading
import time
from typing import Tuple, Any

from adagio.cache import NO_OP_CACHE, WorkflowResultCache
from adagio.exceptions import AbortedError
from adagio.instances import (TaskContext, WorkflowContext, _ConfigVar, _Input,
                              _Output, _State, _Task)
from adagio.shells.interfaceless import function_to_taskspec
from pytest import raises
from triad.exceptions import InvalidOperationError


def test_task_skip():
    ts = build_task(example_helper1, t1, inputs=dict(a=1), configs=dict(b="xx"))
    assert _State.CREATED == ts.state
    ts.skip()
    assert _State.SKIPPED == ts.state
    raises(InvalidOperationError, lambda: ts.skip())
    ts.run()  # run is no op after skip
    assert _State.SKIPPED == ts.state


def test_task_run_finished():
    ts = build_task(example_helper1, t2, inputs=dict(a=1), configs=dict(b="xx"))
    assert _State.CREATED == ts.state
    ts.run()
    assert _State.FINISHED == ts.state
    assert ts.outputs["_0"].is_successful
    assert 3 == ts.outputs["_0"].value
    assert ts.outputs["_1"].is_successful
    assert -1 == ts.outputs["_1"].value


def test_task_run_cached():
    cache = MockCache()
    ts = build_task(example_helper1, t2, inputs=dict(a=1),
                    configs=dict(b="xx"), cache=cache)
    id1 = ts.__uuid__()
    oid1 = ts.outputs["_0"].__uuid__()
    assert 1 == cache.get_called  # tried first output, not found, quit
    assert 0 == cache.set_called
    assert 0 == cache.skip_called
    assert _State.CREATED == ts.state
    ts.run()
    assert _State.FINISHED == ts.state
    assert ts.outputs["_0"].is_successful
    assert 3 == ts.outputs["_0"].value
    assert ts.outputs["_1"].is_successful
    assert -1 == ts.outputs["_1"].value
    assert 1 == cache.get_called
    assert 2 == cache.set_called  # set both outputs
    assert 0 == cache.skip_called
    assert id1 == ts.__uuid__()
    assert oid1 == ts.outputs["_0"].__uuid__()

    assert 2 == len(cache.tb)
    ts = build_task(example_helper1, t2, inputs=dict(a=1),
                    configs=dict(b="xx"), cache=cache)
    assert id1 == ts.__uuid__()
    assert oid1 == ts.outputs["_0"].__uuid__()
    assert 3 == cache.get_called  # got both cached results
    assert 2 == cache.set_called
    assert 0 == cache.skip_called
    assert _State.FINISHED == ts.state
    assert ts.outputs["_0"].is_successful
    assert 3 == ts.outputs["_0"].value
    assert ts.outputs["_1"].is_successful
    assert -1 == ts.outputs["_1"].value


def test_task_run_cached_with_skip():
    cache = MockCache()
    ts = build_task(example_helper1, t1, inputs=dict(a=1),
                    configs=dict(b="xx"), cache=cache)
    id1 = ts.__uuid__()
    oid1 = ts.outputs["_0"].__uuid__()
    assert 1 == cache.get_called  # tried first output, not found, quit
    assert 0 == cache.set_called
    assert 0 == cache.skip_called
    assert _State.CREATED == ts.state
    ts.run()
    assert _State.FINISHED == ts.state
    assert ts.outputs["_0"].is_successful
    assert 3 == ts.outputs["_0"].value
    assert ts.outputs["_1"].is_skipped
    assert 1 == cache.get_called
    assert 1 == cache.set_called
    assert 1 == cache.skip_called
    assert id1 == ts.__uuid__()
    assert oid1 == ts.outputs["_0"].__uuid__()

    assert 2 == len(cache.tb)
    ts = build_task(example_helper1, t1, inputs=dict(a=1),
                    configs=dict(b="xx"), cache=cache)
    assert id1 == ts.__uuid__()
    assert oid1 == ts.outputs["_0"].__uuid__()
    assert 3 == cache.get_called  # got both cached results
    assert 1 == cache.set_called
    assert 1 == cache.skip_called
    assert _State.FINISHED == ts.state
    assert ts.outputs["_0"].is_successful
    assert 3 == ts.outputs["_0"].value
    assert ts.outputs["_1"].is_skipped


def test_task_run_cached_non_deterministic():
    cache = MockCache()
    ts = build_task(example_helper1, t2, inputs=dict(a=1),
                    configs=dict(b="xx"), cache=cache, deterministic=False)
    id1 = ts.__uuid__()
    oid1 = ts.outputs["_0"].__uuid__()
    assert 0 == cache.get_called
    assert 0 == cache.set_called
    assert 0 == cache.skip_called
    assert _State.CREATED == ts.state
    ts.run()
    assert _State.FINISHED == ts.state
    assert ts.outputs["_0"].is_successful
    assert 3 == ts.outputs["_0"].value
    assert ts.outputs["_1"].is_successful
    assert -1 == ts.outputs["_1"].value
    assert 0 == cache.get_called
    assert 0 == cache.set_called  # set both outputs
    assert 0 == cache.skip_called
    assert id1 == ts.__uuid__()
    assert oid1 == ts.outputs["_0"].__uuid__()

    assert 0 == len(cache.tb)
    ts = build_task(example_helper1, t2, inputs=dict(a=1),
                    configs=dict(b="xx"), cache=cache, deterministic=False)
    assert id1 != ts.__uuid__()
    assert oid1 != ts.outputs["_0"].__uuid__()
    assert 0 == cache.get_called  # got both cached results
    assert 0 == cache.set_called
    assert 0 == cache.skip_called
    assert _State.CREATED == ts.state


def test_task_run_failed():
    ts = build_task(example_helper1, t3, inputs=dict(a=1), configs=dict(b="xx"))
    assert _State.CREATED == ts.state
    ts.run()
    assert _State.FAILED == ts.state
    assert isinstance(ts._exception, SyntaxError)
    assert ts.outputs["_0"].is_successful
    assert 3 == ts.outputs["_0"].value
    assert ts.outputs["_1"].is_failed
    assert isinstance(ts.outputs["_1"].exception, SyntaxError)


def test_task_run_aborted():
    # Abort before run
    ts = build_task(example_helper1, t3, inputs=dict(a=1), configs=dict(b="xx"))
    assert _State.CREATED == ts.state
    ts.request_abort()
    ts.run()
    assert _State.ABORTED == ts.state
    assert ts.outputs["_0"].is_failed
    assert isinstance(ts.outputs["_0"].exception, AbortedError)
    assert ts.outputs["_1"].is_failed
    assert isinstance(ts.outputs["_1"].exception, AbortedError)
    # Abort during run
    ts = build_task(example_helper1, t4, inputs=dict(a=1), configs=dict(b="xx"))
    assert _State.CREATED == ts.state
    x = threading.Thread(target=ts.run)
    x.start()
    time.sleep(0.1)
    ts.request_abort()
    x.join()
    assert _State.ABORTED == ts.state
    assert ts.outputs["_0"].is_successful
    assert 3 == ts.outputs["_0"].value
    assert ts.outputs["_1"].is_failed
    assert isinstance(ts.outputs["_1"].exception, AbortedError)


def t1(ctx: TaskContext):
    a = ctx.inputs.get_or_throw("a", int)
    b = ctx.configs.get_or_throw("b", str)
    ctx.outputs["_0"] = a + len(b)


def t2(ctx: TaskContext):
    ctx.ensure_all_ready()
    a = ctx.inputs.get_or_throw("a", int)
    b = ctx.configs.get_or_throw("b", str)
    ctx.outputs["_0"] = a + len(b)
    ctx.outputs["_1"] = a - len(b)
    ctx.log.info("done")


def t3(ctx: TaskContext):
    a = ctx.inputs.get_or_throw("a", int)
    b = ctx.configs.get_or_throw("b", str)
    ctx.outputs["_0"] = a + len(b)
    raise SyntaxError("Expected")


def t4(ctx: TaskContext):
    a = ctx.inputs.get_or_throw("a", int)
    b = ctx.configs.get_or_throw("b", str)
    ctx.outputs["_0"] = a + len(b)
    for i in range(5):
        time.sleep(0.1)
        if ctx.abort_requested:
            raise AbortedError()
    raise SyntaxError("Expected")


def build_task(example_func, func, inputs=None, configs=None, cache=NO_OP_CACHE, deterministic=True):
    ts = function_to_taskspec(example_func, lambda ds: [
        d["data_type"] is str for d in ds])
    ts.func = func
    ts.deterministic = deterministic
    wfctx = WorkflowContext(cache=cache)
    t = _Task("taskname", ts, wfctx)
    if inputs is not None:
        for k, v in inputs.items():
            t.inputs[k].dependency = 1  # set a dummy value so will not complaint
            t.inputs[k]._cached = True
            t.inputs[k]._cached_value = v
    if configs is not None:
        for k, v in configs.items():
            t.configs[k].set(v)
    t.update_by_cache()
    return t


class MockCache(WorkflowResultCache):
    def __init__(self):
        self.tb = dict()
        self.set_called = 0
        self.skip_called = 0
        self.get_called = 0

    def set(self, key: str, value: Any) -> None:
        self.tb[key] = (False, value)
        print("set", key)
        self.set_called += 1

    def skip(self, key: str) -> None:
        self.tb[key] = (True, None)
        self.skip_called += 1

    def get(self, key: str) -> Tuple[bool, bool, Any]:
        self.get_called += 1
        if key not in self.tb:
            print("not get", key)
            return False, False, None
        x = self.tb[key]
        print("get", key)
        return True, x[0], x[1]


def example_helper1(a: int, b: str = "x") -> Tuple[int, int]:
    return 1 + len(b), 1 - len(b)
