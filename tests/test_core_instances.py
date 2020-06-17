import threading
import time
from collections import OrderedDict
from threading import RLock
from time import sleep
from typing import Any, Tuple

from adagio.exceptions import AbortedError
from adagio.instances import (NoOpCache, ParallelExecutionEngine, TaskContext,
                              WorkflowContext, WorkflowHooks,
                              WorkflowResultCache, _ConfigVar, _Input,
                              _make_top_level_workflow, _Output, _State, _Task,
                              _Workflow)
from adagio.shells.interfaceless import function_to_taskspec
from adagio.specs import InputSpec, OutputSpec, WorkflowSpec, _NodeSpec
from pytest import raises
from triad.exceptions import InvalidOperationError
from timeit import timeit


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


def test_task_run_finished_native():
    ts = build_task(example_helper1, None, inputs=dict(a=1), configs=dict(b="xx"))
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
    # Abort before run will skip
    ts = build_task(example_helper1, t3, inputs=dict(a=1), configs=dict(b="xx"))
    assert _State.CREATED == ts.state
    ts.ctx.abort()
    ts.run()
    assert _State.SKIPPED == ts.state
    assert ts.outputs["_0"].is_skipped
    assert ts.outputs["_1"].is_skipped
    # Abort during run will abort
    ts = build_task(example_helper1, t4, inputs=dict(a=1), configs=dict(b="xx"))
    assert _State.CREATED == ts.state
    x = threading.Thread(target=ts.run)
    x.start()
    time.sleep(0.1)
    ts.ctx.abort()
    x.join()
    assert _State.ABORTED == ts.state
    assert ts.outputs["_0"].is_successful
    assert 3 == ts.outputs["_0"].value
    assert ts.outputs["_1"].is_failed
    assert isinstance(ts.outputs["_1"].exception, AbortedError)


def test_workflow_build():
    """
       a   d       j
       |   |      / \
       b   e     |   k
       |   |     |   |
       c   |     aa  bb <----- sub start
        \ /      |   |  \
         f       a   b   |
         |       |  /|   |
         g       | | c   |
        / \      |  \    |
       h   i     cc  dd  ee <----- sub end
    """
    s1 = SimpleSpec(["aa", "bb"], ["cc", "dd", "ee"])
    s1.add("a", example_helper_task1, "*aa")
    s1.add("b", example_helper_task1, "*bb")
    s1.add("c", example_helper_task1)
    s1.link("cc", "a._0")
    s1.link("dd", "b._0")
    s1.link("ee", "bb")
    s = SimpleSpec()
    s.add("a", example_helper_task0)
    s.add("b", example_helper_task1)
    s.add("c", example_helper_task1)
    s.add("d", example_helper_task0)
    s.add("e", example_helper_task1)
    s.add("f", example_helper_task2, "c", "e")
    s.add("g", example_helper_task3)
    s.add("h", example_helper_task3, "g._0")
    s.add("i", example_helper_task3, "g._1")
    s.add("j", example_helper_task0)
    s.add("k", example_helper_task1)
    s.add_task("x", s1, {"aa": "j._0", "bb": "k._0"})
    ctx = WorkflowContext()
    raises(InvalidOperationError, lambda: _make_top_level_workflow(s1, ctx, {}))
    wf = _make_top_level_workflow(s, ctx, {})
    assert wf.tasks["a"].__uuid__() == wf.tasks["d"].__uuid__()
    assert wf.tasks["a"].execution_id != wf.tasks["d"].execution_id
    assert {wf.tasks["j"]} == wf.tasks["x"].tasks["a"].upstream

    #assert 14 == len(ctx._tasks)

    def assert_dep(node, up, down):
        assert set(list(up)) == (up if isinstance(up, set) else set(
            t.name for t in wf.tasks[node].upstream))
        assert set(list(down)) == (down if isinstance(down, set) else set(
            t.name for t in wf.tasks[node].downstream))

    assert_dep("a", "", "b")
    assert_dep("b", "a", "c")
    assert_dep("c", "b", "f")
    assert_dep("d", "", "e")
    assert_dep("e", "d", "f")
    assert_dep("f", "ce", "g")
    assert_dep("g", "f", "hi")
    assert_dep("h", "g", "")
    assert_dep("i", "g", "")
    assert_dep("j", "", "ak")  # downstream are always tasks no workflow (x)
    assert_dep("k", "j", "b")
    assert_dep("x", "jk", "")


def test_workflow_run():
    """
       a   d       j
       |   |      / \
       b   e     |   k
       |   |     |   |
       c   |     aa  bb <----- sub start
        \ /      |   |  \
         f       _a  _b  |
         |       |  /|   |
         g       | | _c  |
        / \      |  \    |
       h   i     cc  dd  ee <----- sub end
                     |   |
                     l   m
    """
    s1 = SimpleSpec(["aa", "bb"], ["cc", "dd", "ee"])
    s1.add("_a", example_helper_task1, "*aa")
    s1.add("_b", example_helper_task1, "*bb")
    s1.add("_c", example_helper_task1)
    s1.link("cc", "_a._0")
    s1.link("dd", "_b._0")
    s1.link("ee", "bb")
    s = SimpleSpec()
    s.add("a", example_helper_task0)
    s.add("b", example_helper_task1)
    s.add("c", example_helper_task1)
    s.add("d", example_helper_task0)
    s.add("e", example_helper_task1)
    s.add("f", example_helper_task2, "c", "e")
    s.add("g", example_helper_task3)
    s.add("h", example_helper_task1, "g._0")
    s.add("i", example_helper_task1, "g._1")
    s.add("j", example_helper_task0)
    s.add("k", example_helper_task1)
    s.add_task("x", s1, {"aa": "j._0", "bb": "k._0"})
    s.add("l", example_helper_task1, "x.dd")
    s.add("m", example_helper_task1, "x.ee")
    hooks = MockHooks(None)
    ctx = WorkflowContext(hooks=hooks)
    ctx.run(s, {})
    expected = {'a': 10, 'b': 11, 'c': 12, 'd': 10, 'e': 11, 'f': 23,
                'h': 34, 'i': 14, 'j': 10, 'k': 11, '_a': 11, '_b': 12,
                '_c': 13, 'l': 13, 'm': 12}
    for k, v in expected.items():
        assert v == hooks.res[k]


def test_workflow_run_parallel():
    # simplest case
    s = SimpleSpec()
    s.add("a", wait_task0)
    s.add("b", wait_task0)
    hooks = MockHooks(None)
    ctx = WorkflowContext(hooks=hooks)
    ctx._engine = ParallelExecutionEngine(2, ctx)
    ctx.run(s, {})
    expected = {'a': 1, 'b': 1}
    for k, v in expected.items():
        assert v == hooks.res[k]
    # with exception
    s = SimpleSpec()
    s.add("a", wait_task0)
    s.add("b", wait_task0e)
    hooks = MockHooks(None)
    ctx = WorkflowContext(hooks=hooks)
    ctx._engine = ParallelExecutionEngine(2, ctx)
    with raises(NotImplementedError):
        ctx.run(s, {})
    expected = {'a': 1}
    for k, v in expected.items():
        assert v == hooks.res[k]
    # on failure should cause abort
    s = SimpleSpec()
    s.add("a", wait_task0)
    s.add("b", wait_task0e)
    s.add("c", wait_task1, "a")
    s.add("d", wait_task1, "b")
    s.add("e", wait_task1, "d")
    s.add("f", wait_task1, "c")
    hooks = MockHooks(None)
    ctx = WorkflowContext(hooks=hooks)
    ctx._engine = ParallelExecutionEngine(10, ctx)

    def run():
        with raises(NotImplementedError):
            ctx.run(s, {})

    t = timeit(run, number=1)
    assert t < 0.2   # only a and b are executed

    expected = {'a': 1}
    for k, v in expected.items():
        assert v == hooks.res[k]
    assert "b" in hooks.failed

    # order of execution
    s = SimpleSpec()
    s.add("a", wait_task0)
    s.add("b", wait_task0)
    s.add("c", wait_task1, "a")
    s.add("d", wait_task1, "b")
    s.add("e", wait_task1, "d")
    s.add("f", wait_task1, "c")
    hooks = MockHooks(None)
    ctx = WorkflowContext(hooks=hooks)
    ctx._engine = ParallelExecutionEngine(2, ctx)
    t = timeit(lambda: ctx.run(s, {}), number=1)
    assert t < 0.4
    assert 3 == hooks.res["e"]
    assert 3 == hooks.res["f"]


def test_workflow_run_with_exception():
    s = SimpleSpec()
    s.add("a", example_helper_task0)
    s.add("b", example_helper_task1e)
    s.add("c", example_helper_task1)
    ctx = WorkflowContext()
    raises(NotImplementedError, lambda: ctx.run(s, {}))


def test_workflow_run_with_cache():
    s = SimpleSpec()
    s.add("a", example_helper_task0)
    s.add("c", example_helper_task1)
    cache = MockCache()
    hooks1 = MockHooks(None)
    ctx = WorkflowContext(cache=cache, hooks=hooks1)
    ctx.run(s, {})
    assert 2 == cache.get_called
    assert 0 == cache.hit
    assert 2 == cache.set_called
    expected = {"a": 10, "c": 11}
    for k, v in expected.items():
        assert v == hooks1.res[k]
    # for the second run, get cache will be called, set will not
    ctx.run(s, {})
    assert 4 == cache.get_called
    assert 2 == cache.hit
    assert 2 == cache.set_called
    expected = {"a": 10, "c": 11}
    for k, v in expected.items():
        assert v == hooks1.res[k]
    # for the third run, get cache will be called, set will not
    hooks2 = MockHooks(None)
    ctx = WorkflowContext(cache=cache, hooks=hooks2)
    ctx.run(s, {})
    assert 6 == cache.get_called
    assert 4 == cache.hit
    assert 2 == cache.set_called
    expected = {"a": 10, "c": 11}
    for k, v in expected.items():
        assert v == hooks1.res[k]
    # TODO: skip is not tested


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


def build_task(example_func, func=None, inputs=None,
               configs=None, cache="NoOpCache",
               deterministic=True, task_name="taskname"):
    ts = function_to_taskspec(example_func, lambda ds: [
        d["data_type"] is str for d in ds])
    ns = _NodeSpec(None, task_name, {}, {}, {})
    ts._node_spec = ns
    if func is not None:
        ts.func = func
    ts.deterministic = deterministic
    wfctx = WorkflowContext(cache=cache)
    t = _Task(ts, wfctx)
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


class SimpleSpec(WorkflowSpec):
    def __init__(self, inputs=[], outputs=[]):
        ip = [InputSpec(x, int, False) for x in inputs]
        op = [OutputSpec(x, int, False) for x in outputs]
        super().__init__(inputs=ip, outputs=op)
        self.cursor = None

    def add(self, name, func, *dep):
        ts = function_to_taskspec(func, lambda ds: [
            d["data_type"] is str for d in ds])
        dependency = {}
        if len(ts.inputs) > 0:
            if len(ts.inputs) == 1:
                if len(dep) == 0:
                    dep = [self.cursor.name]
            for f, t in zip(ts.inputs.keys(), dep):
                if t.startswith("*"):
                    t = t[1:]
                elif "." not in t:
                    t = t + "." + self.tasks[t].outputs.get_key_by_index(0)
                dependency[f] = t
        self.cursor = self.add_task(name, ts, dependency=dependency)


class MockCache(WorkflowResultCache):
    def __init__(self, ctx=None):
        self.tb = dict()
        self.set_called = 0
        self.skip_called = 0
        self.get_called = 0
        self.hit = 0

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
        self.hit += 1
        return True, x[0], x[1]


def example_helper1(a: int, b: str = "x") -> Tuple[int, int]:
    return 1 + len(b), 1 - len(b)


def example_helper_task0() -> int:
    return 10


def example_helper_task1(a: int) -> int:
    return a + 1


def example_helper_task1e(a: int) -> int:
    raise NotImplementedError


def example_helper_task2(a: int, b: int) -> int:
    return a + b


def example_helper_task3(a: int) -> Tuple[int, int]:
    return a + 10, a - 10


def wait_task0() -> int:
    sleep(0.1)
    return 1


def wait_task0e() -> int:
    raise NotImplementedError


def wait_task1(a: int) -> int:
    sleep(0.1)
    return a + 1


def wait_task2(a: int) -> None:
    sleep(0.1)
    print(a + 1)


class MockHooks(WorkflowHooks):
    def __init__(self, wf_ctx: "WorkflowContext"):
        super().__init__(wf_ctx)
        self.lock = RLock()
        self.res = OrderedDict()
        self.skipped = set()
        self.failed = set()

    def on_task_change(
        self,
        task: "_Task",
        old_state: "_State",
        new_state: "_State",
        e=None
    ):
        if new_state == _State.FINISHED and len(task.outputs) == 1:
            with self.lock:
                self.res[task.name] = task.outputs["_0"].value
        if new_state == _State.SKIPPED:
            self.skipped.add(task.name)
        if new_state == _State.FAILED:
            self.failed.add(task.name)
