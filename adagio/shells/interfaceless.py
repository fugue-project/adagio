import inspect
import typing
from typing import Any, Callable, Dict, List, Optional, Union, get_type_hints

from adagio.instances import TaskContext
from adagio.specs import ConfigSpec, InputSpec, OutputSpec, TaskSpec
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import to_function, get_full_type_path


def function_to_taskspec(
    func: Callable,
    is_config: Callable[[List[Dict[str, Any]]], List[bool]],
    deterministic: bool = True,
    lazy: bool = False,
) -> TaskSpec:
    specs = inspect.getfullargspec(func)
    sig = inspect.signature(func)
    annotations = get_type_hints(func)
    assert_or_throw(
        specs.varargs is None and specs.varkw is None and len(specs.kwonlyargs) == 0,
        "Function can't have varargs or kwargs",
    )
    inputs: List[InputSpec] = []
    configs: List[ConfigSpec] = []
    outputs: List[OutputSpec] = []
    arr: List[Dict[str, Any]] = []
    for k, w in sig.parameters.items():
        anno = annotations.get(k, w.annotation)
        a = _parse_annotation(anno)
        a["name"] = k
        if w.default == inspect.Parameter.empty:
            a["required"] = True
        else:
            a["required"] = False
            a["default_value"] = w.default
        arr.append(a)
    cfg = is_config(arr)
    for i in range(len(cfg)):
        if cfg[i]:
            configs.append(ConfigSpec(**arr[i]))
        else:
            assert_or_throw(
                arr[i]["required"], f"{arr[i]}: dependency must not have default value"
            )
            inputs.append(InputSpec(**arr[i]))
    n = 0
    anno = annotations.get("return", sig.return_annotation)
    is_multiple = _is_tuple(anno)
    for x in list(anno.__args__) if is_multiple else [anno]:
        if x == inspect.Parameter.empty or x is type(None):  # noqa: E721
            continue
        a = _parse_annotation(x)
        a["name"] = f"_{n}"
        outputs.append(OutputSpec(**a))
        n += 1
    metadata = dict(__interfaceless_func=get_full_type_path(func))
    return TaskSpec(
        configs,
        inputs,
        outputs,
        _interfaceless_wrapper,
        metadata,
        deterministic=deterministic,
        lazy=lazy,
    )


def _interfaceless_wrapper(ctx: TaskContext) -> None:
    ctx.ensure_all_ready()
    func = to_function(ctx.metadata.get_or_throw("__interfaceless_func", object))
    o = func(**ctx.inputs, **ctx.configs)
    res = list(o) if isinstance(o, tuple) else [o]
    n = 0
    for k in ctx.outputs.keys():
        ctx.outputs[k] = res[n]
        n += 1


def _parse_annotation(anno: Any) -> Dict[str, Any]:
    d = _try_parse(anno)
    if d is not None:
        return d
    assert_or_throw(
        anno.__module__ == "typing", TypeError(f"{anno} is not a valid type")
    )
    nullable = False
    if _is_union(anno):
        tps = set(anno.__args__)
        if type(None) not in tps or len(tps) != 2:
            raise TypeError(f"{anno} can't be converted for TaskSpec")
        anno = [x for x in tps if x is not type(None)][0]  # noqa: E721
        nullable = True
    d = _try_parse(anno)
    if d is not None:
        d["nullable"] = nullable
        return d
    return dict(data_type=_get_origin_type(anno), nullable=nullable)


def _try_parse(anno: Any) -> Optional[Dict[str, Any]]:
    if anno is None or anno == inspect.Parameter.empty or anno is Any:
        return dict(data_type=object, nullable=True)
    if _is_native_type(anno):
        assert_or_throw(
            anno is not type(None),  # noqa: E721
            TypeError(f"{anno} NoneType is invalid"),
        )
        return dict(data_type=anno, nullable=False)
    return None


def _get_origin_type(anno: Any, assert_is_type: bool = True) -> Any:
    # this part looks weird but it's working for 3.6-3.8
    if anno is not None and anno.__module__ == "typing":
        if anno is Any:
            return object
        if hasattr(typing, "get_origin"):  # pragma: no cover
            anno = typing.get_origin(anno)  # type: ignore  # 3.8
        elif hasattr(anno, "__extra__"):  # pragma: no cover
            anno = anno.__extra__  # < 3.7
        elif hasattr(anno, "__origin__"):  # pragma: no cover
            anno = anno.__origin__  # 3.7
    if anno is None:  # pragma: no cover
        anno = type(None)
    if assert_is_type:
        assert_or_throw(
            _is_native_type(anno), TypeError(f"Can't find python type for {anno}")
        )
    return anno


def _is_native_type(anno: Any) -> bool:
    return isinstance(anno, type) and anno.__module__ != "typing"


def _is_tuple(anno: Any):
    return _get_origin_type(anno) is tuple


def _is_union(anno: Any):
    return _get_origin_type(anno, False) is Union
