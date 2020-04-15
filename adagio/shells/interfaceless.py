import inspect
from typing import Any, Callable, Dict, List, Optional, Union, get_type_hints

from adagio.specs import ConfigSpec, InputSpec, OutputSpec, TaskSpec
from triad.utils.assertion import assert_or_throw


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
    return TaskSpec(
        configs, inputs, outputs, func, {}, deterministic=deterministic, lazy=lazy
    )


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
    if isinstance(anno, type):
        assert_or_throw(
            anno is not type(None), TypeError(f"{anno} NoneType is invalid")
        )  # noqa: E721
        return dict(data_type=anno, nullable=False)
    return None


def _get_origin_type(anno: Any) -> type:
    if anno is Any:
        return object
    tp: Any = None
    if hasattr(anno, "get_origin"):
        tp = anno.get_origin()  # python 3.8 # pragma: no cover
    elif hasattr(anno, "__extra__"):
        tp = anno.__extra__  # < 3.7 # pragma: no cover
    elif hasattr(anno, "__origin__"):
        tp = anno.__origin__  # 3.7 # pragma: no cover
    assert_or_throw(
        isinstance(tp, type), TypeError(f"Can't find python type for {anno}")
    )
    return tp


def _is_tuple(anno: Any):
    return hasattr(anno, "__origin__") and anno.__origin__ is tuple


def _is_union(anno: Any):
    return hasattr(anno, "__origin__") and anno.__origin__ is Union
