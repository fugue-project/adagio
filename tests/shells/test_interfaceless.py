from adagio.shells.interfaceless import function_to_taskspec, _get_origin_type, _parse_annotation
from typing import Any, Dict, List, Union, Optional, Tuple
from pytest import raises
import inspect


def test_function_to_taskspec():
    ts = function_to_taskspec(f1, lambda ds: [d["data_type"] is not int for d in ds])
    assert 2 == len(ts.inputs)
    assert "a" in ts.inputs and "b" in ts.inputs
    assert all(r.required for r in ts.inputs.values())
    assert ts.inputs["a"].nullable
    assert not ts.inputs["b"].nullable

    assert 2 == len(ts.configs)
    assert "c" in ts.configs and "d" in ts.configs
    assert not ts.configs["c"].nullable
    assert ts.configs["c"].required
    assert ts.configs["d"].nullable
    assert not ts.configs["d"].required
    assert ts.configs["d"].default_value == "x"
    assert all(r.data_type is str for r in ts.configs.values())

    assert 1 == len(ts.outputs)
    assert "_0" in ts.outputs
    assert not ts.outputs["_0"].nullable
    assert ts.outputs["_0"].data_type is int

    ts = function_to_taskspec(f2, lambda ds: [d["data_type"] is not int for d in ds],
                              deterministic=False, lazy=True)
    assert ts.lazy
    assert not ts.deterministic

    assert 2 == len(ts.inputs)
    assert "a" in ts.inputs and "c" in ts.inputs
    assert 1 == len(ts.configs)
    assert "b" in ts.configs
    assert ts.configs["b"].data_type is object
    assert ts.configs["b"].nullable
    assert 2 == len(ts.outputs)
    assert "_0" in ts.outputs and "_1" in ts.outputs
    assert not ts.outputs["_0"].nullable
    assert ts.outputs["_1"].nullable

    ts = function_to_taskspec(f3, lambda ds: [d["data_type"] is not int for d in ds])
    print(ts.to_json(True))
    assert 0 == len(ts.outputs)

    ts = function_to_taskspec(f4, lambda ds: [d["data_type"] is not int for d in ds])
    print(ts.to_json(True))
    assert 0 == len(ts.outputs)

    ts = function_to_taskspec(f5, lambda ds: [d["data_type"] is not int for d in ds])
    print(ts.to_json(True))
    assert 0 == len(ts.inputs)
    assert 0 == len(ts.outputs)

    # TODO: not tested on func and metadata because that is supposed to change to something else


def test__parse_annotation():
    assert dict(data_type=object, nullable=True) == _parse_annotation(None)
    assert dict(data_type=object, nullable=True) == _parse_annotation(
        inspect.Parameter.empty)
    assert dict(data_type=object, nullable=True) == _parse_annotation(Any)
    assert dict(data_type=int, nullable=False) == _parse_annotation(int)
    assert dict(data_type=dict, nullable=False) == _parse_annotation(Dict[str, Any])
    assert dict(data_type=object, nullable=True) == _parse_annotation(
        Optional[Any])
    assert dict(data_type=str, nullable=True) == _parse_annotation(
        Optional[str])
    assert dict(data_type=dict, nullable=True) == _parse_annotation(
        Optional[Dict[str, Any]])
    assert dict(data_type=dict, nullable=True) == _parse_annotation(
        Union[None, Dict[str, Any]])
    assert dict(data_type=dict, nullable=True) == _parse_annotation(
        Union[Dict[str, Any], None])
    assert dict(data_type=dict, nullable=True) == _parse_annotation(
        Union[Dict[str, Any], None, None])
    assert dict(data_type=dict, nullable=False) == _parse_annotation(
        Union[Dict[str, Any]])
    raises(TypeError, lambda: _parse_annotation(Union[Dict[str, Any], List[str]]))
    raises(TypeError, lambda: _parse_annotation(Union[Dict[str, Any], List[str], None]))
    raises(TypeError, lambda: _parse_annotation(Union[None]))
    raises(TypeError, lambda: _parse_annotation(Union[None, None]))
    raises(TypeError, lambda: _parse_annotation(type(None)))


def test__get_origin_type():
    assert _get_origin_type(Any) is object
    assert _get_origin_type(Dict[str, Any]) is dict
    assert _get_origin_type(List[str]) is list
    assert _get_origin_type(List[Any]) is list
    assert _get_origin_type(Tuple[int, str]) is tuple
    assert _get_origin_type(Union[int, str], False) is Union
    assert _get_origin_type(Union[None]) is type(None)
    assert _get_origin_type(int) is int


def f1(a: Optional[int], b: "int", c: str, d: "Optional[str]" = "x") -> "int":
    return a + b


def f2(a: int, b, c: int) -> "Tuple[int,Optional[str]]":
    return a + c


def f3(a: int, b: str, c: int):
    pass


def f4(a: int, b: str, c: int) -> None:
    pass


def f5():
    pass
