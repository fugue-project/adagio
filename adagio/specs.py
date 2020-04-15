import json
from typing import Any, List, Set, Type, TypeVar

from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import (
    as_type,
    get_full_type_path,
    to_function,
    to_timedelta,
    to_type,
)
from triad.utils.hash import to_uuid
from triad.utils.string import assert_triad_var_name


class OutputSpec(object):
    def __init__(self, name: str, data_type: Any, nullable: bool, metadata: Any = None):
        self.name = assert_triad_var_name(name)
        self.data_type = to_type(data_type)
        self.nullable = nullable
        self.metadata = ParamDict(metadata, deep=True)

    def __uuid__(self) -> str:
        return to_uuid([self.__dict__[x] for x in self.attributes])

    def __repr__(self) -> str:
        return self.name

    def validate_value(self, obj: Any) -> Any:
        if obj is not None:
            assert_or_throw(
                isinstance(obj, self.data_type),
                TypeError(f"{obj} mismatches type {self.paramdict}"),
            )
            return obj
        assert_or_throw(self.nullable, f"Can't set None to {self}")
        return obj

    def validate_spec(self, spec: "OutputSpec") -> "OutputSpec":
        if not self.nullable:
            assert_or_throw(
                not spec.nullable,
                TypeError(f"{self} - {spec} are not compatible on nullable"),
            )
        assert_or_throw(
            issubclass(spec.data_type, self.data_type),
            TypeError(f"{self} - {spec} are not compatible on data_type"),
        )
        return spec

    @property
    def attributes(self) -> List[str]:
        return ["name", "data_type", "nullable", "metadata"]

    @property
    def paramdict(self) -> ParamDict:
        return ParamDict((x, self.__dict__[x]) for x in self.attributes)

    @property
    def jsondict(self) -> ParamDict:
        res = ParamDict()
        for k, v in self.paramdict.items():
            if isinstance(v, type):
                v = get_full_type_path(v)
            res[k] = v
        return res


class ConfigSpec(OutputSpec):
    def __init__(
        self,
        name: str,
        data_type: Any,
        nullable: bool,
        required: bool = True,
        default_value: Any = None,
        metadata: Any = None,
    ):
        super().__init__(name, data_type, nullable, metadata)
        self.required = required
        self.default_value = default_value
        if required:
            assert_or_throw(
                default_value is None, "required var can't have default_value"
            )
        elif default_value is None:
            assert_or_throw(
                nullable, "default_value can't be None because it's not nullable"
            )
        else:
            self.default_value = as_type(self.default_value, self.data_type)

    def validate_value(self, obj: Any) -> Any:
        if obj is not None:
            return super().validate_value(obj)
        assert_or_throw(self.nullable, f"Can't set None to {self.paramdict}")
        return obj

    @property
    def attributes(self) -> List[str]:
        return [
            "name",
            "data_type",
            "nullable",
            "required",
            "default_value",
            "metadata",
        ]


class InputSpec(ConfigSpec):
    def __init__(
        self,
        name: str,
        data_type: Any,
        nullable: bool,
        required: bool = True,
        default_value: Any = None,
        timeout: Any = 0,
        default_on_timeout: bool = False,
        metadata: Any = None,
    ):
        super().__init__(name, data_type, nullable, required, default_value, metadata)
        self.timeout = to_timedelta(timeout).total_seconds()
        self.default_on_timeout = default_on_timeout
        assert_or_throw(self.timeout >= 0, "timeout can't be negative")
        if required:
            assert_or_throw(
                not default_on_timeout, "default is not allowed for required input"
            )

    @property
    def attributes(self) -> List[str]:
        return [
            "name",
            "data_type",
            "nullable",
            "required",
            "default_value",
            "timeout",
            "default_on_timeout",
            "metadata",
        ]


T = TypeVar("T", bound="OutputSpec")


class TaskSpec(object):
    def __init__(
        self,
        configs: Any,
        inputs: Any,
        outputs: Any,
        func: Any,
        metadata: Any = None,
        deterministic: bool = True,
        lazy: bool = False,
    ):
        self.configs = self._parse_spec_collection(configs, ConfigSpec)
        self.inputs = self._parse_spec_collection(inputs, InputSpec)
        self.outputs = self._parse_spec_collection(outputs, OutputSpec)
        self.metadata = ParamDict(metadata, deep=True)
        self.func = to_function(func)
        self.deterministic = deterministic
        self.lazy = lazy

    def __uuid__(self) -> str:
        return to_uuid(
            self.configs,
            self.inputs,
            self.outputs,
            get_full_type_path(self.func),
            self.metadata,
            self.deterministic,
            self.lazy,
        )

    def to_json(self, indent: bool = False) -> str:
        if not indent:
            return json.dumps(self.jsondict, separators=(",", ":"))
        else:
            return json.dumps(self.jsondict, indent=4)

    @property
    def jsondict(self) -> ParamDict:
        return ParamDict(
            dict(
                configs=[c.jsondict for c in self.configs.values()],
                inputs=[c.jsondict for c in self.inputs.values()],
                outputs=[c.jsondict for c in self.outputs.values()],
                func=get_full_type_path(self.func),
                metadata=self.metadata,
                deterministic=self.deterministic,
                lazy=self.lazy,
            )
        )

    def _parse_spec(self, obj: Any, to_type: Type[T]) -> T:
        if isinstance(obj, to_type):
            return obj
        if isinstance(obj, str):
            obj = json.loads(obj)
        assert_or_throw(isinstance(obj, dict), f"{obj} is not dict")
        return to_type(**obj)

    def _parse_spec_collection(
        self, obj: Any, to_type: Type[T]
    ) -> IndexedOrderedDict[str, T]:
        res: IndexedOrderedDict[str, T] = IndexedOrderedDict()
        assert_or_throw(isinstance(obj, List), "Spec collection must be a list")
        for v in obj:
            s = self._parse_spec(v, to_type)
            assert_or_throw(s.name not in res, KeyError(f"Duplicated key {s.name}"))
            res[s.name] = s
        return res


class _WorkflowSpecNode(object):
    def __init__(
        self, workflow: "WorkflowSpec", name: str, task: TaskSpec, links: List[str]
    ):
        self.workflow = workflow
        self.name = assert_triad_var_name(name)
        self.task = task
        self.links: List[str] = []
        self._linked: Set[str] = set()
        for l in links:
            self._link(l)

    def __uuid__(self) -> str:
        return to_uuid(self.name, self.task, sorted(self.links))

    def _link(self, expr: str) -> None:
        e = expr.split(",", 1)
        from_expr, to_expr = e[0], e[1]
        assert_or_throw(from_expr not in self._linked, f"{from_expr} is already linked")
        f = from_expr.split(".", 1)
        t = to_expr.split(".", 1)
        if f[0] == "input":
            assert_or_throw(
                f[1] in self.task.inputs, f"{f[1]} is not an input of {self.task}"
            )
            if len(t) == 1:
                assert_or_throw(
                    t[0] in self.workflow.inputs,
                    f"{t[0]} is not an input of the workflow",
                )
                self.task.inputs[f[1]].validate_spec(self.workflow.inputs[t[0]])
            else:  # len(t) == 2
                assert_or_throw(
                    t[0] != self.name, f"{to_expr} tries to connect to self"
                )
                node = self.workflow.nodes[t[0]]
                assert_or_throw(
                    t[1] in node.task.outputs, f"{t[1]} is not an output of {node}"
                )
                self.task.inputs[f[1]].validate_spec(node.task.outputs[t[1]])
        elif f[0] == "config":
            assert_or_throw(
                f[1] in self.task.configs, f"{f[1]} is not a config of {self.task}"
            )
            assert_or_throw(
                to_expr in self.workflow.configs,
                f"{to_expr} is not a config of the workflow",
            )
            self.task.configs[f[1]].validate_spec(self.workflow.configs[to_expr])
        else:
            raise SyntaxError(f"{from_expr} is an invalid expression")
        self._linked.add(from_expr)
        self.links.append(expr)

    @property
    def jsondict(self) -> ParamDict:
        return dict(name=self.name, task=self.task.jsondict, links=self.links)

    def validate(self) -> None:
        defined = set(  # noqa: C401
            x.split(".")[1] for x in self._linked if x.startswith("input.")
        )
        expected = set(self.task.inputs.keys())
        diff = expected.difference(defined)
        assert_or_throw(len(diff) == 0, f"Inputs {diff} are not linked")


class WorkflowSpec(TaskSpec):
    def __init__(
        self,
        configs: Any,
        inputs: Any,
        outputs: Any,
        metadata=None,
        deterministic: bool = True,
        lazy: bool = True,
    ):
        super().__init__(
            configs,
            inputs,
            outputs,
            _no_op,
            metadata=metadata,
            deterministic=deterministic,
            lazy=lazy,
        )
        self.nodes: IndexedOrderedDict[str, _WorkflowSpecNode] = {}
        self.links: List[str] = []
        self._linked: Set[str] = set()

    def __uuid__(self) -> str:
        return to_uuid(super().__uuid__(), self.nodes, sorted(self.links))

    def add_task(
        self, name: str, task: TaskSpec, links: List[str]
    ) -> _WorkflowSpecNode:
        assert_or_throw(
            name not in self.nodes, KeyError(f"{name} already exists in workflow")
        )
        node = _WorkflowSpecNode(self, name, task, links)
        self.nodes[name] = node
        return node

    def link(self, expr: str):
        e = expr.split(",", 1)
        f, to_expr = e[0], e[1]
        assert_or_throw(f not in self._linked, f"{f} is already linked")
        t = to_expr.split(".", 1)
        assert_or_throw(f in self.outputs, f"{f} is not an output of the workflow")
        if len(t) == 1:
            assert_or_throw(
                t[0] in self.inputs, f"{t[0]} is not an input of the workflow"
            )
            self.outputs[f].validate_spec(self.inputs[t[0]])
        else:  # len(t) == 2
            node = self.nodes[t[0]]
            assert_or_throw(
                t[1] in node.task.outputs, f"{t[1]} is not an output of {node}"
            )
            self.outputs[f].validate_spec(node.task.outputs[t[1]])
        self._linked.add(f)
        self.links.append(expr)

    @property
    def jsondict(self) -> ParamDict:
        d = super().jsondict
        d["nodes"] = [x.jsondict for x in self.nodes.values()]
        d["links"] = self.links
        return d

    def validate(self) -> None:
        for n in self.nodes.values():
            n.validate()
        defined = set(self._linked)
        expected = set(self.outputs.keys())
        diff = expected.difference(defined)
        assert_or_throw(len(diff) == 0, f"Outputs {diff} are not linked")


def _no_op(self, *args, **kwargs):  # pragma: no cover
    pass
