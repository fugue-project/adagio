import json
from typing import Any, Dict, List, Optional, Type, TypeVar

from adagio.exceptions import DependencyDefinitionError, DependencyNotDefinedError
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.utils.assertion import assert_or_throw as aot
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
        self.metadata.set_readonly()

    def __uuid__(self) -> str:
        return to_uuid([self.__dict__[x] for x in self.attributes])

    def __repr__(self) -> str:
        return self.name

    def validate_value(self, obj: Any) -> Any:
        if obj is not None:
            aot(
                isinstance(obj, self.data_type),
                TypeError(f"{obj} mismatches type {self.paramdict}"),
            )
            return obj
        aot(self.nullable, f"Can't set None to {self}")
        return obj

    def validate_spec(self, spec: "OutputSpec") -> "OutputSpec":
        if not self.nullable:
            aot(
                not spec.nullable,
                TypeError(f"{self} - {spec} are not compatible on nullable"),
            )
        aot(
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
            aot(default_value is None, "required var can't have default_value")
        elif default_value is None:
            aot(nullable, "default_value can't be None because it's not nullable")
        else:
            self.default_value = as_type(self.default_value, self.data_type)

    def validate_value(self, obj: Any) -> Any:
        if obj is not None:
            return super().validate_value(obj)
        aot(self.nullable, f"Can't set None to {self.paramdict}")
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
        aot(self.timeout >= 0, "timeout can't be negative")
        if required:
            aot(not default_on_timeout, "default is not allowed for required input")

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
        aot(isinstance(obj, dict), f"{obj} is not dict")
        return to_type(**obj)

    def _parse_spec_collection(
        self, obj: Any, to_type: Type[T]
    ) -> IndexedOrderedDict[str, T]:
        res: IndexedOrderedDict[str, T] = IndexedOrderedDict()
        aot(isinstance(obj, List), "Spec collection must be a list")
        for v in obj:
            s = self._parse_spec(v, to_type)
            aot(s.name not in res, KeyError(f"Duplicated key {s.name}"))
            res[s.name] = s
        return res


class _WorkflowSpecNode(object):
    def __init__(
        self,
        workflow: "WorkflowSpec",
        name: str,
        task: Any,
        dependency: Optional[Dict[str, str]],
        config: Optional[Dict[str, Any]],
        config_dependency: Optional[Dict[str, str]],
    ):
        if isinstance(task, TaskSpec):
            _t: TaskSpec = task
        elif isinstance(task, dict):
            _t = TaskSpec(**task)
        else:  # pragma: no cover
            raise TypeError(f"{task} is not a valid TaskSpec")
        self.workflow = workflow
        self.name = assert_triad_var_name(name)
        self.task = _t
        self.dependency = dependency or {}
        self.config = config or {}
        self.config_dependency = config_dependency or {}
        try:
            self._validate_config()
            self._validate_dependency()
        except DependencyDefinitionError:
            raise
        except Exception as e:
            raise DependencyDefinitionError(e)

    def __uuid__(self) -> str:
        return to_uuid(
            self.name, self.task, self.dependency, self.config, self.config_dependency
        )

    @property
    def jsondict(self) -> ParamDict:
        return dict(
            name=self.name,
            task=self.task.jsondict,
            dependency=self.dependency,
            config=self.config,
            config_dependency=self.config_dependency,
        )

    def _validate_dependency(self):
        if set(self.dependency.keys()) != set(self.task.inputs.keys()):
            raise DependencyNotDefinedError(
                self.name + " input", self.task.inputs.keys(), self.dependency.keys()
            )
        for k, v in self.dependency.items():
            t = v.split(".", 1)
            if len(t) == 1:
                aot(
                    t[0] in self.workflow.inputs,
                    f"{t[0]} is not an input of the workflow",
                )
                self.task.inputs[k].validate_spec(self.workflow.inputs[t[0]])
            else:  # len(t) == 2
                aot(t[0] != self.name, f"{v} tries to connect to self node {self.name}")
                node = self.workflow.nodes[t[0]]
                aot(
                    t[1] in node.task.outputs,
                    f"{t[1]} is not an output of node {node.name}",
                )
                self.task.inputs[k].validate_spec(node.task.outputs[t[1]])

    def _validate_config(self):
        for k, v in self.config.items():
            self.task.configs[k].validate_value(v)
        defined = set(self.config.keys())
        for k, t in self.config_dependency.items():
            aot(k not in defined, f"can't redefine config {k} in node {self.name}")
            defined.add(k)
            aot(t in self.workflow.configs, f"{t} is not a config of the workflow")
            self.task.configs[k].validate_spec(self.workflow.configs[t])
        for k in set(self.task.configs.keys()).difference(defined):
            aot(
                not self.task.configs[k].required,
                f"config {k} in node {self.name} is required but not defined",
            )


class WorkflowSpec(TaskSpec):
    def __init__(
        self,
        configs: Any,
        inputs: Any,
        outputs: Any,
        metadata=None,
        deterministic: bool = True,
        lazy: bool = True,
        nodes: Optional[List[Any]] = None,
        internal_dependency: Optional[Dict[str, str]] = None,
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
        self.internal_dependency: Dict[str, str] = {}
        if nodes is not None:
            for n in nodes:
                self.add_task(**n)
        if internal_dependency is not None:
            for k, v in internal_dependency.items():
                self.link(k, v)

    def __uuid__(self) -> str:
        return to_uuid(super().__uuid__(), self.nodes, self.internal_dependency)

    def add_task(
        self,
        name: str,
        task: Any,
        dependency: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
        config_dependency: Optional[Dict[str, str]] = None,
    ) -> _WorkflowSpecNode:
        aot(name not in self.nodes, KeyError(f"{name} already exists in workflow"))
        node = _WorkflowSpecNode(
            self, name, task, dependency, config, config_dependency
        )
        self.nodes[name] = node
        return node

    def link(self, output: str, to_expr: str):
        try:
            aot(output in self.outputs, f"{output} is not an output of the workflow")
            aot(output not in self.internal_dependency, f"{output} is already defined")
            t = to_expr.split(".", 1)
            if len(t) == 1:
                aot(t[0] in self.inputs, f"{t[0]} is not an input of the workflow")
                self.outputs[output].validate_spec(self.inputs[t[0]])
            else:  # len(t) == 2
                node = self.nodes[t[0]]
                aot(t[1] in node.task.outputs, f"{t[1]} is not an output of {node}")
                self.outputs[output].validate_spec(node.task.outputs[t[1]])
            self.internal_dependency[output] = to_expr
        except Exception as e:
            raise DependencyDefinitionError(e)

    @property
    def jsondict(self) -> ParamDict:
        d = super().jsondict
        d["nodes"] = [x.jsondict for x in self.nodes.values()]
        d["internal_dependency"] = self.internal_dependency
        del d["func"]
        return d

    def validate(self) -> None:
        if set(self.outputs.keys()) != set(self.internal_dependency.keys()):
            raise DependencyNotDefinedError(
                "workflow output", self.outputs.keys(), self.internal_dependency.keys()
            )


def json_to_taskspec(json_str: str) -> TaskSpec:
    d = json.loads(json_str)
    if "nodes" in d:
        return WorkflowSpec(**d)
    else:
        return TaskSpec(**d)


def _no_op(self, *args, **kwargs):  # pragma: no cover
    pass
