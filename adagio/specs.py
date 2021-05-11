import json
from typing import Any, Dict, List, Optional, Type, TypeVar

from adagio.exceptions import DependencyDefinitionError, DependencyNotDefinedError
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw as aot, assert_arg_not_none
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
                lambda: TypeError(f"{obj} mismatches type {self.paramdict}"),
            )
            return obj
        aot(self.nullable, lambda: f"Can't set None to {self}")
        return obj

    def validate_spec(self, spec: "OutputSpec") -> "OutputSpec":
        if not self.nullable:
            aot(
                not spec.nullable,
                lambda: TypeError(f"{self} - {spec} are not compatible on nullable"),
            )
        aot(
            issubclass(spec.data_type, self.data_type),
            lambda: TypeError(f"{self} - {spec} are not compatible on data_type"),
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
        aot(self.nullable, lambda: f"Can't set None to {self.paramdict}")
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
        self._node_spec: Optional["_NodeSpec"] = None

    @property
    def node_spec(self) -> "_NodeSpec":
        if self._node_spec is not None:
            return self._node_spec
        raise InvalidOperationError(  # pragma: no cover
            f"node_spec is not set for {self}"
        )

    @property
    def name(self) -> str:
        return self.node_spec.name

    @property
    def parent_workflow(self) -> "WorkflowSpec":
        return self.node_spec.workflow

    def __uuid__(self) -> str:
        return to_uuid(
            self.configs,
            self.inputs,
            self.outputs,
            get_full_type_path(self.func),
            self.metadata,
            self.deterministic,
            self.lazy,
            self._node_spec,
        )

    def to_json(self, indent: bool = False) -> str:
        if not indent:
            return json.dumps(self.jsondict, separators=(",", ":"))
        else:
            return json.dumps(self.jsondict, indent=4)

    @property
    def jsondict(self) -> ParamDict:
        res = ParamDict(
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
        if self._node_spec is not None:
            res["node_spec"] = self.node_spec.jsondict
        return res

    def _validate_dependency(self):
        if set(self.node_spec.dependency.keys()) != set(self.inputs.keys()):
            raise DependencyNotDefinedError(
                self.name + " input",
                self.inputs.keys(),
                self.node_spec.dependency.keys(),
            )
        for k, v in self.node_spec.dependency.items():
            t = v.split(".", 1)
            if len(t) == 1:
                aot(
                    t[0] in self.parent_workflow.inputs,
                    lambda: f"{t[0]} is not an input of the workflow",
                )
                self.inputs[k].validate_spec(self.parent_workflow.inputs[t[0]])
            else:  # len(t) == 2
                aot(
                    t[0] != self.name,
                    lambda: f"{v} tries to connect to self node {self.name}",
                )
                task = self.parent_workflow.tasks[t[0]]
                aot(
                    t[1] in task.outputs,
                    lambda: f"{t[1]} is not an output of node {task.name}",
                )
                self.inputs[k].validate_spec(task.outputs[t[1]])

    def _validate_config(self):
        for k, v in self.node_spec.config.items():
            self.configs[k].validate_value(v)
        defined = set(self.node_spec.config.keys())
        for k, t in self.node_spec.config_dependency.items():
            aot(
                k not in defined,
                lambda: f"can't redefine config {k} in node {self.name}",
            )
            defined.add(k)
            aot(
                t in self.parent_workflow.configs,
                lambda: f"{t} is not a config of the workflow",
            )
            self.configs[k].validate_spec(self.parent_workflow.configs[t])
        for k in set(self.configs.keys()).difference(defined):
            aot(
                not self.configs[k].required,
                lambda: f"config {k} in node {self.name} is required but not defined",
            )

    def _parse_spec(self, obj: Any, to_type: Type[T]) -> T:
        if isinstance(obj, to_type):
            return obj
        if isinstance(obj, str):
            obj = json.loads(obj)
        aot(isinstance(obj, dict), lambda: f"{obj} is not dict")
        return to_type(**obj)

    def _parse_spec_collection(
        self, obj: Any, to_type: Type[T]
    ) -> IndexedOrderedDict[str, T]:
        res: IndexedOrderedDict[str, T] = IndexedOrderedDict()
        if obj is None:
            return res
        aot(isinstance(obj, List), "Spec collection must be a list")
        for v in obj:
            s = self._parse_spec(v, to_type)
            aot(s.name not in res, KeyError(f"Duplicated key {s.name}"))
            res[s.name] = s
        return res


class WorkflowSpec(TaskSpec):
    def __init__(
        self,
        configs: Any = None,
        inputs: Any = None,
        outputs: Any = None,
        metadata=None,
        deterministic: bool = True,
        lazy: bool = True,
        tasks: Optional[List[Any]] = None,
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
        self.tasks: IndexedOrderedDict[str, TaskSpec] = {}
        self.internal_dependency: Dict[str, str] = {}
        if tasks is not None:
            for t in tasks:
                self._append_task(to_taskspec(t, self))
        if internal_dependency is not None:
            for k, v in internal_dependency.items():
                self.link(k, v)

    def __uuid__(self) -> str:
        return to_uuid(super().__uuid__(), self.tasks, self.internal_dependency)

    def add_task(
        self,
        name: str,
        task: Any,
        dependency: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
        config_dependency: Optional[Dict[str, str]] = None,
    ) -> TaskSpec:
        _t = to_taskspec(task)
        aot(_t._node_spec is None, "node_spec must not be set")
        _t._node_spec = _NodeSpec(self, name, dependency, config, config_dependency)
        return self._append_task(_t)

    def link(self, output: str, to_expr: str):
        try:
            aot(
                output in self.outputs,
                lambda: f"{output} is not an output of the workflow",
            )
            aot(
                output not in self.internal_dependency,
                lambda: f"{output} is already defined",
            )
            t = to_expr.split(".", 1)
            if len(t) == 1:
                aot(
                    t[0] in self.inputs,
                    lambda: f"{t[0]} is not an input of the workflow",
                )
                self.outputs[output].validate_spec(self.inputs[t[0]])
            else:  # len(t) == 2
                node = self.tasks[t[0]]
                aot(t[1] in node.outputs, lambda: f"{t[1]} is not an output of {node}")
                self.outputs[output].validate_spec(node.outputs[t[1]])
            self.internal_dependency[output] = to_expr
        except Exception as e:
            raise DependencyDefinitionError(e)

    @property
    def jsondict(self) -> ParamDict:
        d = super().jsondict
        d["tasks"] = [x.jsondict for x in self.tasks.values()]
        d["internal_dependency"] = self.internal_dependency
        del d["func"]
        return d

    def validate(self) -> None:
        if set(self.outputs.keys()) != set(self.internal_dependency.keys()):
            raise DependencyNotDefinedError(
                "workflow output", self.outputs.keys(), self.internal_dependency.keys()
            )

    def _append_task(self, task: TaskSpec) -> TaskSpec:
        name = task.name
        assert_triad_var_name(name)
        aot(
            name not in self.tasks,
            lambda: KeyError(f"{name} already exists in workflow"),
        )
        aot(
            task.parent_workflow is self,
            lambda: InvalidOperationError(f"{task} has mismatching node_spec"),
        )
        try:
            task._validate_config()
            task._validate_dependency()
        except DependencyDefinitionError:
            raise
        except Exception as e:
            raise DependencyDefinitionError(e)
        self.tasks[name] = task
        return task


def to_taskspec(
    obj: Any, parent_workflow_spec: Optional[WorkflowSpec] = None
) -> TaskSpec:
    assert_arg_not_none(obj, "obj")
    if isinstance(obj, str):
        return to_taskspec(json.loads(obj))
    if isinstance(obj, TaskSpec):
        return obj
    if isinstance(obj, Dict):
        d: Dict[str, Any] = dict(obj)
        node_spec: Optional[_NodeSpec] = None
        if "node_spec" in d:
            aot(
                parent_workflow_spec is not None,
                lambda: InvalidOperationError("parent workflow must be set"),
            )
            node_spec = _NodeSpec(
                workflow=parent_workflow_spec, **d["node_spec"]  # type: ignore
            )
            del d["node_spec"]
        if "tasks" in d:
            ts: TaskSpec = WorkflowSpec(**d)
        else:
            ts = TaskSpec(**d)
        if node_spec is not None:
            ts._node_spec = node_spec
        return ts
    raise TypeError(f"can't convert {obj} to TaskSpec")  # pragma: no cover


def _no_op(self, *args, **kwargs):  # pragma: no cover
    pass


class _NodeSpec(object):
    def __init__(
        self,
        workflow: "WorkflowSpec",
        name: str,
        dependency: Optional[Dict[str, str]],
        config: Optional[Dict[str, Any]],
        config_dependency: Optional[Dict[str, str]],
    ):
        self.workflow = workflow
        self.name = name
        self.dependency = dependency or {}
        self.config = config or {}
        self.config_dependency = config_dependency or {}

    def __uuid__(self) -> str:
        # self.name is not part of uuid because same uuid in spec means
        # with the same dependency, same config, and same func, they should
        # have the same result, and uuid is the identifier of that result
        return to_uuid(self.dependency, self.config, self.config_dependency)

    @property
    def jsondict(self) -> ParamDict:
        return dict(
            name=self.name,
            dependency=self.dependency,
            config=self.config,
            config_dependency=self.config_dependency,
        )
