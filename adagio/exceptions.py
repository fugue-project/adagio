from typing import Iterable


class AdagioError(Exception):
    pass


class WorkflowBug(AdagioError):
    pass


class CompileError(AdagioError):
    pass


class DependencyDefinitionError(CompileError):
    pass


class DependencyNotDefinedError(DependencyDefinitionError):
    def __init__(self, name: str, expected: Iterable[str], actual: Iterable[str]):
        s = f"expected {sorted(expected)}, actual {sorted(actual)}"
        super().__init__(f"Task {name} dependencies not well defined: " + s)


class WorkflowRuntimeError(AdagioError):
    pass


class AbortedError(WorkflowRuntimeError):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SkippedError(WorkflowRuntimeError):
    pass
