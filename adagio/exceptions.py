from typing import Iterable


class AdagioError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class WorkflowBug(AdagioError):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CompileError(AdagioError):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DependencyDefinitionError(CompileError):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DependencyNotDefinedError(DependencyDefinitionError):
    def __init__(self, name: str, expected: Iterable[str], actual: Iterable[str]):
        s = f"expected {sorted(expected)}, actual {sorted(actual)}"
        super().__init__(f"Task {name} dependencies not well defined: " + s)


class WorkflowRuntimeError(AdagioError):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AbortedError(WorkflowRuntimeError):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SkippedError(WorkflowRuntimeError):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def assert_on_compile(bool_expr: bool, msg: str) -> None:
    if not bool_expr:
        raise CompileError(msg)
