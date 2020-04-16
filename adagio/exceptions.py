class AdagioError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CompileError(AdagioError):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SkippedError(AdagioError):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def assert_on_compile(bool_expr: bool, msg: str) -> None:
    if not bool_expr:
        raise CompileError(msg)
