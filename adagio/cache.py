from abc import ABC, abstractmethod
from typing import Any, Tuple


class WorkflowResultCache(ABC):
    """Interface for cachine workflow task outputs. This cache is
    normally for cross execution retrieval.

    The implementation should be thread safe, and all methods should catch all
    exceptions and not raise.
    """

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set `key` with `value`

        :param key: uuid string
        :param value: any value
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def skip(self, key: str) -> None:
        """Skip `key`

        :param key: uuid string
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def get(self, key: str) -> Tuple[bool, bool, Any]:
        """Try to get value for `key`

        :param key: uuid string
        :return: <hasvalue>, <skipped>, <value>
        """
        raise NotImplementedError  # pragma: no cover


class NoOpCache(WorkflowResultCache):
    """Dummy WorkflowResultCache doing nothing
    """

    def set(self, key: str, value: Any) -> None:
        """Set `key` with `value`

        :param key: uuid string
        :param value: any value
        """
        return

    def skip(self, key: str) -> None:
        """Skip `key`

        :param key: uuid string
        """
        return

    def get(self, key: str) -> Tuple[bool, bool, Any]:
        """Try to get value for `key`

        :param key: uuid string
        :return: <hasvalue>, <skipped>, <value>
        """
        return False, False, None


NO_OP_CACHE = NoOpCache()
