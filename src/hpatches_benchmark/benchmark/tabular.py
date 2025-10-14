from abc import ABC, abstractmethod
from typing import Any

__all__ = ['Tabular']

class Tabular(ABC):
    """
    Interface for classes which can be displayed in a table.
    """
    @property
    @abstractmethod
    def table_headings(self) -> list[str]:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def table_body(self) -> list[Any]:
        raise NotImplementedError()
