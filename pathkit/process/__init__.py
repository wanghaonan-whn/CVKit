from typing import TYPE_CHECKING

__all__ = ["AnnotationUtils", "XMLDocument"]

if TYPE_CHECKING:
    from pathkit.process.annotation import AnnotationUtils
    from pathkit.process.xmldocument import XMLDocument


def __getattr__(name: str):
    if name == "AnnotationUtils":
        from pathkit.process.annotation import AnnotationUtils

        return AnnotationUtils
    if name == "XMLDocument":
        from pathkit.process.xmldocument import XMLDocument

        return XMLDocument
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
