from typing import TYPE_CHECKING

__all__ = ["VOCAnnotationUtils", "XMLDocument"]

if TYPE_CHECKING:
    from pathkit.process.hbb.voc import VOCAnnotationUtils
    from pathkit.process.io.xml import XMLDocument


def __getattr__(name: str):
    if name == "AnnotationUtils":
        from pathkit.process.hbb.voc import VOCAnnotationUtils

        return VOCAnnotationUtils
    if name == "XMLDocument":
        from pathkit.process.io.xml import XMLDocument

        return XMLDocument
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
