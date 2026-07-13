from typing import TYPE_CHECKING

__all__ = ["VOCAnnotationUtils", "XMLDocument"]

if TYPE_CHECKING:
    from core.annotation.hbb.voc import VOCAnnotationUtils
    from core.annotation.io.xml import XMLDocument


def __getattr__(name: str):
    if name == "AnnotationUtils":
        from core.annotation.hbb.voc import VOCAnnotationUtils

        return VOCAnnotationUtils
    if name == "XMLDocument":
        from core.annotation.io.xml import XMLDocument

        return XMLDocument
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
