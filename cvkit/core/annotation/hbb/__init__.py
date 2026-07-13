from typing import TYPE_CHECKING

__all__ = ["VOCAnnotationUtils", "XMLDocument"]

if TYPE_CHECKING:
    from cvkit.core.annotation.hbb.voc import VOCAnnotationUtils
    from cvkit.core.annotation.io import XMLDocument


def __getattr__(name: str):
    if name == "AnnotationUtils":
        from cvkit.core.annotation.hbb.voc import VOCAnnotationUtils

        return VOCAnnotationUtils
    if name == "XMLDocument":
        from cvkit.core.annotation.io import XMLDocument

        return XMLDocument
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
