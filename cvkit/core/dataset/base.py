from pathlib import Path


class Dataset:
    IMAGE_EXTS = (".jpg", ".jpeg", ".png")

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.label_dir = self.data_dir / "labels"

    def find_image(self, stem: str) -> Path | None:
        return next(
            (
                image_file
                for image_file in self.image_dir.glob(f"{stem}.*")
                if image_file.suffix.lower() in self.IMAGE_EXTS
            ),
            None,
        )