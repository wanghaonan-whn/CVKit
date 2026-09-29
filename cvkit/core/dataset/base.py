from pathlib import Path


class Dataset:
    def __init__(self, data_dir: str | Path, image_ext: str = ".jpg") -> None:
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.label_dir = self.data_dir / "labels"
        self.image_ext = image_ext.lower()

    def find_image(self, stem: str) -> Path | None:
        image_file = self.image_dir / f"{stem}{self.image_ext}"
        return image_file if image_file.exists() else None
