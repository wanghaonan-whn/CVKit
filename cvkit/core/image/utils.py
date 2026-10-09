import hashlib
from pathlib import Path


class ImageUtils:
    @staticmethod
    def calculate_file_hash(file_path: str | Path) -> str:
        file_path = Path(file_path)
        hasher = hashlib.sha256()

        with file_path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                hasher.update(chunk)

        return hasher.hexdigest()