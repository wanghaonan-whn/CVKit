# cvkit

A lightweight computer-vision utility toolkit.

## Modules

- `cvkit.core.annotation`: annotation IO and format helpers
- `cvkit.core.dataset`: dataset utilities
- `cvkit.augment`: image augmentation utilities
- `cvkit.ocr`: OCR-related utilities

## Build Wheel

Install build tool:

```bash
python -m pip install build
```

Build wheel package in project root:

```bash
python -m build --wheel
```

After build completes, the wheel file will be generated in `dist/`.
