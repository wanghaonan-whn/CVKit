# pathkit

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

## Error Rules

- query methods such as `exists()` return boolean values
- filesystem-dependent methods such as `stat()` and `samefile()` raise exceptions when paths do not exist
- `relative_to()` raises `ValueError` when the target path is not under the base path
- scanning methods default to `on_permission_error="skip"`
