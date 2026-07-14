# MV-GoTrack offline-capture patch

Apply this patch from the root of the approved private MV-GoTrack checkout.
It was generated against commit `a9f033734c0bdf2d191265d22ea732a914c861f6` and changes only:

- `archive/run_multiview_gotrack_anchor_online_multi_object.py`: sequential camera micro-batches while keeping all camera observations for one triangulation and rigid fit.
- `utils/renderer_nvdiffrast.py`: require CUDA nvdiffrast instead of silently falling back to the much slower Pyrender renderer.

It intentionally does **not** include the local `external/bop_toolkit` or `external/dinov2` submodule pointer changes.

```bash
cd /path/to/MV-GoTrack
git rev-parse HEAD  # must be a9f033734c0bdf2d191265d22ea732a914c861f6
git apply --check /path/to/autodex/patches/MV-GoTrack-offline-capture.patch
git apply /path/to/autodex/patches/MV-GoTrack-offline-capture.patch
```

After applying, run the AutoDex preflight in the `gotrack` environment. Do not apply the legacy `MV-GoTrack-renderer-fix.patch`; it targets an older renderer revision.
