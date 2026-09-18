# MV-GoTrack offline-capture patch

Apply this patch from the root of the approved private MV-GoTrack checkout.
It was generated against commit `a9f033734c0bdf2d191265d22ea732a914c861f6` and contains only the following offline-pipeline changes:

- `archive/run_multiview_gotrack_anchor_online_multi_object.py`: sequential camera micro-batches while keeping all camera observations for one triangulation and rigid fit.
- `utils/renderer_nvdiffrast.py`: require CUDA nvdiffrast instead of silently falling back to the much slower Pyrender renderer.
- `scripts/onboard_custom_mesh_for_foundpose.py`: preserve UV texture maps in an embedded GLB for FoundPose template rendering while retaining the conventional PLY artifact for BOP metadata and compatibility.

## UV-textured FoundPose onboarding

The original onboarding path always rendered templates from a PLY. Because PLY does not carry an OBJ/MTL UV texture map, onboarding first sampled the texture at mesh vertices and stored the result as vertex colors. That approximation is adequate for dense scans, but sparse meshes can lose small labels, logos, and other rotation-breaking appearance evidence.

With this patch, onboarding behaves as follows:

- An input with a valid UV map and material texture is exported as an embedded-texture `obj_XXXXXX.glb`; template RGB/depth/mask rendering uses this GLB.
- A vertex-coloured `obj_XXXXXX.ply` is still written for BOP-compatible geometry and metadata consumers.
- Meshes without a usable UV texture continue to render from PLY, so dense vertex-coloured inputs keep the previous behavior.
- Reusing an existing output chooses the GLB only when it is present; otherwise it falls back to the PLY.
- `summary.json` records `render_model_path`, making the actual template-render input auditable.

This improves the fidelity of the template RGB evidence, but does not guarantee a different final pose. FoundPose uses template appearance to establish DINO correspondences and generate PnP hypotheses; the offline multiview wrapper subsequently performs global silhouette scoring/refinement and uses DINO reprojection only to rerank near-tied mask hypotheses. In a local `instant_rice` A/B study, the UV templates were sharper but the final poses changed by less than `0.6°` and `0.2 mm`, because its dense 253k-vertex PLY already preserved the package texture reasonably well and the silhouette stage converged to the same orientation. The change is expected to matter most for sparse UV meshes such as labelled cans, not for every textured object.

It intentionally does **not** include the local `external/bop_toolkit` or `external/dinov2` submodule pointer changes.

```bash
cd /path/to/MV-GoTrack
git rev-parse HEAD  # must be a9f033734c0bdf2d191265d22ea732a914c861f6
git apply --check /path/to/autodex/patches/MV-GoTrack-offline-capture.patch
git apply /path/to/autodex/patches/MV-GoTrack-offline-capture.patch
```

After applying, run the AutoDex preflight in the `gotrack` environment. Do not apply the legacy `MV-GoTrack-renderer-fix.patch`; it targets an older renderer revision.
