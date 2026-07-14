# Offline capture: FoundPose + GoTrack 설치 매뉴얼

이 문서는 capture archive 한 개에 대해 아래의 **offline** 파이프라인을 새 Linux PC에서
재현하기 위한 기준 문서다. 실시간 daemon, planner, FoundationStereo,
FoundationPose의 기존 `src/process/pose.py`는 범위 밖이다.

```text
videos/ + cam_param/
  -> undistorted_video/                 (gotrack env)
  -> frame-0 SAM3 mask                  (sam3 env)
  -> FoundPose initial world pose        (gotrack env)
  -> mask-free GoTrack trajectory        (gotrack env)
  -> optional reprojection grid video    (gotrack env)
```

출력은 capture의 기존 `videos/`, `cam_param/` 또는 과거 pipeline 결과를 바꾸지
않는다. 각 단계는 새 output directory만 만든다.

## 먼저 알아둘 현재 상태

이 저장소 전체를 그대로 clone한 것만으로는 위 파이프라인을 재현할 수 **없다**.
이것은 설치 순서 문제가 아니라 version-control 경계의 문제다.

1. `autodex/perception/thirdparty/MV-GoTrack/`는 top-level `.gitignore`에 의해
   추적되지 않는다. 현재 사용 중인 수정(카메라 micro-batch 및 nvdiffrast renderer
   수정)은 그 별도 working tree에만 있다.
2. `gotrack_checkpoint.pt`(약 1.5 GB)와 FoundPose asset cache는 의도적으로 git에
   넣지 않는다.
3. 기존 `scripts/setup_gotrack_cu128_capture.sh`는 과거 운영 PC용이다. `~/AutoDex`,
   `~/anaconda3`, `gotrack_cu128`를 고정하고 `git reset --hard origin/main`을
   수행하므로, 새 포크/새 PC 설치에 사용하면 안 된다.
4. MV-GoTrack의 원본 `environment.yml`(PyTorch 2.0/CUDA 11.7)과 실제 검증한 환경
   (Python 3.10, PyTorch 2.11.0+cu128, xformers 0.0.35)는 서로 다르다. 둘을 섞어
   설치하면 재현성이 없다.

따라서 아래 절차의 출발점은 **top-level AutoDex 포크와 MV-GoTrack 포크를 각각
commit으로 고정한 뒤**이다. 설치 전에 다음 두 값을 release note 또는 manifest에
기록한다.

```bash
git -C /path/to/autodex rev-parse HEAD
git -C /path/to/autodex/autodex/perception/thirdparty/MV-GoTrack rev-parse HEAD
```

권장하는 최종 구조는 MV-GoTrack을 top-level repo의 git submodule로 등록하거나,
최소한 `thirdparty/MV-GoTrack.lock`에 fork URL과 commit SHA를 추적하는 것이다.
그 전까지는 "동일 pipeline"의 정확한 의미를 보장할 수 없다.

## 1. 시스템 전제

- Ubuntu Linux, NVIDIA driver가 설치되어 있고 `nvidia-smi`가 동작해야 한다.
- Conda 또는 Miniconda가 있어야 한다.
- CUDA 12.8 wheel을 사용한 현재 검증 조합은 CUDA-capable GPU가 필요하다.
  GPU driver 호환성은 먼저 `nvidia-smi`로 확인한다.
- `git`, `git-lfs`, `ffmpeg`를 설치한다. EGL headless renderer가 필요하므로
  일반적인 NVIDIA OpenGL/EGL driver도 필요하다.

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs ffmpeg
git lfs install
nvidia-smi
```

`apt`가 404를 내면 먼저 `sudo apt-get update`를 다시 실행한다. Blender는 이
offline tracking pipeline의 필수 의존성이 아니다.

## 2. 코드 checkout

`<AUTODEX_FORK_URL>`, `<AUTODEX_COMMIT>`, `<MV_GOTRACK_FORK_URL>`,
`<MV_GOTRACK_COMMIT>`은 실험에 사용한 commit으로 치환한다. 아직 local-only인
MV-GoTrack 수정은 먼저 그 fork에 commit/push해야 한다.

```bash
export REPO="$HOME/src/autodex"
git clone --recurse-submodules <AUTODEX_FORK_URL> "$REPO"
cd "$REPO"
git checkout <AUTODEX_COMMIT>
git submodule update --init --recursive

mkdir -p autodex/perception/thirdparty
git clone --recurse-submodules <MV_GOTRACK_FORK_URL> \
  autodex/perception/thirdparty/MV-GoTrack
git -C autodex/perception/thirdparty/MV-GoTrack checkout <MV_GOTRACK_COMMIT>
git -C autodex/perception/thirdparty/MV-GoTrack submodule update --init --recursive
```

중요: 설치 스크립트가 clone한 repo에 `git reset --hard`를 실행해서는 안 된다.
포크의 branch tip 대신 명시적 commit을 checkout한다.

private fork에 직접 commit할 권한이 없는 경우, 승인된 base commit
`a9f033734c0bdf2d191265d22ea732a914c861f6`을 checkout한 뒤 public AutoDex에 포함된
patch를 적용한다. 이 patch는 GoTrack runner의 camera micro-batch와 CUDA-only
nvdiffrast renderer만 바꾼다.

```bash
cd "$REPO/autodex/perception/thirdparty/MV-GoTrack"
git checkout a9f033734c0bdf2d191265d22ea732a914c861f6
git apply --check "$REPO/patches/MV-GoTrack-offline-capture.patch"
git apply "$REPO/patches/MV-GoTrack-offline-capture.patch"
```

상세한 patch 범위와 재적용 방법은
[MV-GoTrack-offline-capture.md](../patches/MV-GoTrack-offline-capture.md)를 참고한다.
기존 `MV-GoTrack-renderer-fix.patch`는 이전 renderer revision용이므로 이 offline
pipeline에 적용하면 안 된다.

## 3. 모델 파일

GoTrack checkpoint는 MV-GoTrack root에 정확히 이 이름으로 놓아야 한다.
현재 검증한 파일의 SHA-256은 아래와 같다.

```bash
export GOTRACK_DIR="$REPO/autodex/perception/thirdparty/MV-GoTrack"
rsync -avP <ASSET_HOST>:/path/to/gotrack_checkpoint.pt \
  "$GOTRACK_DIR/gotrack_checkpoint.pt"
sha256sum "$GOTRACK_DIR/gotrack_checkpoint.pt"
# f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086
```

SAM3 weight는 조직의 승인된 Hugging Face cache 또는 shared weight store에서
준비한다. 접근 token이 필요한 모델이면 새 PC에서도 해당 계정으로 login해야 한다.
FoundPose `assets/object_repre/.../repre.pth`는 object mesh와 onboarding 옵션에
종속된다. 기본 위치는 `~/shared_data/mesh_blender/<object>/foundpose_assets/`이며,
모든 PC가 같은 NAS를 mount하면 자동으로 공유된다. 먼저 아래 전처리 CLI로 object당
한 번 생성한다(현재 798-view onboarding은 object당 약 20–21분이었다).

```bash
conda run --no-capture-output -n gotrack python -u \
  src/process/onboard_foundpose_mesh.py \
  --object-name <object> \
  --reference-intrinsics-json /path/to/one/capture/cam_param/intrinsics.json
```

object별 output lock이 있으므로 여러 PC는 **서로 다른 object**를 동시에 처리할 수
있다. 같은 object를 동시에 실행하지 않는다.

### 여러 PC에 object 전처리 분배

`inspire_dftp`처럼 camera calibration family가 capture campaign마다 다른 경우에는
mesh 전역 cache를 공유하지 않는다. campaign의 각 `<object>/0/cam_param/intrinsics.json`을
reference로 쓰고, cache는 episode 폴더들과 나란한
`<campaign>/<object>/foundpose_assets/`에 저장한다. controller를 이 PC의 tmux 안에서
실행한다. worker는 `local` 또는 `user@ip`로 지정하며, IP는 반드시 실제 장비 주소로
바꾼다. 각 worker는 한 번에 object 하나만 처리한다. 개별 SSH/onboarding 실패는 다른
object를 멈추지 않고 최대 3회 재시도된다. 완료 representation은 자동 skip되고
상태/로그는 NAS mesh root에 저장되므로 같은 `--run-name`으로 재실행하면 이어서 진행한다.

```bash
tmux new -s foundpose-onboard
cd "$REPO"
python -u scripts/distribute_foundpose_onboard.py \
  --scenario-root-rel capture/eccv2026/inspire_dftp \
  --workers local capture13@192.168.0.<IP13> capture14@192.168.0.<IP14> \
            capture15@192.168.0.<IP15> capture18@192.168.0.<IP18> \
  --run-name onboarding_batch_01
```

다른 터미널에서는 SSH polling 없이 shared state만 읽어 진행 상황을 볼 수 있다.

```bash
watch -n 5 python scripts/foundpose_onboard_status.py \
  --state-dir ~/shared_data/mesh_blender/.foundpose_onboard_runs/onboarding_batch_01
```

## 4. Conda 환경

### 자동 환경 설치

SAM3와 GoTrack/FoundPose 환경의 검증된 직접 runtime package 목록은
`requirements/offline_capture/`에 version으로 고정했다. private MV-GoTrack checkout과
승인된 patch를 준비한 **뒤** 아래만 실행한다. 이 script는 git clone, `git apply`,
`git reset`을 절대 수행하지 않는다.

```bash
cd "$REPO"
bash scripts/setup_offline_capture_env.sh \
  --gotrack-dir "$REPO/autodex/perception/thirdparty/MV-GoTrack"
```

`--skip-gotrack` 또는 `--skip-sam3`로 한 환경만 설치할 수 있다. 설치 후 script는
CUDA, nvdiffrast context, checkpoint SHA, private patch의 micro-batch marker 및 Python
imports를 검사한다. 이미 조직에서 관리하는 호환 SAM3 environment가 설치되어 있으면
`--skip-sam3`를 사용해도 된다. 일반 preflight는 외부 SAM3 checkout도 허용하며,
setup script로 레포 내부 SAM3를 설치한 경우에만 그 경로를 엄격히 검사한다. 수동 검사는
다음과 같다.

MV-GoTrack의 BOP/DINOv2 metadata에는 원본 Torch 2.0/CUDA 11.7 pin이 남아 있다.
설치 script는 이들의 source만 `--no-deps`로 설치하므로, `gotrack`의 Torch CUDA 12.8
및 NumPy 2.x를 절대 downgrade하지 않는다.

```bash
conda run -n gotrack python scripts/check_offline_capture_setup.py --component gotrack
conda run -n sam3 python scripts/check_offline_capture_setup.py --component sam3
```

Blackwell (SM120) GPU라면 위 설치 뒤 `scripts/setup_gotrack_blackwell_xformers.sh`를
쓰되, `ENV_NAME=gotrack`, `CONDA_DIR=<실제 conda root>`를 명시한다. 이 script도
현재는 build helper이며 코드/asset 설치 스크립트가 아니다.

## 5. 설치 검증

모델 inference 전에 이 검증을 통과해야 한다. `MPLCONFIGDIR`는 multiprocessing
시 matplotlib cache 권한 경고를 피한다.

```bash
export MPLCONFIGDIR=/tmp/matplotlib-gotrack
mkdir -p "$MPLCONFIGDIR"

conda run -n gotrack python - <<PY
import sys, torch
sys.path.insert(0, "$GOTRACK_DIR")
import nvdiffrast.torch as dr
from utils import renderer_nvdiffrast
assert torch.cuda.is_available(), "CUDA is unavailable"
print("torch:", torch.__version__, "gpu:", torch.cuda.get_device_name(0))
print("nvdiffrast context:", dr.RasterizeCudaContext())
PY

conda run -n sam3 python - <<PY
import cv2, sam3, torch
assert torch.cuda.is_available(), "CUDA is unavailable"
print("SAM3:", sam3.__version__, "OpenCV:", cv2.__version__)
PY

cd "$REPO"
conda run -n gotrack python src/process/gotrack_capture.py --help
conda run -n gotrack python src/process/foundpose_init_capture.py --help
conda run -n sam3 python src/process/mask.py --help
```

## 6. capture 하나를 실행하는 순서

아래의 모든 `<...>`를 실제 값으로 바꾼다. calibration은 `cam_param/intrinsics.json`
의 `original_intrinsics`, `intrinsics_undistort`, `dist_params`와
`extrinsics.json`을 요구한다.

```bash
export CAPTURE=/path/to/capture/object/episode
export MESH=/path/to/mesh.obj
export OBJ=object_name
export FRAME_OUT="$CAPTURE/foundpose_frame_000000_$OBJ"
export INIT_OUT="$CAPTURE/foundpose_$OBJ"
export TRACK_OUT="$CAPTURE/gotrack_${OBJ}_12cam"

cd "$REPO"
conda run --no-capture-output -n gotrack python -u \
  src/process/undistort_capture_videos.py --capture-dir "$CAPTURE"

conda run --no-capture-output -n sam3 python -u src/process/mask.py \
  --capture_dir "$CAPTURE" --frame-index 0 --prompt "${OBJ//_/ }" \
  --video-dir "$CAPTURE/undistorted_video" --frame-output-dir "$FRAME_OUT"

conda run --no-capture-output -n gotrack python -u \
  src/process/foundpose_init_capture.py \
  --capture-dir "$CAPTURE" --frame-dir "$FRAME_OUT" \
  --mesh "$MESH" --object-name "$OBJ" --output-dir "$INIT_OUT"

conda run --no-capture-output -n gotrack python -u \
  src/process/gotrack_capture.py \
  --capture-dir "$CAPTURE" --video-dir "$CAPTURE/undistorted_video" \
  --mesh "$MESH" --init-pose "$INIT_OUT/init_pose_world.npy" \
  --object-name "$OBJ" --num-cameras 12 --max-frames -1 \
  --output-dir "$TRACK_OUT"
```

GoTrack이 OOM이면 `--num-cameras 8`로 새로운 `--output-dir`에 재실행하거나,
선택 camera 수보다 작은 `--camera-micro-batch-size`를 쓴다. 예를 들어 21개
camera 전체 관측을 유지하면서 GPU refinement만 8개씩 처리하려면
`--num-cameras 21 --camera-micro-batch-size 8`이다. 이 옵션이 동작하려면 해당
변경을 포함한 **MV-GoTrack fork commit**이어야 한다.

## 7. 결과와 보존 규칙

```text
<CAPTURE>/undistorted_video/                         # generated, raw videos/ 불변
<FRAME_OUT>/images/*.png, masks/*.png               # SAM3 first-frame inputs
<INIT_OUT>/init_pose_world.npy                      # FoundPose 6D initial world pose
~/shared_data/mesh_blender/<OBJ>/foundpose_assets/object_repre/v1/<OBJ>/1/repre.pth
                                                    # reusable per-object cache
<TRACK_OUT>/gotrack_output/<OBJ>/world_pose_records.json  # per-frame 6D world poses
```

`gotrack_capture.py`는 이미 존재하는 output directory를 의도적으로 거부한다.
재실행할 때 기존 결과를 지우지 말고 새로운 이름을 사용한다. FoundPose asset cache를
다른 위치의 cache를 써야 할 때만 `--assets-root`를 명시한다.

## 커밋 전 정리 체크리스트

- [ ] top-level의 `src/process/gotrack_capture.py`, `foundpose_init_capture.py`,
  `undistort_capture_videos.py`, grid renderer와 `mask.py` 변경을 commit한다.
- [ ] MV-GoTrack의 `archive/run_multiview_gotrack_anchor_online_multi_object.py`
  micro-batch 변경과 `utils/renderer_nvdiffrast.py` 변경을 별도 fork에 commit한다.
- [ ] `utils/renderer_nvdiffrast.py.orig` 및 `.rej`는 patch 충돌 잔재이므로
  커밋/배포하지 말고 renderer 최종본을 테스트한 뒤 제거한다.
- [ ] `patches/MV-GoTrack-renderer-fix.patch`는 현재 local renderer diff와 일치하지
  않는다. fork commit 방식으로 대체하거나, 최종 renderer commit에서 patch를 다시
  생성한다.
- [ ] 실제 새 PC에서 1-frame smoke test와 짧은 GoTrack run을 한 번 수행하고,
  AutoDex SHA, MV-GoTrack SHA, checkpoint SHA, `conda list --explicit`을 run manifest에
  기록한다.

위 체크리스트가 끝나면 이 문서를 바탕으로 `scripts/bootstrap_offline_capture.sh`와
preflight checker를 추가하는 것이 안전하다. 그 전에는 기존 setup script를 자동화의
기준으로 사용하지 않는다.
