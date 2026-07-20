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
  -> optional sparse debug sheet / Viser inspection (gotrack env)
  -> optional reprojection grid video    (gotrack env; exceptional cases only)
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

따라서 아래 절차의 출발점은 **top-level object_tracking 포크와 MV-GoTrack checkout을 각각
commit으로 고정한 뒤**이다. 설치 전에 다음 두 값을 release note 또는 manifest에
기록한다.

```bash
git -C "$HOME/object_tracking" rev-parse HEAD
git -C "$HOME/object_tracking/autodex/perception/thirdparty/MV-GoTrack" rev-parse HEAD
```

MV-GoTrack은 현재 top-level과 **별도 git 경계**이며 top-level `.gitignore` 대상이다.
필요한 private 변경은 [MV-GoTrack-offline-capture.patch](../patches/MV-GoTrack-offline-capture.patch)로
관리한다. top-level commit에 MV-GoTrack 변경이 자동 포함된다고 가정하면 안 된다.

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

현재 public repo와 검증된 private base를 쓰는 설치 예시는 아래와 같다. top-level의
submodule은 정상 절차대로 초기화하되, **MV-GoTrack 자체에는 절대로**
`--recurse-submodules`를 붙이거나 `git submodule update --init --recursive`를 실행하지 않는다.
MV-GoTrack이 가리키는 과거 BOP submodule SHA `481265...`는 upstream에서 사라져 있다.

```bash
export REPO="$HOME/object_tracking"
git clone --recurse-submodules https://github.com/chas0612/object_tracking.git "$REPO"
cd "$REPO"
git checkout tracking-session-progress
git submodule update --init --recursive

export GOTRACK_DIR="$REPO/autodex/perception/thirdparty/MV-GoTrack"
mkdir -p "$(dirname "$GOTRACK_DIR")"
git clone https://github.com/gunhee1113/MV-GoTrack "$GOTRACK_DIR"
git -C "$GOTRACK_DIR" checkout a9f033734c0bdf2d191265d22ea732a914c861f6

git clone https://github.com/thodan/bop_toolkit.git "$GOTRACK_DIR/external/bop_toolkit"
git -C "$GOTRACK_DIR/external/bop_toolkit" checkout cea62d651c7e395b2e1962b9749e4e89693c6ac4
git clone https://github.com/facebookresearch/dinov2.git "$GOTRACK_DIR/external/dinov2"
git -C "$GOTRACK_DIR/external/dinov2" checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
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

GoTrack checkpoint는 MV-GoTrack root에 정확히 이 이름으로 놓아야 한다. Git LFS
fetch 대신 공식 raw URL을 사용한다. 중단된 다운로드는 같은 명령을 재실행하면 이어받는다.

```bash
export GOTRACK_DIR="$REPO/autodex/perception/thirdparty/MV-GoTrack"
curl -L --fail --retry 3 --continue-at - \
  -o "$GOTRACK_DIR/gotrack_checkpoint.pt.part" \
  https://github.com/facebookresearch/gotrack/raw/refs/heads/main/gotrack_checkpoint.pt
mv "$GOTRACK_DIR/gotrack_checkpoint.pt.part" "$GOTRACK_DIR/gotrack_checkpoint.pt"
sha256sum "$GOTRACK_DIR/gotrack_checkpoint.pt"
# f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086
```

SAM3 weight는 조직의 승인된 Hugging Face cache 또는 shared weight store에서
준비한다. 접근 token이 필요한 모델이면 새 PC에서도 해당 계정으로 login해야 한다.
FoundPose `assets/object_repre/.../repre.pth`는 object mesh, onboarding 옵션 및
reference calibration에 종속된다. 일반 cache의 기본 위치는
`~/shared_data/mesh_blender/<object>/foundpose_assets/`이다. 다만
`inspire_dftp`는 campaign calibration을 보존하기 위해
`~/shared_data/capture/eccv2026/inspire_dftp/<object>/foundpose_assets/`를 사용한다.
`foundpose_init_capture.py`는 episode 부모의 campaign cache를 자동으로 우선한다.
먼저 아래 전처리 CLI로 object당 한 번 생성한다(57 viewpoints × 14 rotations = 798 templates).

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

이 연구실의 원격 SSH 포트는 `77`이며 scheduler가 이를 사용한다. 실행 전에 각 worker에
host key를 등록하고 key-based login을 확인해야 한다. scheduler는 `BatchMode=yes`로
실행하므로 비밀번호 또는 host-key 확인 prompt가 필요한 연결은 즉시 실패한다. SSAA=4는
2048 기준 내부 약 8192×8192 render라 전력·발열 부담이 크다. 재개 첫 시도는 3대 이하로
제한하고, 별도 tmux에서 GPU 온도·전력·RAM을 기록한다.

```bash
tmux new -s foundpose-onboard
cd "$REPO"
python -u scripts/distribute_foundpose_onboard.py \
  --scenario-root-rel capture/eccv2026/inspire_dftp \
  --workers local capture13@192.168.0.<IP13> capture14@192.168.0.<IP14> \
            capture15@192.168.0.<IP15> capture18@192.168.0.<IP18> \
  --run-name inspire_dftp_onboarding_05
```

다른 터미널에서는 SSH polling 없이 shared state만 읽어 진행 상황을 볼 수 있다.

```bash
watch -n 5 python scripts/foundpose_onboard_status.py \
  --state-dir ~/shared_data/mesh_blender/.foundpose_onboard_runs/inspire_dftp_onboarding_05
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

### 실패한 기존 설치 복구

이미 `dinov2` 설치와 `nvdiffrast_cuda_context=ok`가 확인된 PC에서는 BOP/DINOv2를
다시 설치하지 말고 checkpoint 다운로드와 preflight만 먼저 수행한다. 이전 설치가
NumPy 1.x로 내려갔거나 DINOv2 metadata가 Torch를 바꿨다면 다음 순서로 복구한다.

```bash
conda run --no-capture-output -n gotrack python -m pip install --force-reinstall --no-deps numpy==2.2.6
conda run --no-capture-output -n gotrack python -m pip install --no-deps -e "$GOTRACK_DIR/external/bop_toolkit"
conda run --no-capture-output -n gotrack python -m pip install --no-deps "$GOTRACK_DIR/external/dinov2"
conda run -n gotrack python scripts/check_offline_capture_setup.py \
  --component gotrack --gotrack-dir "$GOTRACK_DIR"
```

`not our ref 481265...` 오류로 external source가 불완전할 때만 기존 디렉터리를
timestamp 이름으로 보관한 뒤, 2절의 BOP/DINOv2 clone/checkout 명령을 다시 실행한다.
`pip install "$GOTRACK_DIR/external/dinov2"`처럼 `--no-deps` 없는 설치는 금지한다.

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

GoTrack은 모든 AVI의 frame count/FPS로 계산한 duration을 비교해 median에서 기본 1초 이상
벗어난 camera만 자동 제외한다. 몇 frame 누락은 허용하며, 과거 특정 serial을 하드코딩해
제외하지 않는다. `run_manifest.json`의 `video_timings`, `rejected_cameras`에서 판단 근거를
확인할 수 있다. 필요하면 `--max-video-duration-skew-sec`으로 threshold를 조절하거나 `0`으로
끄고, 정말 제외할 camera만 `--exclude-cameras`로 명시한다.

GoTrack이 OOM이면 `--num-cameras 8`로 새로운 `--output-dir`에 재실행하거나,
선택 camera 수보다 작은 `--camera-micro-batch-size`를 쓴다. 예를 들어 21개
camera 전체 관측을 유지하면서 GPU refinement만 8개씩 처리하려면
`--num-cameras 21 --camera-micro-batch-size 8`이다. 이 옵션이 동작하려면 해당
변경을 포함한 **MV-GoTrack fork commit**이어야 한다.

### 여러 PC에 다른 robot capture를 분배

`scripts/distribute_foundpose_gotrack.py`는 robot 하나의 root
`<robot-root>/<object>/<episode>`를 task로 분해한다. 각 task는 target robot의
cache를 만들지 않고, 명시적으로
`capture/eccv2026/inspire_dftp/<object>/foundpose_assets`의 `repre.pth`를 사용한다.
worker는 shared-storage atomic claim으로 episode 하나씩만 점유하며, 다음 단계를
순서대로 실행한다: undistort → SAM3 frame-0 mask → FoundPose init → GoTrack.
GoTrack 재시도는 새 `attempt_NN` output directory를 사용하므로 partial output을
덮어쓰지 않는다.

먼저 dry-run으로 mesh/cache/episode 조건을 확인하고 queue를 만든다.

```bash
cd "$REPO"
python -u scripts/distribute_foundpose_gotrack.py \
  --mode init --schedule-id hand_taeyun_right_01 \
  --target-root-rel capture/eccv2026/hand_taeyun/right \
  --num-cameras 22 --camera-micro-batch-size 11 --max-frames -1 --dry-run

python -u scripts/distribute_foundpose_gotrack.py \
  --mode init --schedule-id hand_taeyun_right_01 \
  --target-root-rel capture/eccv2026/hand_taeyun/right \
  --num-cameras 22 --camera-micro-batch-size 11 --max-frames -1
```

그 다음 controller는 remote worker를 `nohup`으로 detach한다. controller가 종료돼도
remote worker는 shared queue가 빌 때까지 계속 실행한다. SSH는 port 77과 key-based
login이 준비되어 있어야 한다.

```bash
python -u scripts/distribute_foundpose_gotrack.py \
  --mode launch --schedule-id hand_taeyun_right_01 \
  --workers capture13@192.168.0.<IP13> capture14@192.168.0.<IP14> \
            capture18@192.168.0.<IP18>

python -u scripts/distribute_foundpose_gotrack.py \
  --mode status --schedule-id hand_taeyun_right_01
```

실패 task는 기본으로 재시도하지 않는다. 원인을 log에서 확인한 뒤 worker 또는 launch에
`--retry-failed`를 추가한다. task당 기본 최대 시도 횟수는 2회다.

### 빠른 결과 검증: sparse sheet와 Viser

전체 camera와 모든 frame을 MP4 grid로 인코딩하는 것은 대규모 run의 기본 QA 방식으로
비싸다. 우선 아래 순서를 사용한다.

1. `world_pose_records.json`에 pose 누락, NaN 또는 큰 frame 간 jump가 없는지 수치적으로 확인한다.
2. 대표 camera와 시작/중간/끝 frame만 담은 **한 장의 reprojection contact sheet**를 만든다.
3. sheet에서 의심스러운 episode만 Viser로 자세히 보고, 필요한 짧은 구간에만 grid video를
   만든다.

`render_gotrack_debug_sheet.py`는 기본 여섯 camera와 세 frame을 GPU로 overlay하지만,
최종 JPEG/PNG 한 장만 저장하며 intermediate frame이나 video를 만들지 않는다.

```bash
export RECORDS="$TRACK_OUT/gotrack_output/$OBJ/world_pose_records.json"

conda run --no-capture-output -n gotrack python -u \
  src/process/render_gotrack_debug_sheet.py \
  --capture-dir "$CAPTURE" --object-mesh "$MESH" \
  --gotrack-records "$RECORDS" \
  --output "$TRACK_OUT/debug_sheet.jpg"
```

특정 움직임 구간만 볼 때는 `--frame-indices 120 150 180`처럼 frame을 지정한다.
더 작은 sheet가 필요하면 `--max-cameras 4 --cell-width 360`을 추가한다.

`view_gotrack_viser.py`는 output을 쓰지 않는 interactive 검사 도구다. object trajectory와
calibration camera frustum을 띄우며, 기본은 모든 camera, camera image 비표시, 30 FPS 재생이다.
camera image를 띄우지 않으면 video decode/전송이 없어 가장 빠르다. browser의 `Show camera
frames`를 켜거나 `--show-camera-images`를 주면 선택 frame의 기존 undistorted video만 읽는다.

```bash
conda run --no-capture-output -n gotrack python -u \
  src/process/view_gotrack_viser.py \
  --capture-dir "$CAPTURE" --object-mesh "$MESH" \
  --gotrack-records "$RECORDS" --port 8080
```

Viser는 gotrack environment에 별도로 필요하다.

```bash
conda run -n gotrack python -m pip install "viser[urdf]"
```

Inspire capture에서는 `C2R.npy`를 사용해 robot-base 좌표계가 기본 scene frame이 된다.
episode의 arm/hand recording과 로봇 URDF가 있으면 robot motion도 함께 표시한다. arm/hand
asset과 robot helper는 이 repository가 아니라 **Paradex repository**에서 제공되므로, viewer를
실행하는 PC에 Paradex checkout 및 해당 Python import가 준비돼 있어야 한다. object/camera만
검사하려면 `--no-robot`을 사용한다.

### Texture 없는 mesh

FoundPose의 `make_mesh_tensors()`는 UV와 material은 있지만 texture image가 없는 OBJ도 지원한다.
이 경우 material color(없으면 중립 회색)로 만든 1×1 RGB texture를 사용해 UV renderer 경로를
유지한다. 따라서 이런 object의 실패는 즉시 재시도하지 말고, worker가 이 top-level
FoundationPose 수정이 포함된 commit을 받고 있는지 먼저 확인한다.

## 7. 결과와 보존 규칙

```text
<CAPTURE>/undistorted_video/                         # generated, raw videos/ 불변
<FRAME_OUT>/images/*.png, masks/*.png               # SAM3 first-frame inputs
<INIT_OUT>/init_pose_world.npy                      # FoundPose 6D initial world pose
<CAPTURE 부모>/foundpose_assets/object_repre/v1/<OBJ>/1/repre.pth
                                                    # campaign cache가 있으면 우선 사용
~/shared_data/mesh_blender/<OBJ>/foundpose_assets/... # campaign cache가 없을 때 fallback
<TRACK_OUT>/gotrack_output/<OBJ>/world_pose_records.json  # per-frame 6D world poses
```

`gotrack_capture.py`는 이미 존재하는 output directory를 의도적으로 거부한다.
재실행할 때 기존 결과를 지우지 말고 새로운 이름을 사용한다. FoundPose asset cache를
다른 위치의 cache를 써야 할 때만 `--assets-root`를 명시한다.

## 커밋 전 정리 체크리스트

- [ ] top-level의 `src/process/gotrack_capture.py`, `foundpose_init_capture.py`,
  `undistort_capture_videos.py`, debug sheet/Viser/grid renderer와 `mask.py` 변경을 commit한다.
- [ ] MV-GoTrack base가 `a9f033734c0bdf2d191265d22ea732a914c861f6`이고,
  `patches/MV-GoTrack-offline-capture.patch`가 적용됐는지 확인한다.
- [ ] `utils/renderer_nvdiffrast.py.orig` 및 `.rej`가 있다면 patch 충돌 잔재다.
  배포 전에 renderer 최종본을 검증하고 private checkout에서 제거한다.
- [ ] 실제 새 PC에서 1-frame smoke test와 짧은 GoTrack run을 한 번 수행하고,
  AutoDex SHA, MV-GoTrack SHA, checkpoint SHA, `conda list --explicit`을 run manifest에
  기록한다.

위 체크리스트가 끝나면 이 문서를 바탕으로 `scripts/bootstrap_offline_capture.sh`와
preflight checker를 추가하는 것이 안전하다. 그 전에는 기존 setup script를 자동화의
기준으로 사용하지 않는다.
