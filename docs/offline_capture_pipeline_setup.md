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
reference calibration에 종속된다. canonical cache의 기본 위치는
`~/shared_data/mesh_new/<object>/foundpose_assets/`이다. 현재 canonical template은
`inspire_dftp/<object>/0` calibration으로 만든다. 다른 intrinsic group에서도 runtime은
각 view의 calibration을 사용하지만, 성능 검증 전에는 같은 cache가 완전히 동등하다고
가정하지 않는다. `foundpose_init_capture.py`는 mesh 옆 canonical cache를 자동으로 우선한다.
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

`inspire_dftp`의 각 `<object>/0/cam_param/intrinsics.json`을 reference로 쓰고, cache는
`mesh_new/<object>/foundpose_assets/`에 저장한다. controller를 이 PC의 tmux 안에서
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

GoTrack 완료 뒤의 자동 debug sheet와 Viser 확인은 gotrack environment의
`transforms3d`, `viser[urdf]`도 사용한다. 최신 setup script는 이를 함께 설치한다.
이미 만들어 둔 worker environment에는 다음 한 번만 실행하면 된다.

```bash
conda run --no-capture-output -n gotrack python -m pip install \
  transforms3d==0.4.2 "viser[urdf]==1.0.30"
```

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
export FRAME_INDEX=30  # frame 0 can contain a stale capture on this platform
export FRAME_OUT="$CAPTURE/foundpose_frame_$(printf '%06d' "$FRAME_INDEX")_$OBJ"
export INIT_OUT="$CAPTURE/foundpose_$OBJ"
export TRACK_OUT="$CAPTURE/gotrack_${OBJ}_12cam"

cd "$REPO"
conda run --no-capture-output -n gotrack python -u \
  src/process/undistort_capture_videos.py --capture-dir "$CAPTURE"

conda run --no-capture-output -n sam3 python -u src/process/mask.py \
  --capture_dir "$CAPTURE" --frame-index "$FRAME_INDEX" --prompt "${OBJ//_/ }" \
  --video-dir "$CAPTURE/undistorted_video" --frame-output-dir "$FRAME_OUT"

conda run --no-capture-output -n gotrack python -u \
  src/process/foundpose_init_capture.py \
  --capture-dir "$CAPTURE" --frame-dir "$FRAME_OUT" \
  --mesh "$MESH" --object-name "$OBJ" --output-dir "$INIT_OUT"

conda run --no-capture-output -n gotrack python -u \
  src/process/gotrack_capture.py \
  --capture-dir "$CAPTURE" --video-dir "$CAPTURE/undistorted_video" \
  --mesh "$MESH" --init-pose "$INIT_OUT/init_pose_world.npy" \
  --object-name "$OBJ" --init-frame-index "$FRAME_INDEX" --bidirectional \
  --num-cameras 12 --max-frames -1 \
  --output-dir "$TRACK_OUT"
```

기본 bootstrap frame은 30이다. `gotrack_capture.py`는 이 frame에서 FoundPose로
초기화한 뒤 정방향 tracking과 `0..30` 역방향 tracking을 실행하고, 원래 시간순의
`gotrack_output/<object>/world_pose_records.json` 하나로 합친다. 따라서 stale frame-0
문제를 피하면서도 frame 0부터 pose를 제공한다. seed 이전 pose가 필요 없는 짧은
smoke test에만 `--forward-only`를 사용한다.

대칭성 때문에 시작 회전이 애매한 case study에는 `foundpose_init_capture.py`의
`--pose-selection-mode hybrid`를 시험할 수 있다. 이는 FoundPose를 다시 여러 번
실행하지 않고, 이미 계산한 camera별 PnP quality·cross-view pose consensus·silhouette
IoU를 함께 점수화한다. 기본값 `silhouette`은 기존 결과와 호환된다. 각 실행은
`<INIT_OUT>/candidate_bank.json`에 상위 후보와 score를 남긴다. 완전한 시각적 대칭은
어떤 selector도 의미상 올바른 회전을 보장하지 않으므로, `hybrid`는 검증 후에만
schedule 기본값으로 승격한다.

후보 순위 자체를 검토할 때는 GoTrack을 후보마다 실행하지 않는다.
`view_foundpose_candidates_viser.py`는 기존 `candidate_bank.json`의 pose를 읽어
bootstrap frame의 robot·camera scene에 후보를 색상별로 등록한다. 모든 rank가 기본으로
표시되며, GUI checkbox로 하나씩 숨기거나 다시 겹쳐 볼 수 있다. NAS output은 추가로 만들지 않는다.

```bash
conda run --no-capture-output -n gotrack python -u \
  src/process/view_foundpose_candidates_viser.py \
  --capture-dir "$CAPTURE" --object-mesh "$MESH" \
  --candidate-bank "$INIT_OUT/candidate_bank.json" --port 8080
```

기본 frame은 같은 directory의 `result.json`에서 추론하고, 없으면 30을 사용한다.
camera RGB가 필요할 때만 `--show-camera-images`를 추가한다. 후보를 고른 뒤 선택한
zero-based rank 하나에만 `--foundpose-candidate-rank`를 주어 GoTrack을 실행한다.

상위 FoundPose 후보 자체에 올바른 회전이 없을 때는 opt-in `global` selector를 사용할 수
있다. 이 mode는 모든 per-camera FoundPose pose에서 translation medoid를 구하고, 원래 후보
pose·후보 rotation과 medoid translation의 조합·저편향 SO(3) rotation 256개를 저해상도
multi-view silhouette으로 평가한다. camera별 IoU 양끝 15%를 제외한 robust mean으로 coarse
순위를 정한다. 점수만으로 인접 pose가 상위 slot을 독점하지 않도록 기본 35도 이상의 회전
간격을 강제하고, 서로 다른 pose basin 상위 5개만 기존 full-resolution silhouette optimizer로
정밀화한다. 최종
pose 하나만 GoTrack으로 넘어가므로 GoTrack 실행 횟수는 늘지 않는다.

```bash
python -u scripts/distribute_foundpose_gotrack.py \
  --mode init --schedule-id <new-schedule-id> \
  --target-root-rel capture/<campaign>/<robot> \
  --objects <object> --episodes <episode> \
  --foundpose-selection-mode global --max-attempts 1
```

기본 `silhouette` 동작에는 영향이 없으며, global mode의 주요 비용은 저해상도 rotation
search와 다섯 번의 silhouette refinement다. 필요하면
`--foundpose-global-rotation-count`, `--foundpose-global-coarse-max-side`,
`--foundpose-global-refine-top-k`, `--foundpose-global-min-rotation-separation-deg`로 조절한다.
`global_coarse_bank.json`에는 refinement 전의 서로 다른 seed들이, global
`candidate_bank.json`에는 refinement 후에도 중복 제거된 pose들이 저장된다. 후자의 pose는 이미
각각 refinement된 결과이므로 candidate Viser에서 바로 비교할 수 있다.

FoundPose는 camera별로 검색한 상위 template 각각에 대해 PnP를 계산하지만 원래 interface는
내부 PnP 1등 하나만 반환한다. global mode에서는 동일한 DINO feature/correspondence를 재사용해
camera별 PnP 대안을 기본 5개까지 회수한 뒤 coarse multiview pool에 추가한다. 모든 대안을 비싼
refinement에 넣지는 않고, 저해상도 silhouette score와 pose diversity를 통과한 소수만 기존
refinement를 받는다. primary candidate들만 translation medoid 계산에 사용하므로 낮은 순위의
오류 translation이 global search 중심을 이동시키지 않는다. 후보 수는
`--foundpose-per-view-candidates`로 조절하며, 결과는
`per_view_pose_candidates_world.npz`와 `candidate_bank.json`에서 확인한다. 기존
`silhouette`/`consensus`/`hybrid` mode는 camera별 1등만 사용한다.

작은 handle이나 고리만 symmetry를 깨는 object는 전체 silhouette IoU가 몸체 면적에 지배될 수
있다. global mode는 refinement된 후보 중 25도 이상 떨어진 회전이 robust IoU 0.005 이내에
있을 때만 near-symmetry fallback을 켠다. 이때 diverse seed를 최대 12개까지 추가 refinement하고,
후보 render들의 union과 intersection이 다른 pixel을 자동 symmetry-breaking region으로 삼는다.
이 영역을 512px resolution에서 확장해 mask mismatch를 계산하고, 기본 0.7 weight로 전체 IoU와
합쳐 최종 순위를 다시 정한다. 일반적인 비대칭 object에는 이 추가 비용이 발생하지 않는다.
관련 조절 옵션은 `--foundpose-global-asymmetry-refine-top-k`,
`--foundpose-global-asymmetry-max-side`, `--foundpose-global-asymmetry-score-margin`,
`--foundpose-global-asymmetry-weight`다.

silhouette와 작은 disagreement region으로도 회전 후보를 구분하지 못하는 case study에는
실험 옵션 `--foundpose-global-dino-rerank`를 사용할 수 있다. FoundPose가 각 camera에서
PnP를 만들 때 이미 계산한 DINO 2D-to-object-3D correspondence를 버리지 않고, refinement된
world-pose 후보를 모든 crop camera에 다시 투영해 soft reprojection score를 계산한다. 새
backbone pass나 FoundPose core matching 변경은 없으며 GoTrack 실행 횟수도 늘지 않는다.
기존 mask/asymmetry 최상점에서 기본 0.02 이내인 후보만 mask-tied 후보로 간주하고 그 안에서만
DINO score를 tie-break로 사용한다. 비교 가능한 candidate와 camera evidence가 부족하면 기존
mask 순서를 유지한다. 이 기능은 아직 기본값이 아니며, 검증 schedule에서만 켜는 것이 안전하다.
`candidate_bank.json`의 `dino_reprojection_score`, `dino_median_error_px`,
`dino_valid_views`, `dino_per_view`에 판단 근거가 저장된다. margin과 crop-pixel threshold는
각각 `--foundpose-global-dino-score-margin`,
`--foundpose-global-dino-inlier-threshold-px`로 조절한다.

GoTrack은 모든 AVI의 frame count/FPS로 계산한 duration을 비교해 median에서 기본 1초 이상
벗어난 camera만 자동 제외한다. 몇 frame 누락은 허용하며, 과거 특정 serial을 하드코딩해
제외하지 않는다. `run_manifest.json`의 `video_timings`, `rejected_cameras`에서 판단 근거를
확인할 수 있다. 필요하면 `--max-video-duration-skew-sec`으로 threshold를 조절하거나 `0`으로
끄고, 정말 제외할 camera만 `--exclude-cameras`로 명시한다.

기본값 `--camera-micro-batch-size 0`은 선택 camera 전체를 한 번에 refinement한다.
GoTrack이 OOM이면 `--num-cameras 8`로 새로운 `--output-dir`에 재실행하거나,
선택 camera 수보다 작은 micro-batch를 쓴다. 예를 들어 21개 camera 전체 관측을
유지하면서 GPU refinement만 8개씩 처리하려면
`--num-cameras 21 --camera-micro-batch-size 8`이다. 이 옵션이 동작하려면 해당
변경을 포함한 **MV-GoTrack fork commit**이어야 한다.

### 여러 PC에 다른 robot capture를 분배

`scripts/distribute_foundpose_gotrack.py`는 robot 하나의 root
`<robot-root>/<object>/<episode>`를 task로 분해한다. 각 task는 target robot의
cache를 만들지 않고, 명시적으로
`mesh_new/<object>/foundpose_assets`의 `repre.pth`를 사용한다.
worker는 shared-storage atomic claim으로 episode 하나씩만 점유하며, 다음 단계를
순서대로 실행한다: undistort → SAM3 frame-30 mask → FoundPose init → GoTrack
정방향+역방향 병합. `--init-frame-index`로 seed를 바꿀 수 있다.

`completed`는 더 이상 pose 하나만 있어도 되지 않는다. 기본적으로 valid pose coverage가
50% 이상이고 마지막 missing 구간이 30 frame 이하여야 한다. 그보다 심한 중단은
곧바로 task 전체 재시도로 넘기기 전에 기본 tail recovery를 시도한다. 반면 pose가 전 frame에
있지만 의미상 drift한 경우는 자동 실패로 단정하지 않는다.

tail recovery는 선택 camera들이 공통으로 가진 마지막 frame에서 시작해 30 frame 간격으로
앞쪽 seed를 탐색한다. 각 seed에 전체 선택 camera의 SAM3를 실행하고 최소 6-view mask 및
최소 3-view FoundPose pose를 얻은 첫 지점에서 GoTrack을 한 번만 실행한다. 이 pass는 seed에서 끝까지의 짧은
정방향 구간과, 기존 마지막 정상 pose보다 30 frame 앞까지의 역방향 구간만 만든다. 다음 조건을
모두 만족할 때만 기존 `world_pose_records.json`의 잃어버린 suffix를 교체한다.

- 기존/복구 track이 overlap에서 최소 3 pose를 공유한다.
- overlap 중앙값 차이가 translation 3 cm, rotation 15도 이하다. late FoundPose가 대칭으로
  인해 상수 회전 offset을 선택한 경우에는 실패 경계에서 먼 overlap frame들에서 그 offset의
  dispersion이 5도 이하일 때만 object-frame 회전을 정렬한다.
- 복구 suffix pose coverage가 90% 이상이고 마지막 missing이 30 frame 이하다.

publish 전 원본 records는 `gotrack_tracking/pre_tail_recovery_world_pose_records.json`에
보존된다. 탐색 및 거부 근거는 `attempt_NN/tail_recovery/recovery_manifest.json`에 남는다.
복구를 끄려면 `--no-tail-recovery`를 사용한다. 주요 조절 옵션은
`--tail-recovery-frame-step`, `--tail-recovery-max-seed-attempts`,
`--tail-recovery-min-mask-views`, `--tail-recovery-overlap-frames`다. FoundPose가 성공한 뒤
reverse bridge가 거부되더라도 다른 seed에서 비싼 GoTrack을 반복하지 않고 task를 failed로
남긴다.

tail recovery는 실제 영상에서 물체가 보이지 않는 구간의 pose를 만들어내지 않는다. 끝까지
물체가 복귀하지 않았거나 역방향 track이 기존 정상 구간에 연결되지 않으면 자동 병합하지 않는다.
GoTrack 재시도는 새 `attempt_NN` output directory를 사용하므로 partial output을
덮어쓰지 않는다.

옛 schedule의 process-exit 기반 `completed` 결과는 다음 audit으로 점검한다.

```bash
python -u scripts/audit_gotrack_completeness.py --all-schedules --only-incomplete
```

먼저 dry-run으로 mesh/cache/episode 조건을 확인하고 queue를 만든다.

```bash
cd "$REPO"
python -u scripts/distribute_foundpose_gotrack.py \
  --mode init --schedule-id hand_taeyun_right_01 \
  --target-root-rel capture/eccv2026/hand_taeyun/right \
  --num-cameras 22 --max-frames -1 --dry-run

python -u scripts/distribute_foundpose_gotrack.py \
  --mode init --schedule-id hand_taeyun_right_01 \
  --target-root-rel capture/eccv2026/hand_taeyun/right \
  --num-cameras 22 --max-frames -1
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

Viser는 gotrack environment에 포함되며, 기존 environment를 수동 보완할 때만 다음을 실행한다.

```bash
conda run -n gotrack python -m pip install "viser[urdf]==1.0.30"
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
~/shared_data/mesh_new/<OBJ>/foundpose_assets/object_repre/v1/<OBJ>/1/repre.pth
                                                    # canonical cache, mesh 옆에서 우선 사용
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
