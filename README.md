# EOIR Fire Detector

NXP i.MX8M Plus 환경을 우선 대상으로 하는 듀얼 카메라 화재 감지 파이프라인입니다. RGB(객체 탐지)와 IR(화점 탐지)을 동시에 처리하고, TCP로 전송하거나 GUI에서 모니터링/제어할 수 있습니다.

## 핵심 기능
- 듀얼 카메라: RGB(V4L2/GStreamer) + IR(PureThermal, Y16)
- YOLOv8 TFLite 추론: NPU delegate 사용 가능, CPU/XNNPACK 대체 경로
- IR 화점 탐지: RAW16 기반 온도 분석, 파라미터 런타임 조정
- GUI: 탭 기반 설정(Input/Inference/IR Hotspot/Overlay/Capture), 실시간 프리뷰/플롯/로그
- CLI: 키보드 단축키로 카메라 회전/반전 제어
- TCP 송신: RGB/IR/IR16/Det 프레임을 JPEG로 전송
- 캡처: RGB/IR 비디오, RAW16 npy, 메타데이터, 옵션으로 추론 결과 JSONL

## 요구사항
### 하드웨어
- 보드: NXP i.MX8M Plus (Vivante NPU) 권장
- 카메라: RGB `/dev/video*`, IR PureThermal(`/dev/video*`, VID:PID 1e4e:0100)

### 소프트웨어
- Python 3.10+
- OpenCV (GStreamer 지원), PyQt6(GUI), tflite-runtime, pyyaml, numpy
- 보드용 NPU delegate `.so` (예: `/usr/lib/libvx_delegate.so`)

## 설치 / 환경 준비
### PC(개발/테스트)
1. `python3 -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. 기본 PC 프로파일은 `configs/config_pc.yaml` (모델/라벨 경로는 `./model/` 기준). 다른 프로파일을 쓰려면 `CONFIG_PATH` 환경변수로 지정.

### 보드(i.MX8M Plus)
- BSP에 포함된 OpenCV / tflite-runtime 사용을 권장합니다. `requirements.txt`의 `opencv-python-headless`/`tflite-runtime` 라인은 PC 기본값이므로, 보드에서 충돌 시 생략하고 `numpy`, `pyyaml` 등만 설치하세요.
- delegate 경로(`DELEGATE`)가 실제 `.so` 파일을 가리키는지 확인하세요. 보통 `/usr/lib/libvx_delegate.so`.
- 나머지 Python 패키지는 `pip install -r requirements.txt`로 설치 가능하나, 보드별 패키지 제공 정책에 따라 조정하세요.

## 실행 방법

### 기본 실행
```bash
# CLI 모드 (기본)
CONFIG_PATH=configs/config.yaml python3 app.py

# CLI + Display 모드 (OpenCV 윈도우 표시)
CONFIG_PATH=configs/config.yaml python3 app.py
# config.yaml에서 DISPLAY.ENABLED: true 설정 필요

# GUI 모드
CONFIG_PATH=configs/config.yaml APP_MODE=gui python3 app.py
# 또는
CONFIG_PATH=configs/config.yaml python3 app.py --mode gui
```

### 환경변수 옵션

#### 필수 환경변수
- `CONFIG_PATH`: 설정 파일 경로 (기본값: `configs/config.yaml`)

#### 실행 모드 제어
- `APP_MODE`: 실행 모드 선택
  - `cli`: CLI 모드 (기본값, 키보드 제어 가능)
  - `gui`: GUI 모드 (PyQt6 필요)

#### 입력 소스 오버라이드 (RGB)
- `RGB_INPUT_MODE`: RGB 입력 모드
  - `live`: 실제 카메라 (기본값)
  - `video`: 비디오 파일
  - `mock`: 테스트용 모의 입력
- `RGB_VIDEO_PATH`: 비디오 파일 경로 (MODE=video일 때)
  - 단일 파일: `/path/to/video.mp4`
  - 여러 파일: `/path/to/video1.mp4;/path/to/video2.mp4` (세미콜론 구분)
- `RGB_LOOP`: 비디오 반복 재생 (true/false, 1/0)
- `RGB_FRAME_INTERVAL_MS`: 프레임 간격 (밀리초, 재생 속도 조절)
- `RGB_DEVICE`: 카메라 장치 오버라이드 (예: 0, /dev/video5)

#### 입력 소스 오버라이드 (IR)
- `IR_INPUT_MODE`: IR 입력 모드
  - `live`: 실제 카메라 (기본값)
  - `video`: 비디오 파일
  - `mock`: 테스트용 모의 입력
- `IR_VIDEO_PATH`: 비디오 파일 경로 (MODE=video일 때)
- `IR_LOOP`: 비디오 반복 재생 (true/false, 1/0)
- `IR_FRAME_INTERVAL_MS`: 프레임 간격 (밀리초)

#### 화재 탐지 시각화 모드
- `FUSION_VIS_MODE`: 화재 annotation 표시 모드
  - `test`: 모든 탐지 결과 표시 (기본값)
    - EO-only (RGB만 탐지): 노란색 박스
    - 확정 화재 (IR+RGB 탐지): 빨간색 박스
  - `temp`: 확정 화재만 노란색으로 표시
    - EO-only 박스는 숨김
    - 확정 화재는 노란색으로 표시

### 실행 예제

#### 1. 보드에서 실제 카메라로 실행 (CLI)
```bash
CONFIG_PATH=configs/config.yaml python3 app.py
```

#### 2. PC에서 GUI로 실행
```bash
CONFIG_PATH=configs/config_pc.yaml APP_MODE=gui python3 app.py
```

#### 3. 비디오 파일로 테스트 (RGB만)
```bash
RGB_INPUT_MODE=video RGB_VIDEO_PATH=/data/fire.mp4 RGB_LOOP=true IR_INPUT_MODE=mock python3 app.py
```

#### 4. 여러 비디오 파일 순차 재생
```bash
RGB_INPUT_MODE=video RGB_VIDEO_PATH="/data/video1.mp4;/data/video2.mp4" python3 app.py
```

#### 5. 특정 카메라 장치 지정
```bash
RGB_DEVICE=/dev/video5 python3 app.py
```

## 설정 파일 (config.yaml)

### 카메라 설정 (CAMERA)

#### IR 카메라
```yaml
CAMERA:
  IR:
    FPS: 9                    # 목표 FPS
    RES: [160, 120]           # 해상도 [width, height]
    SLEEP: 0.11               # 프레임 간 대기 시간 (초)
    DEVICE: "/dev/video3"     # 장치 경로
    ROTATE: 0                 # 회전 각도 (0, 90, 180, 270)
    FLIP_H: false             # 좌우반전
    FLIP_V: false             # 상하반전
    TAU: 0.95                 # 대기 투과율 (실내: 0.95, 야외: 0.3~0.7)
    FIRE_DETECTION: true      # 화점 탐지 활성화
    FIRE_MIN_TEMP: 80         # 화점 최소 온도 (섭씨)
    FIRE_THR: 20              # 보정 온도 임계값
    FIRE_RAW_THR: 5           # raw 온도 임계값
```

#### RGB 카메라 (보드용)
```yaml
CAMERA:
  RGB_FRONT:
    FPS: 30                   # 목표 FPS
    RES: [640, 480]           # 해상도 [width, height]
    SLEEP: 0.033              # 프레임 간 대기 시간 (초)
    DEVICE: "/dev/video5"     # 장치 경로
    ROTATE: 0                 # 회전 각도 (0, 90, 180, 270)
    FLIP_H: false             # 좌우반전
    FLIP_V: false             # 상하반전
```

**중요:** RGB 해상도는 Step 제약이 있습니다
- 너비: 16의 배수 (예: 640, 1280, 1920)
- 높이: 8의 배수 (예: 480, 720, 1080)
- 잘못된 예: 960x540 (540은 8의 배수 아님)
- 올바른 예: 960x544, 640x480, 1280x720

#### RGB 카메라 (PC용)
```yaml
CAMERA:
  RGB_PC:
    FPS: 30
    RES: [1280, 720]
    SLEEP: 0.033
    DEVICE: 0                 # PC 웹캠 인덱스
```

### 모델 및 Delegate 설정
```yaml
MODEL: /root/lk_fire/model/8n_640_v2/best_full_integer_quant.tflite
LABEL: /root/lk_fire/model/labels.txt
DELEGATE: "/usr/lib/libvx_delegate.so"  # NPU delegate (비우면 CPU/XNNPACK)
TARGET_RES: [960, 540]                   # 탐지 전 리사이징 해상도
```

### 입력 소스 설정 (INPUT)
```yaml
INPUT:
  RGB:
    MODE: live              # live | video | mock
    VIDEO_PATH: ""          # 비디오 파일 경로
    LOOP: true              # 비디오 반복 재생
    FRAME_INTERVAL_MS: null # 프레임 간격 (null=원본 속도)
    COLOR: [0, 255, 0]      # 박스 색상 (BGR)
  IR:
    MODE: live
    VIDEO_PATH: ""
    LOOP: true
    FRAME_INTERVAL_MS: null
```

### 화재 탐지 설정 (STATE)
```yaml
STATE:
  FIRE:
    NMS: 0.1                # Non-Maximum Suppression 임계값
    WINDOW: 50              # 슬라이딩 윈도우 크기
    THRESHOLD: 60           # 화재 판정 임계값
    CONFIDENCE: 0.2         # 최소 신뢰도
    MIN_DUR: 10.0           # 최소 지속 시간 (초)
    ACTIVE_DUR: 2.0         # 활성 지속 시간 (초)
    INACTIVE_DUR: 10.0      # 비활성 지속 시간 (초)
    DET_MODE: 1             # 탐지 모드
  BUFFERS:
    RAW16: 100              # RAW16 버퍼 크기
    RAW: 50                 # RAW 버퍼 크기
    DET: 100                # 탐지 버퍼 크기
  DET_SLEEP: 0.11           # 탐지 스레드 대기 시간
```

### 네트워크 설정 (SERVER)
```yaml
SERVER:
  IP: '192.168.50.178'      # 수신 서버 IP
  PORT: 9999                # 수신 서버 포트
  COMP_RATIO: 70            # JPEG 압축 품질 (0-100)
```

### 디스플레이 설정 (DISPLAY)
```yaml
DISPLAY:
  ENABLED: false            # OpenCV 윈도우 표시 (CLI 전용)
  WINDOW_NAME: "Vision AI Display"
```

**CLI Display 모드:**
- CLI 모드에서 OpenCV 윈도우로 실시간 영상 표시
- RGB/Det/IR/Overlay 4분할 화면 표시
- `ENABLED: true` 설정 후 실행
- X11 포워딩이 가능한 환경에서 사용 (로컬 PC, SSH -X)
- 임베디드 보드에서는 HDMI 출력이 있거나 X11이 설정된 경우 사용 가능
- GUI 모드와는 별개 (GUI는 PyQt6, Display는 OpenCV)

**참고:**
- 임베디드 보드에서 Display 모드는 성능 저하 가능성 있음
- 원격에서 사용 시: `ssh -X user@host` 또는 VNC/X11 포워딩 필요
- 헤드리스 환경에서는 비활성화 권장

### 동기화 설정 (SYNC)
```yaml
SYNC:
  ENABLED: false            # IR/RGB 타임스탬프 동기화
  MAX_DIFF_MS: 120          # 최대 허용 시간차 (밀리초)
```

### 캡처 설정 (CAPTURE)
```yaml
CAPTURE:
  OUTPUT_DIR: "./capture_session"
  DURATION_SEC: null        # 캡처 지속 시간 (초, null=무제한)
  MAX_FRAMES: null          # 최대 프레임 수 (null=무제한)
  MAX_DIFF_MS: 80           # IR/RGB 최대 시간차 (밀리초)
  SAVE_RGB_VIDEO: true      # RGB 비디오 저장
  SAVE_IR_VIDEO: true       # IR 비디오 저장
  SAVE_IR_RAW16: true       # IR RAW16 데이터 저장
  RGB_CODEC: "mp4v"         # RGB 코덱
  IR_CODEC: "mp4v"          # IR 코덱
```

### 좌표 보정 설정 (COORD)
```yaml
COORD:
  OFFSET_X: 0.0             # X축 오프셋
  OFFSET_Y: 0.0             # Y축 오프셋
  SCALE: null               # 스케일 보정
```

## CLI 키보드 제어

CLI 모드에서는 키보드로 실시간 제어가 가능합니다. 앱을 포그라운드로 실행해야 합니다.

### IR 카메라 제어
- `1`: IR 시계방향 90도 회전
- `2`: IR 좌우반전 토글
- `3`: IR 상하반전 토글

### RGB 카메라 제어
- `4`: RGB 시계방향 90도 회전
- `5`: RGB 좌우반전 토글
- `6`: RGB 상하반전 토글

### 양쪽 카메라 제어
- `7`: 양쪽 좌우반전 토글
- `8`: 양쪽 상하반전 토글

### 기타
- `s`: 현재 상태 표시
- `h`: 도움말 표시
- `q`: 애플리케이션 종료

**참고:**
- 백그라운드(&)로 실행하면 키보드 입력이 작동하지 않습니다
- SSH 터미널에서 직접 키를 누르면 즉시 반영됩니다
- Config 파일의 ROTATE/FLIP 설정은 초기값만 지정하고, 런타임에 키보드로 자유롭게 변경 가능합니다

## GUI 모드

### 실행
```bash
CONFIG_PATH=configs/config.yaml APP_MODE=gui python3 app.py
```

### 탭 구성
- **Input**: RGB/IR 모드(live/video/mock), 경로, Loop, Device 선택(+Browse 버튼)
- **Inference**: 모델/라벨/Delegate 경로, 클래스 필터(smoke/fire), 적용 시 탐지 워커 재시작
- **IR Hotspot**: 화점 탐지 on/off, MinTemp, Thr, RawThr, Tau 런타임 적용
- **Overlay**: IR↔RGB 정렬 Offset/Scale 조정, Nudge 버튼
- **Capture**: 출력 경로/Duration/MaxFrames 설정, `Start Capture`로 `capture.py` 실행

### 상단 버튼
- IR/RGB 회전 버튼 (90도씩)
- Start/Stop Sender (TCP 전송)
- Start/Stop Capture (캡처)

### 화면 구성
- **프리뷰**: RGB/Det/IR/Overlay 4분할 화면
- **상태 라벨**: Det/IR/RGB FPS, SYNC 상태 표시
- **플롯**: Det/RGB/IR FPS 롤링 그래프
- **로그 창**: GUI 이벤트/오류 표시

## 성능 최적화

### 임베디드 보드에서 성능 문제 발생 시

#### 1. RGB 해상도 낮추기
CPU 부담이 클 경우 해상도를 낮춰서 성능을 개선할 수 있습니다:
```yaml
CAMERA:
  RGB_FRONT:
    RES: [640, 480]   # 1920x1080 대신 640x480 사용
```

권장 해상도:
- 저사양: 640x480 (VGA)
- 중급: 1280x720 (720p)
- 고사양: 1920x1080 (1080p)

#### 2. FPS 조정
```yaml
CAMERA:
  RGB_FRONT:
    FPS: 15          # 30fps 대신 15fps 사용
    SLEEP: 0.066     # 1/15초
```

#### 3. TARGET_RES 조정
탐지 전 리사이징 해상도를 낮춰서 NPU 부담 감소:
```yaml
TARGET_RES: [640, 480]   # 기본 [960, 540] 대신
```

#### 4. JPEG 압축률 조정
네트워크 대역폭이 문제일 경우:
```yaml
SERVER:
  COMP_RATIO: 50     # 기본 70 대신 낮은 품질로 압축
```

## 수신(Receiver)

### 기본 실행
```bash
python3 receiver.py
```

### 옵션
- 기본 포트: 9999
- 저장 경로:
  - RGB: `save/visible/`
  - IR: `save/lwir/`

송신 측(SERVER.IP, SERVER.PORT)과 수신 측 포트를 맞춰주세요.

## 캡처 & 재사용

### CLI에서 캡처
```bash
python3 capture.py --output ./capture_session [--duration SEC] [--max-frames N] [--save-det]
```

### 옵션
- `--output DIR`: 출력 디렉토리
- `--duration SEC`: 캡처 지속 시간 (초)
- `--max-frames N`: 최대 프레임 수
- `--save-det`: 탐지 결과도 저장 (JSONL)

### 저장되는 파일
- `rgb.mp4`: RGB 비디오
- `ir_vis.mp4`: IR 가시화 비디오
- `ir16/*.npy`: IR RAW16 데이터 (프레임별)
- `metadata.csv`: 타임스탬프 및 메타데이터
- `det.jsonl`: 탐지 결과 (옵션)

### 재사용 예제
```python
from utils.capture_loader import CaptureLoader

for item in CaptureLoader("./capture_session"):
    rgb = item["rgb"]           # RGB 프레임 (numpy array)
    ir = item["ir"]             # IR 가시화 프레임
    ir_raw = item["ir_raw"]     # IR RAW16 데이터
    # 처리...
```

## 테스트

### 기본 테스트
```bash
pip install -r requirements-dev.txt
pytest
```

### 특정 테스트
```bash
# 비디오 소스 테스트 (sample/fire_sample.mp4 필요)
pytest tests/test_video_sources.py

# 특정 테스트 케이스
pytest tests/test_video_sources.py::test_rgb_video_source
```

## NPU/Delegate

### Delegate 사용
- `DELEGATE` 경로가 지정되어 있으면 NPU 사용 시도
- `.so` 파일 로드 실패 시 자동으로 CPU/XNNPACK으로 폴백
- i.MX8M Plus: `/usr/lib/libvx_delegate.so`

### CPU 전용 모드
```yaml
DELEGATE: ""    # 빈 문자열 또는 주석 처리
```

## 자주 겪는 문제

### RGB 장치가 다시 안 열릴 때
1. 장치 점유 확인: `fuser /dev/video5`
2. VideoCapture.release() 확인 (최근 수정으로 처리됨)
3. 컨테이너 실행 시 `--device /dev/videoX` 포함 확인

### IR가 mock→live 전환 후 멈출 때
1. `IRCamera.stop()`에서 cleanup 완료 (최근 수정됨)
2. `/dev/video*` 인덱스 변동 여부 확인
3. `v4l2-ctl --list-devices`로 장치 확인

### Delegate 로드 실패로 CPU로 떨어질 때
1. `DELEGATE` 경로 존재 여부 확인
2. 보드용 `.so` 파일인지 확인
3. 로그에서 delegate 로드 메시지 확인

### GStreamer 파이프라인 실패
1. RGB 해상도가 Step 제약을 만족하는지 확인 (너비: 16의 배수, 높이: 8의 배수)
2. GStreamer 플러그인 설치 확인: `gst-inspect-1.0 v4l2src`
3. 장치가 NV12 포맷을 지원하는지 확인: `v4l2-ctl -d /dev/video5 --list-formats-ext`

### PyQt6 버전 충돌 (GUI 모드)
1. 시스템 Qt 버전과 PyQt6 버전 확인
2. 보드에서는 CLI 모드 사용 권장 (GUI는 개발/테스트용)
3. `pip install PyQt6==X.Y.Z` 버전 다운그레이드 시도

### CPU 사용량이 높을 때
1. RGB 해상도 낮추기 (성능 최적화 섹션 참고)
2. FPS 낮추기
3. TARGET_RES 낮추기
4. 불필요한 기능 비활성화 (DISPLAY, SYNC 등)

## 리포지토리 구조

```
lk_fire/
├── app.py                  # 메인 엔트리 포인트
├── capture.py              # 캡처 스크립트
├── receiver.py             # TCP 수신 서버
├── sender.py               # TCP 송신 모듈
├── camera/                 # 카메라 소스
│   ├── rgbcam.py          # RGB 카메라
│   ├── ircam.py           # IR 카메라 인터페이스
│   ├── purethermal/       # PureThermal 드라이버
│   ├── frame_source.py    # 프레임 소스 베이스 클래스
│   └── device_selector.py # 장치 자동 선택
├── detector/
│   └── tflite.py          # YOLOv8 TFLite 워커
├── gui/
│   └── app_gui.py         # PyQt6 GUI
├── core/
│   ├── state.py           # 카메라 상태 관리
│   └── util.py            # 유틸리티 함수
├── configs/
│   ├── config.yaml        # 보드용 설정
│   ├── config_pc.yaml     # PC용 설정
│   ├── schema.py          # 설정 스키마
│   └── get_cfg.py         # 설정 로더
├── utils/
│   └── capture_loader.py  # 캡처 재생 로더
├── tests/                 # 테스트
└── model/                 # TFLite 모델 및 라벨
```

## 라이센스

(라이센스 정보 추가 예정)
