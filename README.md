# EOIR Fire Detector

듀얼 카메라(RGB/IR) 기반 화재 감지 파이프라인. RGB는 YOLOv8 TFLite 추론, IR은 hotspot(온도) 감지를 수행하며, 융합 결과를 CLI/GUI/TCP로 활용합니다. 우선 대상은 NXP i.MX8M Plus 보드이지만 PC에서도 mock/video 입력으로 개발·테스트할 수 있습니다.

## 빠른 시작
- PC 개발/테스트
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  CONFIG_PATH=configs/config_pc.yaml python3 app.py
  ```
- 보드(i.MX8M Plus)
  ```bash
  pip install -r requirements.txt  # 보드에 맞게 opencv/tflite 항목은 조정
  CONFIG_PATH=configs/config.yaml python3 app.py
  ```
- GUI 모드: `CONFIG_PATH=configs/config_pc.yaml APP_MODE=gui python3 app.py`
- Display 모드(OpenCV 창): `CONFIG_PATH=configs/config_pc.yaml python3 app.py` 후 `DISPLAY.ENABLED: true` 설정

## 주요 기능
- RGB/IR 동시 입력: V4L2/GStreamer RGB, PureThermal IR(Y16)
- TFLite 추론: NPU delegate(libvx) 우선, CPU/XNNPACK 폴백
- IR hotspot 감지 + EO-IR 융합(Phase 1 완료)
- TCP 송신: RGB/IR/IR16/Det 프레임 JPEG 전송
- 캡처/재생: RGB/IR 비디오, RAW16(npys), 메타데이터 저장/로더
- GUI(Python/PyQt6): 입력/추론/IR/Overlay/Capture 제어·모니터링

## 실행/설정 요약
- 필수 환경변수: `CONFIG_PATH` (기본 `configs/config.yaml`)
- 실행 모드: `APP_MODE=cli|gui` (`--mode gui`도 지원)
- 입력 오버라이드: `RGB_INPUT_MODE=live|video|mock`, `RGB_VIDEO_PATH`, `IR_INPUT_MODE`, `IR_VIDEO_PATH`, `RGB_DEVICE`, `IR_LOOP/RGB_LOOP`
- 시각화 모드: `FUSION_VIS_MODE=test|temp` (확정 화재만 노란색으로 보고 싶을 때 `temp`)
- Delegate/모델: `MODEL`, `LABEL`, `DELEGATE`(예: `/usr/lib/libvx_delegate.so`). PC 테스트는 `model/` 내 기본 경로 사용.
- RGB 해상도 제약: 너비 16배수, 높이 8배수(예: 640x480, 1280x720). 잘못된 예: 960x540.
- 대표 실행:
  - CLI 기본: `CONFIG_PATH=configs/config.yaml python3 app.py`
  - GUI: `CONFIG_PATH=configs/config_pc.yaml APP_MODE=gui python3 app.py`
  - 비디오 재생 테스트: `RGB_INPUT_MODE=video RGB_VIDEO_PATH=/path/video.mp4 RGB_LOOP=true IR_INPUT_MODE=mock python3 app.py`

## 캡처 & 재생
- 캡처: `python3 capture.py --output ./capture_session [--duration SEC] [--max-frames N] [--save-det]`
- 저장물: `rgb.mp4`, `ir_vis.mp4`, `ir16/*.npy`, `metadata.csv`, `det.jsonl`(옵션)
- 재생 예시:
  ```python
  from utils.capture_loader import CaptureLoader
  for item in CaptureLoader("./capture_session"):
      rgb = item["rgb"]; ir = item["ir"]; ir_raw = item["ir_raw"]
  ```

## 테스트
- 기본: `pip install -r requirements-dev.txt && pytest`
- 샘플 영상 필요 시: `sample/fire_sample.mp4`가 없으면 `tests/test_video_sources.py` 일부가 skip 됩니다. `test_fire_fusion.py`는 항상 실행 가능.

## 문제 해결 빠른 단서
- Delegate 로드 실패 시: 경로 존재 여부와 보드용 `.so` 확인 → CPU/XNNPACK으로 자동 폴백
- GStreamer 오류: RGB 해상도 Step 제약 확인, `gst-inspect-1.0 v4l2src`
- IR/RGB 동기화: `SYNC` 설정(`ENABLED`, `MAX_DIFF_MS`)을 조정
- 더 자세한 로드맵/설계는 `docs/FIRE_FUSION_ROADMAP.md`, `docs/pyqt_gui_design.md` 참고

## 저장소 구조
```
app.py                  # 메인 엔트리
capture.py              # 캡처 스크립트
receiver.py / sender.py # TCP 수신 / 송신
display.py              # CLI 디스플레이
camera/                 # RGB/IR 소스, PureThermal
core/                   # 상태, 융합, 버퍼, 좌표 매핑
detector/               # TFLite 워커
gui/                    # PyQt GUI
configs/                # 설정 및 스키마
utils/                  # 캡처 로더
tests/                  # 스모크 테스트
model/                  # TFLite 모델/라벨 (대용량)
sample/                 # 샘플 영상/이미지
docs/                   # 로드맵 및 GUI 설계
```

## 라이선스
MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.
