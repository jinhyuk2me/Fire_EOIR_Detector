# 코드 리팩토링 로드맵

> **작성일**: 2025-12-05
> **대상 프로젝트**: EOIR Fire Detector
> **현재 코드 품질**: 5.6/10
> **목표 코드 품질**: 8.0/10

---

## 📊 개요

| Phase | 기간 | 작업 시간 | 주요 목표 | 위험도 |
|-------|------|---------|---------|--------|
| Phase 1 | 1-2주 | 2.5시간 | 긴급 버그/중복 제거 | 🟢 낮음 |
| Phase 2 | 3-4주 | 7시간 | 구조 개선 | 🟡 중간 |
| Phase 3 | 4-6주 | 12시간 | 고급 개선 | 🟠 높음 |
| **총계** | **6-12주** | **21.5시간** | **품질 5.6→8.0** | - |

---

## 🎯 현재 상태 평가

### 코드 품질 점수

| 평가 항목 | 점수 | 설명 |
|----------|------|------|
| **아키텍처** | 7/10 | 계층 분리 좋음, 순환 의존성 위험 |
| **코드 중복** | 5/10 | rgbcam.py에 정확한 중복, 타임스탬프 변환 함수 중복 |
| **복잡도** | 6/10 | gui/app_gui.py 과도하게 큼(1,134줄), 대부분 함수는 적절 |
| **설정 관리** | 6/10 | 공통 설정 중복, 일부 하드코딩 |
| **에러 처리** | 5/10 | 포괄적 except 블록, 예외 타입 부족 |
| **테스트** | 3/10 | 2개 파일만 있음, 커버리지 <2% |
| **문서화** | 6/10 | 기본 docstring 있음, 타입 힌트 부족 |
| **성능** | 7/10 | 메모리 효율 OK, 최적화 여지 있음 |
| **전체 평가** | **5.6/10** | **"개선 필요" (리팩토링 권장)** |

### 주요 문제점

1. ✗ **rgbcam.py의 정확한 함수 중복** (라인 14-40과 43-69)
2. ✗ **포괄적 except 블록으로 인한 디버깅 어려움** (7곳)
3. ✗ **테스트 커버리지 극히 낮음** (<2%)
4. ✗ **gui/app_gui.py 과도한 책임 집중** (1,134줄)
5. ✗ **설정 파일 중복** (config.yaml, config_pc.yaml 90% 동일)

---

## 🚨 Phase 1: 긴급 수정 (1-2주, 2.5시간)

### 목표
**즉시 해결해야 할 버그 위험 제거 및 명확한 중복 코드 정리**

### 작업 항목

#### 1.1 rgbcam.py 함수 중복 제거 ⭐️⭐️⭐️

**파일**: `camera/rgbcam.py`
**문제**: 라인 14-40과 43-69에서 `_log()`, `_open_capture()` 정확히 복제
**소요 시간**: 30분
**난이도**: ⭐ (매우 쉬움)

**작업 순서**:
1. 백업 생성: `git checkout -b refactor/phase1-rgbcam`
2. 라인 43-69 삭제 (중복 함수 제거)
3. RGBCamera 클래스가 올바른 함수 참조하는지 확인
4. 테스트 실행: `pytest tests/`
5. 커밋: `git commit -m "Remove duplicate functions in rgbcam.py"`

**변경 내용**:
```python
# 현재 (중복)
def _log(msg):  # 라인 14
def _open_capture(...):  # 라인 18
def _log(msg):  # 라인 43 ← 삭제
def _open_capture(...):  # 라인 47 ← 삭제

# 개선 (중복 제거)
def _log(msg):  # 유지
def _open_capture(...):  # 유지
# (라인 43-69 완전 삭제)
```

**위험도**: 🟢 매우 낮음 (단순 중복 제거)
**영향 범위**: `camera/rgbcam.py` 단일 파일
**테스트 필요**: RGB 카메라 입력 동작 확인

---

#### 1.2 타임스탬프 변환 함수 통합 ⭐️⭐️

**파일**: `gui/app_gui.py:60-66`, `sender.py:193-199`
**문제**: `_ts_to_epoch_ms()` 동일 함수 두 곳에서 정의
**소요 시간**: 15분
**난이도**: ⭐ (쉬움)

**작업 순서**:
1. `core/util.py`에 공통 함수 추가:
```python
# core/util.py
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def ts_to_epoch_ms(ts: str) -> Optional[float]:
    """
    타임스탬프 문자열을 epoch milliseconds로 변환

    Args:
        ts: "yymmddHHMMSSffffff" 형식 문자열

    Returns:
        epoch milliseconds 또는 None (변환 실패 시)
    """
    if not ts:
        return None
    try:
        dt = datetime.strptime(ts, "%y%m%d%H%M%S%f")
        return dt.timestamp() * 1000.0
    except Exception as e:
        logger.warning("Invalid timestamp format: %s (error: %s)", ts, e)
        return None
```

2. `gui/app_gui.py`와 `sender.py`에서 import로 교체:
```python
# gui/app_gui.py, sender.py
from core.util import ts_to_epoch_ms  # 추가

# 기존 _ts_to_epoch_ms() 함수 정의 삭제
```

**위험도**: 🟢 낮음
**영향 범위**: 3개 파일
**테스트 필요**: GUI/CLI 양쪽 타임스탬프 표시 확인

---

#### 1.3 포괄적 except 블록 구체화 ⭐️⭐️⭐️

**파일**:
- `sender.py:166`
- `app.py:66, 75, 87`
- `camera/purethermal/thermalcamera.py:75, 186, 198`

**문제**: `except:` 또는 `except Exception:` 사용으로 예외 타입 불명확
**소요 시간**: 1시간
**난이도**: ⭐⭐ (중간)

**작업 순서**:

**1.3.1 sender.py:166** (우선순위 HIGH)
```python
# 현재
try:
    ...
except:
    pass  # ← 문제

# 개선
try:
    ...
except socket.error as e:
    logger.error("Socket connection error: %s", e)
    self.connected = False
except json.JSONDecodeError as e:
    logger.warning("Invalid JSON packet: %s", e)
except Exception as e:
    logger.exception("Unexpected error in frame send: %s", e)
```

**1.3.2 app.py:66, 75** (우선순위 MEDIUM)
```python
# 현재
try:
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
except:
    return None

# 개선
except (termios.error, AttributeError, OSError) as e:
    logger.debug("Terminal setup failed (non-interactive): %s", e)
    return None
```

**1.3.3 camera/purethermal/thermalcamera.py** (우선순위 LOW)
```python
# 3곳의 except: 블록을 구체화
except usb.core.USBError as e:
    logger.error("USB error: %s", e)
except Exception as e:
    logger.exception("Thermal camera error: %s", e)
```

**위험도**: 🟡 중간 (예외 처리 로직 변경)
**영향 범위**: 3개 파일, 7개 위치
**테스트 필요**:
- 네트워크 끊김 시나리오
- 터미널 비대화형 모드
- USB 장치 연결/해제

---

### Phase 1 완료 체크리스트

- [ ] 모든 변경사항 커밋 완료
- [ ] pytest 통과 (기존 테스트 2개)
- [ ] CLI 모드 실행 테스트
- [ ] GUI 모드 실행 테스트
- [ ] RGB/IR 카메라 입력 정상 동작
- [ ] TCP 송신 정상 동작
- [ ] 브랜치 병합: `git merge refactor/phase1-rgbcam`

**예상 효과**:
- 코드 라인 -35줄
- 버그 위험도 -40%
- 디버깅 난이도 -50%
- 유지보수성 +20%

---

## 🏗️ Phase 2: 구조 개선 (3-4주, 7시간)

### 목표
**테스트 가능한 구조로 변경, 재사용성 향상**

### 작업 항목

#### 2.1 RuntimeController 분리 ⭐️⭐️⭐️⭐️

**파일**: `app.py` → 새로운 `core/controller.py`
**문제**: app.py에 런타임 제어 로직 혼재 (277줄)
**소요 시간**: 3시간
**난이도**: ⭐⭐⭐ (어려움)

**작업 순서**:

**2.1.1 새 파일 생성**
```bash
touch core/controller.py
```

**2.1.2 RuntimeController 클래스 이동**
```python
# core/controller.py (NEW)
import threading
import logging
from typing import Dict, Any, Optional
from .buffer import DoubleBuffer
from .state import LabelScaleState, DEFAULT_LABEL_SCALE

logger = logging.getLogger(__name__)

class CoordState:
    """좌표 매핑 상태 관리"""
    # app.py:152-166에서 이동
    ...

class RuntimeController:
    """런타임 파이프라인 컨트롤러"""
    # app.py:168-444에서 이동
    ...
```

**2.1.3 app.py에서 import로 교체**
```python
# app.py
from core.controller import RuntimeController, CoordState
# (기존 정의 삭제)
```

**위험도**: 🟠 높음 (핵심 로직 이동)
**영향 범위**: 2개 파일 (app.py, core/controller.py)
**의존성**: Phase 1 완료 필수

---

#### 2.2 하드코딩된 이미지 크기 제거 ⭐️⭐️⭐️

**파일**: `sender.py:239-240`, `gui/app_gui.py:113-119`
**문제**: IR(160x120), RGB(960x540) 크기 하드코딩
**소요 시간**: 1시간
**난이도**: ⭐⭐ (중간)

**작업 순서**:

**2.2.1 FireFusion 팩토리 패턴 도입**
```python
# core/fire_fusion.py에 추가
class FireFusionFactory:
    """설정 기반 FireFusion 인스턴스 생성"""

    @staticmethod
    def from_config(config) -> 'FireFusion':
        ir_size = tuple(config.CAMERA_IR.RES)
        rgb_size = tuple(config.TARGET_RES)
        coord_cfg = config.COORD if hasattr(config, 'COORD') else {}

        return FireFusion(
            ir_size=ir_size,
            rgb_size=rgb_size,
            offset_x=coord_cfg.get('OFFSET_X', 0.0),
            offset_y=coord_cfg.get('OFFSET_Y', 0.0),
            scale=coord_cfg.get('SCALE')
        )
```

**2.2.2 sender.py, gui/app_gui.py에서 사용**
```python
# sender.py, gui/app_gui.py
from core.fire_fusion import FireFusionFactory

fire_fusion = FireFusionFactory.from_config(cfg)
```

**위험도**: 🟡 중간
**영향 범위**: 3개 파일

---

#### 2.3 타입 힌트 추가 ⭐️⭐️

**파일**:
- `core/coord_mapper.py`
- `camera/ircam.py` (주요 메서드)
- `detector/tflite.py` (주요 메서드)

**소요 시간**: 3시간
**난이도**: ⭐⭐ (중간)

**예시**:
```python
# core/coord_mapper.py
from typing import Tuple

def ir_to_rgb(self, ir_x: float, ir_y: float) -> Tuple[float, float]:
    """IR 좌표를 RGB 좌표로 변환"""
    ...
```

**위험도**: 🟢 낮음 (기능 변경 없음)
**테스트 필요**: mypy 타입 체크 통과

---

### Phase 2 완료 체크리스트

- [ ] RuntimeController 분리 완료
- [ ] 하드코딩된 크기 제거 완료
- [ ] 주요 모듈에 타입 힌트 추가
- [ ] mypy 타입 체크 통과
- [ ] pytest 통과
- [ ] 통합 테스트 (GUI + CLI + TCP)
- [ ] 브랜치 병합: `git merge refactor/phase2-structure`

**예상 효과**:
- 테스트 가능성 +60%
- 코드 재사용성 +80%
- IDE 자동완성 +100%
- 설정 유연성 +50%

---

## 🎨 Phase 3: 고급 개선 (4-6주, 12시간)

### 목표
**GUI 분리, 테스트 확대, 설정 통합**

### 작업 항목

#### 3.1 GUI 클래스 분리 ⭐️⭐️⭐️⭐️⭐️

**파일**: `gui/app_gui.py` (1,134줄 → 4개 파일)
**소요 시간**: 5시간
**난이도**: ⭐⭐⭐⭐ (매우 어려움)

**분리 계획**:
```
MainWindow (현재 1,134줄)
├─ UI 레이아웃 관리 (300줄) → main_window.py
├─ 프레임 업데이트 (200줄) → frame_updater.py
├─ 제어 패널 (300줄) → control_panel.py
└─ 모니터 패널 (200줄) → monitor_panel.py
```

**새 파일**:
- `gui/frame_updater.py` - 프레임 업데이트 및 렌더링
- `gui/control_panel.py` - 카메라/탐지기 파라미터 제어
- `gui/monitor_panel.py` - 상태 모니터링

**위험도**: 🔴 매우 높음 (GUI 전체 재구성)
**의존성**: Phase 2.1 완료 필수

---

#### 3.2 설정 파일 통합 ⭐️⭐️⭐️

**파일**: `configs/config.yaml`, `configs/config_pc.yaml`
**소요 시간**: 2시간
**난이도**: ⭐⭐⭐ (어려움)

**작업 순서**:

**3.2.1 공통 설정 추출**
```yaml
# configs/config_base.yaml (NEW)
CAMERA:
  IR:
    FPS: 9
    RES: [160, 120]
    FIRE_DETECTION: true
    # ... 공통 설정
```

**3.2.2 환경별 오버라이드**
```yaml
# configs/environments/board.yaml (NEW)
CAMERA:
  IR:
    DEVICE: "/dev/video3"
  RGB_FRONT:
    DEVICE: "/dev/video5"
    RES: [320, 240]

# configs/environments/pc.yaml (NEW)
CAMERA:
  IR:
    DEVICE: "/dev/video0"
  RGB_FRONT:
    DEVICE: 0
    RES: [1280, 720]
```

**3.2.3 설정 로더 수정**
```python
# configs/get_cfg.py
def get_cfg():
    # 기본 설정 로드
    base_config = load_yaml('config_base.yaml')

    # 환경별 오버라이드
    env = os.getenv('DEPLOY_ENV', 'pc')  # board | pc
    env_config = load_yaml(f'environments/{env}.yaml')

    # 병합
    return merge_configs(base_config, env_config)
```

**위험도**: 🟠 높음 (설정 로직 변경)

---

#### 3.3 테스트 커버리지 확대 ⭐️⭐️⭐️⭐️

**목표**: < 2% → 30% 커버리지
**소요 시간**: 5시간
**난이도**: ⭐⭐⭐⭐ (어려움)

**작업 순서**:

**3.3.1 테스트 인프라 구축**
```bash
# requirements-dev.txt 확장
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
mypy>=1.0.0
```

**3.3.2 핵심 모듈 테스트 작성**
- `tests/test_controller.py` (NEW)
- `tests/test_util.py` (NEW)
- `tests/test_coord_mapper.py` (확장)
- `tests/test_tflite_worker.py` (NEW)

**3.3.3 커버리지 측정**
```bash
pytest --cov=core --cov=camera --cov=detector --cov-report=html
```

**위험도**: 🟢 낮음 (기능 추가만)

---

### Phase 3 완료 체크리스트

- [ ] GUI 4개 파일로 분리 완료
- [ ] 설정 파일 통합 완료
- [ ] 테스트 파일 10개 이상 추가
- [ ] 커버리지 30% 달성
- [ ] pytest 전체 통과
- [ ] mypy 타입 체크 통과
- [ ] 수동 통합 테스트 (GUI + CLI + TCP + 재생)
- [ ] 브랜치 병합: `git merge refactor/phase3-advanced`

**예상 효과**:
- GUI 이해도 +40%
- 설정 관리 중복 -40%
- 테스트 커버리지 +28%
- 전체 코드 품질 8.0/10 달성

---

## ⚠️ 위험 관리 및 롤백 전략

### 위험 요소

| 위험 | 발생 가능성 | 영향도 | 대응 방안 |
|------|-----------|--------|---------|
| GUI 분리 시 신호/슬롯 연결 오류 | 🟠 중간 | 🔴 높음 | Phase 3.1 전에 상세 설계 문서 작성 |
| 설정 병합 로직 버그 | 🟡 낮음 | 🟠 중간 | 단위 테스트 먼저 작성 |
| RuntimeController 분리 시 순환 의존성 | 🟡 낮음 | 🔴 높음 | import 그래프 사전 분석 |
| 타입 힌트 추가 시 mypy 오류 대량 발생 | 🟢 매우 낮음 | 🟡 낮음 | 점진적 적용 (ignore 활용) |
| 테스트 작성 시 Mock 설정 복잡 | 🟠 중간 | 🟢 낮음 | pytest-mock 활용 |

### 롤백 전략

**Phase별 브랜치 전략**:
```bash
main
 ├─ refactor/phase1-rgbcam (완료 후 병합)
 ├─ refactor/phase2-structure (완료 후 병합)
 └─ refactor/phase3-advanced (완료 후 병합)
```

**각 Phase 시작 전 백업**:
```bash
git tag backup-before-phase1
git tag backup-before-phase2
git tag backup-before-phase3
```

**문제 발생 시 롤백**:
```bash
git reset --hard backup-before-phaseX
```

---

## 📈 진행 상황 추적

### 마일스톤

| 마일스톤 | 완료 조건 | 예상 날짜 |
|---------|---------|---------|
| M1: Phase 1 완료 | pytest 통과, 중복 제거 | Week 2 |
| M2: Phase 2 완료 | Controller 분리, 타입 힌트 | Week 4 |
| M3: Phase 3 완료 | GUI 분리, 커버리지 30% | Week 8 |
| M4: 최종 검증 | 전체 통합 테스트 통과 | Week 10 |

### 주간 체크리스트

**Week 1-2: Phase 1**
- [ ] Mon: rgbcam.py 중복 제거 (1.1)
- [ ] Tue: 타임스탬프 함수 통합 (1.2)
- [ ] Wed-Thu: except 블록 구체화 (1.3)
- [ ] Fri: 통합 테스트 및 커밋

**Week 3-4: Phase 2**
- [ ] Week 3: RuntimeController 분리 (2.1)
- [ ] Week 4 Mon-Tue: 하드코딩 제거 (2.2)
- [ ] Week 4 Wed-Fri: 타입 힌트 추가 (2.3)

**Week 5-8: Phase 3**
- [ ] Week 5-6: GUI 분리 (3.1)
- [ ] Week 7: 설정 통합 (3.2)
- [ ] Week 8: 테스트 확대 (3.3)

---

## 🎯 최종 목표 및 성공 지표

### 코드 품질 지표

| 지표 | 현재 | Phase 1 | Phase 2 | Phase 3 | 목표 |
|------|------|---------|---------|---------|------|
| **전체 평가** | 5.6/10 | 6.2/10 | 7.0/10 | **8.0/10** | 8.0+ |
| 코드 중복 | 5/10 | **9/10** | 9/10 | 9/10 | 9+ |
| 복잡도 | 6/10 | 6/10 | 7/10 | **8/10** | 8+ |
| 에러 처리 | 5/10 | **8/10** | 8/10 | 8/10 | 8+ |
| 테스트 커버리지 | 3/10 | 3/10 | 4/10 | **7/10** | 7+ |
| 문서화 | 6/10 | 6/10 | **8/10** | 9/10 | 8+ |

### 정량적 지표

| 항목 | 현재 | 목표 |
|------|------|------|
| 총 코드 라인 | ~6,500 | ~6,000 (중복 제거) |
| 테스트 커버리지 | <2% | 30%+ |
| 타입 힌트 비율 | ~10% | 60%+ |
| 평균 함수 길이 | ~45줄 | ~30줄 |
| 최대 클래스 크기 | 1,134줄 | <400줄 |

---

## 📝 변경 이력

| 날짜 | 버전 | 내용 |
|------|------|------|
| 2025-12-05 | 1.0 | 초안 작성 - 3단계 리팩토링 로드맵 |
