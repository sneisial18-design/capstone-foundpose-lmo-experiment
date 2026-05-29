# AI Coding Tools Practice 보고서

주제: FoundPose: Unseen Object Pose Estimation with Foundation Features  
과목명: 스마트팩토리캡스톤디자인1  
작성자: 수정 필요  
학번: 수정 필요  
GitHub 주소: https://github.com/sneisial18-design/capstone-foundpose-lmo-experiment/tree/main/foundpose  
사용한 AI 코딩 툴: Codex

## 1. 개요

이번 과제에서는 ECCV 2024 논문인 **FoundPose: Unseen Object Pose Estimation with Foundation Features**의 공식 오픈소스 구현을 대상으로, AI 코딩 툴을 이용해 코드 구조를 분석하고 LM-O 데이터셋 기반 실행 과정을 정리하였다. FoundPose는 단일 RGB 이미지에서 물체의 6DoF pose를 추정하는 방법으로, 새로운 물체에 대해 별도의 학습이나 fine-tuning 없이 3D CAD 모델과 foundation model feature를 활용한다.

FoundPose의 핵심은 DINOv2 feature를 이용하여 입력 이미지와 미리 렌더링한 물체 템플릿 사이의 2D-3D correspondence를 찾고, 이를 PnP RANSAC으로 변환하여 물체의 회전 행렬과 이동 벡터를 추정하는 것이다. 본 실습에서는 LM-O 데이터셋의 8개 객체에 대해 템플릿 생성, 객체 표현 생성, 추론, BOP 제출 형식 CSV 생성까지의 흐름을 재현하였다.

| 항목 | 내용 |
|---|---|
| 논문 제목 | FoundPose: Unseen Object Pose Estimation with Foundation Features |
| 발표 | ECCV 2024 |
| 저자 | Evin Pinar Ornek, Yann Labbe, Bugra Tekin, Lingni Ma, Cem Keskin, Christian Forster, Tomas Hodan |
| 공식 코드 | https://github.com/facebookresearch/foundpose |
| 프로젝트 페이지 | https://evinpinar.github.io/foundpose/ |
| 실습 데이터셋 | BOP LM-O(lmo) |
| 주요 모델 | DINOv2 ViT-S/14 register 모델 |
| 주요 산출물 | template 이미지, object representation, estimated pose JSON, BOP CSV |

## 2. 논문 및 알고리즘 분석

### 2.1 문제 정의

6DoF object pose estimation은 이미지 속 물체가 카메라 좌표계에서 어떤 위치와 방향을 갖는지 추정하는 문제이다. 6DoF는 3차원 회전과 3차원 이동을 의미하며, 로봇 조작, 증강현실, 산업 자동화 검사 등에서 중요한 역할을 한다.

기존 방법은 특정 객체나 데이터셋에 대해 많은 학습 데이터가 필요하거나, 새 물체를 사용할 때 재학습이 필요한 경우가 많다. FoundPose는 이 문제를 줄이기 위해 self-supervised 방식으로 사전 학습된 DINOv2 feature를 사용한다. 즉, 물체별 학습 데이터 없이도 3D 모델만 있으면 새로운 물체를 빠르게 등록하고 pose를 추정할 수 있다.

### 2.2 전체 파이프라인

FoundPose의 실행 흐름은 다음 네 단계로 정리할 수 있다.

1. **템플릿 렌더링**

   LM-O 객체의 3D 모델을 여러 viewpoint와 in-plane rotation으로 렌더링한다. 각 템플릿은 RGB 이미지, mask, depth map, camera parameter로 저장된다.

2. **객체 표현 생성**

   렌더링된 템플릿에서 DINOv2 patch descriptor를 추출한다. 추출된 feature는 PCA로 차원을 축소하고 clustering을 통해 bag-of-words 또는 TF-IDF 기반 template descriptor로 구성된다.

3. **추론**

   테스트 이미지에서 CNOS-FastSAM detection mask를 사용해 객체 crop을 만든다. 이후 DINOv2 feature를 추출하고, TF-IDF 방식으로 상위 템플릿을 검색한다. 검색된 템플릿과 입력 이미지 사이에서 cyclic buddies matching을 수행하여 대응점을 찾는다.

4. **Pose 계산 및 제출 파일 생성**

   2D 이미지 좌표와 3D 모델 좌표의 correspondence를 OpenCV PnP RANSAC에 입력하여 회전 행렬 `R`과 이동 벡터 `t`를 계산한다. 최종 결과는 BOP benchmark 형식의 CSV 파일로 저장된다.

### 2.3 FoundPose의 특징

| 특징 | 설명 |
|---|---|
| Training-free | 새 객체에 대한 task-specific training 없이 실행 가능 |
| Foundation feature 활용 | DINOv2의 일반화된 patch descriptor를 이용 |
| Model-based pose estimation | RGB 이미지와 3D CAD 모델을 함께 사용 |
| Template retrieval | 모든 템플릿을 brute-force로 비교하지 않고 TF-IDF 기반 검색을 적용 |
| PnP 기반 pose 산출 | 2D-3D correspondence를 이용해 최종 6DoF pose 계산 |

## 3. 코드 구조 분석

본 실습 저장소는 공식 `facebookresearch/foundpose` 구현을 기반으로 하며, `external/bop_toolkit`과 `external/dinov2`를 함께 사용한다. 주요 파일 구조는 다음과 같다.

```text
foundpose/
├── README.md                         # 논문, 설치, 실행 방법 설명
├── conda_foundpose_mps.yaml          # macOS MPS 환경 설정
├── conda_foundpose_gpu.yaml          # CUDA GPU 환경 설정
├── env_vars.sh                       # REPO_PATH, BOP_PATH, PYTHONPATH 설정
├── configs/
│   ├── gen_templates/lmo.json        # LM-O 템플릿 생성 설정
│   ├── gen_repre/lmo.json            # LM-O 객체 표현 생성 설정
│   └── infer/lmo.json                # LM-O 추론 설정
├── scripts/
│   ├── gen_templates.py              # 3D 모델 기반 템플릿 렌더링
│   ├── gen_repre.py                  # DINOv2 feature 기반 object representation 생성
│   ├── infer.py                      # 템플릿 검색, correspondence, PnP 추론
│   └── prepare_bop_submission.py     # estimated pose JSON을 BOP CSV로 변환
├── utils/
│   ├── dinov2_utils.py               # DINOv2 feature extractor
│   ├── corresp_util.py               # 2D-3D correspondence 생성
│   ├── pnp_util.py                   # PnP pose estimation
│   ├── repre_util.py                 # 객체 표현 관련 유틸리티
│   └── renderer.py                   # 템플릿 렌더링 유틸리티
├── external/
│   ├── bop_toolkit/                  # BOP 데이터셋/평가 도구
│   └── dinov2/                       # DINOv2 backbone 코드
└── outputs/
    ├── templates/v1/lmo/             # 렌더링된 LM-O 템플릿
    ├── object_repre/lmo/v1/          # 객체별 representation
    ├── inference/lmo_v1/             # 객체별 추론 결과
    └── results/                      # BOP 제출 형식 CSV
```

### 3.1 주요 설정 파일

| 파일 | 주요 설정 |
|---|---|
| `configs/gen_templates/lmo.json` | LM-O 객체 ID, viewpoint 수, in-plane rotation 수, crop size, lighting |
| `configs/gen_repre/lmo.json` | DINOv2 extractor, PCA 256차원, cluster 2048개, TF-IDF descriptor |
| `configs/infer/lmo.json` | detection 사용 여부, top-5 template matching, cyclic buddies matching, PnP RANSAC |
| `env_vars.sh` | `REPO_PATH=/Users/sneisial/foundpose`, `BOP_PATH=/Users/sneisial/bop_datasets` |

### 3.2 주요 스크립트 역할

| 파일 | 역할 |
|---|---|
| `scripts/gen_templates.py` | 3D mesh를 여러 시점에서 렌더링하여 RGB, mask, depth 템플릿 생성 |
| `scripts/gen_repre.py` | 템플릿 이미지에서 DINOv2 feature를 추출하고 PCA/cluster/TF-IDF 표현 생성 |
| `scripts/infer.py` | 입력 이미지와 템플릿 표현을 매칭하여 pose 추정 |
| `scripts/prepare_bop_submission.py` | 객체별 `estimated-poses.json`을 하나의 BOP CSV로 통합 |
| `utils/dinov2_utils.py` | DINOv2 모델을 로드하고 patch token feature map 생성 |
| `utils/corresp_util.py` | query feature와 template feature 사이의 correspondence 계산 |
| `utils/pnp_util.py` | correspondence를 이용한 PnP RANSAC pose 계산 |

## 4. 순차적 구현 매뉴얼

### 4.1 저장소 준비

공식 저장소는 submodule을 포함하므로 다음과 같이 clone해야 한다.

```bash
git clone --recurse-submodules https://github.com/facebookresearch/foundpose
cd foundpose
```

이미 clone한 경우에는 다음 명령으로 submodule을 받을 수 있다.

```bash
git submodule update --init --recursive
```

### 4.2 Conda 환경 생성

macOS에서 Apple Silicon 또는 MPS backend를 사용할 경우 다음 환경 파일을 사용한다.

```bash
conda env create -f conda_foundpose_mps.yaml
conda activate foundpose_mps
```

CUDA GPU 환경에서는 다음 파일을 사용한다.

```bash
conda env create -f conda_foundpose_gpu.yaml
conda activate foundpose_gpu
```

### 4.3 환경 변수 설정

FoundPose는 `REPO_PATH`, `BOP_PATH`, `PYTHONPATH` 설정이 필요하다. 본 실습에서는 다음과 같이 설정하였다.

```bash
export REPO_PATH=/Users/sneisial/foundpose
export BOP_PATH=/Users/sneisial/bop_datasets
export PYTHONPATH=$REPO_PATH:$REPO_PATH/external/bop_toolkit:$REPO_PATH/external/dinov2:$PYTHONPATH
```

이 설정은 `env_vars.sh`에 저장되어 있으며, conda 활성화 스크립트에 등록하면 매번 자동으로 적용할 수 있다.

### 4.4 BOP 데이터셋 및 detection 준비

LM-O 실험에는 BOP 데이터셋의 `lmo` 폴더가 필요하다. 최소한 `camera.json`, `models`, `models_eval`, `test` 폴더가 있어야 한다. 또한 `configs/infer/lmo.json`에서 `use_detections=true`이므로 BOP 2023 Task 4의 CNOS-FastSAM detection 결과도 필요하다.

예상 구조는 다음과 같다.

```text
bop_datasets/
├── lmo/
│   ├── camera.json
│   ├── models/
│   ├── models_eval/
│   └── test/
└── detections/
    └── cnos-fastsam/
        └── cnos-fastsam_lmo_test.json
```

BOP toolkit 설정 파일에서는 다음 경로를 사용하도록 구성되어 있다.

```text
datasets_path = /Users/sneisial/bop_datasets
results_path  = /Users/sneisial/foundpose/outputs/results
eval_path     = /Users/sneisial/foundpose/outputs/eval
output_path   = /Users/sneisial/foundpose/outputs
```

### 4.5 템플릿 생성

LM-O 객체 템플릿은 다음 명령으로 생성한다.

```bash
python scripts/gen_templates.py --opts-path configs/gen_templates/lmo.json
```

주요 설정은 다음과 같다.

| 설정 | 값 |
|---|---|
| object dataset | `lmo` |
| object ids | 1, 5, 6, 8, 9, 10, 11, 12 |
| viewpoint 수 | 최소 57 |
| in-plane rotation 수 | 14 |
| crop size | 420 x 420 |
| background | black |
| light type | multi directional |

현재 저장소의 산출물을 확인한 결과, 각 객체마다 `57 x 14 = 798`개의 템플릿이 생성되었다. 각 템플릿은 RGB, mask, depth로 저장된다.

| 객체 ID | RGB | Mask | Depth |
|---:|---:|---:|---:|
| 1 | 798 | 798 | 798 |
| 5 | 798 | 798 | 798 |
| 6 | 798 | 798 | 798 |
| 8 | 798 | 798 | 798 |
| 9 | 798 | 798 | 798 |
| 10 | 798 | 798 | 798 |
| 11 | 798 | 798 | 798 |
| 12 | 798 | 798 | 798 |

### 4.6 객체 표현 생성

템플릿 생성 후 DINOv2 feature 기반 object representation을 생성한다.

```bash
python scripts/gen_repre.py --opts-path configs/gen_repre/lmo.json
```

주요 설정은 다음과 같다.

| 설정 | 값 |
|---|---|
| extractor | `dinov2_version=vits14-reg_stride=14_facet=token_layer=9_logbin=0_norm=1` |
| grid cell size | 14.0 |
| PCA | 사용 |
| PCA components | 256 |
| feature clustering | 사용 |
| cluster 수 | 2048 |
| template descriptor | TF-IDF |

실행 결과는 `outputs/object_repre/lmo/v1/{object_id}/` 아래에 저장된다. 각 객체 폴더에는 `config.json`과 `repre.pth`가 생성된다.

### 4.7 추론 실행

LM-O 테스트 이미지에 대해 pose estimation을 수행한다.

```bash
python scripts/infer.py --opts-path configs/infer/lmo.json
```

추론 단계의 핵심 설정은 다음과 같다.

| 설정 | 값 |
|---|---|
| detection 사용 | true |
| crop size | 420 x 420 |
| template matching | TF-IDF |
| top-N templates | 5 |
| feature matching | cyclic buddies |
| top-K buddies | 300 |
| PnP | OpenCV |
| RANSAC iteration | 400 |
| inlier threshold | 10.0 |
| 최종 pose 선택 | best coarse |
| visualization | true |

추론 결과는 객체별 `estimated-poses.json`과 시각화 이미지로 저장된다.

| 객체 ID | 추론 결과 수 |
|---:|---:|
| 1 | 139 |
| 5 | 180 |
| 6 | 90 |
| 8 | 189 |
| 9 | 168 |
| 10 | 139 |
| 11 | 103 |
| 12 | 143 |
| 합계 | 1151 |

### 4.8 BOP 제출 파일 생성

객체별 `estimated-poses.json`을 하나의 CSV로 합치기 위해 다음 스크립트를 실행한다.

```bash
python scripts/prepare_bop_submission.py
```

최종 CSV는 다음 형식으로 저장된다.

```text
scene_id,im_id,obj_id,score,R,t,time
```

현재 저장소에는 다음 파일이 생성되어 있다.

```text
outputs/inference/lmo_v1/coarse_lmo-estimated-poses.csv
outputs/results/coarse_lmo-estimated-poses.csv
outputs/results/foundpose_lmo-test.csv
outputs/results/foundpose-lmo-test.csv
```

`outputs/results/foundpose_lmo-test.csv` 기준으로 확인한 결과, header를 제외한 pose prediction은 총 1151개이며 평균 score는 약 0.2504, 평균 실행 시간은 약 3.0211초로 계산되었다.

## 5. 발생 오류 및 AI 코딩 툴 활용 내용

이번 과제에서는 Codex를 이용해 README, 설정 파일, 출력 폴더, 코드 구조를 분석하고 보고서 형태로 정리하였다. 대표적인 프롬프트와 해결 내용은 다음과 같다.

### 5.1 저장소 구조 파악

**입력한 프롬프트**

```text
보고서 예시를 줄게. 일단 보고서부터 만들어줘.
내 타겟논문은 GitHub 저장소를 보면 알 수 있어.
```

**분석 내용**

Codex가 `README.md`, `configs/`, `scripts/`, `utils/`, `outputs/` 구조를 확인하였다. 이를 통해 대상 논문이 FoundPose이며, 공식 구현이 LM-O 데이터셋을 대상으로 템플릿 생성, representation 생성, 추론, BOP 제출 파일 생성 단계를 제공한다는 것을 파악하였다.

### 5.2 환경 변수 문제

**발생 가능 오류**

```text
ModuleNotFoundError: No module named 'bop_toolkit_lib'
ModuleNotFoundError: No module named 'dinov2'
```

**원인**

`external/bop_toolkit`과 `external/dinov2`가 `PYTHONPATH`에 포함되지 않으면 FoundPose 내부 모듈에서 import 오류가 발생한다.

**해결 방법**

`env_vars.sh`에 다음 경로를 등록한다.

```bash
export PYTHONPATH=$REPO_PATH:$REPO_PATH/external/bop_toolkit:$REPO_PATH/external/dinov2:$PYTHONPATH
```

### 5.3 BOP 데이터셋 경로 문제

**발생 가능 오류**

```text
FileNotFoundError: camera.json 또는 models 폴더를 찾을 수 없음
```

**원인**

BOP toolkit은 `BOP_PATH` 또는 `external/bop_toolkit/bop_toolkit_lib/config.py`의 `datasets_path`를 기준으로 데이터셋을 찾는다. 실제 LM-O 데이터셋 경로와 설정 경로가 다르면 오류가 발생한다.

**해결 방법**

`BOP_PATH=/Users/sneisial/bop_datasets`로 설정하고, `lmo` 데이터셋과 detection 파일을 해당 경로에 맞게 배치한다.

### 5.4 템플릿 출력 폴더 중복 문제

**발생 가능 오류**

```text
ValueError: Output directory already exists
```

**원인**

`scripts/gen_templates.py`는 출력 폴더가 이미 존재하고 `overwrite=false`이면 기존 결과를 덮어쓰지 않는다.

**해결 방법**

반복 실험에서는 설정의 `overwrite` 옵션을 true로 두거나, 기존 출력 폴더를 백업한 뒤 새로 생성한다. 현재 산출물의 `config.json`에는 `overwrite=true`가 기록되어 있다.

### 5.5 DINOv2 weight 로딩 문제

**발생 가능 오류**

```text
RuntimeError 또는 network error while loading pretrained DINOv2 weights
```

**원인**

`utils/dinov2_utils.py`는 DINOv2 backbone을 `pretrained=True`로 생성한다. 실행 환경에 인터넷 접근이 없거나 weight cache가 없으면 모델 로딩이 실패할 수 있다.

**해결 방법**

처음 실행할 때는 인터넷이 가능한 환경에서 DINOv2 weight를 내려받거나, 이미 내려받은 weight cache를 같은 환경에 유지한다.

### 5.6 BOP 제출 파일명 문제

**발생 가능 혼동**

```text
coarse_lmo-estimated-poses.csv
foundpose_lmo-test.csv
foundpose-lmo-test.csv
```

**원인**

`prepare_bop_submission.py`는 기본적으로 `coarse_lmo-estimated-poses.csv`를 생성한다. BOP 평가 또는 제출 스크립트에서 요구하는 파일명이 다르면 같은 내용을 다른 이름으로 복사해야 한다.

**해결 방법**

최종 제출 시에는 평가 스크립트가 요구하는 이름을 확인하고, `outputs/results/`에 해당 이름으로 CSV를 배치한다. 현재 저장소에는 세 파일이 모두 존재한다.

## 6. 실험 결과 정리

본 실습에서 확인한 산출물은 다음과 같다.

| 산출 단계 | 결과 |
|---|---|
| 템플릿 생성 | 8개 객체 각각 RGB/mask/depth 798개 생성 |
| 객체 표현 생성 | 8개 객체 각각 `repre.pth` 생성 |
| 추론 | 객체별 `estimated-poses.json` 및 시각화 PNG 생성 |
| BOP CSV | `outputs/results/foundpose_lmo-test.csv` 생성 |
| 총 pose prediction 수 | 1151개 |
| 평균 score | 약 0.2504 |
| 평균 time | 약 3.0211초 |

또한 PPT 마지막 부분의 실행 캡처를 확인한 결과, BOP evaluation까지 수행된 로그가 포함되어 있었다. 캡처 화면의 `FINAL SCORES` 기준 정량 결과는 다음과 같다.

| 평가 항목 | 실행 결과 |
|---|---:|
| `bop19_average_recall_mssd` | 0.2591 |
| `bop19_average_recall_mspd` | 0.5315 |
| `bop19_average_recall` | 0.3953 |
| Average Recall(%) | 약 39.53% |
| `bop19_average_time_per_image` | 약 3.0486초 |
| Evaluation time | 약 7.06초 |

PPT의 결과 분석 슬라이드에서는 논문 공식 수치를 AR 39.6%로 정리하고, 로컬 재현 수치를 AR 39.53%로 제시하였다. 오차는 약 -0.07%p로 매우 작다. 추론 속도는 논문 기준 1.7초보다 느린 약 3.04초로 나타났는데, 이는 실험 장치 차이, GPU/CPU 사용 여부, 코어 할당, macOS 로컬 환경의 부동소수점 연산 차이 등이 영향을 준 것으로 해석할 수 있다.

### 6.1 실행 화면 증빙

아래 이미지는 제공된 PPT의 마지막 부분에서 추출한 실행 캡처이다.

![온라인 추론 로그](assets/inference_log.png)

위 화면은 `python scripts/infer.py --opts-path configs/infer/lmo.json` 실행 중 객체 1에 대해 pose를 추정하는 로그이다. DINOv2 feature extraction, template matching, correspondence 생성, coarse pose 계산, pose evaluation 시간이 순서대로 출력된다.

![6D pose 추정 시각화](assets/pose_visualization.png)

위 화면은 `outputs/inference/lmo_v1/12/` 아래의 추론 시각화 결과이다. 입력 이미지 위의 추정 pose overlay, 검색된 template, feature correspondence가 함께 표시되어 있어 FoundPose의 2D-3D matching 기반 추론 과정을 확인할 수 있다.

![BOP evaluation final scores](assets/evaluation_scores.png)

위 화면은 BOP evaluation의 최종 점수 로그이다. `bop19_average_recall=0.395294...`와 `bop19_average_time_per_image=3.04856...`가 확인된다.

## 7. 결론

FoundPose는 새로운 객체에 대해 별도 학습 없이 DINOv2 foundation feature와 CAD 모델 기반 템플릿을 이용해 6DoF pose를 추정하는 방법이다. 본 과제에서는 AI 코딩 툴을 이용하여 공식 오픈소스의 구조를 분석하고, LM-O 데이터셋을 대상으로 템플릿 생성, 객체 표현 생성, 추론, BOP 제출 파일 생성 과정을 정리하였다.

실습 결과 8개 LM-O 객체에 대해 각 798개의 렌더링 템플릿이 생성되었고, DINOv2 feature 기반 representation과 총 1151개의 pose prediction CSV가 생성된 것을 확인하였다. PPT 실행 캡처 기준 BOP evaluation 결과는 Average Recall 약 39.53%로, 정리된 논문 공식 수치 39.6%와 거의 동일한 수준이다.

이번 실습을 통해 FoundPose의 핵심 구현은 단순한 end-to-end 학습 코드가 아니라, 3D 모델 렌더링, foundation feature 추출, template retrieval, feature correspondence, PnP pose estimation이 단계적으로 연결된 파이프라인이라는 점을 확인하였다. AI 코딩 툴은 특히 복잡한 저장소 구조 파악, 설정 파일 해석, 산출물 요약, 보고서화 과정에서 효과적으로 활용되었다.

## 참고 자료

1. FoundPose 공식 GitHub: https://github.com/facebookresearch/foundpose
2. FoundPose 프로젝트 페이지: https://evinpinar.github.io/foundpose/
3. FoundPose arXiv: https://arxiv.org/abs/2311.18809
4. BOP 데이터셋: https://bop.felk.cvut.cz/datasets/
5. DINOv2 GitHub: https://github.com/facebookresearch/dinov2
