# PROMPTS.md

## 프로젝트 정보

- 과제명: AI Coding Tools Practice
- 사용한 AI 코딩 툴: Codex
- 대상 논문: FoundPose: Unseen Object Pose Estimation with Foundation Features
- 공식 구현: https://github.com/facebookresearch/foundpose
- 제출 저장소: https://github.com/sneisial18-design/capstone-foundpose-lmo-experiment/tree/main/foundpose
- 실험 데이터셋: BOP LM-O(lmo)

## 작업 목표

FoundPose 공식 오픈소스 구현을 대상으로 코드 구조, 실행 절차, 실험 결과를 분석하고 과제 제출용 보고서를 작성하였다. 또한 PPT에 포함된 실행 캡처를 확인하여 실제 로컬 실행 결과와 BOP evaluation 점수를 보고서에 반영하였다.

## 프롬프트 로그

### 1. 과제 요구사항 파악

**User prompt**

```text
과제를 할건데 도와줘
```

**Codex response / action**

- 과제 주제, 제출 형식, 현재 진행 상황을 확인해야 한다고 안내하였다.
- 이후 사용자가 과제 안내 이미지와 요구사항을 제공하였다.

### 2. 제출물 범위 확인

**User prompt**

```text
보고서 및 프롬프트 로그를 만들어야해
```

**Codex response / action**

- 과제 안내에서 보고서와 `PROMPTS.md` 제출이 필요하다는 점을 확인하였다.
- 보고서에는 매뉴얼, 분석 내용, 실행 결과, AI 코딩 툴 사용 내역을 포함해야 한다고 판단하였다.

### 3. 예시 보고서 기반 형식 분석

**User prompt**

```text
보고서 예시를 줄게 일단 보고서 부터 만들어줘
내 타겟논문은 https://github.com/sneisial18-design/capstone-foundpose-lmo-experiment/tree/main/foundpose 여기 보면 알 수 있어
```

**Codex response / action**

- 사용자가 제공한 예시 문서들을 확인하였다.
- 예시 보고서의 공통 구조를 `개요 -> 코드 구조 분석 -> 순차적 구현 매뉴얼 -> 오류 및 프롬프트 기반 해결 과정 -> 실험 결과 -> 결론`으로 정리하였다.
- GitHub 저장소의 `foundpose` 폴더와 로컬 프로젝트를 확인하여 대상 논문이 FoundPose임을 파악하였다.

### 4. FoundPose 저장소 구조 분석

**Codex internal task**

```text
README.md, configs, scripts, utils, outputs 폴더를 확인해서 FoundPose 구현 흐름을 분석한다.
```

**Codex response / action**

- `README.md`에서 논문 정보, 환경 설정, BOP 데이터셋 구성, 실행 명령을 확인하였다.
- `configs/gen_templates/lmo.json`, `configs/gen_repre/lmo.json`, `configs/infer/lmo.json`의 주요 파라미터를 정리하였다.
- `scripts/gen_templates.py`, `scripts/gen_repre.py`, `scripts/infer.py`, `scripts/prepare_bop_submission.py`의 역할을 보고서에 반영하였다.

### 5. 실행 산출물 분석

**Codex internal task**

```text
outputs 폴더에 생성된 템플릿, object representation, inference 결과, BOP CSV 파일을 확인한다.
```

**Codex response / action**

- LM-O 객체 ID `1, 5, 6, 8, 9, 10, 11, 12`에 대해 템플릿과 representation이 생성되어 있음을 확인하였다.
- 각 객체마다 RGB, mask, depth 템플릿이 각각 798개씩 생성된 것을 계산하였다.
- `outputs/results/foundpose_lmo-test.csv` 기준 pose prediction이 1151개임을 확인하였다.
- 평균 score는 약 `0.2504`, 평균 time은 약 `3.0211초`로 정리하였다.

### 6. 보고서 초안 생성

**Codex internal task**

```text
예시 보고서 형식에 맞춰 FoundPose 구현 보고서를 Markdown과 Word 형식으로 작성한다.
```

**Codex response / action**

- `reports/FoundPose_AI코딩툴_구현보고서.md`를 작성하였다.
- `pandoc`을 사용하여 `reports/FoundPose_AI코딩툴_구현보고서.docx`로 변환하였다.
- `textutil`로 Word 문서에 내용이 정상적으로 들어갔는지 확인하였다.

### 7. PPT 실행 캡처 확인

**User prompt**

```text
여기 마지막 부분에 내가 코드를 실행한 사진들이 있어 이것도 확인
```

**Codex response / action**

- `FoundPose_Unseen_Object_Pose_Estimation_with_Foundation_Features-4.pptx`를 압축 해제하여 마지막 슬라이드의 텍스트와 이미지 관계를 확인하였다.
- 31장에는 온라인 추론 로그와 6D pose 추정 시각화가 포함되어 있음을 확인하였다.
- 32장에는 BOP evaluation 최종 점수 화면이 포함되어 있음을 확인하였다.
- 33장에는 결과 분석 표가 포함되어 있으며, AR 39.53%와 평균 추론 시간 약 3.04초가 정리되어 있음을 확인하였다.

### 8. 평가 결과 보강

**Codex internal task**

```text
PPT 실행 캡처의 FINAL SCORES를 보고서 결과 장에 반영한다.
```

**Codex response / action**

- PPT 이미지에서 다음 값을 확인하였다.
  - `bop19_average_recall_mssd = 0.2591`
  - `bop19_average_recall_mspd = 0.5315`
  - `bop19_average_recall = 0.3953`
  - `bop19_average_time_per_image = 3.0486초`
- 기존 보고서의 "AR 미확인" 문장을 제거하고, 로컬 재현 AR 약 39.53%를 반영하였다.
- PPT에서 추출한 실행 캡처 3장을 `reports/assets/`에 저장하고 Word 문서에 포함하였다.

### 9. GitHub 업로드 요청

**User prompt**

```text
내 깃허브에 저걸 prompt.md로 업로드 가능?
```

**Codex response / action**

- 과제 안내의 파일명 요구사항에 맞춰 `foundpose/PROMPTS.md`로 업로드하기로 판단하였다.
- GitHub 저장소 `sneisial18-design/capstone-foundpose-lmo-experiment` 접근 권한을 확인하였다.
- 기존 `foundpose/PROMPTS.md` 또는 `foundpose/prompt.md` 파일이 없는 것을 확인하였다.
- 본 파일을 작성한 뒤 GitHub `main` 브랜치에 업로드하였다.

## 주요 분석 결과 요약

### 코드 구조

- `scripts/gen_templates.py`: LM-O 3D 모델을 여러 viewpoint와 in-plane rotation으로 렌더링한다.
- `scripts/gen_repre.py`: 렌더링 템플릿에서 DINOv2 feature를 추출하고 PCA, clustering, TF-IDF 기반 object representation을 생성한다.
- `scripts/infer.py`: 테스트 이미지와 object representation을 매칭하고 PnP/RANSAC으로 6D pose를 추정한다.
- `scripts/prepare_bop_submission.py`: 객체별 `estimated-poses.json`을 BOP 제출 형식 CSV로 변환한다.

### 실행 결과

- LM-O 8개 객체에 대해 각 객체별 798개 템플릿이 생성되었다.
- 총 1151개의 pose prediction이 생성되었다.
- PPT 실행 캡처 기준 BOP Average Recall은 약 39.53%이다.
- 평균 추론 시간은 약 3.0486초이다.

### 보고서 산출물

- `reports/FoundPose_AI코딩툴_구현보고서.md`
- `reports/FoundPose_AI코딩툴_구현보고서.docx`
- `reports/assets/inference_log.png`
- `reports/assets/pose_visualization.png`
- `reports/assets/evaluation_scores.png`

## 정리

Codex를 이용하여 FoundPose 오픈소스 구현의 구조를 분석하고, LM-O 데이터셋 기반 실행 흐름과 산출물을 보고서로 정리하였다. 추가로 PPT 실행 캡처를 확인하여 실제 BOP evaluation 결과를 반영했으며, 최종적으로 과제 제출용 프롬프트 로그를 `PROMPTS.md`로 작성하였다.
