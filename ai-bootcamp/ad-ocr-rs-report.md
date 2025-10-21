---
title: 'AI 실전 학습 경진대회 보고서 (AD / OCR / RS)'
layout: page
icon: fas fa-chart-line
permalink: /ai-bootcamp/ad-ocr-rs-full-report/
toc: true
tags:
  - Anomaly Detection
  - OCR
  - Recommender System
  - AD
  - 이상탐지
  - 문자인식
  - 추천시스템
  - AI부트캠프
  - 패스트캠퍼스
  - 업스테이지패스트캠퍼스
  - UpstageAILab
---

## 1주차 – AD/OCR/RS 온라인 강의

**강의 주제**: AD/OCR/RS 개념 및 다양한 방법을 학습하고 실전 적용 사례를 파악한다.

### 📌 Anomaly Detection – 중요 개념 및 원리 (7)

1. **정의·문제 설정**: 정상 패턴에서 벗어난 샘플 탐지. 극심한 불균형·라벨 부족 환경.
2. **Autoencoder 계열**: AE/VAE/MemAE 재구성 오차 기반 이상 판단.
3. **Isolation Forest**: 무작위 분할 트리로 고립 길이로 이상치 측정.
4. **통계적 방법**: Z-score, IQR, MAD; 다변량은 Robust Covariance.
5. **시계열**: STL/ARIMA/Prophet, LSTM/TCN; 계절·추세 제거 후 탐지.
6. **평가**: PR-AUC, F1, Recall@k 중심. ROC-AUC 단독 사용 지양.
7. **운영**: 동적 임계값, 알림 규칙, 인간 검증 루프(HITL).

#### 느낀 점 & 적용 가능성

- 전처리·스케일링·드리프트 관리가 모델만큼 중요.
- 제조/보안/금융 등 실시간 모니터링에 즉시 유효.

#### 추가 학습 계획

- AE vs IF 성능 비교, 비용 기반 임계값 최적화.
- 시계열 이상탐지용 TCN 실험 및 drift 모니터링.

### 📌 OCR – 중요 개념 및 원리 (7)

1. **파이프라인**: 검출 → 인식 → 후처리.
2. **전처리**: 이진화·노이즈 제거·왜곡 보정·정규화.
3. **CRNN+CTC**: CNN 특징 + BiLSTM 시퀀스 + CTC loss 정렬.
4. **Transformer 계열**: TrOCR/SATRN/ViT 기반.
5. **다국어/한글**: 자모 결합·복합자·띄어쓰기 처리 이슈.
6. **라벨링/증강**: bbox/라인/문자 라벨, synthetic data.
7. **평가**: CER/WER/단어정확도/라인정확도.

#### 느낀 점 & 적용 가능성

- 검출 미스는 인식 개선으로 대체 어려움 → 검출 품질이 병목.
- 문서 자동화, 영수증 처리, 차량번호판, OCR+LLM까지 확장 가능.

#### 추가 학습 계획

- PaddleOCR/TrOCR 파인튜닝, 한글 특화 augment.
- 사전·언어모델 기반 후처리 비교.

### 📌 Recommender System – 중요 개념 및 원리 (7)

1. **협업 필터링**: 유사 사용자/아이템 기반 추천.
2. **행렬분해(MF/BPR)**: 잠재요인 내적으로 선호도 예측.
3. **Neural CF/Two-tower**: 임베딩+MLP/대조학습으로 클릭·구매 예측.
4. **콘텐츠/지식 그래프**: 메타데이터·KG로 콜드스타트 완화.
5. **후보생성→재랭킹**: Recall→Rerank, 다양성/공정성 제약.
6. **평가**: AUC/NDCG/Recall@k vs 온라인 A/B.
7. **운영**: 피처 스토어·실시간 피드·피드백 루프.

#### 느낀 점 & 적용 가능성

- 데이터 파이프라인·로그 설계가 품질을 좌우.
- 이커머스/콘텐츠/광고/뉴스 등 광범위 적용.

#### 추가 학습 계획

- LightFM/NCF 비교, 재랭킹(XGBoost/GBDT) 실험.
- KG/LLM 보조, 장기효과 메트릭 도입.

## 2주차 – 데이터 분석(EDA) 정리

### 🔎 AD – 데이터 탐색 및 이해

#### 데이터 구성 요소

- **필드**: timestamp, sensor_id, value, line_id, status, temperature, vibration
- **샘플 수**: 200,384건 (기간: 2025-09-01 ~ 2025-10-10)
- **결측/이상치**: 결측 2.1% (주로 sensor_id=5, vibration), 라벨 이상 이벤트 0.7%
- **주기/계절성**: 일별 24h 주기, 주말 사용량 18% 감소

#### 변수 설명

**timestamp**(s), **sensor_id**(1~10), **value**(0~100, 정규화 필요),  
**temperature**(℃, 평균 63.2, 표준편차 6.8), **vibration**(g-RMS),  
**status**(0/1).

*파생*: rolling mean(10분), std(10분), lag1~3, derivative, zscore, hour-of-day, day-of-week.

#### 탐색적 분석 & 관계

| 분석 항목 | 결과 요약 | 메모 |
|---|---|---|
| 분포/분산 | value, vibration long-tail; temperature는 근사 정규 | 로그 변환 및 robust scaler 적용 |
| 상관/공분산 | temperature–vibration r=0.73 | 공동 이상 가능성 → 다변량 모델 적합 |
| 시간 패턴 | 02~05시 잡음 증가, 월초 설비 점검 후 노이즈 감소 | 이 구간 필터링/마스킹 |
| 이상치 후보 | vibration 상위 0.5% spike, 특정 라인(line_id=3)에 집중 | 실제 경고 로그와 82% 일치 |

#### 시각화 결과 & 인사이트

- Boxplot: vibration 극단치가 sensor_id 3,7에서 현저
- Heatmap: temp–vib 강한 양의 상관
- Line plot: 10/03에 레벨 쉬프트 및 단기 spike

#### 특징 & 추후 모델링 계획

- 후보: AE/VAE, Isolation Forest, One-Class SVM
- 임계값: validation 기준 재구성 오차 상위 1% → anomaly
- 목표: Precision ≥ 0.92, Recall ≥ 0.85
- 운영: drift 모니터링, 1일 단위 rolling 재학습

### 🔎 OCR – 데이터 탐색 및 이해

#### 데이터 구성 요소

- **이미지**: 28,600장, 해상도 1024×768, 컬러 3채널
- **라벨**: 단어/라인 bbox + 텍스트, 총 410k 토큰
- **언어**: 한글 68%, 영문 27%, 숫자/기타 5%
- **클래스 불균형**: 희귀 한자/특수기호 1% 미만

#### 전처리·증강

- 이진화(Adaptive), 디노이징(Non-local Means), 기울임 보정(Deskew)
- 리사이즈 32×128 패딩, 정규화
- 증강: blur, affine, perspective, jpeg artifacts, gaussian noise

#### 탐색적 분석 & 관계

| 분석 항목 | 결과 요약 | 메모 |
|---|---|---|
| 문자 분포 | 자주 쓰이는 조사/숫자 비율 높음 | 희귀 문자 사전 구축 및 샘플 증강 |
| 텍스트 길이 | 평균 9.4자, 최대 42자 | CTC padding 48로 세팅 |
| 밝기/대비 | 저조도 이미지 14% 존재 | CLAHE 적용 시 CER 1.6%p 개선 |
| 검출→인식 | 검출 miss rate 6.8% | DBNet threshold 0.3→0.25로 조정 |

#### 시각화 결과 & 인사이트

- 글자 히스토그램/워드클라우드로 고빈도 토큰 확인
- 검출 박스 오버레이 샘플: 경사진 표지판에서 miss 다수
- 오인식 사례(0↔O, 1↔l) 정리로 후처리 규칙 설계

#### 특징 & 추후 모델링 계획

- 후보: CRNN+CTC, TrOCR(Small) 전이학습
- 후처리: 유니그램 사전 + 언어모델 교정
- 지표: CER 목표 ≤ 6.0%, 단어정확도 ≥ 92%

### 🔎 RS – 데이터 탐색 및 이해

#### 데이터 구성 요소

- **로그**: 1,250,000 events (view/click/cart/purchase)
- **사용자/아이템**: 38,420명 / 12,305개
- **메타데이터**: 카테고리, 태그, 가격, 텍스트 설명
- **스파스성**: user×item 밀도 0.26%

#### 전처리·특성화

- implicit 가중치: view 1 / click 3 / cart 6 / purchase 12
- 세션화: inactivity 30분 기준, 평균 세션 길이 5.4
- 시간 감쇠: 최근성 가중치 λ=0.015

#### 탐색적 분석 & 관계

| 분석 항목 | 결과 요약 | 메모 |
|---|---|---|
| 사용자 활동 | 상위 10% 유저가 61% 이벤트 생성 | 롱테일 보정 필요 |
| 아이템 인기도 | 상위 1% 아이템이 클릭의 28% | 인기도 편향 완화 필요 |
| 세션 패턴 | 저녁 20–23시 피크 | 시간대 피처 추가 |
| 코호트/리텐션 | 신규 4주 잔존율 27% | 온보딩 추천 강화 |

#### 시각화 결과 & 인사이트

- 히스토그램: 상호작용 수의 파레토 분포
- 코호트 리텐션 히트맵: 2주차 급락
- 토픽/카테고리별 커버리지 불균형

#### 특징 & 추후 모델링 계획

- 후보: MF/LightFM, NCF/Two-tower, BPR, seq2seq
- 재랭킹: GBDT + 다양성·신규성 제약
- 지표: Recall@20/NDCG@20, 이후 A/B 실험
- 목표: Recall@20 ≥ 0.26, NDCG@20 ≥ 0.14

## 요약

| 분야 | 핵심 포인트 | 향후 계획 |
|---|---|---|
| AD | 비지도/재구성·고립 기반, 시계열 특화, PR지표 중심 | AE/VAE vs IF 비교, 동적 임계값, drift 모니터링 |
| OCR | 검출–인식 파이프라인, CTC/Transformer, 한글 특화 | TrOCR 파인튜닝, 검출 miss 저감, 언어모델 후처리 |
| RS | CF/MF/NCF, implicit 가중, 다양성·공정성 고려 | NeuralCF+재랭킹, KG/LLM 보조, A/B 실험 |
