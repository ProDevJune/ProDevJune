---
title: 'CV Gen 모델 가이드'
layout: page
icon: fas fa-robot
permalink: /ai-bootcamp/cvgen-model-guide/
toc: true
tags:
  - CV Gen
  - 생성형AI
  - Stable Diffusion
  - Text-to-Image
  - ControlNet
  - Diffusion Model
  - Computer Vision
---

# 🧠 CV Gen 모델 가이드

## 1. 개요 (Overview)

**CV Gen 모델**은 "Computer Vision Generation"의 약자로, **이미지 생성 또는 변환**을 중심으로 한 **컴퓨터 비전 생성형 AI 모델**을 의미합니다.  
텍스트 설명 기반 이미지 생성이나, 기존 이미지 편집 등에서 두각을 나타냅니다.

---

## 2. 핵심 개념 (Core Concepts)

### 2.1 Text-to-Image Generation
- 입력: 텍스트
- 출력: 해당 설명에 맞는 이미지
- 핵심 기술: CLIP, Diffusion

### 2.2 Latent Diffusion Models (LDM)
- 고해상도 이미지를 압축된 공간에서 생성  
- Stable Diffusion이 대표 사례

### 2.3 이미지 변환 기반 생성
- 입력 이미지를 변형 (예: 낮 → 밤, 흑백 → 컬러)
- pix2pix, CycleGAN 등 사용

### 2.4 고급 제어 기술
- ControlNet: 입력 조건 기반 제어
- LoRA: 소량 학습으로 성능 유지

---

## 3. 주요 모델 비교

| 모델 | 기관 | 기술 | 특징 |
|------|------|------|------|
| DALL·E 2/3 | OpenAI | CLIP + Diffusion | 정교한 텍스트 해석 |
| Stable Diffusion | Stability AI | LDM | 오픈소스, 확장성 |
| Imagen | Google | T5 + Diffusion | 고품질 출력 |
| Midjourney | Midjourney Labs | 비공개 | 예술적 품질 우수 |
| ControlNet | Tencent | 조건 기반 제어 | 다양한 입력 제어 가능 |

---

## 4. 아키텍처 및 파이프라인

```plaintext
[Text Input] → [Text Encoder] → [Conditioned Diffusion Model] → [Decoder] → [Output Image]
```

---

## 5. 학습 방식

- Pretraining: 텍스트-이미지 쌍 대규모 학습
- Contrastive Learning: 텍스트/이미지 동시 임베딩
- Diffusion Training: 노이즈 제거 기반 생성 학습

---

## 6. 주요 응용

| 분야 | 예시 |
|------|------|
| 콘텐츠 제작 | 웹툰, 썸네일 |
| 패션/인테리어 | 착용 시뮬, 가구 배치 |
| 의료 | 합성 CT 생성 |
| 게임/메타버스 | 캐릭터 생성 |
| 자율주행 | 시뮬레이션 환경 |

---

## 7. 실전 코드 예시

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.to("cuda")
prompt = "a futuristic city in cyberpunk style"
image = pipe(prompt).images[0]
image.save("cyberpunk_city.png")
```

---

## 8. 최신 동향

- DALL·E 3 + GPT 연동
- DreamBooth, LoRA 확산
- ControlNet, SDXL, PhotoMaker
- Foundation Model 통합

---

## 9. 장단점

✅ 직관적인 프롬프트, 창작 효율  
❌ 편향, 윤리 문제, 실사 한계

---

## 10. 추천 자료

- 📄 논문: Rombach et al. (2022) *Latent Diffusion Models*  
- 🧪 코드: https://github.com/CompVis/stable-diffusion  
- 📚 강의: HuggingFace Course, FastCampus

---

## ✅ 요약 정리

- CV Gen은 생성형 비전 모델의 핵심
- Diffusion + 텍스트 인식 (CLIP) 구조가 핵심
- 다양한 모델 비교 및 실전 코드 습득 중요

---

## 🔑 핵심어

`CV Gen`, `Diffusion`, `Stable Diffusion`, `ControlNet`, `Text-to-Image`, `Latent Space`, `Prompt Engineering`

---

**📁 저장 경로 예시**:  
`/ComputerVision/GenAI/01_CV_Gen_모델_완전정복.md`
