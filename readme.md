# BISKIT: GNN 기반 상권 분석 및 생존율 예측 플랫폼

> 창업자를 위한 데이터 기반 의사결정 도구
>
> Graph Neural Network를 활용하여 특정 위치에서 업종별 1~5년 폐업률을 예측하고, 지속 가능한 창업 입지를 추천합니다.

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Spring Boot](https://img.shields.io/badge/Spring%20Boot-6DB33F?style=flat&logo=spring-boot&logoColor=white)](https://spring.io/projects/spring-boot)
[![Next.js](https://img.shields.io/badge/Next.js-000000?style=flat&logo=next.js&logoColor=white)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev/)

---

## 🎯 프로젝트 개요

**BISKIT**(Business Starter KIT)은 예비 창업자가 **데이터 기반 의사결정**을 할 수 있도록 돕는 GNN 기반 상권 분석 플랫폼입니다.

### 💡 문제 정의
- 2025년 경제 침체로 **취업난 심화**, 금리 인하로 **창업 진입 장벽 하락**
- 신규 창업의 **폐업률 역대 최고** 수준 기록
- 창업자 수 증가에도 **지속 가능성을 높이기 위한 데이터 기반 도구 부재**

### 🚀 해결 방안
- **GNN 기반 생존율 예측**: 위치와 업종을 입력하면 1~5년 폐업률 예측
- **지역 환경 피처 분석**: 인구, 교통, 주변 상권 등 다차원 데이터 활용
- **Explainable AI**: LLM을 활용한 예측 근거 자연어 설명 제공

### 🎪 주요 사용 사례
- **입지 선정**: 특정 위치에서 어떤 업종이 생존율이 높은지 확인
- **업종 비교**: 같은 위치에서 업종별 폐업률 순위 비교
- **위험도 진단**: 이미 정해진 업종의 위치별 위험도 파악
- **근거 확인**: 왜 이 위치가 위험/안전한지 AI 설명 제공

---

## ✨ 주요 기능

### 🎯 GNN 기반 생존율 예측
- **위치 기반 예측**: 지도에서 클릭한 위치의 업종별 폐업률 예측
- **다년도 예측**: 1년~5년차 누적 폐업률 제공
- **업종 랭킹**: 해당 위치에서 생존율 높은 업종 순위
- **XAI 설명**: 예측 근거를 자연어로 설명

### 🗺️ 지도 기반 인터페이스
- **카카오맵 연동**: 대화형 지도에서 위치 선택
- **상권 시각화**: 마커, 클러스터로 상가 분포 표시
- **범위 검색**: 원/사각형/다각형 영역 내 분석

### 💬 실시간 커뮤니케이션
- **WebSocket 채팅**: STOMP 프로토콜 기반 실시간 메시징
- **채팅방 관리**: 관심 상권별 커뮤니티 기능
- **메시지 영속성**: Redis 캐싱 + MySQL 저장

### 👤 사용자 관리
- **Google OAuth2**: 소셜 로그인
- **JWT 인증**: 액세스/리프레시 토큰 기반 세션
- **검색 기록**: 과거 검색 저장 및 재검색
- **즐겨찾기**: 관심 위치 저장

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  Next.js 15     │◄────►│ Spring Boot 3.5 │◄────►│    MySQL 8.0    │
│  Frontend       │      │   Backend API   │      │    Database     │
│  (TypeScript)   │      │   (Java 21)     │      │                 │
└─────────────────┘      └────────┬────────┘      └─────────────────┘
         │                         │
         │ WebSocket               │ REST API
         │ (실시간 채팅)            ▼
         │              ┌─────────────────────┐   ┌─────────────────┐
         │              │   FastAPI AI Engine │   │     Redis       │
         │              │   (Python 3.x)      │   │   (Session/     │
         └─────────────►│                     │   │    Cache)       │
                        │  • PyTorch          │   └─────────────────┘
                        │  • PyG (GraphSAGE)  │
                        │  • OpenAI API       │
                        │  • KDTree           │
                        └─────────────────────┘
```

### 주요 컴포넌트

| 계층 | 기술 스택 | 역할 |
|------|----------|------|
| **Frontend** | Next.js 15, React 19, TypeScript, Zustand, Tailwind CSS | 지도 기반 UI, 상권 시각화 |
| **Backend API** | Spring Boot 3.5, Java 21, JWT, WebSocket, JPA | 비즈니스 로직, 인증, 실시간 채팅 |
| **AI Engine** | FastAPI, PyTorch, PyTorch Geometric, OpenAI | GNN 추론, 생존율 예측, XAI 설명 |
| **Database** | MySQL 8.0 | 상가 정보, 사용자 데이터 영속성 |
| **Cache** | Redis 7 | 세션 관리, API 캐싱 |

---

## 📂 프로젝트 구조

```
BISKIT/
├── ai/              # FastAPI 기반 AI/ML 추론 엔진
├── backend/         # Spring Boot 백엔드
├── frontend/        # Next.js 프론트엔드
├── mysql/           # MySQL 초기화 스크립트
├── docs/            # 상세 문서
│   ├── ARCHITECTURE.md   # AI/ML 아키텍처 상세
│   ├── DATASET.md        # 데이터셋 구조
│   └── API.md            # API 문서
├── exec/            # 배포 설정 문서
├── docker-compose.yml
└── README.md
```

---

## 🛠 기술 스택

### 🤖 AI/ML (Python 3.x)
```yaml
Framework: FastAPI 0.115.4
Deep Learning:
  - PyTorch 2.5.1
  - PyTorch Geometric 2.6.1 (GraphSAGE)
Data Processing:
  - NumPy 2.2.6
  - Pandas 2.3.2
  - SciPy 1.15.3 (KDTree)
LLM Integration: OpenAI 1.108.2
```

### 🔧 Backend (Java 21 + Spring Boot 3.5.5)
```yaml
Framework: Spring Boot 3.5.5
Authentication:
  - JWT (io.jsonwebtoken 0.12.6)
  - OAuth2 Client (Google)
Real-time: WebSocket (STOMP)
Database:
  - Spring Data JPA
  - MySQL Connector
Cache: Spring Data Redis
API Client: WebFlux
Documentation: SpringDoc OpenAPI 2.6.0
```

### 🎨 Frontend (Next.js 15 + React 19)
```yaml
Framework: Next.js 15.5.2 + React 19.1.0
Language: TypeScript 5.x
UI Library: Radix UI + Tailwind CSS 4.x
State Management: Zustand 5.0.8
Server State: TanStack Query 5.87
Form Handling: React Hook Form 7.62 + Zod 4.1
HTTP Client: Axios 1.11
Map: Kakao Maps SDK
Real-time: @stomp/stompjs 7.2
```

### 🐳 DevOps
```yaml
Containerization: Docker + Docker Compose
CI/CD: Jenkins Pipeline
Database: MySQL 8.0 (linux/amd64)
Cache: Redis 7 Alpine
Monitoring: Spring Actuator
```

---

## 🚀 빠른 시작

### 1. 전체 서비스 실행 (Docker Compose)

```bash
# 환경 변수 설정
cp .env.example .env
# .env 파일에 필요한 값 입력 (DB, JWT, OAuth, API 키 등)

# 개발 환경 실행
docker-compose up --build

# 백그라운드 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f ai
```

### 2. 개별 서비스 실행

**AI Engine (FastAPI)**
```bash
cd ai
pip install -r requirements.txt

# 데이터 파일 필요: data/survival_gnn.pt, data/survival_meta.json
python -m uvicorn app.main:app --reload --port 8000
# http://localhost:8000/docs → FastAPI Swagger UI
```

**Backend (Spring Boot)**
```bash
cd backend
./gradlew bootRun
# http://localhost:8080/swagger-ui.html → API 문서
```

**Frontend (Next.js)**
```bash
cd frontend
yarn install
yarn dev
# http://localhost:3000
```

---

## 👥 팀 소개 (Team BISKIT)

| 이름 | 역할 | 담당 영역 |
|------|------|----------|
| 성기원 | 팀장, 백엔드 | 전체 일정 관리, 추천 시스템 구현, ERD 및 DB 스키마 설계 |
| 강건 | 프론트엔드, 백엔드 | 전체 프론트엔드 개발, 상가 검색 및 AI 추천 연동, DB 최적화 |
| 김승민 | 백엔드 | 로그인, 실시간 채팅 기능 (Google OAuth2, JWT, WebSocket) |
| 이승주 | 프론트엔드, 인프라 | Docker, Jenkins, 배포/운영 환경 구성, API 설계 |
| 강한설 | AI, DS | GNN 모델링, 데이터 전처리 |
| 문종원 | AI, DS | GNN 모델링, 데이터 전처리 |

---

## 📚 상세 문서

- **[AI/ML 아키텍처](docs/ARCHITECTURE.md)** - GNN 모델 상세, 추론 알고리즘, 하이퍼파라미터
- **[데이터셋 구조](docs/DATASET.md)** - 데이터 소스, 그래프 구조, 특징 벡터
- **[API 문서](docs/API.md)** - 엔드포인트, 요청/응답 예시

---

## ⚠️ 한계 및 주의사항

| 항목 | 설명 |
|------|------|
| **시간적 한계** | 과거 데이터 기반 학습으로 급격한 경제 변화 미반영 |
| **공간적 한계** | 행정동 단위 집계로 미세한 골목 상권 특성 손실 가능 |
| **카테고리 불균형** | 일부 업종은 데이터 부족으로 예측 정확도 낮을 수 있음 |
| **외부 변수** | 정책, 재개발, 팬데믹 등 외생 변수 미고려 |
| **확률적 추정** | 결과는 통계적 경향이며 개별 사업 성공을 보장하지 않음 |

---

## 📄 라이선스 및 면책 조항

- 상권/폐업률 데이터 출처: 공공데이터 포털, 소상공인시장진흥공단
- GNN 모델은 과거 데이터 기반 학습으로 실제 미래와 차이 가능
- **본 서비스는 의사결정 참고용 도구이며, 최종 창업 판단의 책임은 사용자에게 있습니다.**
