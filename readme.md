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

## 🤖 AI/ML 핵심 기술

### 1. SurvivalGNN 모델 아키텍처

**GraphSAGE 기반 생존 분석 모델**

```python
class SurvivalGNN(nn.Module):
    def __init__(self, in_dim, hid=64, out_hazards=5):
        self.conv1 = SAGEConv(in_dim, hid)      # 1차 이웃 집계
        self.conv2 = SAGEConv(hid, hid)         # 2차 이웃 집계
        self.head = nn.Linear(hid, out_hazards) # 5개년 hazard 예측
```

- **입력**: 환경 피처 + 업종 One-Hot (약 100+ 차원)
- **출력**: 1~5년차 hazard rate (연도별 폐업 확률)
- **활성화**: ReLU → Sigmoid (hazard를 0~1로 변환)

### 2. 동적 서브그래프 생성 (Augmented Subgraph)

**임의 좌표에서 예측 가능한 가상 노드 기법**

```
┌─────────────────────────────────────────────────────────────┐
│  쿼리 좌표 (lat, lon) + 업종 (category_id)                    │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────┐    KDTree Query     ┌─────────────────┐    │
│  │ 가상 노드   │ ◄─────────────────► │ 인근 k개 지역    │    │
│  │ (Virtual)   │    거리 기반 가중치   │ (Region Nodes)  │    │
│  └─────────────┘                      └─────────────────┘    │
│       │                                                      │
│       ▼                                                      │
│  가중 평균으로 환경 피처 블렌딩                                │
│  + 업종 One-Hot 인코딩                                        │
│  + 경쟁 밀도 점수 주입                                        │
└─────────────────────────────────────────────────────────────┘
```

**핵심 알고리즘**
```python
# 1. 인근 지역 가중치 계산 (거리 역수)
w = 1.0 / (distance + 1e-6)
w /= w.sum()  # 정규화

# 2. 환경 피처 블렌딩
env_mix = Σ (w_i × region_vector_i)

# 3. 경쟁 밀도 점수 (Gaussian Weighted)
competition_score = Σ exp(-(d²) / (2σ²))
normalized_competition = 1.0 / (1.0 + 0.05 × competition_score)
```

### 3. Hazard-to-Survival 변환

**연도별 누적 생존율 계산**

```python
# Hazard Rate → Survival Probability
def hazard_to_survival(hazards):
    survival = 1.0
    S = []
    for p in hazards:  # p = 해당 연도 폐업 확률
        survival *= (1.0 - p)
        S.append(survival)
    return S  # [S1, S2, S3, S4, S5]

# 예시:
# hazard = [0.15, 0.12, 0.10, 0.08, 0.07]
# survival = [0.85, 0.75, 0.67, 0.62, 0.58]
# failure = [0.15, 0.25, 0.33, 0.38, 0.42]
```

### 4. 환경 피처 구성

**7개 핵심 환경 피처**

| 피처명 | 설명 | 영향 |
|--------|------|------|
| `pop` | 거주 인구 | 소비자 풀 크기 |
| `work_pop` | 직장 인구 (출퇴근) | 평일 유동 인구 |
| `bus_stop_count` | 버스 정류장 수 | 접근성 |
| `school_count` | 학교 수 | 학생 인구 밀도 |
| `library_count` | 도서관 수 | 문화 시설 밀집도 |
| `nightview_count` | 야간 유동 스팟 | 야간 상권 활성도 |
| `subway_traffic` | 지하철 유동량 | 대중교통 접근성 |

**피처 정규화**
```python
# Min-Max Normalization (0~1)
normalized_value = raw_value / region_max[feature_name]
```

### 5. Explainable AI (XAI) - 예측 근거 설명

**2단계 설명 생성 파이프라인**

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  피처 기여도    │ ──► │  영향도 정량화   │ ──► │  LLM 자연어화   │
│  분석          │     │  (버킷 분류)     │     │  (GPT 설명)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**피처 영향도 분류**
```python
def _bucket(score: float):
    if score >= 0.60: return "매우 큼"
    elif score >= 0.35: return "큼"
    elif score >= 0.20: return "보통"
    elif score > 0.05: return "약함"
    else: return "매우 약함"
```

**LLM 프롬프트 구성**
```python
sys_prompt = """
너는 상권 데이터 분석가야. 아래 정보를 바탕으로
'왜 이 업종의 폐업 위험이 그렇게 나왔는지'를
비전문가(점주)도 이해할 수 있는 한국어로 설명해.

규칙:
1) '엣지, 노드, 임베딩' 같은 기술 용어 사용 금지
2) 숫자 대신 강도로 표현 (매우 높음, 보통 등)
3) 장점/리스크 균형 있게 요약
"""
```

**응답 예시**
```json
{
  "explain": "해당 위치는 거주 인구가 많고 지하철 유동량이 높아
              접근성이 좋습니다. 다만 주변에 동일 업종 점포가
              다소 밀집해 있어 경쟁이 예상됩니다. 종합적으로
              해당 업종의 폐업 위험은 보통 수준으로 판단됩니다."
}
```

---

## 📊 AI 추천 알고리즘 Pipeline

```mermaid
graph TD
    A[사용자 입력<br/>위치 + 업종] --> B[KDTree Query<br/>인근 k개 지역 탐색]
    B --> C[거리 기반 가중치 계산<br/>w = 1/(d + ε)]
    C --> D[환경 피처 블렌딩<br/>가중 평균]
    D --> E[경쟁 밀도 계산<br/>Gaussian Weighted]
    E --> F[가상 노드 생성<br/>env_vec + cat_onehot]
    F --> G[서브그래프 구성<br/>Virtual ↔ Region Edges]
    G --> H[SurvivalGNN 추론<br/>GraphSAGE 2-layer]
    H --> I[Hazard → Survival<br/>누적 생존율 변환]
    I --> J[업종별 순위 정렬<br/>5년차 폐업률 기준]
    J --> K[XAI 설명 생성<br/>피처 기여도 + LLM]
    K --> L[최종 결과 반환]
```

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

## 📂 프로젝트 구조

```
BISKIT/
├── ai/                                  # 🤖 AI/ML 추론 엔진 (FastAPI)
│   ├── app/
│   │   ├── main.py                     # FastAPI 애플리케이션 엔트리
│   │   │   ├── POST /api/v1/ai/location    # 위치별 전체 업종 분석
│   │   │   ├── POST /api/v1/ai/job         # 특정 업종 상세 분석
│   │   │   └── POST /api/v1/ai/gms         # LLM 설명 생성
│   │   ├── core/
│   │   │   ├── model.py                # SurvivalGNN 모델 정의
│   │   │   ├── subgraph.py             # 동적 서브그래프 생성
│   │   │   │   ├── build_augmented_subgraph_for_category()
│   │   │   │   └── predict_hazards_at_location()
│   │   │   ├── location.py             # 업종 랭킹 로직
│   │   │   ├── explain.py              # XAI 피처 분석
│   │   │   ├── gms.py                  # LLM 설명 생성
│   │   │   ├── data_io.py              # 데이터 로딩/컨텍스트
│   │   │   └── utils.py                # 유틸리티 함수
│   │   └── schemas/
│   │       └── single.py               # Pydantic 요청/응답 모델
│   ├── requirements.txt                 # Python 의존성
│   └── Dockerfile                       # 컨테이너 이미지
│
├── backend/                             # 🔧 Spring Boot 백엔드
│   ├── src/main/java/com/example/backend/
│   │   ├── auth/                        # JWT + OAuth2 인증
│   │   ├── chat/                        # WebSocket 실시간 채팅
│   │   │   ├── controller/             # REST + STOMP 컨트롤러
│   │   │   ├── service/                # 채팅 비즈니스 로직
│   │   │   └── entity/                 # Room, Message 엔티티
│   │   ├── recommend/                   # AI 서버 연동
│   │   │   └── dto/                    # Request/Response DTO
│   │   └── user/                        # 사용자 프로필 관리
│   ├── build.gradle.kts                 # Gradle 빌드 설정
│   └── application-{profile}.yml        # 환경별 설정
│
├── frontend/                            # 🎨 Next.js 프론트엔드
│   ├── src/
│   │   ├── app/                         # App Router 페이지
│   │   ├── features/
│   │   │   ├── map/                    # 카카오맵 컴포넌트
│   │   │   │   ├── kakao-map.tsx      # 지도 메인
│   │   │   │   ├── MarkerPopup.tsx    # 상가 정보 팝업
│   │   │   │   └── ClusterPopup.tsx   # 클러스터 팝업
│   │   │   ├── stores/                 # 상가 목록/필터
│   │   │   ├── chat/                   # 채팅 UI
│   │   │   └── auth/                   # Google OAuth
│   │   ├── store/                       # Zustand 전역 상태
│   │   └── api/                         # Axios API 클라이언트
│   ├── package.json
│   └── next.config.ts
│
├── mysql/                               # 📦 MySQL 초기화 스크립트
├── exec/                                # 🚀 배포 설정 문서
├── docker-compose.yml                   # 개발 환경 오케스트레이션
└── README.md
```

---

## 🛠 기술 스택

### 🤖 AI/ML (Python 3.x)
```yaml
Framework: FastAPI 0.115.4              # 고성능 비동기 API 서버
Deep Learning:
  - PyTorch 2.5.1                       # 딥러닝 프레임워크
  - PyTorch Geometric 2.6.1             # GNN 라이브러리 (GraphSAGE)
Data Processing:
  - NumPy 2.2.6                         # 다차원 배열 연산
  - Pandas 2.3.2                        # 데이터 전처리
  - SciPy 1.15.3                        # KDTree 공간 검색
LLM Integration: OpenAI 1.108.2         # GPT 기반 설명 생성
Validation: Pydantic 2.9.2              # 입력 데이터 검증
Server: Uvicorn 0.30.6                  # ASGI 서버
```

**주요 알고리즘**
- GraphSAGE (Sampling and Aggregating) 기반 GNN
- KDTree 기반 공간 검색 및 이웃 탐색
- Gaussian Weighted 경쟁 밀도 계산
- Hazard-to-Survival 누적 확률 변환

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
API Client: WebFlux (비동기 AI 서버 호출)
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
Real-time: @stomp/stompjs 7.2 (WebSocket)
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

### 3. API 테스트

**위치별 전체 업종 분석**
```bash
curl -X POST "http://localhost:8000/api/v1/ai/location" \
  -H "Content-Type: application/json" \
  -d '{
    "building_id": "BD001",
    "lat": 37.5665,
    "lng": 126.9780
  }'
```

**응답 예시**
```json
{
  "success": true,
  "status": 200,
  "body": {
    "building_id": "BD001",
    "lat": 37.5665,
    "lng": 126.978,
    "data": [
      {
        "rank": 1,
        "category": "편의점",
        "1": 8.45,
        "2": 15.23,
        "3": 21.87,
        "4": 27.54,
        "5": 32.18
      },
      {
        "rank": 2,
        "category": "커피전문점",
        "1": 12.34,
        "2": 22.56,
        "3": 31.45,
        "4": 38.92,
        "5": 45.67
      }
    ]
  }
}
```

**특정 업종 XAI 설명**
```bash
curl -X POST "http://localhost:8000/api/v1/ai/gms" \
  -H "Content-Type: application/json" \
  -d '{
    "building_id": "BD001",
    "lat": 37.5665,
    "lng": 126.9780,
    "category": 101
  }'
```

**응답 예시**
```json
{
  "success": true,
  "status": 200,
  "body": {
    "building_id": "BD001",
    "lat": 37.5665,
    "lng": 126.978,
    "explain": "해당 위치는 지하철 유동량이 높고 직장 인구가 많아
               점심/저녁 시간대 수요가 기대됩니다. 거주 인구는
               보통 수준이며, 인근에 동일 업종 점포가 다소 있어
               경쟁이 예상됩니다. 종합적으로 해당 업종의 5년 내
               폐업 위험은 보통 수준으로 판단됩니다."
  }
}
```

---

## 🔬 AI 추론 엔진 상세

### 1. 서브그래프 생성 (Subgraph Construction)

#### 인근 지역 탐색 (KDTree Query)
```python
# k개 인근 지역 노드 탐색
d, idxs = node_tree.query([lat, lon], k=k_region)

# 거리 기반 가중치 (역수)
w = 1.0 / (d + 1e-6)
w /= w.sum()  # 정규화
```

#### 환경 피처 블렌딩
```python
# 가중 평균으로 가상 노드 환경 벡터 생성
env_mix = np.zeros(env_feat_count)
for region_code, weight in zip(codes, w):
    env_mix += weight * region_vector(region_code)

# 감마 보정 및 게인 적용
env_mix = np.power(env_mix, env_gamma)
env_mix = np.clip(env_mix * env_gain, 0.0, 1.0)
```

#### 경쟁 밀도 점수 주입
```python
# 300m 반경 내 상가 탐색
radius_deg = 0.3 / 111.1  # ~300m
nearby_indices = store_tree.query_ball_point([lat, lon], r=radius_deg)

# Gaussian Weighted 경쟁 점수
sigma = 0.1  # km
weights = np.exp(-(distances**2) / (2 * sigma**2))
competition_score = np.sum(weights)

# 정규화 (높을수록 경쟁 심함 → 낮은 점수)
normalized = 1.0 / (1.0 + 0.05 * competition_score)
```

### 2. GNN 추론 (Model Inference)

#### 모델 구조
```
Input: [env_features(7) + category_onehot(N)] → ~100+ dims
    │
    ▼
SAGEConv(in_dim → 64) + ReLU
    │
    ▼
SAGEConv(64 → 64) + ReLU
    │
    ▼
Linear(64 → 5) → Sigmoid
    │
    ▼
Output: [hazard_1y, hazard_2y, hazard_3y, hazard_4y, hazard_5y]
```

#### 추론 코드
```python
@torch.no_grad()
async def predict_hazards_at_location(ctx, lat, lon, cid, knobs):
    # 서브그래프 생성
    sub, v_idx = await build_augmented_subgraph_for_category(
        ctx, lat, lon, cid, knobs
    )

    # 차원 보정
    sub.x = _ensure_feature_dim(sub.x, target_dim)

    # 모델 추론
    ctx.model.eval()
    logits, _ = ctx.model(sub.x, sub.edge_index)
    hazard = torch.sigmoid(logits[v_idx]).numpy()

    # 생존율/폐업률 변환
    S, F = _hazard_to_survival_and_failure(hazard)

    return {"hazard": hazard, "survival": S, "failure": F}
```

### 3. 하이퍼파라미터 설정

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `k_region` | 5 | 인근 지역 탐색 개수 |
| `k_max_ratio` | 2.0 | 최대 거리 비율 제한 |
| `edge_gain` | 3.0 | 엣지 가중치 증폭 |
| `env_gain` | 1.0 | 환경 피처 게인 |
| `env_gamma` | 1.0 | 환경 피처 감마 보정 |
| `DISTANCE_SIGMA_KM` | 0.1 | 경쟁 점수 가우시안 σ |

---

## 📊 데이터셋 구조

### 데이터 소스

| 데이터 | 파일 | 설명 |
|--------|------|------|
| **상가 데이터** | `STORE.parquet` | 위치, 업종, 행정동 코드 등 |
| **인구 데이터** | `POPULATION.parquet` | 행정동별 상주인구, 직장인구 |
| **인프라 데이터** | - | 버스정류장, 학교, 도서관, 지하철역 위치 및 유동인구 |
| **보행 데이터** | `CROSSWALK.parquet` | 행정동 경계 및 지역 좌표 |
| **야간 조명 데이터** | `NIGHTVIEW.parquet` | 상권 활성도 지표 |

### 그래프 구조

```
노드(Node):
  - 지역 노드: 행정동 단위 (환경 특징 포함)
  - 점포 노드: 개별 상가 (업종 원핫 인코딩)

엣지(Edge):
  - 점포 ↔ 소속 행정동 (양방향)
  - 가상 노드 ↔ 인근 지역 노드 (추론 시 동적 생성)
```

### 특징 벡터 (Feature Vector)

```python
feature_dim = env_features + category_onehot

env_features (7차원):
  - pop: 상주인구 (정규화)
  - work_pop: 직장인구 (정규화)
  - bus_stop_count: 버스정류장 수
  - school_count: 학교 수
  - library_count: 도서관 수
  - nightview_count: 야간 조명 밀도
  - subway_traffic: 지하철 유동인구

category_onehot (N차원):
  - 업종 원핫 인코딩 (N = 업종 개수)
```

### 데이터 전처리 파이프라인

```
Parquet 로드 → 좌표계 변환(WGS84) → KDTree 구축 →
행정동 매핑 → 환경 특징 집계 → 정규화 → 그래프 구축
```

---

## 💡 핵심 성과 및 차별점

### 🎯 기술적 성과

1. **GraphSAGE 기반 생존 분석**
   - 노드 특성 집계(Aggregation)로 인근 상권 영향 반영
   - 2-layer 구조로 2-hop 이웃 정보 활용
   - Half-precision (FP16) 추론으로 메모리/속도 최적화

2. **동적 서브그래프 기법**
   - 학습 시 없던 임의 좌표에서도 예측 가능
   - 가상 노드 + 인근 지역 노드로 서브그래프 구성
   - KDTree 기반 O(log N) 이웃 탐색

3. **다차원 피처 엔지니어링**
   - 인구, 교통, 경쟁 밀도 등 7개 환경 피처
   - Gaussian Weighted 경쟁 점수로 지역 포화도 반영
   - 업종 One-Hot 인코딩으로 업종 특성 구분

4. **Explainable AI 구현**
   - 피처별 기여도 분석 (영향도 버킷 분류)
   - LLM 기반 자연어 설명 생성 (비전문가 친화적)
   - 설명 실패 시 규칙 기반 폴백 제공

### 🚀 확장 가능성

- **시계열 확장**: 시점별 상권 변화 반영 (Temporal GNN)
- **피드백 학습**: 실제 폐업 데이터로 모델 재학습
- **다중 타겟**: 매출 예측, 임대료 예측 등 확장

### 📈 기대 효과

- **의사결정 지원**: 데이터 기반 입지 선정으로 창업 리스크 감소
- **정량적 비교**: 업종별 폐업률 순위로 객관적 비교 가능
- **근거 제공**: XAI 설명으로 예측 신뢰도 향상

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
