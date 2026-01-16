# BISKIT: GNN 기반 상권 분석 및 추천 시스템

**BISKIT**(Business Starter KIT)은 **Graph Neural Network(GNN)**를 활용하여
상권의 폐업률을 예측하고 최적의 창업 입지를 추천하는 **딥러닝 기반 의사결정 지원 시스템**입니다.

## 🎯 ML/AI 개발자를 위한 프로젝트 하이라이트

- **GNN 모델**: 공간적 관계와 업종 간 상호작용을 그래프로 모델링한 폐업률 예측 시스템
- **시공간 데이터 처리**: 위치 정보, 업종 카테고리, 시계열 특성을 결합한 멀티모달 데이터 학습
- **실시간 추론**: FastAPI 기반의 고성능 모델 서빙 인프라
- **프로덕션 ML 파이프라인**: 데이터 전처리부터 모델 배포까지 End-to-End ML 시스템 구현

---

## 배경 (Background)

2025년 경제 침체로 인해 **취업난**이 심화되고,

정부의 경기 부양을 위한 **금리 인하 기조**로 창업 진입 장벽이 낮아지면서 **창업에 대한 관심**이 높아지고 있습니다.

그러나 올해 **신규 창업의 폐업률**은 역대 최고 수준을 기록하며,

창업자 수는 증가하고 있음에도 사업의 **지속 가능성**을 높이기 위한 지원과 대책이 시급한 상황입니다.

BISKIT은 이러한 문제를 해소하기 위해

**폐업률 데이터를 활용한 상권 분석**과 **AI 기반 추천 시스템**을 결합하여

**지속 가능한 창업을 돕는 도구**를 목표로 합니다.

---

## 주요 기능 (Features)

### 1. 추천 및 분석 시스템

- **GNN 기반 상권 분석**
    - 선택한 지역에 대해 다양한 업종의 분석 정보와 **생존율 / 폐업률 예측**을 제공합니다.
    - 지도에서 특정 위치를 클릭하면 **1~5년 예상 폐업률과 그 이유**를 조회할 수 있습니다.
- **쿼리 타입별 추천 로직**
    - **좌표 입력**
        - 해당 위치의 폐업률 정보 조회
        - 지속 가능성이 높은 **업종 추천**
    - **좌표 + 업종 입력**
        - 해당 위치에서 특정 업종의 폐업률 정보 조회
    - **범위 + 업종 입력 (원/사각형/다각형)**
        - 선택한 범위 내 업종의 폐업률 정보 조회
        - 폐업률이 낮은, 즉 **지속 가능성이 높은 위치 추천**

---

### 2. 지도 기반 UI (Map & Visualization)

- **카카오맵(Kakao Map)** 기반의 대화형 지도 제공
- 사용자가 지도에서 위치를 직접 선택하여
    - 상권 현황
    - 업종 분포
    - 폐업률/생존율 예측 결과
        
        **즉시 시각적으로 확인**할 수 있습니다.
        
- 분석 결과를 지도 상에 **시각적으로 표현**해 직관적인 의사결정이 가능합니다.

---

### 3. 로그인 및 사용자 관리 (Authentication & User)

- **Google OAuth2**를 활용한 소셜 로그인
- **JWT(액세스/리프레시 토큰)** 기반의 안전한 인증/인가 및 세션 관리
- **검색 기록 저장/관리 기능**
    - 과거 검색 기록을 조회하고, 동일 조건으로 재검색 가능
- **건물 찜(즐겨찾기) 기능**
    - 관심 있는 건물/위치를 저장해 빠르게 다시 조회할 수 있습니다.

---

### 4. 채팅 (Chat)

- 채팅방 개설 및 참여 기능
- 관심 상권, 업종, 창업 아이템 등에 대해 **실시간 정보 공유**
- 같은 관심사를 가진 예비 창업자 간 **커뮤니티 기능** 제공

---

### 5. AI 추천 (AI Recommendation)

- **창업 선호 테스트**를 통해 사용자의 성향, 선호 업종, 리스크 성향 등을 분석합니다.
- 분석 결과를 바탕으로
    - 추천 업종
    - 추천 상권/위치
        
        를 제안합니다.
        
- 추천된 업종은 드롭다운 리스트의 **상단에 노출**되어 입력 편의성을 높입니다.

---

## 🛠️ 기술 스택 (Tech Stack)

이 프로젝트는 Docker를 활용해 완전히 컨테이너화된 여러 서비스로 구성된 **모노레포(Monorepo)** 구조입니다.

| 서비스 | 기술 스택 |
| --- | --- |
| **프론트엔드** | Next.js, React, TypeScript (TSX), Tailwind CSS, Zustand, React Query |
| **백엔드** | Spring Boot, Java 21, Spring Security, JPA, JWT, WebSocket |
| **AI** | FastAPI, Python, PyTorch, Pandas, OpenAI |
| **데이터베이스** | MySQL, Redis |
| **데브옵스** | Docker, Docker Compose, Jenkins, Nginx |

---

## 👥 팀 소개 (Team BISKIT)

| 이름 | 역할 | 담당 영역 |
| --- | --- | --- |
| 성기원 | 팀장, 백엔드 | 전체 일정 관리, 추천 시스템 구현,  ERD 및 DB 스키마 설계 |
| 강건 | 프론트엔드, 백엔드 | 전체 프론트엔드 개발 및 상가 검색 및 AI 추천 시스템 개발, DB 설계 및 최적화 |
| 김승민 | 백엔드 | Login, 실시간 채팅 기능 전체 개발 (Google OAuth2, JWT, WebSocket) |
| 이승주 | 프론트엔드, 인프라 | Docker, Jenkins, 배포/운영 환경 구성, API 설계 |
| 강한설 | AI, DS | GNN 모델링, 데이터 전처리 |
| 문종원 | AI, DS | GNN 모델링, 데이터 전처리 |

---

## 📂 프로젝트 구조 (Project Structure)

```
.
├─ frontend/        # Next.js 기반 프론트엔드
├─ backend/         # Spring Boot 백엔드
├─ ai/              # FastAPI 기반 AI/GNN 서비스
├─ mysql/           # 상가 & 카테고리 데이터베이스 정보
├─ exec/            # Docker & 배포 관련 설정 문서
└─ README.md

```

---

## 🚀 시작하기 (Getting Started)

전체 애플리케이션은 **Docker Compose**로 실행되도록 설계되었습니다.

### 1. 필수 조건 (Prerequisites)

- Docker
- Docker Compose

### 2. 환경 변수 파일(.env) 생성

루트 디렉터리에 `.env` 파일을 생성하고, 필요한 환경 변수를 설정합니다.

(예: DB 접속 정보, JWT 시크릿, OAuth 클라이언트 ID, Redis 설정 등)

### 3. Docker Compose로 빌드 및 실행

아래 명령어를 루트 디렉터리에서 실행합니다.

```
docker-compose up --build -d

```

### 4. 애플리케이션 접속

- 프론트엔드: [http://localhost:3000](http://localhost:3000/) (또는 설정한 포트)
- 백엔드 / AI 서비스 / 기타 포트는 `docker-compose.yml` 설정에 따릅니다.

### 5. 애플리케이션 중지하기

```
docker-compose down

```

---

---

## 📊 데이터 및 모델 상세 (Data & Model Architecture)

### 데이터셋 구조

**데이터 소스**
- **상가 데이터** (STORE.parquet): 위치, 업종, 행정동 코드 등
- **인구 데이터** (POPULATION.parquet): 행정동별 상주인구, 직장인구
- **인프라 데이터**: 버스정류장, 학교, 도서관, 지하철역 위치 및 유동인구
- **보행 데이터** (CROSSWALK.parquet): 행정동 경계 및 지역 좌표
- **야간 조명 데이터** (NIGHTVIEW.parquet): 상권 활성도 지표

**그래프 구조**
```
노드(Node):
  - 지역 노드: 행정동 단위 (환경 특징 포함)
  - 점포 노드: 개별 상가 (업종 원핫 인코딩)

엣지(Edge):
  - 점포 ↔ 소속 행정동 (양방향)
  - 가상 노드 ↔ 인근 지역 노드 (추론 시 동적 생성)
```

**특징 벡터 (Feature Vector)**
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

**데이터 전처리 파이프라인**
```
Parquet 로드 → 좌표계 변환(WGS84) → KDTree 구축 →
행정동 매핑 → 환경 특징 집계 → 정규화 → 그래프 구축
```

### GNN 모델 아키텍처

**모델: Survival GNN** ([model.py:4-15](ai/app/core/model.py#L4-L15))

```python
class SurvivalGNN(nn.Module):
    def __init__(self, in_dim, hid=64, out_hazards=5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid)      # 1st GraphSAGE layer
        self.conv2 = SAGEConv(hid, hid)         # 2nd GraphSAGE layer
        self.head = nn.Linear(hid, out_hazards) # Output: 5-year hazards

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        logits = self.head(h)
        return logits, h
```

**모델 특징**
- **기반**: GraphSAGE (inductive learning, 새로운 노드 추론 가능)
- **레이어**: 2개 GraphSAGE Convolution → ReLU 활성화
- **출력**: 5개 시간 구간별 폐업 위험도 (Hazard Probabilities)
  - 1년차, 2년차, 3년차, 4년차, 5년차 각각의 폐업 확률
- **손실 함수**: Binary Cross-Entropy (생존 분석)
- **추론 모드**: FP16 (half precision) 경량화

**추론 프로세스** ([subgraph.py:144-159](ai/app/core/subgraph.py#L144-L159))
```
1. 입력: (위도, 경도, 업종 ID)
2. 서브그래프 구축:
   - KDTree로 인근 k개 행정동 검색 (k=20)
   - 거리 기반 가중치 계산 (역거리 가중)
   - 가상 노드 생성 (환경 특징 + 업종 원핫)
   - 가상 노드 ↔ 지역 노드 엣지 생성
3. GNN Forward Pass:
   - 서브그래프에 대해 GraphSAGE 실행
   - 가상 노드의 임베딩 추출
4. 출력:
   - hazard: 연도별 폐업 확률 [p1, p2, p3, p4, p5]
   - survival: 누적 생존율 S(t) = ∏(1 - p_i)
   - failure: 누적 폐업률 F(t) = 1 - S(t)
```

**경쟁 밀도 특징** ([subgraph.py:86-118](ai/app/core/subgraph.py#L86-L118))
- 반경 300m 내 동일 업종 밀집도를 가우시안 커널로 계산
- 경쟁 점수를 환경 특징에 주입하여 포화 지역 페널티 반영

### AI 서빙 아키텍처

**FastAPI 엔드포인트** ([main.py](ai/app/main.py))
```
POST /api/v1/ai/location
  - 입력: {lat, lng, building_id}
  - 출력: 모든 업종의 폐업률 순위 (추천)

POST /api/v1/ai/job
  - 입력: {lat, lng, category}
  - 출력: 특정 업종의 상세 폐업률 분석

POST /api/v1/ai/gms
  - 입력: {lat, lng, category}
  - 출력: LLM 기반 폐업 위험 요인 설명
```

**기술 스택**
- **모델 학습**: PyTorch, PyTorch Geometric (GraphSAGE)
- **데이터 처리**: Pandas, NumPy, GeoPandas, scipy.spatial.cKDTree
- **모델 서빙**: FastAPI, Uvicorn (비동기 처리)
- **설명 생성**: OpenAI API (GPT 기반 해석)

### 성능 및 최적화

**추론 최적화**
- **FP16 양자화**: 모델 가중치 half precision 변환 (메모리 ↓, 속도 ↑)
- **동적 서브그래프 샘플링**: 전체 그래프가 아닌 인근 k개 노드만 로드
- **KDTree 기반 공간 인덱싱**: O(log N) 시간 복잡도로 인근 지역 검색
- **메모리 관리**: 그래프 구축 후 원본 DataFrame 해제

**확장성**
- Docker 컨테이너화로 독립 배포
- GPU 사용 가능 (CUDA 자동 감지)
- 비동기 API로 동시 요청 처리

### 한계 및 주의사항

- **시간적 한계**: 과거 데이터 기반 학습으로 급격한 경제 변화 미반영
- **공간적 한계**: 행정동 단위 집계로 미세한 골목 상권 특성 손실 가능
- **카테고리 불균형**: 일부 업종은 데이터 부족으로 예측 정확도 낮을 수 있음
- **외부 변수**: 정책, 재개발, 팬데믹 등 외생 변수 미고려
- **확률적 추정**: 결과는 통계적 경향이며 개별 사업 성공을 보장하지 않음

**본 서비스는 의사결정 참고용 도구이며, 최종 창업 판단의 책임은 사용자에게 있습니다.**

## 기여 & 문의 (Contributing)

본 프로젝트는 **신규 창업자의 데이터 기반 의사결정**을 지원하기 위해 개발되었습니다.

- 버그 제보, 기능 제안, 질문 등은 이슈 트래커를 통해 남겨 주세요.
    
    Team BISKIT