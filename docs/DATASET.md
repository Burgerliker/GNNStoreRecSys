# 데이터셋 구조

## 데이터 소스

| 데이터 | 파일 | 설명 |
|--------|------|------|
| **상가 데이터** | `STORE.parquet` | 위치, 업종, 행정동 코드 등 |
| **인구 데이터** | `POPULATION.parquet` | 행정동별 상주인구, 직장인구 |
| **인프라 데이터** | - | 버스정류장, 학교, 도서관, 지하철역 위치 및 유동인구 |
| **보행 데이터** | `CROSSWALK.parquet` | 행정동 경계 및 지역 좌표 |
| **야간 조명 데이터** | `NIGHTVIEW.parquet` | 상권 활성도 지표 |

## 그래프 구조

```
노드(Node):
  - 지역 노드: 행정동 단위 (환경 특징 포함)
  - 점포 노드: 개별 상가 (업종 원핫 인코딩)

엣지(Edge):
  - 점포 ↔ 소속 행정동 (양방향)
  - 가상 노드 ↔ 인근 지역 노드 (추론 시 동적 생성)
```

## 특징 벡터 (Feature Vector)

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

## 데이터 전처리 파이프라인

```
Parquet 로드 → 좌표계 변환(WGS84) → KDTree 구축 →
행정동 매핑 → 환경 특징 집계 → 정규화 → 그래프 구축
```

---

## 데이터 출처

- 상권/폐업률 데이터: 공공데이터 포털, 소상공인시장진흥공단
- 인구 데이터: 행정안전부 주민등록 인구통계
- 인프라 데이터: 국토교통부, 서울열린데이터광장
