# API 문서

## API 엔드포인트

### 1. 위치별 전체 업종 분석

**Endpoint:** `POST /api/v1/ai/location`

**설명:** 특정 위치에서 모든 업종의 폐업률을 분석하고 순위를 반환

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/ai/location" \
  -H "Content-Type: application/json" \
  -d '{
    "building_id": "BD001",
    "lat": 37.5665,
    "lng": 126.9780
  }'
```

**Response:**
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

---

### 2. 특정 업종 상세 분석

**Endpoint:** `POST /api/v1/ai/job`

**설명:** 특정 위치에서 특정 업종의 상세 폐업률 분석

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/ai/job" \
  -H "Content-Type: application/json" \
  -d '{
    "building_id": "BD001",
    "lat": 37.5665,
    "lng": 126.9780,
    "category": 101
  }'
```

**Response:**
```json
{
  "success": true,
  "status": 200,
  "body": {
    "building_id": "BD001",
    "lat": 37.5665,
    "lng": 126.978,
    "category": 101,
    "failure_rates": {
      "1": 12.5,
      "2": 23.4,
      "3": 32.1,
      "4": 39.8,
      "5": 45.2
    }
  }
}
```

---

### 3. XAI 설명 생성

**Endpoint:** `POST /api/v1/ai/gms`

**설명:** 특정 업종의 폐업 위험 요인을 자연어로 설명

**Request:**
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

**Response:**
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

## API 문서 (Swagger)

FastAPI는 자동으로 대화형 API 문서를 제공합니다:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 에러 응답

```json
{
  "success": false,
  "status": 500,
  "message": "Error message here"
}
```

**주요 에러 코드:**
- `400`: 잘못된 요청 (좌표 범위, 필수 필드 누락 등)
- `404`: 업종 ID를 찾을 수 없음
- `500`: 서버 내부 오류 (모델 추론 실패 등)
