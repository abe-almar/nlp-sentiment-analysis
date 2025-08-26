# 🧠 NLP Sentiment Analysis API

[![Powered by TensorFlow](https://img.shields.io/badge/Powered_by-TensorFlow-orange)](https://www.tensorflow.org/)

## 📝 Overview

This FastAPI application performs entity-level sentiment analysis on tweets. Given a tweet and an entity mentioned in the tweet, the model predicts the sentiment toward that entity as Positive, Negative, or Neutral, along with a confidence score.

---

## 🚀 Getting Started

### 🛠️ Run with Docker

Ensure you're in the project root directory where the *Dockerfile* is located.

1. **Build the Docker image**
   ```bash
   docker build -t nlp-sentiment-analysis .

2.  **Run the container image**
     ```
     docker run -p 80:80 nlp-sentiment-analysis
     ```

## 🌐 Base URL
`http://127.0.0.1/`

## 📡 API Endpoints
### 🔍 GET /isalive — Health Check
- Description:
Health check to confirm the API is running.

- Example using curl:
```bash
curl -X GET -H "Content-type: application/json" '127.0.0.1/isalive'
```

Sample Response:
```json
{
"health_check":"OK",
"message":"NLP Sentiment Analysis API"
}
```

### POST /predict — Predict Brain Cancer Type
- Description:
Provide a tweet and a target entity. The API returns the sentiment expressed in the tweet about that specific entity and the confidence level of the prediction.
- Request:

`multipart/form-data` with **one required** file:

| Field              | Type     | Required | Description                          |
|--------------------|----------|----------|--------------------------------------|
| text              | string   | ✅       | 	The tweet text |                   |
| entity             | string   | ✅       | The entity whose sentiment is to be analyzed |



- Example using curl:
```bash
curl -X POST "http://127.0.0.1:80/predict" \
-H "Content-Type: application/json" \
-d '{"entity": "NBA2K", "text": "@NBA2K game sucks... down by 2 with 38 seconds left and my team intentionally fouls"}'
```
- Successful Response:
```json
{
  "entity": "NBA2K",
  "predicted_class": "Negative",
  "confidence": 0.99998
}


```
- entity: Echoes back the input entity
- predicted_sentiment: One of "Positive", "Negative", or "Neutral"
- confidence: Model's confidence score (float between 0 and 1)

### GET /docs — Swagger UI
- Description:
Interactive Swagger UI to explore and test the API endpoints.

### GET /redoc — ReDoc UI
- Description:
Alternative API documentation with ReDoc.

## ❌ Error Handling
### Internal Server Error (500)
Response (500 - Internal Server Error):
```json
{
  "detail": "Internal server error."
}
```
This occurs if there is an issue processing the uploaded files or prediction logic.

## 📝 Additional Notes

- Both tweet and entity fields are required in the request payload.

- The API expects valid UTF-8 text input.

- Recommended improvements for production:

  - Add request schema validation and better user feedback
  
  - Secure with API key or OAuth if exposed externally
  
  - Enable logging and monitoring
  
  - Add rate limiting to prevent abuse