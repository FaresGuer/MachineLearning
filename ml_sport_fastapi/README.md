# ML Sport FastAPI

FastAPI service for football analytics and player recommendations.

**Live demo:** https://machinelearning-y0fz.onrender.com/ui

## What it does

- DSO1: predicts player market value
- DSO2: predicts the best player position
- DSO4: recommends similar players
- Built-in browser UI at `/ui`
- Swagger docs at `/docs`
- Optional Gemini-powered football chat at `/chat/football`

## Project structure

- `app.py` - FastAPI application and API routes
- `static/index.html` - browser UI
- `artifacts/` - trained models, scalers, feature lists, and recommendation data
- `requirements.txt` - Python dependencies
- `Dockerfile` - container build for deployment

## Required artifacts

Place these files in `artifacts/` before starting the app:

- `modele_retenu_dso1.pkl`
- `scaler_dso1.pkl`
- `features_dso1.json`
- `modele_retenu_dso2.pkl`
- `scaler_dso2.pkl`
- `features_dso2.json`
- `label_encoder_dso2.pkl` - optional
- `rec_df.parquet`
- `rec_features.json`

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open these URLs after startup:

- `http://localhost:8000/docs`
- `http://localhost:8000/ui`
- `http://localhost:8000/health`

## Environment variables

- `GEMINI_API_KEY` - required for the Gemini chat endpoint
- `GEMINI_MODEL` - optional Gemini model name, defaults to `gemini-2.5-flash`
- `PORT` - deployment port used by container platforms such as Render

If `GEMINI_API_KEY` is missing, the chat endpoint returns a 503 response and the UI falls back to a local football helper reply.

## API endpoints

### Health

`GET /health`

### Feature metadata

`GET /meta/features`

### Predict player value

`POST /predict/value`

Example:

```json
{
  "features": {
    "reactions": 75,
    "composure": 72
  }
}
```

Missing features are filled automatically with `0.0`.

### Predict best position

`POST /predict/position`

Example:

```json
{
  "features": {
    "goals": 5,
    "assists": 3
  }
}
```

Missing features are filled automatically with `0.0`.

### Find similar players

`POST /recommend/similar`

Example:

```json
{
  "player_name": "Kevin De Bruyne",
  "top_n": 5,
  "value_tolerance": 0.3
}
```

### Football chat

`POST /chat/football`

Example:

```json
{
  "message": "What should a midfielder be good at?"
}
```

## Docker

```powershell
docker build -t ml-sport-api .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key_here ml-sport-api
```

## Deploying to Render

This project is already deployed at https://machinelearning-y0fz.onrender.com/ui.

To deploy your own fork:

1. Push the repository to GitHub.
2. Create a new Render Web Service from the repo.
3. In Render environment variables, add:
   - `GEMINI_API_KEY` - your Google Gemini API key for chat support
   - `GEMINI_MODEL` - set to `gemini-2.5-flash` (or another supported Gemini model)
4. Keep the container port driven by `PORT`.
5. Redeploy after pushing changes.

## Notes

- The UI loads feature names dynamically from `/meta/features`.
- The form automatically defaults missing feature values to zero.
- The chat UI uses Gemini when available and falls back to a simple local reply when it is not.
