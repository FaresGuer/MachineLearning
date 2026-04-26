# ML Sport FastAPI

API pour exposer les modeles du notebook:
- DSO1: prediction de valeur marchande
- DSO2: prediction de poste
- DSO4: recommandation de joueurs similaires

## 1) Artifacts attendus

Place ces fichiers dans `artifacts/`:

- `modele_retenu_dso1.pkl`
- `scaler_dso1.pkl`
- `features_dso1.json`
- `modele_retenu_dso2.pkl`
- `scaler_dso2.pkl`
- `features_dso2.json`
- `label_encoder_dso2.pkl` (optionnel mais recommande)
- `rec_df.parquet`
- `rec_features.json`

## 2) Lancer en local

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Docs Swagger:
- `http://localhost:8000/docs`

Mini web UI:
- `http://localhost:8000/ui`

## 3) Exemple de payloads

### DSO1

```json
{
  "features": {
    "reactions": 75,
    "composure": 72
  }
}
```

Important: `features` doit contenir exactement toutes les colonnes de `features_dso1.json`.

### DSO2

```json
{
  "features": {
    "goals": 5,
    "assists": 3
  }
}
```

Important: `features` doit contenir exactement toutes les colonnes de `features_dso2.json`.

### DSO4

```json
{
  "player_name": "Kevin De Bruyne",
  "top_n": 5,
  "value_tolerance": 0.3
}
```

## 4) Docker

```powershell
docker build -t ml-sport-api .
docker run -p 8000:8000 ml-sport-api
```
