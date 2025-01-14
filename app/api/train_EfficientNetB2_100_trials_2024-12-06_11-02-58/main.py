from autogluon.multimodal import MultiModalPredictor
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pathlib import Path
from zipfile import ZipFile
import uuid
import shutil
import imghdr
import sqlite3
import json

# Set up FastAPI app
app = FastAPI()

# Load predictor
predictor = MultiModalPredictor.load("./train_EfficientNetB2_100_trials_2024-12-06_11-02-58")

# Set up directories and SQLite database
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
DB_PATH = "prediction_history.db"

# Initialize SQLite database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    filename TEXT,
    predictions TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()
conn.close()

ALLOWED_EXTENSIONS = {"jpeg", "png", "jpg"}

def predict_image(image_filepath):
    proba = predictor.predict_proba({'image': [image_filepath]}, realtime=True)
    proba = proba.flatten()

    prediction_data = {
        "empty_bunch": float(proba[0]),
        "overripe": float(proba[1]),
        "ripe": float(proba[2]),
        "rotten": float(proba[3]),
        "underripe": float(proba[4]),
        "unripe": float(proba[5])
    }
    return prediction_data

@app.post("/tandan-predict/zip")
async def tandan_predict_zip(file: UploadFile = File(...)):
    # Validate uploaded file
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are allowed.")

    unique_dir = UPLOAD_DIR / str(uuid.uuid4())
    unique_dir.mkdir()
    zip_path = unique_dir / file.filename

    try:
        with zip_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract the zip file
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(unique_dir)

        prediction_results = []
        for image_file in unique_dir.iterdir():
            if image_file.suffix[1:].lower() in ALLOWED_EXTENSIONS and imghdr.what(str(image_file)):
                prediction_data = predict_image(str(image_file))
                prediction_results.append({"filename": image_file.name, "predictions": prediction_data})

                # Save to history
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO predictions (id, filename, predictions) VALUES (?, ?, ?)",
                    (str(uuid.uuid4()), image_file.name, json.dumps(prediction_data))
                )
                conn.commit()
                conn.close()
    finally:
        # Clean up files
        shutil.rmtree(unique_dir)

    return JSONResponse(content={"predictions": prediction_results})

@app.get("/tandan-predict/history")
async def get_prediction_history(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1)):
    offset = (page - 1) * page_size
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, predictions, timestamp FROM predictions ORDER BY timestamp DESC LIMIT ? OFFSET ?", (page_size, offset))
    records = cursor.fetchall()
    conn.close()

    history = [
        {
            "id": record[0],
            "filename": record[1],
            "predictions": json.loads(record[2]),
            "timestamp": record[3]
        }
        for record in records
    ]

    return JSONResponse(content={"history": history, "page": page, "page_size": page_size})
