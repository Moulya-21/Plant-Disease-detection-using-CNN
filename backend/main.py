import io
import os
import time
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from backend.model_service import (
    get_model,
    generate_gradcam_and_predict,
    get_curated_samples,
    get_sample_image_file,
    get_model_metrics
)
from backend.disease_info import CLASS_NAMES, DISEASE_DATABASE
from backend.auth_service import router as auth_router, get_current_user_from_token

app = FastAPI(
    title="PlantCare AI - Plant Disease Recognition & Grad-CAM Explainability",
    description="Full-stack AI diagnostic engine for crop disease identification and neural visual explainability with robust TTA.",
    version="2.1.0"
)

# Enable CORS for Vite and local clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount authentication router
app.include_router(auth_router)

# In-memory recent scan store (keeps last 20 scans)
recent_scans = []

class PredictBase64Request(BaseModel):
    image_base64: str
    alpha: Optional[float] = 0.45


@app.on_event("startup")
async def startup_event():
    # Warm up the model on startup
    try:
        print("Initializing TensorFlow model and caching weights...")
        get_model()
        print("TensorFlow model loaded successfully.")
    except Exception as e:
        print(f"Warning: Model preload failed: {e}")


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "service": "PlantCare AI Engine",
        "version": "2.0.0",
        "total_classes": len(CLASS_NAMES),
        "timestamp": time.time()
    }


@app.get("/api/classes")
def get_classes():
    """
    Returns full catalog of all 38 classes with disease details, severity, and treatments.
    """
    classes_list = []
    for cname in CLASS_NAMES:
        info = DISEASE_DATABASE.get(cname, {})
        classes_list.append({
            "raw_class": cname,
            "crop": info.get("crop", cname.split("___")[0].replace("_", " ")),
            "condition": info.get("condition", cname.split("___")[-1].replace("_", " ")),
            "is_healthy": info.get("is_healthy", "healthy" in cname.lower()),
            "severity": info.get("severity", "Unknown"),
            "pathogen": info.get("pathogen", "N/A"),
            "symptoms": info.get("symptoms", ""),
            "organic_treatment": info.get("organic_treatment", ""),
            "chemical_treatment": info.get("chemical_treatment", ""),
            "prevention": info.get("prevention", "")
        })
    return {"classes": classes_list, "total": len(classes_list)}


@app.get("/api/samples")
def get_samples():
    """
    Returns curated sample leaves from dataset/valid.
    """
    samples = get_curated_samples()
    return {"samples": samples, "count": len(samples)}


@app.get("/api/sample-image/{class_name}")
def get_sample_image(class_name: str):
    """
    Streams a sample image file from disk.
    """
    image_path = get_sample_image_file(class_name)
    if not image_path or not os.path.isfile(image_path):
        raise HTTPException(status_code=404, detail=f"Sample image for class {class_name} not found")
    return FileResponse(image_path)


@app.post("/api/predict")
async def predict_image(
    file: Optional[UploadFile] = File(None),
    alpha: float = Form(0.45),
    sample_class: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None)
):
    """
    Analyzes an uploaded leaf image or sample class and generates Grad-CAM explainability.
    Requires an active authenticated agronomist session.
    """
    user = get_current_user_from_token(authorization)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please sign in to run plant disease diagnostics."
        )

    try:
        if sample_class:
            img_path = get_sample_image_file(sample_class)
            if not img_path or not os.path.isfile(img_path):
                raise HTTPException(status_code=404, detail="Requested sample not found.")
            pil_image = Image.open(img_path)
            filename = os.path.basename(img_path)
        elif file:
            content = await file.read()
            pil_image = Image.open(io.BytesIO(content))
            filename = file.filename
        else:
            raise HTTPException(status_code=400, detail="Provide an image file or a sample_class.")

        result = generate_gradcam_and_predict(pil_image, alpha=float(alpha))

        # Store in recent scans
        scan_entry = {
            "id": int(time.time() * 1000),
            "timestamp": time.strftime("%H:%M:%S - %b %d"),
            "crop": result["prediction"]["crop"],
            "condition": result["prediction"]["condition"],
            "is_healthy": result["prediction"]["is_healthy"],
            "confidence": result["prediction"]["confidence"],
            "confidence_status": result["prediction"]["confidence_status"],
            "thumbnail_base64": result["images"]["original_base64"],
            "filename": filename,
            "user_email": user.get("email"),
            "user_name": user.get("full_name")
        }
        recent_scans.insert(0, scan_entry)
        if len(recent_scans) > 20:
            recent_scans.pop()

        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics")
@app.get("/api/model-info")
def get_metrics():
    """
    Returns training history, accuracy/loss curves, and CNN layer topology.
    """
    try:
        return get_model_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/recent-scans")
def get_recent_scans():
    """
    Returns the recent scan history.
    """
    return {"scans": recent_scans}


# Serve built frontend if exists
FRONTEND_DIST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")