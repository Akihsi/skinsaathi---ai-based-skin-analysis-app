from flask import Flask, render_template, request, session, redirect, url_for
import os
import uuid
import base64
import pandas as pd
from PIL import Image
import joblib
from ultralytics import YOLO

from utils import (
    clean_label,
    save_base64_image,
    detect_and_crop_face,
    predict_image,
    predict_questionnaire,
    gather_recommendations_from_csv,
)

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Required for session

# Paths and folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DATASET_PATH = os.path.join(BASE_DIR, "data", "skincare_dataset.csv")

# Load models
skin_type_model = YOLO(os.path.join(BASE_DIR, "models", "skin_type_model.pt"))
skin_disease_model = YOLO(os.path.join(BASE_DIR, "models", "skin_disease_model.pt"))
face_detector = YOLO(os.path.join(BASE_DIR, "models", "face.pt"))

QUESTIONNAIRE_MODEL_PATH = os.path.join(BASE_DIR, "models", "logistic_model.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

questionnaire_model = joblib.load(QUESTIONNAIRE_MODEL_PATH) if os.path.exists(QUESTIONNAIRE_MODEL_PATH) else None
label_encoder = joblib.load(LABEL_ENCODER_PATH) if os.path.exists(LABEL_ENCODER_PATH) else None


# -----------------------------
# Step 1: Questionnaire
# -----------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# -----------------------------
# Step 2: Upload Image
# -----------------------------
@app.route("/upload", methods=["POST"])
def upload():
    # Store questionnaire data in session
    session["questionnaire"] = {
        "hydration": request.form.get("hydration"),
        "diet": request.form.get("diet"),
        "sleep": request.form.get("sleep"),
        "stress": request.form.get("stress"),
        "pollution": request.form.get("pollution"),
        "climate": request.form.get("climate"),
        "routine": request.form.get("routine"),
    }
    return render_template("upload.html")


# -----------------------------
# Step 3: Analyze
# -----------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    questionnaire_data = session.get("questionnaire", {})

    # Prepare questionnaire inputs
    try:
        q_inputs = [int(questionnaire_data.get(k, 0)) for k in [
            "hydration", "diet", "sleep", "stress", "pollution", "climate", "routine"
        ]]
    except Exception:
        q_inputs = [0] * 7

    # Get image data
    frames = {"left": None, "right": None, "front": None}
    filenames = {"left": None, "right": None, "front": None}

    # A) Camera (base64)
    for side in ("left", "right", "front"):
        b64_field = request.form.get(f"frame_{side}")
        if b64_field:
            fname, path = save_base64_image(b64_field, upload_dir=UPLOAD_FOLDER, prefix=side)
            frames[side], filenames[side] = path, fname

    # B) File upload
    for side in ("left", "right", "front"):
        if frames[side] is None:
            f = request.files.get(side)
            if f and f.filename:
                fname = f"{uuid.uuid4().hex}_{f.filename}"
                path = os.path.join(UPLOAD_FOLDER, fname)
                f.save(path)
                frames[side], filenames[side] = path, fname

    # C) Fallback to single image
    single_uploaded = False
    if all(v is None for v in frames.values()):
        f = request.files.get("image")
        if f and f.filename:
            fname = f"{uuid.uuid4().hex}_{f.filename}"
            path = os.path.join(UPLOAD_FOLDER, fname)
            f.save(path)
            frames["front"], filenames["front"], single_uploaded = path, fname, True

    if all(v is None for v in frames.values()):
        return "No image(s) provided.", 400

    # Predict skin type & disease
    weights = {"left": 0.3, "right": 0.3, "front": 0.2}
    skin_type_scores = {}
    skin_disease_scores = {}

    for side, path in frames.items():
        if not path:
            continue
        crop_path = detect_and_crop_face(face_detector, path, upload_dir=UPLOAD_FOLDER, prefix=f"{side}_crop")

        try:
            st_raw = predict_image(skin_type_model, crop_path or path)
            sd_raw = predict_image(skin_disease_model, crop_path or path)
        except Exception:
            st_raw, sd_raw = None, None

        if st_raw:
            st = clean_label(st_raw)
            skin_type_scores[st] = skin_type_scores.get(st, 0) + weights.get(side, 0)
        if sd_raw:
            sd = clean_label(sd_raw)
            skin_disease_scores[sd] = skin_disease_scores.get(sd, 0) + weights.get(side, 0)

    # Add questionnaire prediction (0.2 weight)
    if questionnaire_model and label_encoder:
        try:
            q_pred = predict_questionnaire(questionnaire_model, q_inputs)
            q_label = label_encoder.inverse_transform([q_pred])[0] if not isinstance(q_pred, str) else q_pred
            q_label = clean_label(q_label)
            skin_type_scores[q_label] = skin_type_scores.get(q_label, 0) + 0.2
        except:
            pass

    # Final prediction
    final_skin_type = max(skin_type_scores, key=skin_type_scores.get) if skin_type_scores else "Unknown"
    final_disease = max(skin_disease_scores, key=skin_disease_scores.get) if skin_disease_scores else "Unknown"

    # Fallback check
    if single_uploaded and frames["front"]:
        if final_skin_type == "Unknown":
            try: final_skin_type = clean_label(predict_image(skin_type_model, frames["front"]))
            except: pass
        if final_disease == "Unknown":
            try: final_disease = clean_label(predict_image(skin_disease_model, frames["front"]))
            except: pass

    # Recommendations
    try:
        df = pd.read_csv(DATASET_PATH)
    except Exception as e:
        return f"Failed to load product dataset: {e}", 500

    products = gather_recommendations_from_csv(df, final_skin_type, final_disease)
    display_fname = filenames.get("front") or filenames.get("left") or filenames.get("right")

    # Clear session
    session.pop("questionnaire", None)

    return render_template("result.html",
                           skin_type=final_skin_type,
                           skin_disease=final_disease,
                           recommendations=products,
                           image_filename=display_fname)


if __name__ == "__main__":
    app.run(debug=True)
