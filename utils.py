# utils.py
import os
import uuid
import base64
import io
from typing import Optional, List
from PIL import Image
import numpy as np


def clean_label(label: str) -> str:
    """Turn 'hand_eczema' -> 'Hand Eczema'."""
    return str(label).replace("_", " ").title()


def save_pil_image(img: Image.Image, upload_dir: str, prefix: str = "img"):
    """Save a PIL image to /static/uploads and return (filename, full_path)."""
    os.makedirs(upload_dir, exist_ok=True)
    fname = f"{prefix}_{uuid.uuid4().hex}.jpg"
    full = os.path.join(upload_dir, fname)
    img.save(full, format="JPEG")
    return fname, full


def save_base64_image(dataurl: str, upload_dir: str, prefix: str = "cam"):
    """
    Accepts 'data:image/png;base64,...' or raw base64. Returns (filename, full_path).
    """
    if dataurl.startswith("data:"):
        _, b64 = dataurl.split(",", 1)
    else:
        b64 = dataurl
    b = base64.b64decode(b64)
    img = Image.open(io.BytesIO(b)).convert("RGB")
    return save_pil_image(img, upload_dir, prefix=prefix)


def detect_and_crop_face(face_detector, image_path: str, upload_dir: str, prefix: str = "crop") -> Optional[str]:
    """
    Use a YOLO face detector to crop the largest detected face.
    Returns saved crop path or None if no face detected.
    """
    try:
        results = face_detector(image_path)
        if not results or len(results[0].boxes) == 0:
            return None

        # pick the largest box
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if boxes.size == 0:
            return None
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = int(np.argmax(areas))
        x1, y1, x2, y2 = boxes[idx].astype(int)

        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face_img = img.crop((x1, y1, x2, y2))
        # Optional: slightly pad / resize if you want
        return save_pil_image(face_img, upload_dir, prefix=prefix)[1]
    except Exception:
        return None


def predict_image(model, image_path: str) -> str:
    """
    Run YOLO classification and return its top-1 class name.
    """
    res = model(image_path)
    label_idx = res[0].probs.top1
    class_name = res[0].names[label_idx]
    return class_name


def predict_questionnaire(model, inputs: list):
    """
    Run classic ML model's predict() on a 1-row feature list.
    """
    return model.predict([inputs])[0]


def gather_recommendations_from_csv(df, skin_type: str, disease: str) -> List[str]:
    """
    Find rows where SkinType & disease match (case-insensitive) and return
    the non-empty product names from the product columns.
    Expected columns: SkinType, disease, cleanser, toner, serum, moisturizer, sunscreen, treatment
    """
    if df is None or df.empty:
        return ["No exact recommendation found."]

    # Normalize column names
    df_cols_norm = {c.lower().strip(): c for c in df.columns}
    # Required columns
    needed = ["skintype", "disease"]
    for need in needed:
        if need not in df_cols_norm:
            return ["No exact recommendation found."]

    # Product columns (only add those that actually exist in the CSV)
    possible_product_cols = ["cleanser", "toner", "serum", "moisturizer", "sunscreen", "treatment"]
    product_cols = [df_cols_norm[c] for c in possible_product_cols if c in df_cols_norm]

    # Filter rows
    st_col = df_cols_norm["skintype"]
    ds_col = df_cols_norm["disease"]
    mask = (df[st_col].str.lower() == skin_type.lower()) & (df[ds_col].str.lower() == disease.lower())
    sub = df.loc[mask]

    if sub.empty:
        return ["No exact recommendation found."]

    # Collect all non-empty product entries
    picks = []
    for _, row in sub.iterrows():
        for pc in product_cols:
            val = str(row[pc]).strip()
            if val and val.lower() not in ("nan", "none"):
                picks.append(val)

    # Deduplicate but keep order
    seen = set()
    out = []
    for p in picks:
        if p not in seen:
            seen.add(p)
            out.append(p)

    return out if out else ["No exact recommendation found."]
