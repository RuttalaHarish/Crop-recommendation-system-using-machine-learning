# app.py
from flask import Flask, request, render_template
import numpy as np
import os
import joblib
from pathlib import Path

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
STAND_SCALER_PATH = os.path.join(BASE_DIR, "standscaler.pkl")
MINMAX_SCALER_PATH = os.path.join(BASE_DIR, "minmaxscaler.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")  # optional

# --- Load model & scalers (fail gracefully) ---
model = sc = ms = label_encoder = None

def safe_load(path, loader=joblib.load):
    """Try to load a file with joblib; print and return None on failure."""
    try:
        if Path(path).exists():
            return loader(path)
        else:
            print(f"[WARN] File not found: {path}")
            return None
    except Exception as e:
        print(f"[ERROR] Failed to load {path!r}: {e}")
        return None

# Use joblib for sklearn objects
model = safe_load(MODEL_PATH)
sc = safe_load(STAND_SCALER_PATH)
ms = safe_load(MINMAX_SCALER_PATH)
label_encoder = safe_load(LABEL_ENCODER_PATH)  # may be None (optional)

if model is not None:
    print("Model loaded.")
if sc is not None:
    print("Standard scaler loaded.")
if ms is not None:
    print("MinMax scaler loaded.")
if label_encoder is not None:
    print("Label encoder loaded.")

# --- Flask app ---
app = Flask(__name__)

@app.route("/")
def index():
    """
    index.html should include a form that POSTs to /predict with these field names:
    'Nitrogen', 'Phosporus', 'Potassium', 'Temperature', 'Humidity', 'Ph', 'Rainfall'
    """
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Check that required objects are available
    if model is None or sc is None or ms is None:
        result = ("Server configuration error: model or scaler not loaded. "
                  "Ensure model.pkl, standscaler.pkl and minmaxscaler.pkl exist in the app folder.")
        return render_template("index.html", result=result)

    # Expected input fields (must match your index.html form names)
    fields = ['Nitrogen', 'Phosporus', 'Potassium', 'Temperature', 'Humidity', 'Ph', 'Rainfall']
    values = []
    for field in fields:
        raw = request.form.get(field)
        if raw is None or raw.strip() == "":
            return render_template("index.html", result=f"Please provide a value for {field}.")
        try:
            val = float(raw)
        except ValueError:
            return render_template("index.html", result=f"Invalid numeric value for {field}: '{raw}'")
        values.append(val)

    # Prepare features
    single_pred = np.array(values).reshape(1, -1)

    # Scale features (same order used during training): MinMax -> Standard
    try:
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)
    except Exception as e:
        return render_template("index.html", result=f"Error while scaling features: {e}")

    # Predict
    try:
        prediction = model.predict(final_features)
    except Exception as e:
        return render_template("index.html", result=f"Error during model prediction: {e}")

    pred = prediction[0]

    # Try to resolve prediction to crop name:
    crop_name = None

    # 1) If label encoder exists, invert-transform the integer label
    if label_encoder is not None:
        try:
            # If model returns numpy int, convert to python int
            crop_name = label_encoder.inverse_transform([int(pred)])[0]
        except Exception as e:
            # fallback to None and continue
            print(f"[WARN] label_encoder inverse transform failed: {e}")
            crop_name = None

    # 2) If label encoder not present or failed, accept string outputs directly
    if crop_name is None:
        if isinstance(pred, str):
            crop_name = pred

    # 3) Final fallback: numeric label -> name mapping (if you used numeric labels without encoder)
    if crop_name is None:
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
            8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
            19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }
        try:
            crop_name = crop_dict.get(int(pred))
        except Exception:
            crop_name = None

    # Prepare result message
    if crop_name:
        result = f"{crop_name} is the best crop to be cultivated right there."
    else:
        result = f"Sorry, could not determine the best crop. Model output: {pred}"

    return render_template("index.html", result=result)


if __name__ == "__main__":
    # Development server (fine for local testing)
    app.run(debug=True)
