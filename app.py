import json, io, os
import torch, timm
import numpy as np
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse

app = FastAPI()

CLASS_NAMES = json.load(open("class_names.json"))
NUM_CLASSES = len(CLASS_NAMES)

model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()
print(f"✅ Model ready — {NUM_CLASSES} classes")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

TREATMENTS = {
    "healthy": "Your plant is healthy! Keep up good care.",
    "blight":  "Apply copper-based fungicide. Remove infected leaves.",
    "rust":    "Apply sulfur-based fungicide. Improve air circulation.",
    "spot":    "Remove affected leaves. Apply neem oil every 7 days.",
    "mildew":  "Apply potassium bicarbonate. Reduce humidity.",
    "rot":     "Remove infected parts. Apply fungicide. Improve drainage.",
    "virus":   "Remove infected plants. Control insects.",
    "scorch":  "Ensure adequate watering. Avoid water stress.",
}

# Lowered to 0.20 — model confidence is genuinely low on real leaves
# The real fix is retraining the model with more data
CONFIDENCE_THRESHOLD = 0.20

# Classes that should always be rejected even if confidence is high
NON_LEAF_CLASSES = {
    "background_without_leaves",
    "background without leaves",
    "not a plant",
    "not plant",
    "invalid",
}


def format_class_name(raw: str) -> str:
    if "___" in raw:
        disease_part = raw.split("___")[1]
    else:
        disease_part = raw
    return disease_part.replace("_", " ").strip().title()


def get_treatment(raw: str) -> str:
    name_lower = raw.lower()
    for key, advice in TREATMENTS.items():
        if key in name_lower:
            return advice
    return "Consult a local agricultural expert."


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>Plant Disease Detection</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { font-family: Arial; background: #f0f7f0; min-height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 20px; }
            .card { background: white; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); padding: 40px; width: 100%; max-width: 600px; }
            h1 { color: #2E7D32; text-align: center; font-size: 26px; margin-bottom: 6px; }
            p.sub { text-align: center; color: #777; margin-bottom: 24px; font-size: 14px; }
            .upload-area { border: 2px dashed #2E7D32; border-radius: 10px; padding: 20px; text-align: center; cursor: pointer; margin-bottom: 16px; }
            #preview { max-width: 100%; max-height: 220px; border-radius: 8px; margin: 12px auto; display: none; }
            button { width: 100%; background: #2E7D32; color: white; border: none; padding: 14px; border-radius: 10px; font-size: 16px; cursor: pointer; }
            button:hover { background: #1B5E20; }
            button:disabled { background: #aaa; }
            #result { margin-top: 20px; padding: 20px; background: #E8F5E9; border-radius: 10px; display: none; }
            #result.error-result { background: #FFEBEE; }
            .disease { font-size: 20px; font-weight: bold; color: #1B5E20; margin-bottom: 6px; }
            .disease.error { color: #C62828; }
            .conf { color: #555; font-size: 14px; margin-bottom: 10px; }
            .treat { background: white; padding: 12px; border-radius: 8px; font-size: 14px; margin-bottom: 12px; }
            .top3-item { font-size: 13px; padding: 4px 0; color: #444; }
            .footer { text-align: center; color: #aaa; font-size: 12px; margin-top: 20px; }
            #top3Label { font-size: 13px; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🌿 Plant Disease Detection</h1>
            <p class="sub">Upload a leaf photo — AI will detect the disease instantly</p>
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                📁 Click to select leaf image
                <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="previewImg(event)">
            </div>
            <img id="preview">
            <button id="btn" onclick="predict()">🔍 Detect Disease</button>
            <div id="result">
                <div class="disease" id="diseaseName"></div>
                <div class="conf"    id="confScore"></div>
                <div class="treat"   id="treatment"></div>
                <b id="top3Label" style="font-size:13px; display:none">Top 3 Predictions:</b>
                <div id="top3"></div>
            </div>
        </div>
        <div class="footer">
            BABA GURU NANAK UNIVERSITY<br>
            Ali Haider | Fasi-Ur-Rehman | M.Hassan | Aysha Hassan
        </div>
        <script>
            function previewImg(e) {
                const img = document.getElementById("preview");
                img.src = URL.createObjectURL(e.target.files[0]);
                img.style.display = "block";
            }
            async function predict() {
                const file = document.getElementById("fileInput").files[0];
                if (!file) { alert("Please select an image!"); return; }
                const btn = document.getElementById("btn");
                btn.disabled = true; btn.innerText = "⏳ Analyzing...";
                const fd = new FormData();
                fd.append("file", file);
                try {
                    const res  = await fetch("/predict", { method: "POST", body: fd });
                    const data = await res.json();

                    const resultDiv = document.getElementById("result");
                    const diseaseEl = document.getElementById("diseaseName");
                    const top3Label = document.getElementById("top3Label");
                    const top3Div   = document.getElementById("top3");

                    diseaseEl.innerText = data.disease;
                    document.getElementById("confScore").innerText = "Confidence: " + data.confidence + "%";
                    document.getElementById("treatment").innerText = data.treatment;

                    # if (data.top3 && data.top3.length > 0) {
                    #     let html = "";
                    #     const medals = ["🥇","🥈","🥉"];
                    #     data.top3.forEach((x,i) => html += `<div class='top3-item'>${medals[i]} ${x.disease} — ${x.probability}%</div>`);
                    #     top3Div.innerHTML       = html;
                    #     top3Label.style.display = "block";
                    #     diseaseEl.classList.remove("error");
                    #     resultDiv.classList.remove("error-result");
                    # } else {
                    #     top3Div.innerHTML       = "";
                    #     top3Label.style.display = "none";
                    #     diseaseEl.classList.add("error");
                    #     resultDiv.classList.add("error-result");
                    # }

                    resultDiv.style.display = "block";
                } catch(e) { alert("Error: " + e); }
                finally { btn.disabled = false; btn.innerText = "🔍 Detect Disease"; }
            }
        </script>
    </body>
    </html>
    """


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img    = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    top3_p, top3_i = torch.topk(probs, 3)

    top_class_raw = CLASS_NAMES[top3_i[0].item()]
    top_conf      = top3_p[0].item()

    # Print to Railway logs for debugging — remove after fixing
    print("=" * 40)
    print(f"TOP PREDICTION : {top_class_raw} → {top_conf*100:.2f}%")
    for i in range(min(5, len(top3_i))):
        print(f"  #{i+1}: {CLASS_NAMES[top3_i[i].item()]} → {top3_p[i].item()*100:.2f}%")
    print("=" * 40)

    # Reject if top class is a known non-leaf class (even with high confidence)
    is_non_leaf_class = top_class_raw.lower().strip() in NON_LEAF_CLASSES

    # Reject if confidence is below threshold
    is_low_confidence = top_conf < CONFIDENCE_THRESHOLD

    if is_low_confidence or is_non_leaf_class:
        return {
            "disease":    "❌ Not a Plant Leaf",
            "confidence": round(top_conf * 100, 2),
            "treatment":  "⚠️ Please upload a clear photo of a plant leaf.",
            "top3":       []
        }

    return {
        "disease":    format_class_name(top_class_raw),
        "confidence": round(top_conf * 100, 2),
        "treatment":  get_treatment(top_class_raw),
        "top3": [
            {
                "disease":     format_class_name(CLASS_NAMES[top3_i[i].item()]),
                "probability": round(top3_p[i].item() * 100, 2)
            }
            for i in range(3)
        ]
    }
