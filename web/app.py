"""Minimal local demo server for the trained landslide-susceptibility models.
Run: python app.py, then open http://127.0.0.1:5000
No live satellite fetch -- see model_utils.py for why.
"""
import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, jsonify, render_template_string, request

from model_utils import SCENARIOS, load_checkpoint, make_demo_patch, make_patch_from_image, predict

app = Flask(__name__)
MODELS = {
    "baseline": load_checkpoint("baseline", "baseline_unet.pt"),
    "comparison": load_checkpoint("comparison", "fold_cluster0_comparison.pt"),
}

PAGE = """
<!doctype html><html><head><meta charset="utf-8">
<title>Landslide Susceptibility -- Local Demo</title>
<style>
  body{font-family:system-ui,sans-serif;max-width:900px;margin:32px auto;padding:0 16px;color:#222}
  h1{font-size:1.3rem}
  .banner{background:#fff3cd;border:1px solid #e0c46c;padding:10px 14px;border-radius:6px;font-size:0.9rem;margin-bottom:20px}
  .controls{display:flex;gap:10px;align-items:center;margin-bottom:20px;flex-wrap:wrap}
  select,button{padding:8px 12px;font-size:0.95rem}
  button{cursor:pointer;background:#2c5f2d;color:#fff;border:none;border-radius:5px}
  button:disabled{opacity:0.6;cursor:wait}
  .imgs{display:flex;gap:20px;flex-wrap:wrap}
  .imgs figure{margin:0;flex:1;min-width:220px}
  .imgs img{width:100%;border:1px solid #ccc;border-radius:4px}
  figcaption{font-size:0.8rem;color:#555;margin-top:4px}
  .stats{margin-top:14px;font-family:monospace;font-size:0.9rem;background:#f5f5f0;padding:10px;border-radius:6px}
</style></head>
<body>
  <h1>Landslide Susceptibility -- Local Demo</h1>
  <div class="banner">Synthetic demo input (no real Sentinel-2/DEM tile fetched) --
    runs the actual trained checkpoint, on a procedurally-generated 16-channel
    patch matching one of three terrain scenarios. Not a real prediction on a
    real location.</div>
  <label>Model:
    <select id="model"><option value="baseline">Baseline (U-Net)</option>
      <option value="comparison">Comparison (DeepLabV3+)</option></select>
  </label>

  <div class="controls" style="margin-top:16px">
    <label>Scenario:
      <select id="scenario">{% for s in scenarios %}<option value="{{s}}">{{s}}</option>{% endfor %}</select>
    </label>
    <button id="run">Run prediction</button>
  </div>

  <div class="controls">
    <label>Upload an image: <input type="file" id="file" accept="image/*"></label>
    <label>Assumed slope (deg): <input type="number" id="slope" value="15" min="0" max="60" style="width:60px"></label>
    <button id="runUpload">Run on uploaded image</button>
  </div>
  <div class="banner" style="background:#fde2e2;border-color:#e08c8c">Uploaded-image mode is a bigger
    approximation than the scenarios above: only 3 of the 16 model inputs are real (RGB), the other 13
    (NIR/SWIR bands, indices, terrain) are crude heuristics guessed from those 3 values, not real
    spectral or elevation data. Interaction demo only.</div>

  <div class="imgs">
    <figure><img id="rgb" src=""><figcaption>Input (RGB preview)</figcaption></figure>
    <figure><img id="mask" src=""><figcaption>Predicted susceptibility (probability heatmap)</figcaption></figure>
  </div>
  <div class="stats" id="stats"></div>
<script>
function showResult(data) {
  document.getElementById('rgb').src = 'data:image/png;base64,' + data.rgb;
  document.getElementById('mask').src = 'data:image/png;base64,' + data.mask;
  document.getElementById('stats').textContent =
    `mean probability: ${data.mean_prob.toFixed(4)}   max probability: ${data.max_prob.toFixed(4)}`;
}
async function runButton(btn, fn) {
  btn.disabled = true; const old = btn.textContent; btn.textContent = 'Running...';
  try { showResult(await fn()); } finally { btn.disabled = false; btn.textContent = old; }
}
document.getElementById('run').onclick = (e) => runButton(e.target, async () => {
  const model = document.getElementById('model').value;
  const scenario = document.getElementById('scenario').value;
  const res = await fetch(`/predict?model=${model}&scenario=${scenario}`);
  return res.json();
});
document.getElementById('runUpload').onclick = (e) => runButton(e.target, async () => {
  const file = document.getElementById('file').files[0];
  if (!file) { alert('choose an image first'); throw new Error('no file'); }
  const model = document.getElementById('model').value;
  const slope = document.getElementById('slope').value;
  const body = new FormData();
  body.append('image', file);
  const res = await fetch(`/predict_upload?model=${model}&slope_deg=${slope}`, {method: 'POST', body});
  return res.json();
});
</script>
</body></html>
"""


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.route("/")
def index():
    return render_template_string(PAGE, scenarios=list(SCENARIOS))


def render_prediction(model_key, patch, rgb):
    probs = predict(MODELS[model_key], patch)
    fig1, ax1 = plt.subplots(figsize=(3, 3))
    ax1.imshow(rgb); ax1.axis("off")
    fig2, ax2 = plt.subplots(figsize=(3, 3))
    im = ax2.imshow(probs, cmap="inferno", vmin=0, vmax=1); ax2.axis("off")
    fig2.colorbar(im, ax=ax2, fraction=0.046)
    return jsonify(rgb=fig_to_b64(fig1), mask=fig_to_b64(fig2),
                    mean_prob=float(probs.mean()), max_prob=float(probs.max()))


@app.route("/predict")
def predict_route():
    scenario = request.args.get("scenario", "steep_bare_slope")
    model_key = request.args.get("model", "baseline")
    patch, rgb = make_demo_patch(scenario)
    return render_prediction(model_key, patch, rgb)


@app.route("/predict_upload", methods=["POST"])
def predict_upload_route():
    model_key = request.args.get("model", "baseline")
    slope_deg = float(request.args.get("slope_deg", 15.0))
    image_file = request.files["image"]
    patch, rgb = make_patch_from_image(image_file.read(), slope_deg=slope_deg)
    return render_prediction(model_key, patch, rgb)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
