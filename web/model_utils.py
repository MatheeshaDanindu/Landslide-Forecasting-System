"""Loads the trained checkpoints and builds synthetic demo patches for the
web demo. No live satellite fetch here -- see README in this folder.
"""
import io

import numpy as np
import torch
import segmentation_models_pytorch as smp

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
CKPT_DIR = REPO_ROOT / "Trained Model"
IN_CHANNELS = 16
PATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARCH = {"baseline": ("Unet", "resnet34"), "comparison": ("DeepLabV3Plus", "resnet34")}


def build_model(spec_key):
    arch_name, encoder = ARCH[spec_key]
    arch_cls = getattr(smp, arch_name)
    model = arch_cls(encoder_name=encoder, encoder_weights=None, in_channels=IN_CHANNELS, classes=1)
    return model.to(DEVICE)


def load_checkpoint(spec_key, filename):
    model = build_model(spec_key)
    state = torch.load(CKPT_DIR / filename, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


SCENARIOS = {
    "steep_bare_slope": dict(slope_deg=38, veg=0.15, bare=0.75, seed=1),
    "flat_vegetated": dict(slope_deg=4, veg=0.85, bare=0.05, seed=2),
    "moderate_mixed": dict(slope_deg=22, veg=0.45, bare=0.35, seed=3),
}


def make_demo_patch(scenario_key):
    cfg = SCENARIOS[scenario_key]
    rng = np.random.default_rng(cfg["seed"])
    h = w = PATCH_SIZE
    noise = lambda scale: rng.normal(0, scale, (h, w)).astype(np.float32)

    veg, bare = cfg["veg"], cfg["bare"]
    bands = {
        "B02": 0.05 + 0.05 * bare + noise(0.01), "B03": 0.06 + 0.06 * bare + noise(0.01),
        "B04": 0.05 + 0.10 * bare + noise(0.01), "B05": 0.10 + 0.05 * veg + noise(0.01),
        "B06": 0.20 + 0.15 * veg + noise(0.01), "B07": 0.25 + 0.15 * veg + noise(0.01),
        "B08": 0.10 + 0.35 * veg + noise(0.02), "B8A": 0.25 + 0.15 * veg + noise(0.01),
        "B11": 0.15 + 0.20 * bare - 0.05 * veg + noise(0.01), "B12": 0.10 + 0.20 * bare - 0.05 * veg + noise(0.01),
    }
    band_order = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    channels = [np.clip(bands[b], 0, 1) for b in band_order]

    ndvi = (bands["B08"] - bands["B04"]) / (bands["B08"] + bands["B04"] + 1e-6)
    bsi = ((bands["B11"] + bands["B04"]) - (bands["B08"] + bands["B02"])) / \
          ((bands["B11"] + bands["B04"]) + (bands["B08"] + bands["B02"]) + 1e-6)
    mndwi = (bands["B03"] - bands["B11"]) / (bands["B03"] + bands["B11"] + 1e-6)
    channels += [ndvi, bsi, mndwi]

    slope = np.full((h, w), cfg["slope_deg"], dtype=np.float32) + noise(3.0)
    aspect = np.clip(rng.uniform(0, 360, (h, w)).astype(np.float32), 0, 360)
    curvature = noise(0.02)
    channels += [slope, aspect, curvature]

    patch = np.stack(channels).astype(np.float32)
    rgb_preview = np.stack([bands["B04"], bands["B03"], bands["B02"]], axis=-1)
    rgb_preview = np.clip(rgb_preview / max(rgb_preview.max(), 1e-6), 0, 1)
    return torch.from_numpy(patch), rgb_preview


# ponytail: a real prediction needs true multispectral bands + a real DEM.
# An uploaded photo only has RGB -- the other 13 channels below are crude
# heuristics from those 3 values, not reconstructed spectra. Demo only.
BAND_ORDER = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]


def make_patch_from_image(image_bytes, slope_deg=15.0):
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((PATCH_SIZE, PATCH_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    nir_like = np.clip(1.4 * g - 0.2 * r, 0, 1)          # greener -> more NIR-bright
    swir_like = np.clip(0.6 * r - 0.3 * g + 0.3, 0, 1)   # less green -> more soil/SWIR-bright
    bands = {"B02": b, "B03": g, "B04": r,
             "B05": (r + g) / 2, "B06": (r + g) / 2, "B07": nir_like,
             "B08": nir_like, "B8A": nir_like, "B11": swir_like, "B12": swir_like}
    channels = [bands[k] for k in BAND_ORDER]

    ndvi = (bands["B08"] - bands["B04"]) / (bands["B08"] + bands["B04"] + 1e-6)
    bsi = ((bands["B11"] + bands["B04"]) - (bands["B08"] + bands["B02"])) / \
          ((bands["B11"] + bands["B04"]) + (bands["B08"] + bands["B02"]) + 1e-6)
    mndwi = (bands["B03"] - bands["B11"]) / (bands["B03"] + bands["B11"] + 1e-6)
    channels += [ndvi, bsi, mndwi]

    h, w = r.shape
    channels += [np.full((h, w), slope_deg, dtype=np.float32),
                 np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)]
    patch = np.stack(channels).astype(np.float32)
    return torch.from_numpy(patch), arr


def predict(model, patch_tensor):
    with torch.no_grad():
        x = patch_tensor.unsqueeze(0).to(DEVICE)
        probs = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return probs


if __name__ == "__main__":
    m = load_checkpoint("baseline", "baseline_unet.pt")
    patch, rgb = make_demo_patch("steep_bare_slope")
    probs = predict(m, patch)
    assert probs.shape == (PATCH_SIZE, PATCH_SIZE)
    assert 0.0 <= probs.min() and probs.max() <= 1.0
    print(f"self-check ok -- mean predicted probability: {probs.mean():.4f}")
