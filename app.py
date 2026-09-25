"""Video Lab — local Gradio front-end for ComfyUI-driven video generation.

Runs fully offline against a local ComfyUI instance (http://127.0.0.1:8188).
No models are loaded in this process; every generation is a ComfyUI API-format
workflow POSTed to /prompt, polled via /history, and fetched via /view.
"""
import io
import json
import os
import random
import time
import uuid

import gradio as gr
import requests
import websocket
from PIL import Image

COMFY_URL = "http://127.0.0.1:8188"
COMFY_WS_URL = "ws://127.0.0.1:8188/ws"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
CLIENT_ID = str(uuid.uuid4())

# ---------------------------------------------------------------------------
# Model files — edit these to match what's in your ComfyUI models/ folders.
# ---------------------------------------------------------------------------
MODEL_FILES = {
    "wan_t2v_unet": "Wan2.1-T2V-1.3B-Q4_K_M.gguf",
    "wan_vace_unet": "Wan2.1-VACE-1.3B-Q4_K_M.gguf",
    "wan_clip": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "wan_vae": "wan_2.1_vae.safetensors",
    "ltx_unet": "ltxv-2b-0.9.6-distilled-04-25-Q6_K.gguf",
    "ltx_clip": "t5xxl_fp8_e4m3fn_scaled.safetensors",
    "ltx_vae": "LTX-Video-0.9.6-VAE-BF16.safetensors",
}

# (config key, human label, ComfyUI loader node, its input name)
MODEL_SLOTS = [
    ("wan_t2v_unet", "Wan T2V 1.3B (diffusion model)", "UnetLoaderGGUF", "unet_name"),
    ("wan_vace_unet", "Wan VACE 1.3B (diffusion model)", "UnetLoaderGGUF", "unet_name"),
    ("wan_clip", "Wan text encoder (umt5_xxl)", "CLIPLoader", "clip_name"),
    ("wan_vae", "Wan VAE", "VAELoader", "vae_name"),
    ("ltx_unet", "LTX-Video 2B distilled (diffusion model)", "UnetLoaderGGUF", "unet_name"),
    ("ltx_clip", "LTX text encoder (t5xxl)", "CLIPLoader", "clip_name"),
    ("ltx_vae", "LTX VAE", "VAELoader", "vae_name"),
]

REQUIRED_NODES = [
    "UnetLoaderGGUF", "CLIPLoader", "VAELoader", "CLIPTextEncode",
    "WanVaceToVideo", "WanImageToVideo", "TrimVideoLatent", "ModelSamplingSD3",
    "KSampler", "EmptyLTXVLatentVideo", "LTXVConditioning", "LTXVAddGuide",
    "LTXVCropGuides", "LTXVPreprocess", "LTXVScheduler", "KSamplerSelect",
    "SamplerCustom", "VAEDecode", "CreateVideo", "SaveVideo", "LoadImage",
]

DEFAULT_NEGATIVE = (
    "blurry, low quality, distorted, deformed, watermark, text, subtitles, "
    "static, worst quality, jpeg artifacts, overexposed, extra limbs"
)

LTX_DEFAULTS = dict(width=768, height=512, frames=97, steps=8, cfg=1.0, fps=24)
WAN_DEFAULTS = dict(width=832, height=480, frames=81, steps=30, cfg=6.0, fps=16)


def snap_wh(value, model):
    m = 32 if model == "ltx" else 16
    return max(m, int(round(value / m)) * m)


def snap_frames(value, model):
    if model == "ltx":
        k = max(0, round((value - 1) / 8))
        return max(1, k * 8 + 1)
    k = max(0, round((value - 1) / 4))
    return max(1, k * 4 + 1)


def model_defaults(model):
    d = dict(LTX_DEFAULTS if model == "ltx" else WAN_DEFAULTS)
    return d["width"], d["height"], d["frames"], d["steps"], d["cfg"]


# ---------------------------------------------------------------------------
# Minimal API-format workflow graph builder
# ---------------------------------------------------------------------------
class Graph:
    def __init__(self):
        self.nodes = {}
        self._n = 0

    def add(self, class_type, **inputs):
        self._n += 1
        nid = str(self._n)
        self.nodes[nid] = {"class_type": class_type, "inputs": inputs}
        return nid

    @staticmethod
    def o(node_id, index=0):
        return [node_id, index]


class ComfyError(Exception):
    pass


class ComfyClient:
    def __init__(self, base_url=COMFY_URL):
        self.base = base_url

    def _get(self, path, **kw):
        try:
            r = requests.get(self.base + path, timeout=kw.pop("timeout", 15), **kw)
        except requests.exceptions.ConnectionError:
            raise ComfyError(
                f"Can't reach ComfyUI at {self.base}. Is it running? "
                f"(start it with: python main.py)"
            )
        return r

    def is_up(self):
        try:
            r = self._get("/system_stats", timeout=3)
            return r.status_code == 200
        except ComfyError:
            return False

    def object_info(self, node_class=None):
        path = f"/object_info/{node_class}" if node_class else "/object_info"
        r = self._get(path)
        if r.status_code != 200:
            raise ComfyError(f"object_info failed: {r.status_code}")
        return r.json()

    def upload_image(self, pil_image, name=None):
        name = name or f"vidlab_{uuid.uuid4().hex}.png"
        buf = io.BytesIO()
        pil_image.convert("RGB").save(buf, format="PNG")
        buf.seek(0)
        try:
            r = requests.post(
                self.base + "/upload/image",
                files={"image": (name, buf, "image/png")},
                data={"overwrite": "true"},
                timeout=30,
            )
        except requests.exceptions.ConnectionError:
            raise ComfyError(f"Can't reach ComfyUI at {self.base} to upload image.")
        if r.status_code != 200:
            raise ComfyError(f"Image upload failed: {r.text}")
        return r.json()["name"]

    def queue(self, graph):
        payload = {"prompt": graph.nodes, "client_id": CLIENT_ID}
        try:
            r = requests.post(self.base + "/prompt", json=payload, timeout=20)
        except requests.exceptions.ConnectionError:
            raise ComfyError(f"Can't reach ComfyUI at {self.base}.")
        if r.status_code != 200:
            try:
                err = r.json()
                msg = err.get("error", {}).get("message", str(err))
                node_errors = err.get("node_errors", {})
                if node_errors:
                    details = "; ".join(
                        f"{nid}: {info.get('errors', info)}"
                        for nid, info in node_errors.items()
                    )
                    msg += f" ({details})"
            except Exception:
                msg = r.text
            raise ComfyError(f"ComfyUI rejected the workflow: {msg}")
        return r.json()["prompt_id"]

    def wait(self, prompt_id, timeout=1800, poll=1.0):
        start = time.time()
        while time.time() - start < timeout:
            r = self._get(f"/history/{prompt_id}")
            data = r.json()
            if prompt_id in data:
                entry = data[prompt_id]
                status = entry.get("status", {})
                if status.get("completed") or status.get("status_str") == "success":
                    return entry
                for msg in status.get("messages", []):
                    if isinstance(msg, list) and len(msg) > 1 and msg[0] == "execution_error":
                        raise ComfyError(f"Execution error: {msg[1]}")
                if status.get("status_str") == "error":
                    raise ComfyError(f"Execution failed: {status}")
            time.sleep(poll)
        raise ComfyError("Timed out waiting for ComfyUI to finish.")

    def run_with_progress(self, graph, progress_cb=None, timeout=3600):
        """Queue a graph and stream live step progress over ComfyUI's websocket.

        progress_cb(value, max_value, rate_s_per_it) is called on every step tick.
        Returns the finished /history entry, same shape as wait().
        """
        ws = websocket.WebSocket()
        try:
            ws.connect(f"{COMFY_WS_URL}?clientId={CLIENT_ID}", timeout=10)
        except Exception as e:
            raise ComfyError(f"Can't open a websocket to ComfyUI at {self.base}: {e}")
        try:
            prompt_id = self.queue(graph)
            start = time.time()
            step_times = []
            last_tick = start
            while True:
                if time.time() - start > timeout:
                    raise ComfyError("Timed out waiting for ComfyUI to finish.")
                ws.settimeout(5)
                try:
                    raw = ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                if not isinstance(raw, str):
                    continue  # binary preview frame, skip
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                mtype, d = msg.get("type"), msg.get("data", {})
                if mtype == "progress_state" and d.get("prompt_id") == prompt_id:
                    running = [n for n in d.get("nodes", {}).values() if n.get("state") == "running"]
                    if running and progress_cb:
                        node = running[0]
                        value, mx = node.get("value", 0), node.get("max", 1) or 1
                        now = time.time()
                        if value > 0:
                            step_times.append(now - last_tick)
                        last_tick = now
                        rate = sum(step_times[-5:]) / len(step_times[-5:]) if step_times else 0.0
                        progress_cb(value, mx, rate)
                elif mtype == "execution_error" and d.get("prompt_id") == prompt_id:
                    raise ComfyError(f"Execution error: {d}")
                elif mtype == "execution_success" and d.get("prompt_id") == prompt_id:
                    break
                elif mtype == "execution_interrupted" and d.get("prompt_id") == prompt_id:
                    raise ComfyError("Generation was interrupted.")
        finally:
            try:
                ws.close()
            except Exception:
                pass
        r = self._get(f"/history/{prompt_id}")
        data = r.json()
        entry = data.get(prompt_id)
        if not entry:
            raise ComfyError("No history entry found after completion.")
        return entry

    def fetch_outputs(self, history_entry):
        files = []
        for node_out in history_entry.get("outputs", {}).values():
            for val in node_out.values():
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict) and "filename" in item:
                            files.append(item)
        saved = []
        for f in files:
            r = self._get(
                "/view",
                params={
                    "filename": f["filename"],
                    "subfolder": f.get("subfolder", ""),
                    "type": f.get("type", "output"),
                },
            )
            if r.status_code == 200:
                path = os.path.join(OUTPUT_DIR, f["filename"])
                with open(path, "wb") as fh:
                    fh.write(r.content)
                saved.append(path)
        return saved


client = ComfyClient()


def center_crop_to_aspect(img, width, height):
    target = width / height
    w, h = img.size
    cur = w / h
    if cur > target:
        new_w = int(h * target)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    elif cur < target:
        new_h = int(w / target)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    return img.resize((width, height), Image.LANCZOS)


def run_and_collect(graph, seed_used, width, height, frames, gr_progress=None):
    def on_tick(value, mx, rate):
        if gr_progress is None:
            return
        eta = rate * (mx - value)
        eta_txt = f"{int(eta // 60)}m {int(eta % 60):02d}s" if rate > 0 else "…"
        desc = (
            f"Step {value}/{mx} · {frames} frames @ {width}x{height} · "
            f"{rate:.1f}s/it · ETA {eta_txt}"
        )
        gr_progress(value / mx if mx else 0, desc=desc)

    if gr_progress is not None:
        gr_progress(0, desc=f"Queuing · {frames} frames @ {width}x{height}")
    entry = client.run_with_progress(graph, progress_cb=on_tick)
    files = client.fetch_outputs(entry)
    videos = [f for f in files if f.lower().endswith((".mp4", ".webm", ".gif"))]
    result = videos[0] if videos else (files[0] if files else None)
    if result is None:
        raise ComfyError("Generation finished but produced no output file.")
    info = f"seed: {seed_used} · resolution: {width}x{height} · {frames} frames"
    return result, info


# ---------------------------------------------------------------------------
# Wan workflow pieces
# ---------------------------------------------------------------------------
def wan_base(g, unet_key, prompt, negative, shift=8.0):
    unet = g.add("UnetLoaderGGUF", unet_name=MODEL_FILES[unet_key])
    model = g.add("ModelSamplingSD3", model=g.o(unet), shift=float(shift))
    clip = g.add("CLIPLoader", clip_name=MODEL_FILES["wan_clip"], type="wan")
    vae = g.add("VAELoader", vae_name=MODEL_FILES["wan_vae"])
    pos = g.add("CLIPTextEncode", text=prompt, clip=g.o(clip))
    neg = g.add("CLIPTextEncode", text=negative, clip=g.o(clip))
    return g.o(model), g.o(vae), g.o(pos), g.o(neg)


def wan_sample_decode(g, model, vae, positive, negative, latent, steps, cfg, seed):
    ks = g.add(
        "KSampler", model=model, seed=int(seed), steps=int(steps), cfg=float(cfg),
        sampler_name="uni_pc", scheduler="simple", positive=positive, negative=negative,
        latent_image=latent, denoise=1.0,
    )
    dec = g.add("VAEDecode", samples=g.o(ks), vae=vae)
    return g.o(dec)


def wan_save(g, images, fps):
    vid = g.add("CreateVideo", images=images, fps=float(fps))
    g.add("SaveVideo", video=g.o(vid), filename_prefix="videolab/wan", format="auto", codec="auto")


def build_wan_t2v(prompt, negative, width, height, length, steps, cfg, seed):
    g = Graph()
    model, vae, pos, neg = wan_base(g, "wan_t2v_unet", prompt, negative)
    i2v = g.add(
        "WanImageToVideo", positive=pos, negative=neg, vae=vae,
        width=int(width), height=int(height), length=int(length), batch_size=1,
    )
    images = wan_sample_decode(g, model, vae, g.o(i2v, 0), g.o(i2v, 1), g.o(i2v, 2), steps, cfg, seed)
    return g, images


def build_wan_reference(prompt, negative, width, height, length, steps, cfg, seed, ref_filename):
    g = Graph()
    model, vae, pos, neg = wan_base(g, "wan_vace_unet", prompt, negative)
    ref_img = g.o(g.add("LoadImage", image=ref_filename), 0)
    vace = g.add(
        "WanVaceToVideo", positive=pos, negative=neg, vae=vae,
        width=int(width), height=int(height), length=int(length), batch_size=1,
        strength=1.0, reference_image=ref_img,
    )
    ks = g.add(
        "KSampler", model=model, seed=int(seed), steps=int(steps), cfg=float(cfg),
        sampler_name="uni_pc", scheduler="simple",
        positive=g.o(vace, 0), negative=g.o(vace, 1), latent_image=g.o(vace, 2), denoise=1.0,
    )
    trim = g.add("TrimVideoLatent", samples=g.o(ks), trim_amount=g.o(vace, 3))
    dec = g.add("VAEDecode", samples=g.o(trim), vae=vae)
    return g, g.o(dec)


# ---------------------------------------------------------------------------
# LTX workflow pieces
# ---------------------------------------------------------------------------
def ltx_base(g, prompt, negative, fps):
    unet = g.add("UnetLoaderGGUF", unet_name=MODEL_FILES["ltx_unet"])
    clip = g.add("CLIPLoader", clip_name=MODEL_FILES["ltx_clip"], type="ltxv")
    vae = g.add("VAELoader", vae_name=MODEL_FILES["ltx_vae"])
    pos = g.add("CLIPTextEncode", text=prompt, clip=g.o(clip))
    neg = g.add("CLIPTextEncode", text=negative, clip=g.o(clip))
    cond = g.add("LTXVConditioning", positive=g.o(pos), negative=g.o(neg), frame_rate=float(fps))
    return g.o(unet), g.o(vae), g.o(cond, 0), g.o(cond, 1)


def ltx_sample_decode(g, model, vae, positive, negative, latent, steps, cfg, seed):
    sigmas = g.add(
        "LTXVScheduler", steps=int(steps), max_shift=2.05, base_shift=0.95,
        stretch=True, terminal=0.1, latent=latent,
    )
    sampler = g.add("KSamplerSelect", sampler_name="euler")
    out = g.add(
        "SamplerCustom", model=model, add_noise=True, noise_seed=int(seed), cfg=float(cfg),
        positive=positive, negative=negative, sampler=g.o(sampler), sigmas=g.o(sigmas),
        latent_image=latent,
    )
    dec = g.add("VAEDecode", samples=g.o(out, 0), vae=vae)
    return g.o(dec)


def ltx_save(g, images, fps):
    vid = g.add("CreateVideo", images=images, fps=float(fps))
    g.add("SaveVideo", video=g.o(vid), filename_prefix="videolab/ltx", format="auto", codec="auto")


def build_ltx_t2v(prompt, negative, width, height, length, steps, cfg, seed, fps):
    g = Graph()
    model, vae, pos, neg = ltx_base(g, prompt, negative, fps)
    latent = g.o(g.add("EmptyLTXVLatentVideo", width=int(width), height=int(height), length=int(length), batch_size=1))
    images = ltx_sample_decode(g, model, vae, pos, neg, latent, steps, cfg, seed)
    return g, images


def build_ltx_guided(prompt, negative, width, height, length, steps, cfg, seed, fps, guides):
    """guides: list of (filename, frame_idx) pinned onto the clip."""
    g = Graph()
    model, vae, pos, neg = ltx_base(g, prompt, negative, fps)
    latent = g.o(g.add("EmptyLTXVLatentVideo", width=int(width), height=int(height), length=int(length), batch_size=1))
    for fname, frame_idx in guides:
        img = g.o(g.add("LoadImage", image=fname))
        pre = g.o(g.add("LTXVPreprocess", image=img, img_compression=35))
        guide = g.add(
            "LTXVAddGuide", positive=pos, negative=neg, vae=vae, latent=latent,
            image=pre, frame_idx=int(frame_idx), strength=1.0,
        )
        pos, neg, latent = g.o(guide, 0), g.o(guide, 1), g.o(guide, 2)
    crop = g.add("LTXVCropGuides", positive=pos, negative=neg, latent=latent)
    pos, neg, latent = g.o(crop, 0), g.o(crop, 1), g.o(crop, 2)
    images = ltx_sample_decode(g, model, vae, pos, neg, latent, steps, cfg, seed)
    return g, images


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------
def resolve_seed(seed):
    return random.randint(0, 2**31 - 1) if seed is None or int(seed) < 0 else int(seed)


def gen_text_to_video(model_choice, prompt, negative, width, height, frames, steps, cfg, seed, fps, progress=gr.Progress()):
    if not client.is_up():
        raise gr.Error(f"ComfyUI isn't running at {COMFY_URL}. Start it first (see README).")
    if not prompt.strip():
        raise gr.Error("Enter a prompt.")
    negative = negative.strip() or DEFAULT_NEGATIVE
    seed_used = resolve_seed(seed)
    try:
        if model_choice == "LTX":
            g, images = build_ltx_t2v(prompt, negative, width, height, frames, steps, cfg, seed_used, fps)
            ltx_save(g, images, fps)
        else:
            g, images = build_wan_t2v(prompt, negative, width, height, frames, steps, cfg, seed_used)
            wan_save(g, images, fps)
        path, info = run_and_collect(g, seed_used, width, height, frames, gr_progress=progress)
    except ComfyError as e:
        raise gr.Error(str(e))
    return path, info


def gen_image_to_video(model_choice, image, prompt, negative, width, height, frames, steps, cfg, seed, fps, progress=gr.Progress()):
    if not client.is_up():
        raise gr.Error(f"ComfyUI isn't running at {COMFY_URL}. Start it first (see README).")
    if image is None:
        raise gr.Error("Upload an image.")
    if not prompt.strip():
        raise gr.Error("Enter a prompt.")
    negative = negative.strip() or DEFAULT_NEGATIVE
    seed_used = resolve_seed(seed)
    cropped = center_crop_to_aspect(image, int(width), int(height))
    try:
        fname = client.upload_image(cropped)
        if model_choice == "LTX":
            g, images = build_ltx_guided(
                prompt, negative, width, height, frames, steps, cfg, seed_used, fps,
                guides=[(fname, 0)],
            )
            ltx_save(g, images, fps)
        else:
            g, images = build_wan_reference(prompt, negative, width, height, frames, steps, cfg, seed_used, fname)
            wan_save(g, images, fps)
        path, info = run_and_collect(g, seed_used, width, height, frames, gr_progress=progress)
    except ComfyError as e:
        raise gr.Error(str(e))
    return path, info


def compose_reference_sheet(images, width, height):
    sheet = Image.new("RGB", (width, height), "white")
    n = len(images)
    slot_w = width // n
    for i, img in enumerate(images):
        cell = center_crop_to_aspect(img, slot_w, height)
        sheet.paste(cell, (i * slot_w, 0))
    return sheet


def gen_compose(images, prompt, negative, width, height, frames, steps, cfg, seed, fps, progress=gr.Progress()):
    if not client.is_up():
        raise gr.Error(f"ComfyUI isn't running at {COMFY_URL}. Start it first (see README).")
    imgs = [i for i in images if i is not None] if images else []
    if len(imgs) < 1:
        raise gr.Error("Upload 1-3 reference images.")
    if not prompt.strip():
        raise gr.Error("Enter a prompt.")
    negative = negative.strip() or DEFAULT_NEGATIVE
    seed_used = resolve_seed(seed)
    sheet = compose_reference_sheet(imgs, int(width), int(height))
    try:
        fname = client.upload_image(sheet)
        g, out_images = build_wan_reference(prompt, negative, width, height, frames, steps, cfg, seed_used, fname)
        wan_save(g, out_images, fps)
        path, info = run_and_collect(g, seed_used, width, height, frames, gr_progress=progress)
    except ComfyError as e:
        raise gr.Error(str(e))
    return sheet, path, info


def gen_keyframes(images, prompt, negative, width, height, frames, steps, cfg, seed, fps, progress=gr.Progress()):
    if not client.is_up():
        raise gr.Error(f"ComfyUI isn't running at {COMFY_URL}. Start it first (see README).")
    imgs = [i for i in images if i is not None] if images else []
    if len(imgs) < 2:
        raise gr.Error("Upload 2-4 keyframe images.")
    negative = negative.strip() or DEFAULT_NEGATIVE
    seed_used = resolve_seed(seed)
    frames = int(frames)
    n = len(imgs)
    last_frame = ((frames - 1) // 8) * 8
    frame_indices = [round(i * last_frame / (n - 1) / 8) * 8 for i in range(n)]
    frame_indices[-1] = last_frame
    try:
        guides = []
        for img, fidx in zip(imgs, frame_indices):
            cropped = center_crop_to_aspect(img, int(width), int(height))
            fname = client.upload_image(cropped)
            guides.append((fname, fidx))
        g, images_out = build_ltx_guided(
            prompt or "smooth animated transition between the reference frames",
            negative, width, height, frames, steps, cfg, seed_used, fps, guides,
        )
        ltx_save(g, images_out, fps)
        path, info = run_and_collect(g, seed_used, width, height, frames, gr_progress=progress)
    except ComfyError as e:
        raise gr.Error(str(e))
    return path, info


def check_setup():
    lines = []
    if not client.is_up():
        lines.append(f"❌ ComfyUI not reachable at {COMFY_URL}. Start it (see README).")
        return "\n".join(lines)
    lines.append(f"✅ ComfyUI reachable at {COMFY_URL}")
    try:
        info = client.object_info()
    except ComfyError as e:
        lines.append(f"❌ {e}")
        return "\n".join(lines)
    for node in REQUIRED_NODES:
        lines.append(f"✅ node {node} available" if node in info else f"❌ node {node} MISSING")
    for key, label, loader, input_name in MODEL_SLOTS:
        options = info.get(loader, {}).get("input", {}).get("required", {}).get(input_name, [[]])[0]
        fname = MODEL_FILES[key]
        if fname in options:
            lines.append(f"✅ {label}: {fname}")
        else:
            lines.append(f"❌ {label}: '{fname}' not found via {loader} ({len(options)} files visible)")
    return "\n".join(lines)


def model_options(loader, input_name):
    try:
        info = client.object_info(loader)
        return info.get(loader, {}).get("input", {}).get("required", {}).get(input_name, [[]])[0]
    except ComfyError:
        return []


def refresh_model_dropdowns():
    updates = []
    for key, label, loader, input_name in MODEL_SLOTS:
        opts = model_options(loader, input_name)
        current = MODEL_FILES[key]
        updates.append(gr.update(choices=opts, value=current if current in opts else (opts[0] if opts else None)))
    return updates


def set_model_file(key, value):
    if value:
        MODEL_FILES[key] = value
    return f"{key} -> {MODEL_FILES[key]}"


def switch_defaults(model_choice):
    w, h, f, s, c = model_defaults("ltx" if model_choice == "LTX" else "wan")
    fps = LTX_DEFAULTS["fps"] if model_choice == "LTX" else WAN_DEFAULTS["fps"]
    return w, h, f, s, c, fps


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def settings_block(model_choice_component=None, ltx_only=False, wan_only=False):
    with gr.Accordion("Settings", open=False):
        with gr.Row():
            width = gr.Slider(64, 1280, value=WAN_DEFAULTS["width"], step=16, label="Width")
            height = gr.Slider(64, 1280, value=WAN_DEFAULTS["height"], step=16, label="Height")
        with gr.Row():
            frames = gr.Slider(1, 257, value=WAN_DEFAULTS["frames"], step=1, label="Frames")
            fps = gr.Slider(8, 30, value=WAN_DEFAULTS["fps"], step=1, label="FPS")
        with gr.Row():
            steps = gr.Slider(1, 60, value=WAN_DEFAULTS["steps"], step=1, label="Steps")
            cfg = gr.Slider(0.0, 15.0, value=WAN_DEFAULTS["cfg"], step=0.1, label="CFG")
        seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
        negative = gr.Textbox(label="Negative prompt", placeholder=DEFAULT_NEGATIVE, lines=2)
    return width, height, frames, fps, steps, cfg, seed, negative


def build_ui():
    with gr.Blocks(title="Video Lab") as demo:
        gr.Markdown("# Video Lab — local ComfyUI video generation")

        with gr.Tab("Text → Video"):
            model_choice = gr.Radio(["LTX", "Wan"], value="Wan", label="Model")
            prompt = gr.Textbox(label="Prompt", lines=3)
            width, height, frames, fps, steps, cfg, seed, negative = settings_block()
            btn = gr.Button("Generate", variant="primary")
            video_out = gr.Video(label="Result")
            info_out = gr.Textbox(label="Run info", interactive=False)
            model_choice.change(
                switch_defaults, inputs=[model_choice],
                outputs=[width, height, frames, steps, cfg, fps],
            )
            btn.click(
                gen_text_to_video,
                inputs=[model_choice, prompt, negative, width, height, frames, steps, cfg, seed, fps],
                outputs=[video_out, info_out],
                concurrency_limit=1,
            )

        with gr.Tab("Image → Video"):
            model_choice2 = gr.Radio(["LTX", "Wan"], value="Wan", label="Model")
            image_in = gr.Image(label="Starting image", type="pil")
            prompt2 = gr.Textbox(label="Prompt", lines=3)
            width2, height2, frames2, fps2, steps2, cfg2, seed2, negative2 = settings_block()
            btn2 = gr.Button("Generate", variant="primary")
            video_out2 = gr.Video(label="Result")
            info_out2 = gr.Textbox(label="Run info", interactive=False)
            model_choice2.change(
                switch_defaults, inputs=[model_choice2],
                outputs=[width2, height2, frames2, steps2, cfg2, fps2],
            )
            btn2.click(
                gen_image_to_video,
                inputs=[model_choice2, image_in, prompt2, negative2, width2, height2, frames2, steps2, cfg2, seed2, fps2],
                outputs=[video_out2, info_out2],
                concurrency_limit=1,
            )

        with gr.Tab("Compose & Animate"):
            gr.Markdown("Upload 1-3 reference images (e.g. person + background). Wan VACE only.")
            with gr.Row():
                ref1 = gr.Image(label="Reference 1", type="pil")
                ref2 = gr.Image(label="Reference 2 (optional)", type="pil")
                ref3 = gr.Image(label="Reference 3 (optional)", type="pil")
            prompt3 = gr.Textbox(label="Prompt", lines=3)
            width3, height3, frames3, fps3, steps3, cfg3, seed3, negative3 = settings_block()
            btn3 = gr.Button("Generate", variant="primary")
            with gr.Row():
                sheet_out = gr.Image(label="Composed reference sheet")
                video_out3 = gr.Video(label="Result")
            info_out3 = gr.Textbox(label="Run info", interactive=False)
            btn3.click(
                gen_compose,
                inputs=[gr.State(None), prompt3, negative3, width3, height3, frames3, steps3, cfg3, seed3, fps3],
                outputs=[sheet_out, video_out3, info_out3],
            ).then(lambda: None)
            # wire the actual image list via a small wrapper below
            def _compose_wrapper(i1, i2, i3, *a, progress=gr.Progress()):
                return gen_compose([i1, i2, i3], *a, progress=progress)
            btn3.click(
                _compose_wrapper,
                inputs=[ref1, ref2, ref3, prompt3, negative3, width3, height3, frames3, steps3, cfg3, seed3, fps3],
                outputs=[sheet_out, video_out3, info_out3],
                concurrency_limit=1,
            )

        with gr.Tab("Keyframes"):
            gr.Markdown("Upload 2-4 images; LTX pins them across the clip and animates the transitions.")
            with gr.Row():
                kf1 = gr.Image(label="Keyframe 1", type="pil")
                kf2 = gr.Image(label="Keyframe 2", type="pil")
                kf3 = gr.Image(label="Keyframe 3 (optional)", type="pil")
                kf4 = gr.Image(label="Keyframe 4 (optional)", type="pil")
            prompt4 = gr.Textbox(label="Prompt (optional)", lines=2)
            width4 = gr.Slider(64, 1280, value=LTX_DEFAULTS["width"], step=32, label="Width")
            height4 = gr.Slider(64, 1280, value=LTX_DEFAULTS["height"], step=32, label="Height")
            frames4 = gr.Slider(9, 257, value=LTX_DEFAULTS["frames"], step=8, label="Frames")
            with gr.Accordion("More settings", open=False):
                fps4 = gr.Slider(8, 30, value=LTX_DEFAULTS["fps"], step=1, label="FPS")
                steps4 = gr.Slider(1, 60, value=LTX_DEFAULTS["steps"], step=1, label="Steps")
                cfg4 = gr.Slider(0.0, 15.0, value=LTX_DEFAULTS["cfg"], step=0.1, label="CFG")
                seed4 = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                negative4 = gr.Textbox(label="Negative prompt", placeholder=DEFAULT_NEGATIVE, lines=2)
            btn4 = gr.Button("Generate", variant="primary")
            video_out4 = gr.Video(label="Result")
            info_out4 = gr.Textbox(label="Run info", interactive=False)

            def _kf_wrapper(i1, i2, i3, i4, *a, progress=gr.Progress()):
                return gen_keyframes([i1, i2, i3, i4], *a, progress=progress)
            btn4.click(
                _kf_wrapper,
                inputs=[kf1, kf2, kf3, kf4, prompt4, negative4, width4, height4, frames4, steps4, cfg4, seed4, fps4],
                outputs=[video_out4, info_out4],
                concurrency_limit=1,
            )

        with gr.Tab("Check setup"):
            check_btn = gr.Button("Run checks")
            check_out = gr.Textbox(label="Status", lines=20, interactive=False)
            check_btn.click(check_setup, outputs=[check_out])

            gr.Markdown("### Model files")
            gr.Markdown("Pick which file each loader uses, then Refresh to pull the list from ComfyUI.")
            refresh_btn = gr.Button("Refresh model lists")
            dropdowns = []
            for key, label, loader, input_name in MODEL_SLOTS:
                dd = gr.Dropdown(choices=[MODEL_FILES[key]], value=MODEL_FILES[key], label=label)
                dd.change(lambda v, k=key: set_model_file(k, v), inputs=[dd], outputs=[])
                dropdowns.append(dd)
            refresh_btn.click(refresh_model_dropdowns, outputs=dropdowns)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
