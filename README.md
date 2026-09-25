# Video Lab

A local, fully offline web app for generating video with Wan 2.1 and LTX-Video, driven
through ComfyUI. No cloud, no internet needed once the models are on disk.

## Every time you want to use it

Two things need to be running: ComfyUI (the backend) and the Gradio app (the web UI).
Use two terminal windows.

**Terminal 1 — start ComfyUI:**
```bash
cd "/home/arghyadutta/Documents/Local Video Generation/ComfyUI"
source ../venv/bin/activate
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python main.py --listen 127.0.0.1 --port 8188
```
Wait for `Starting server` / `To see the GUI go to: http://127.0.0.1:8188`.

If you get `OSError: ... address already in use`, ComfyUI is already running from
before — you don't need a second one, just leave it and move to Terminal 2.

If it crashes with an out-of-memory error mid-generation, add `--lowvram`:
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python main.py --listen 127.0.0.1 --port 8188 --lowvram
```

**Terminal 2 — start the app:**
```bash
cd "/home/arghyadutta/Documents/Local Video Generation"
source venv/bin/activate
python app.py
```
Then open **http://127.0.0.1:7860** in your browser.

Keep both terminals open while you use it — closing either one stops that piece.
Use the **Check setup** tab first if anything seems off; it confirms ComfyUI is
reachable and every configured model file is actually visible to its loader.

> Don't run `app.py` via PyCharm's Run button if PyCharm is installed as a Flatpak —
> its sandboxed Python can shadow the venv's real interpreter. Run it from a normal
> terminal (or PyCharm's Terminal panel) instead.

> **Always start both processes yourself, in your own terminals.** If something
> else launches ComfyUI on your behalf (an assistant, a script, an IDE task) and
> that thing's own process later dies or restarts, ComfyUI dies with it —
> silently, mid-generation, with no warning in the UI. Running it in a terminal
> you own is what keeps a multi-hour render safe.

## Negative prompt

Every tab has a default negative prompt baked in so a blank field doesn't produce
garbage output:

```
blurry, low quality, distorted, deformed, watermark, text, subtitles, static,
worst quality, jpeg artifacts, overexposed, extra limbs
```

You only need to type something in that box if you want to override it for a
specific run (e.g. add "no hands" for a shot with visible hands).

## Model files

Edit the `MODEL_FILES` dict near the top of `app.py` if you rename or swap model
files. You can also change them live from the **Check setup** tab's "Model files"
section (Refresh pulls the current options straight from ComfyUI).

Current setup uses GGUF-quantized diffusion models (lighter on your 8GB VRAM) plus
safetensors text encoders/VAEs:

| Slot | File |
|---|---|
| Wan T2V 1.3B | `Wan2.1-T2V-1.3B-Q4_K_M.gguf` |
| Wan VACE 1.3B | `Wan2.1-VACE-1.3B-Q4_K_M.gguf` |
| Wan text encoder | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` |
| Wan VAE | `wan_2.1_vae.safetensors` |
| LTX-Video 2B distilled | `ltxv-2b-0.9.6-distilled-04-25-Q6_K.gguf` |
| LTX text encoder | `t5xxl_fp8_e4m3fn_scaled.safetensors` |
| LTX VAE | `LTX-Video-0.9.6-VAE-BF16.safetensors` |

## How long a generation takes

Measured on this machine (RTX 4060 Laptop, 8GB):

- **Wan**, default settings (832×480, 81 frames, 30 steps): roughly **15–20 minutes**.
- **LTX**, default settings (768×512, 97 frames, 8 steps): roughly **1–3 minutes**
  (it's a distilled model — much faster per step and needs far fewer steps).

Add ~30–50 seconds the first time you generate in a session (loading models onto
the GPU); later generations in the same ComfyUI session are faster.

For quick iteration on a prompt, drop Steps to ~8 (Wan) or ~4 (LTX) and Frames to
~17–25 in Settings — cuts a run to well under a minute so you can dial in wording
before committing to a full-quality render.

The terminal running ComfyUI shows a live progress bar with `s/it` — multiply that
by remaining steps for a real-time ETA. The web UI itself also shows a live
progress bar (step count, `s/it`, ETA) while a job runs, fed over ComfyUI's
websocket — you don't need to watch the terminal.

## Seeing past runs

Every generation you start — including ones that errored or got interrupted
(e.g. ComfyUI crashed mid-render) — is appended to `runs.jsonl` in the project
root. This is separate from the browser: reloading the page, restarting the
app, or even restarting ComfyUI doesn't lose this history, since it's just a
file on disk.

The **History** tab lists it: timestamp, which tab/model was used, the prompt,
settings, and a status column that tells you how far a run actually got —
`done (30 steps)`, `error at 12/30`, or `interrupted at 9/30 (process
stopped)` for one where ComfyUI itself died before it could report back.
Pick a filename from the dropdown and hit **Load** to preview it — every
output is also sitting directly in `./outputs/` if you'd rather browse there.

## Offline

Nothing here calls out to the internet at runtime. `app.py` only talks to
`127.0.0.1:8188` (ComfyUI), and ComfyUI is launched with `HF_HUB_OFFLINE=1` /
`TRANSFORMERS_OFFLINE=1` so it can't silently reach out even during model loading.
Once the model files above are on disk, this works with no network connection.

## Setup (already done, for reference)

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r ComfyUI/requirements.txt -r ComfyUI/custom_nodes/ComfyUI-GGUF/requirements.txt gradio
```
Model files go under `ComfyUI/models/{diffusion_models,text_encoders,vae}/`.

ComfyUI is pinned to release **v0.7.0** — its current `main` branch depends on a
`comfy_kitchen` package that's incompatible with any released PyTorch build at the
time of writing. If you `git pull` inside `ComfyUI/` and it stops starting with a
`torch.library.custom_op` / `infer_schema` error, check out an older tag again:
```bash
cd ComfyUI && git checkout v0.7.0
```
