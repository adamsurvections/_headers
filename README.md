# Local AI Image Editor

A private image editor that runs on your own computer. There's no account, no uploads to anyone's server, no usage limit and no watermark.

| Tab | What it does | Engine |
|---|---|---|
| **Adjust** | Rotate, flip, crop to square, resize, brightness, contrast, saturation, sharpness, blur, black and white | Pillow (no AI, instant) |
| **AI Edit** | Change a photo with a plain-English instruction: "make it snowy", "turn it into an oil painting" | [InstructPix2Pix](https://huggingface.co/timbrooks/instruct-pix2pix) |
| **AI Inpaint** | Paint over part of an image and describe what should replace it | [Stable Diffusion inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting) |
| **Remove Background** | Cut out the subject onto a transparent background | [rembg](https://github.com/danielgatis/rembg) |

## Requirements

- Python 3.10 to 3.12
- About 10 GB of free disk space for the models
- **Recommended:** an NVIDIA GPU with 6 GB or more of VRAM, or an Apple Silicon Mac (M1 or later)
- It also runs on CPU only, but each AI edit then takes several minutes

## Install

```bash
git clone https://github.com/adamsurvections/_headers.git ai-image-editor
cd ai-image-editor
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
```

**NVIDIA GPU (Windows or Linux):** install the CUDA build of PyTorch first. The exact command for your system is on https://pytorch.org/get-started/locally/. For example:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Then, on every system:
```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Your browser opens at http://127.0.0.1:7860. The first time you use an AI tab, its model downloads (roughly 2 to 5 GB). After that it runs offline.

Options:
- `--port 8000` runs it on a different port
- `--host 0.0.0.0` lets other devices on your home network use it

## Tips

- **Seed:** `-1` gives a different result on every run. To repeat a result you liked, copy the "Seed used" value back into the Seed field.
- **AI Edit:** if the result drifts too far from the original, raise *Keep-original strength*. If the change is too weak, raise *Instruction strength*.
- **Inpaint:** paint a little past the edges of the object you're replacing.
- **Out of memory:** lower *Max working size*.

## Content safety

The Stable Diffusion models include a built-in safety checker that blanks explicit output. It is left **on** on purpose. This tool edits real photos, and a photo editor without that check can be used to make sexual images of real people who never consented. That is illegal in many places and causes real harm.
