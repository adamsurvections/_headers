# Local AI Image Editor

A private image editor that runs on your own computer. There's no account, no uploads to anyone's server, no usage limit and no watermark.

| Tab | What it does | Engine |
|---|---|---|
| **AI Edit** | Upload a photo, describe the change, get the result. You can add a second image to combine ("put the product from image 2 on the table in image 1"). "Keep editing this result" lets you chain edits. | **Best:** [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) · **Light:** [InstructPix2Pix](https://huggingface.co/timbrooks/instruct-pix2pix) |
| **AI Inpaint** | Paint over part of an image and describe what should replace it | [Stable Diffusion inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting) |
| **Adjust** | Rotate, flip, crop to square, resize, brightness, contrast, saturation, sharpness, blur, black and white | Pillow (no AI, instant) |
| **Remove Background** | Cut out the subject onto a transparent background | [rembg](https://github.com/danielgatis/rembg) |

## Requirements

Python 3.10 to 3.12, plus hardware that fits the engine you want to use:

| Engine | Quality | Hardware you need (approximate) | Download |
|---|---|---|---|
| **Best** (Qwen-Image-Edit) | Excellent. It follows detailed instructions, keeps faces and products consistent, and edits text in images. | An NVIDIA GPU with **16 GB+ VRAM** (4-bit mode, on by default) and **32 GB+ system RAM**. Or an Apple Silicon Mac with **64 GB+** unified memory. | ~60 GB |
| **Light** (InstructPix2Pix) | Decent for style and weather changes. Weaker on precise edits. | Any NVIDIA GPU with 6 GB+, any Apple Silicon Mac, or a CPU (slow) | ~2 GB |

The Best engine's hardware figures are estimates for a model of this size and haven't been measured. If it runs out of memory, use the Light engine or rent a cloud GPU (see below).

## Mac mini / Apple Silicon quick start

1. Check your memory: Apple menu → **About This Mac** → **Memory**.
2. Install Python 3.12 from https://www.python.org/downloads/macos/ if you don't have it (`python3 --version` in Terminal).
3. Follow **Install** and **Run** below. Skip the NVIDIA step.

| Your Mac's memory | What runs locally |
|---|---|
| 16 / 24 / 32 GB | **Light** engine, Inpaint, Adjust, Remove Background. The Best engine is disabled automatically so it won't start a 60 GB download it can't use. |
| 64 GB+ (M4 Pro) | Everything, including the **Best** engine. Expect minutes per edit, not seconds. |

For Best-quality edits on a smaller Mac, run the app on a rented GPU (see **No suitable GPU?**) and open it from your Mac's browser.

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

- **Write prompts like instructions to a retoucher.** Say what should change and what must stay the same: *"Replace the background with a sunlit café. Keep the person's face, hair and clothes identical."*
- **Change one thing at a time,** then click **Keep editing this result** for the next change. This works better than one long prompt.
- **Seed:** `-1` gives a different result on every run. To repeat a result you liked, copy the "Seed used" value back into the Seed field.
- **Prompt strength:** raise it if the edit is too weak. Lower it if the image changes too much.
- **Inpaint:** paint a little past the edges of the object you're replacing.
- **Out of memory:** lower *Max working size*.

## No suitable GPU?

You can rent one by the hour (RunPod, Vast.ai, Lambda and similar providers). Pick a 24 GB or 48 GB GPU, clone this repo, and run `python app.py --host 0.0.0.0`. Your images then pass through that provider's machine, so read their privacy terms.

## Licenses

Qwen-Image-Edit is published under Apache 2.0, which allows commercial use. Check the model card for the current terms before you rely on this commercially. The Stable Diffusion models use the CreativeML OpenRAIL-M license.

## Content safety

Every AI output goes through an explicit-content detector (the standard Stable Diffusion safety checker), which blanks sexually explicit images. It is left **on** on purpose. This tool edits real photos, and a photo editor without that check can be used to make sexual images of real people who never consented. That is illegal in many places and causes real harm.
