"""Local AI image editor.

Runs entirely on your own machine: no account, no cloud upload, no usage quota,
no watermark. AI models are downloaded once from Hugging Face and cached locally.

Tabs:
  1. AI Edit      - upload a photo, describe the change, get the edited image
                    Best:  Qwen-Image-Edit-2509 (state-of-the-art open editing model)
                    Light: InstructPix2Pix (runs on modest hardware)
  2. AI Inpaint   - paint over an area and describe what should replace it [SD inpainting]
  3. Adjust       - classic edits (rotate, flip, crop-to-square, resize, colour, blur, sharpen)
  4. Remove BG    - cut the subject out onto a transparent background      [rembg, optional]

Heavy libraries (torch, diffusers, rembg) are imported lazily, so the Adjust tab
works even if you have not installed the AI dependencies yet.
"""

import argparse
import functools

import gradio as gr
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

QWEN_EDIT_MODEL = "Qwen/Qwen-Image-Edit-2509"
INSTRUCT_MODEL = "timbrooks/instruct-pix2pix"
SAFETY_MODEL = "CompVis/stable-diffusion-safety-checker"
ENGINE_BEST = "Best quality (Qwen-Image-Edit)"
ENGINE_LIGHT = "Light / fast (InstructPix2Pix)"
INPAINT_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"


# ---------------------------------------------------------------------------
# Device / model loading
# ---------------------------------------------------------------------------

@functools.cache
def _device():
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def _prepare(pipe):
    device, _ = _device()
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()  # lower VRAM use at a small speed cost
    return pipe


@functools.cache
def _instruct_pipe():
    from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline

    _, dtype = _device()
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(INSTRUCT_MODEL, torch_dtype=dtype)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    return _prepare(pipe)


@functools.cache
def _qwen_pipe(low_vram):
    """Qwen-Image-Edit is ~20B params. low_vram=True loads it 4-bit (NVIDIA only) so it fits ~16 GB cards."""
    import torch
    from diffusers import QwenImageEditPlusPipeline

    device, _ = _device()
    kwargs = {"torch_dtype": torch.bfloat16}
    if low_vram and device == "cuda":  # bitsandbytes is NVIDIA-only; ignored elsewhere
        from diffusers.quantizers import PipelineQuantizationConfig

        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            components_to_quantize=["transformer", "text_encoder"],
        )
    pipe = QwenImageEditPlusPipeline.from_pretrained(QWEN_EDIT_MODEL, **kwargs)
    if device == "cuda":
        pipe.enable_model_cpu_offload()  # only the active sub-model sits on the GPU
    else:
        pipe = pipe.to(device)
    return pipe


@functools.cache
def _safety_checker():
    from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
    from transformers import CLIPImageProcessor

    return StableDiffusionSafetyChecker.from_pretrained(SAFETY_MODEL), CLIPImageProcessor.from_pretrained(SAFETY_MODEL)


def _safe(img):
    """Blank sexually explicit output. The editor works on real photos, so this stays on."""
    checker, processor = _safety_checker()
    clip_input = processor(images=img, return_tensors="pt").pixel_values
    _, flagged = checker(clip_input=clip_input, images=[np.array(img)])
    if flagged[0]:
        gr.Warning("Explicit content was detected, so the output was blanked. Try a different prompt or seed.")
        return Image.new("RGB", img.size)
    return img


@functools.cache
def _inpaint_pipe():
    from diffusers import AutoPipelineForInpainting

    _, dtype = _device()
    pipe = AutoPipelineForInpainting.from_pretrained(INPAINT_MODEL, torch_dtype=dtype)
    return _prepare(pipe)


def _generator(seed):
    import torch

    seed = int(seed)
    if seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    return torch.Generator("cpu").manual_seed(seed), seed


def _fit(img, max_side):
    """Scale so the longest side is <= max_side and both sides are multiples of 8."""
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    w, h = int(w * scale) // 8 * 8, int(h * scale) // 8 * 8
    return img.resize((max(w, 8), max(h, 8)), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Tab 1: classic adjustments (no AI)
# ---------------------------------------------------------------------------

def adjust(img, rotate, flip_h, flip_v, square, width, brightness, contrast, saturation, sharpness, blur, grayscale):
    if img is None:
        raise gr.Error("Upload an image first.")
    img = img.convert("RGBA") if img.mode in ("RGBA", "LA", "P") else img.convert("RGB")

    if rotate:
        img = img.rotate(-rotate, expand=True)
    if flip_h:
        img = ImageOps.mirror(img)
    if flip_v:
        img = ImageOps.flip(img)
    if square:
        side = min(img.size)
        img = ImageOps.fit(img, (side, side), Image.LANCZOS)
    if width and int(width) != img.width:
        width = int(width)
        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)

    alpha = img.getchannel("A") if img.mode == "RGBA" else None
    rgb = img.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    rgb = ImageEnhance.Color(rgb).enhance(saturation)
    rgb = ImageEnhance.Sharpness(rgb).enhance(sharpness)
    if blur:
        rgb = rgb.filter(ImageFilter.GaussianBlur(blur))
    if grayscale:
        rgb = ImageOps.grayscale(rgb).convert("RGB")
    if alpha is not None:
        rgb.putalpha(alpha)
    return rgb


# ---------------------------------------------------------------------------
# Tab 2: instruction-based edit
# ---------------------------------------------------------------------------

def ai_edit(img, ref, instruction, engine, steps, strength, keep, low_vram, seed, progress=gr.Progress(track_tqdm=True)):
    if img is None:
        raise gr.Error("Upload an image first.")
    if not instruction.strip():
        raise gr.Error("Describe the edit you want, e.g. 'replace the background with a beach at sunset'.")
    gen, seed = _generator(seed)
    img = img.convert("RGB")

    if engine == ENGINE_BEST:
        pipe = _qwen_pipe(bool(low_vram) and _device()[0] == "cuda")
        images = [img] + ([ref.convert("RGB")] if ref is not None else [])
        out = pipe(
            image=images if len(images) > 1 else img,
            prompt=instruction,
            negative_prompt=" ",  # enables true classifier-free guidance
            true_cfg_scale=strength,
            num_inference_steps=int(steps),
            generator=gen,
        ).images[0]
    else:
        if ref is not None:
            gr.Warning("The light engine ignores the reference image.")
        pipe = _instruct_pipe()
        out = pipe(
            instruction,
            image=_fit(img, 768),
            num_inference_steps=int(steps),
            guidance_scale=strength * 1.9,  # map the shared 1-10 slider onto this model's range
            image_guidance_scale=keep,
            generator=gen,
        ).images[0]
    return _safe(out), seed


def use_result(img):
    if img is None:
        raise gr.Error("Nothing to keep editing yet.")
    return img


# ---------------------------------------------------------------------------
# Tab 3: inpainting
# ---------------------------------------------------------------------------

def _mask_from_editor(editor):
    """Build a white-on-black mask from every brush stroke layer."""
    bg = editor.get("background")
    if bg is None:
        raise gr.Error("Upload an image first.")
    bg = Image.fromarray(bg) if isinstance(bg, np.ndarray) else bg
    mask = np.zeros((bg.height, bg.width), dtype=bool)
    for layer in editor.get("layers") or []:
        layer = Image.fromarray(layer) if isinstance(layer, np.ndarray) else layer
        layer = layer.convert("RGBA").resize(bg.size)
        mask |= np.array(layer)[:, :, 3] > 0
    if not mask.any():
        raise gr.Error("Paint over the area you want to change first (use the brush tool).")
    return bg.convert("RGB"), Image.fromarray((mask * 255).astype(np.uint8))


def ai_inpaint(editor, prompt, negative, steps, cfg, strength, max_side, seed, progress=gr.Progress(track_tqdm=True)):
    if editor is None:
        raise gr.Error("Upload an image first.")
    orig, full_mask = _mask_from_editor(editor)
    orig_size = orig.size
    image = _fit(orig, max_side)
    mask = full_mask.resize(image.size, Image.NEAREST).filter(ImageFilter.MaxFilter(9))  # grow edges slightly
    pipe = _inpaint_pipe()
    gen, seed = _generator(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative or None,
        image=image,
        mask_image=mask,
        width=image.width,
        height=image.height,
        num_inference_steps=int(steps),
        guidance_scale=cfg,
        strength=strength,
        generator=gen,
    )
    out = result.images[0].resize(orig_size, Image.LANCZOS)
    # Keep untouched pixels identical to the original.
    soft = mask.resize(orig_size).filter(ImageFilter.GaussianBlur(4))
    out = Image.composite(out, orig, soft)
    return _safe(out), mask, seed


# ---------------------------------------------------------------------------
# Tab 4: background removal
# ---------------------------------------------------------------------------

def remove_bg(img):
    if img is None:
        raise gr.Error("Upload an image first.")
    try:
        from rembg import remove
    except ImportError:
        raise gr.Error("Background removal needs the optional 'rembg' package: pip install rembg")
    return remove(img)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="Local AI Image Editor") as demo:
        gr.Markdown(
            "# Local AI Image Editor\n"
            "Everything runs on this computer. The first use of each AI engine downloads its model once "
            "(Best engine: ~60 GB, Light: ~2 GB)."
        )

        with gr.Tab("AI Edit"):
            gr.Markdown("Upload a photo and describe the change. Add an optional second image to combine, e.g. *put the product from image 2 on the table in image 1*.")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        e_in = gr.Image(type="pil", label="Image to edit")
                        e_ref = gr.Image(type="pil", label="Reference image (optional)")
                    e_prompt = gr.Textbox(
                        label="What should change?",
                        lines=3,
                        placeholder="e.g. change the background to a modern office, keep the person exactly the same",
                    )
                    e_engine = gr.Radio([ENGINE_BEST, ENGINE_LIGHT], value=ENGINE_BEST, label="Engine")
                    with gr.Accordion("Settings", open=False):
                        e_steps = gr.Slider(10, 60, 40, step=1, label="Steps (more = slower, finer)")
                        e_str = gr.Slider(1, 10, 4, step=0.5, label="Prompt strength")
                        e_keep = gr.Slider(0.5, 3, 1.5, step=0.1, label="Keep-original strength (light engine)")
                        e_lowvram = gr.Checkbox(True, label="Low-VRAM 4-bit mode (NVIDIA only; untick if you have 48 GB+ VRAM)")
                        e_seed = gr.Number(-1, label="Seed (-1 = random)", precision=0)
                    e_btn = gr.Button("Edit", variant="primary")
                with gr.Column():
                    e_out = gr.Image(type="pil", label="Result", format="png")
                    e_again = gr.Button("Keep editing this result")
                    e_used = gr.Number(label="Seed used", interactive=False)
            e_btn.click(
                ai_edit,
                [e_in, e_ref, e_prompt, e_engine, e_steps, e_str, e_keep, e_lowvram, e_seed],
                [e_out, e_used],
            )
            e_prompt.submit(
                ai_edit,
                [e_in, e_ref, e_prompt, e_engine, e_steps, e_str, e_keep, e_lowvram, e_seed],
                [e_out, e_used],
            )
            e_again.click(use_result, e_out, e_in)

        with gr.Tab("AI Inpaint"):
            gr.Markdown("Upload an image, **paint over** the part to replace, then describe what should go there.")
            with gr.Row():
                with gr.Column():
                    i_in = gr.ImageEditor(
                        type="pil",
                        label="Image + mask",
                        brush=gr.Brush(colors=["#ff00ff"], color_mode="fixed", default_size=40),
                        eraser=gr.Eraser(default_size=40),
                        layers=False,
                    )
                    i_prompt = gr.Textbox(label="What should appear there?", placeholder="e.g. a wooden bench")
                    i_neg = gr.Textbox(label="Avoid (optional)", placeholder="e.g. blurry, distorted")
                    with gr.Accordion("Settings", open=False):
                        i_steps = gr.Slider(5, 100, 30, step=1, label="Steps")
                        i_cfg = gr.Slider(1, 15, 7.5, step=0.5, label="Prompt strength")
                        i_str = gr.Slider(0.1, 1.0, 1.0, step=0.05, label="Change amount (1 = fully replace)")
                        i_max = gr.Slider(256, 1024, 512, step=64, label="Max working size (px)")
                        i_seed = gr.Number(-1, label="Seed (-1 = random)", precision=0)
                    i_btn = gr.Button("Inpaint", variant="primary")
                with gr.Column():
                    i_out = gr.Image(type="pil", label="Result", format="png")
                    i_mask = gr.Image(type="pil", label="Mask used")
                    i_used = gr.Number(label="Seed used", interactive=False)
            i_btn.click(
                ai_inpaint,
                [i_in, i_prompt, i_neg, i_steps, i_cfg, i_str, i_max, i_seed],
                [i_out, i_mask, i_used],
            )

        with gr.Tab("Adjust"):
            with gr.Row():
                with gr.Column():
                    a_in = gr.Image(type="pil", label="Image")
                    with gr.Row():
                        a_rot = gr.Slider(-180, 180, 0, step=90, label="Rotate (degrees)")
                        a_width = gr.Number(0, label="Resize to width (0 = keep)", precision=0)
                    with gr.Row():
                        a_fh = gr.Checkbox(label="Flip horizontal")
                        a_fv = gr.Checkbox(label="Flip vertical")
                        a_sq = gr.Checkbox(label="Crop to square")
                        a_gray = gr.Checkbox(label="Black & white")
                    a_bri = gr.Slider(0, 2, 1, step=0.05, label="Brightness")
                    a_con = gr.Slider(0, 2, 1, step=0.05, label="Contrast")
                    a_sat = gr.Slider(0, 2, 1, step=0.05, label="Saturation")
                    a_shp = gr.Slider(0, 3, 1, step=0.1, label="Sharpness")
                    a_blur = gr.Slider(0, 20, 0, step=0.5, label="Blur")
                    a_btn = gr.Button("Apply", variant="primary")
                a_out = gr.Image(type="pil", label="Result", format="png")
            a_btn.click(
                adjust,
                [a_in, a_rot, a_fh, a_fv, a_sq, a_width, a_bri, a_con, a_sat, a_shp, a_blur, a_gray],
                a_out,
            )

        with gr.Tab("Remove Background"):
            with gr.Row():
                r_in = gr.Image(type="pil", label="Image")
                r_out = gr.Image(type="pil", label="Result", format="png")
            gr.Button("Remove background", variant="primary").click(remove_bg, r_in, r_out)

    return demo


def main():
    parser = argparse.ArgumentParser(description="Local AI image editor")
    parser.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 to allow other devices on your network")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_ui().queue().launch(server_name=args.host, server_port=args.port, inbrowser=True)


if __name__ == "__main__":
    main()
