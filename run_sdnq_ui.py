import torch
import diffusers
import os
import uuid
import random
import gradio as gr
import json
import gc

# --- CRITICAL SYSTEM MEMORY BUFFER ALLOCATION ---
# Prevents CUDA memory fragmentation and tells Windows to reuse VRAM memory pools aggressively
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model
from transformers import CLIPTextModel

# --- CONFIGURATION ---
MODEL_PATH = "C:/ZImage"
CHECKPOINT_FILE = "outputs/checkpoints.json"
GOOD_SEEDS = [
    4224231487544, 487609804978759, 14312527394452, 1063568336641306,
    1011173389061030, 9079599605601, 295920080613453, 509942702529433,
    942195105852109, 1114928983060288, 798085408489866, 94291763623401,
    779870172276697, 430216992284160, 643312781770518
]

# Global cache for pipeline
MODEL_CACHE = {"pipe": None}

def get_or_load_pipe(device_mode="cuda"):
    if MODEL_CACHE["pipe"] is None:
        print("Loading base model components...")
        
        # Enforcing bfloat16 precision parameters
        if device_mode == "cuda" and torch.cuda.is_available():
            dtype = torch.bfloat16
            print("[DEBUG] Initializing engine with bfloat16 precision context.")
        else:
            dtype = torch.float32
            
        pipe = diffusers.ZImagePipeline.from_pretrained(MODEL_PATH, torch_dtype=dtype)
        device = "cuda" if device_mode == "cuda" and torch.cuda.is_available() else "cpu"
        
        # Move the base components to GPU memory initially
        try:
            pipe.to(device)
        except Exception as exc:
            print(f"[WARN] Device placement failed: {exc}")

        if triton_is_available and torch.cuda.is_available():
            try:
                print("Enforcing use_quantized_matmul parameters onto base blocks...")
                pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
                pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)
            except Exception as exc:
                print(f"[WARN] SDNQ setup failed: {exc}")

        try:
            pipe.scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_config(
                pipe.scheduler.config,
                shift=3.0,
                use_beta_sigmas=True
            )
        except Exception as exc:
            print(f"[WARN] FlowMatch scheduler configuration alignment bypassed: {exc}")

        # --- RAM BUFFER / LOW VRAM OFFLOAD MANAGEMENT ---
        try:
            print("[DEBUG] Initializing micro-batch memory tiling optimizations...")
            pipe.enable_attention_slicing(1)
            
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()  # Processes large image grids in micro-tiles to save VRAM
            
            # Mandates System RAM offloading boundaries straight to the CUDA framework
            if device_mode == "cuda" and hasattr(pipe, "enable_sequential_cpu_offload"):
                print("[VRAM BUFFER ACTIVE] Using System RAM as an active streaming swap buffer.")
                pipe.enable_sequential_cpu_offload()
            elif device_mode == "cuda" and hasattr(pipe, "enable_model_cpu_offload"):
                print("[VRAM BUFFER ACTIVE] Enforcing model-level CPU offload strategies.")
                pipe.enable_model_cpu_offload()
        except Exception as e:
            print(f"[WARN] Memory optimization strategy injection bypassed: {e}")

        print("Model ready!")
        MODEL_CACHE["pipe"] = pipe

    return MODEL_CACHE["pipe"]


# --- CHECKPOINT MANAGEMENT ---
def save_checkpoint(name, prompt, neg, res, steps, cfg, seed, lora_name, lora_scale, vae_file, text_encoder_file, transformer_file, scheduler_file):
    if not name or not name.strip():
        return gr.update()

    data = {}
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}

    data[name.strip()] = {
        "prompt": prompt,
        "neg": neg,
        "res": res,
        "steps": steps,
        "cfg": cfg,
        "seed": seed,
        "lora": lora_name,
        "lora_scale": lora_scale,
        "vae": vae_file,
        "text_encoder": text_encoder_file,
        "transformer": transformer_file,
        "scheduler": scheduler_file,
    }

    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=4)

    return gr.update(choices=list(data.keys()), value=name.strip())


def load_checkpoint(name):
    if not name or not os.path.exists(CHECKPOINT_FILE):
        return [gr.update()] * 12

    try:
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
        if name not in data:
            return [gr.update()] * 12

        c = data[name]
        return (
            c.get("prompt", ""),
            c.get("neg", ""),
            c.get("res", 512),
            c.get("steps", 8),
            c.get("cfg", 1.0),
            c.get("seed", 42),
            c.get("lora", "None"),
            c.get("lora_scale", 1.0),
            c.get("vae", "Default"),
            c.get("text_encoder", "Default"),
            c.get("transformer", "Default"),
            c.get("scheduler", "Default"),
        )
    except Exception as exc:
        print(f"[WARN] Failed to load checkpoint: {exc}")
        return [gr.update()] * 12


def delete_checkpoint(name):
    if not name or not os.path.exists(CHECKPOINT_FILE):
        return gr.update()

    try:
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
        if name in data:
            del data[name]
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(data, f, indent=4)
            
        return gr.update(choices=list(data.keys()), value=None)
    except Exception as exc:
        print(f"[WARN] Failed to delete checkpoint: {exc}")
        return gr.update()


# --- UTILITIES ---
def normalize_dropdown_value(value, fallback):
    if value is None:
        return fallback
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else fallback
    return value


def scan_dropdown_choices(path, extensions=None, default_label="Default"):
    if not os.path.exists(path):
        return [default_label]

    files = sorted(
        f for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) or os.path.isdir(os.path.join(path, f))
    )
    filtered = []
    for f in files:
        if f == default_label:
            continue
        full_p = os.path.join(path, f)
        if os.path.isdir(full_p):
            filtered.append(f)
        elif extensions and any(f.lower().endswith(ext.lower()) for ext in extensions):
            filtered.append(f)
        elif not extensions:
            filtered.append(f)
            
    return [default_label] + filtered


def refresh_component_dropdowns():
    checkpoint_choices = []
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                checkpoint_choices = list(json.load(f).keys())
        except Exception:
            pass

    vae_choices = scan_dropdown_choices("C:/ZImage/vae", [".safetensors", ".bin"], "Default")
    lora_choices = scan_dropdown_choices("C:/ZImage/LoRA", [".safetensors", ".bin"], "None")
    text_encoder_choices = scan_dropdown_choices("C:/ZImage/text_encoder", [".safetensors", ".bin", ".json"], "Default")
    transformer_choices = scan_dropdown_choices("C:/ZImage/transformer", [".safetensors", ".bin", ".json"], "Default")
    scheduler_choices = scan_dropdown_choices("C:/ZImage/scheduler", [".json"], "Default")

    return (
        gr.update(choices=checkpoint_choices),
        gr.update(choices=vae_choices, value="Default"),
        gr.update(choices=lora_choices, value="None"),
        gr.update(choices=text_encoder_choices, value="Default"),
        gr.update(choices=transformer_choices, value="Default"),
        gr.update(choices=scheduler_choices, value="Default"),
    )


def get_random_good_seed():
    return random.choice(GOOD_SEEDS)


def get_status(device_mode="cuda"):
    label = "CUDA GPU" if device_mode.lower() == "cuda" else "Integrated CPU"
    return f"**System Status:** Stability Matrix Framework Loaded | Precision: `bfloat16` | RAM Swap Buffer: `Active` | Device: `{label}`"


def toggle_device_mode(current_mode):
    next_mode = "cpu" if current_mode.lower() == "cuda" else "cuda"
    button_label = "Switch to CUDA GPU" if next_mode == "cuda" else "Switch to Integrated CPU"
    return next_mode, button_label, get_status(next_mode)


# --- GENERATION LOGIC ---
def generate(prompt, neg_prompt, res, steps, cfg, seed, lora_scale, lora_name, vae_file, text_encoder_file, transformer_file, scheduler_file, device_mode):
    # Aggressively flush old memory fragments before running the next generation pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    device_mode = (device_mode or "cuda").lower()
    device = "cuda" if device_mode == "cuda" and torch.cuda.is_available() else "cpu"

    lora_name = normalize_dropdown_value(lora_name, "None")
    vae_file = normalize_dropdown_value(vae_file, "Default")
    text_encoder_file = normalize_dropdown_value(text_encoder_file, "Default")
    transformer_file = normalize_dropdown_value(transformer_file, "Default")
    scheduler_file = normalize_dropdown_value(scheduler_file, "Default")

    print("[DEBUG] Initiating streaming generation block...")
    pipe = get_or_load_pipe(device_mode)

    try:
        pipe.unload_lora_weights()
    except Exception:
        pass

    if lora_name and lora_name.lower() != "none":
        path = os.path.join("C:/ZImage/LoRA", lora_name)
        if os.path.exists(path):
            pipe.load_lora_weights(path, adapter_name="current_lora")
            pipe.set_adapters(["current_lora"], adapter_weights=[lora_scale])

    if vae_file and vae_file != "Default":
        vae_dir = os.path.join("C:/ZImage/vae", vae_file)
        if os.path.exists(vae_dir):
            try:
                pipe.vae = diffusers.AutoencoderKL.from_pretrained(vae_dir, torch_dtype=pipe.dtype)
                # If offloading is active, keep custom additions inside the unified CPU offload stream map
                if hasattr(pipe, "enable_sequential_cpu_offload") and device_mode == "cuda":
                    pass
                else:
                    pipe.vae.to(device)
            except Exception as exc:
                print(f"[WARN] Failed to load custom VAE: {exc}")

    if text_encoder_file and text_encoder_file != "Default":
        text_encoder_dir = os.path.join("C:/ZImage/text_encoder", text_encoder_file)
        if os.path.exists(text_encoder_dir):
            try:
                pipe.text_encoder = CLIPTextModel.from_pretrained(text_encoder_dir, torch_dtype=pipe.dtype)
                if triton_is_available and torch.cuda.is_available():
                    pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)
                if not (hasattr(pipe, "enable_sequential_cpu_offload") and device_mode == "cuda"):
                    pipe.text_encoder.to(device)
            except Exception as exc:
                print(f"[WARN] Failed to load custom text encoder: {exc}")

    if transformer_file and transformer_file != "Default":
        transformer_dir = os.path.join("C:/ZImage/transformer", transformer_file)
        if os.path.exists(transformer_dir):
            try:
                if os.path.isfile(transformer_dir):
                    state_dict = torch.load(transformer_dir, map_location="cpu")
                    pipe.transformer.load_state_dict(state_dict, strict=False)
                    del state_dict
                else:
                    pipe.transformer = pipe.transformer.__class__.from_pretrained(transformer_dir, torch_dtype=pipe.dtype)
                
                if triton_is_available and torch.cuda.is_available():
                    pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
                if not (hasattr(pipe, "enable_sequential_cpu_offload") and device_mode == "cuda"):
                    pipe.transformer.to(device)
            except Exception as exc:
                print(f"[WARN] Failed to load custom transformer component: {exc}")

    if scheduler_file and scheduler_file != "Default":
        scheduler_dir = os.path.join("C:/ZImage/scheduler", scheduler_file)
        if os.path.exists(scheduler_dir):
            try:
                pipe.scheduler = pipe.scheduler.__class__.from_pretrained(scheduler_dir)
            except Exception as exc:
                print(f"[WARN] Failed to swap custom scheduler profile matrix: {exc}")

    pipe.vae.enable_tiling()
    generator = torch.Generator(device).manual_seed(int(seed))

    image_obj = pipe(
        prompt=prompt, negative_prompt=neg_prompt,
        height=res, width=res,
        num_inference_steps=steps, guidance_scale=cfg,
        generator=generator
    ).images[0]

    os.makedirs("outputs", exist_ok=True)
    filename = f"outputs/img_{uuid.uuid4().hex[:8]}.png"
    image_obj.save(filename, quality=100, optimize=True)

    # Clean the VRAM cache immediately upon finishing to free the workspace for the next run
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return image_obj


# --- UI LAYOUT DESIGN ---
with gr.Blocks(title="Text-to-Image Pro Studio") as demo:
    device_mode_state = gr.State("cuda")
    status_box = gr.Markdown(get_status())
    gr.Markdown("# 🎨 Text-to-Image Pro Studio (Stability Matrix Optimized Engine)")

    with gr.Row():
        with gr.Column(scale=1):
            ckpt_name = gr.Textbox(label="Checkpoint Name to Save")
            
            with gr.Row():
                save_btn = gr.Button("💾 Save Configuration", variant="secondary")
                delete_btn = gr.Button("❌ Delete Selected", variant="stop")
                
            ckpt_dropdown = gr.Dropdown(label="Load Checkpoint", choices=[])

            prompt = gr.Textbox(label="Prompt", value="a beautiful landscape, highly detailed, 8k resolution")
            neg_prompt = gr.Textbox(label="Negative Prompt", value="blurry, low quality, distorted")
            res = gr.Slider(64, 1536, 512, step=64, label="Resolution Width/Height (Multiples of 64)")
            steps = gr.Slider(1, 30, 8, step=1, label="Steps")

            vae_dropdown = gr.Dropdown(choices=scan_dropdown_choices("C:/ZImage/vae", [".safetensors", ".bin"], "Default"), value="Default", allow_custom_value=True, label="Select VAE Component (Advanced)")
            lora_dropdown = gr.Dropdown(choices=scan_dropdown_choices("C:/ZImage/LoRA", [".safetensors", ".bin"], "None"), value="None", allow_custom_value=True, label="Select LoRA Addon")
            text_encoder_dropdown = gr.Dropdown(choices=scan_dropdown_choices("C:/ZImage/text_encoder", [".safetensors", ".bin", ".json"], "Default"), value="Default", allow_custom_value=True, label="Select Text Encoder Sub-module")
            transformer_dropdown = gr.Dropdown(choices=scan_dropdown_choices("C:/ZImage/transformer", [".safetensors", ".bin", ".json"], "Default"), value="Default", allow_custom_value=True, label="Select Transformer weights")
            scheduler_dropdown = gr.Dropdown(choices=scan_dropdown_choices("C:/ZImage/scheduler", [".json"], "Default"), value="Default", allow_custom_value=True, label="Select Sampler / Scheduler Map")

            with gr.Accordion("Advanced Settings", open=True):
                cfg = gr.Slider(0.0, 15.0, 1.0, step=0.1, label="CFG Scale")
                seed = gr.Number(1223844113, label="Seed Target")
                random_seed_btn = gr.Button("🎲 Random Seed")
                lora_scale = gr.Slider(0.0, 2.0, 1.0, step=0.1, label="LoRA Strength")

            device_toggle_btn = gr.Button("Switch Hardware Target")
            btn = gr.Button("Generate Image", variant="primary")

        with gr.Column(scale=2):
            output = gr.Image(label="Generated Output Result")

    # --- EVENT LINKING MANAGEMENT BLOCK ---
    save_btn.click(
        save_checkpoint,
        inputs=[ckpt_name, prompt, neg_prompt, res, steps, cfg, seed, lora_dropdown, lora_scale, vae_dropdown, text_encoder_dropdown, transformer_dropdown, scheduler_dropdown],
        outputs=ckpt_dropdown
    )

    delete_btn.click(
        delete_checkpoint,
        inputs=[ckpt_dropdown],
        outputs=ckpt_dropdown
    )

    ckpt_dropdown.change(
        load_checkpoint,
        inputs=[ckpt_dropdown],
        outputs=[prompt, neg_prompt, res, steps, cfg, seed, lora_dropdown, lora_scale, vae_dropdown, text_encoder_dropdown, transformer_dropdown, scheduler_dropdown],
    )

    device_toggle_btn.click(
        toggle_device_mode,
        inputs=[device_mode_state],
        outputs=[device_mode_state, device_toggle_btn, status_box],
    )

    random_seed_btn.click(get_random_good_seed, outputs=seed)

    btn.click(
        generate,
        inputs=[prompt, neg_prompt, res, steps, cfg, seed, lora_scale, lora_dropdown, vae_dropdown, text_encoder_dropdown, transformer_dropdown, scheduler_dropdown, device_mode_state],
        outputs=output,
    )

    demo.load(
        refresh_component_dropdowns,
        outputs=[ckpt_dropdown, vae_dropdown, lora_dropdown, text_encoder_dropdown, transformer_dropdown, scheduler_dropdown]
    )

if __name__ == "__main__":
    demo.launch(max_threads=40)