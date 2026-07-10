# 🚀 Low-VRAM Text-to-Image Generation

Welcome! This project explores AI-driven image generation with a specific focus on optimization for budget and limited hardware environments.

## 🖼️ Showcase Gallery
Here are some example images generated using this optimized low-VRAM workflow along with system captures:

<p align="center">
  <img src="assets/img_18b506b0.png" width="45%" alt="Generated Cat Image" />
  <img src="assets/img_c9daffa4.png" width="45%" alt="Generated Landscape Image" />
</p>

<p align="center">
  <img src="assets/folder structure.png" width="45%" alt="Workspace Folders" />
  <img src="assets/cmd scrren.png" width="45%" alt="CLI Terminal Interface" />
</p>

---

## 🎛️ Feature Spotlight: Dynamic Hardware Target Switch & Status Indicator

To give you absolute control over your system's resources, this project splits your hardware controls for maximum clarity:
* **Live Status Indicator (Top of the page):** Keeps you informed in real-time whether the backend execution layer is currently processing tasks on your **CPU** or **GPU**.
* **Hardware Target Button (Bottom of the page):** A dedicated control button that completely switches the execution pipeline on the fly. 
  * If your GPU is running out of memory during a heavy generation task, toggle it to **CPU** mode.
  * When you want maximum speed and have available overhead, click to swap back to **GPU** mode instantly.

---

## 📌 About the Project

This project was born out of a desire to deeply understand the inner mechanics of AI models. Beyond simply running inference, the goal was to investigate how hardware constraints impact performance and how to maximize efficiency through intelligent resource management. It has been an extensive hands-on journey into balancing the interplay between software architecture and hardware limitations.

## ⚙️ How It Works

To ensure smooth generation on hardware with limited VRAM, this project utilizes a **dynamic hotswapping architecture**:

* **Hardware Target Separation:** Places a real-time hardware tracking status display at the top of the interface, while keeping the structural toggle control switch at the bottom of the page layout.
* **Persistent Configuration Storage:** The `outputs/` folder contains a critical `checkpoints.json` configuration file. This tracks all saved checkpoint setups for the app. By default, it comes pre-configured with the **`working 1`** checkpoint profile to get you running out of the box.
* **Memory Efficiency:** The system dynamically swaps assets between the GPU and System RAM on the fly.
* **Space Optimization:** It clears space for new model components by actively managing AI files in real-time.
* **Persistent Performance:** Once loaded into RAM, components remain cached throughout your session to ensure faster subsequent generations, unless you explicitly exit the application.

## 🛠️ Implementation Details

* **User Interface:** Features a clean, user-friendly web interface built with **Gradio**, neatly mapping hardware status rules to the top headers and control switches to the footer actions.
* **Model Pipeline:** Utilizes the [Z-Image-Turbo-SDNQ-uint4-svd-r32](https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32/tree/main) repository. This is a 4-bit (UINT4 with SVD rank 32) quantization of the `Tongyi-MAI/Z-Image-Turbo` model, processed via SDNQ to provide high-quality output while drastically reducing the VRAM footprint.
* **Compatibility:** Defaults to **bf16 (Bfloat16)** to ensure broader hardware compatibility and stability during inference.

---

## 🚀 Getting Started & Usage

### 📋 Prerequisites

1. **Stability Matrix:** You must have Stability Matrix installed at the default location to utilize its bundled virtual environment:
`C:\StabilityMatrix-win-x64\Data\Packages\ComfyUI\venv\Scripts\python.exe`
2. **Project Directory:** The application script and output management folder must be located at:
`C:\ZImage\`

### 📥 1. Automated Model Syncing (Hugging Face)

Because the repository layout remains incredibly lightweight, it does not include the massive weight files by default. Before launching the app for the first time, you **must synchronize the model cache** to your desktop:

1. Open your terminal or command prompt inside the directory (`cd C:\ZImage`).
2. Run the bundled synchronization script:
   ```cmd
   sync_models.bat
