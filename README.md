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

## 🎛️ Feature Spotlight: Dynamic Hardware Target Switch

At the very top of the web page, you will find a **Hardware Target Button** that completely changes the backend execution layer between **CPU** and **GPU** modes on the fly. 
* If your GPU is running out of memory during a heavy generation task, you can toggle it completely to **CPU** mode.
* When you want maximum speed and have available overhead, click to swap back to **GPU** mode instantly.

---

## 📌 About the Project

This project was born out of a desire to deeply understand the inner mechanics of AI models. Beyond simply running inference, the goal was to investigate how hardware constraints impact performance and how to maximize efficiency through intelligent resource management. It has been an extensive hands-on journey into balancing the interplay between software architecture and hardware limitations.

## ⚙️ How It Works

To ensure smooth generation on hardware with limited VRAM, this project utilizes a **dynamic hotswapping architecture**:

* **Hardware Target Control Toggle:** Features a prominent hardware mode switch at the top of the interface page to instantly transition execution pipelines between **CPU** and **GPU** processing.
* **Memory Efficiency:** The system dynamically swaps assets between the GPU and System RAM on the fly.
* **Space Optimization:** It clears space for new model components by actively managing AI files in real-time.
* **Persistent Performance:** Once loaded into RAM, components remain cached throughout your session to ensure faster subsequent generations, unless you explicitly exit the application.

## 🛠️ Implementation Details

* **User Interface:** Features a clean, user-friendly web interface built with **Gradio**, putting the hardware control mode configuration buttons right at the top of the page layout.
* **Model Pipeline:** Utilizes the [Z-Image-Turbo-SDNQ-uint4-svd-r32](https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32/tree/main) repository. This is a 4-bit (UINT4 with SVD rank 32) quantization of the `Tongyi-MAI/Z-Image-Turbo` model, processed via SDNQ to provide high-quality output while drastically reducing the VRAM footprint.
* **Compatibility:** Defaults to **bf16 (Bfloat16)** to ensure broader hardware compatibility and stability during inference.

---

## 🚀 Getting Started & Usage

### 📋 Prerequisites

1. **Stability Matrix:** You must have Stability Matrix installed at the default location to utilize its bundled virtual environment:
`C:\StabilityMatrix-win-x64\Data\Packages\ComfyUI\venv\Scripts\python.exe`
2. **Project Directory:** The application script and output management folder must be located at:
`C:\ZImage\`

### 🏃‍♂️ Running the Application

1. Open your terminal or command prompt.
2. Run the following command to start the UI:
```cmd
"C:\StabilityMatrix-win-x64\Data\Packages\ComfyUI\venv\Scripts\python.exe" "C:\ZImage\run_sdnq_ui.py"
