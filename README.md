# Text-to-Image for Low VRAM

👋 Welcome! This project explores AI-driven image generation with a specific focus on optimization for limited hardware.

## About the Project

This project was born out of a desire to understand the mechanics behind AI models. Beyond just running them, I wanted to investigate how hardware constraints impact performance and how to maximize efficiency through intelligent resource management. It has been an extensive hands-on journey into balancing the interplay between software architecture and hardware limitations.

## How It Works

To ensure smooth generation on hardware with limited VRAM, this project utilizes a **hotswapping architecture**:

* **Memory Efficiency:** The system dynamically swaps assets between the GPU and System RAM.
* **Space Optimization:** It clears space for new model components by managing AI files in real-time.
* **Persistent Performance:** Once loaded into RAM, components remain cached throughout your session to ensure faster subsequent generation, unless you choose to exit the application.

## Implementation Details

* **Interface:** This project features a user-friendly web interface built with **Gradio**.
* **Model:** It utilizes the [Z-Image-Turbo-SDNQ-uint4-svd-r32](https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32) model. This is a 4-bit (UINT4 with SVD rank 32) quantization of the [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) model, processed via SDNQ to provide high-quality output while drastically reducing VRAM requirements.
* **Compatibility:** The system defaults to using **bf16** (Bfloat16) to ensure broader hardware compatibility and stability during inference.

############## Usage ##################

**Prerequisite:** You must have **Stability Matrix** installed at the following location to use the bundled virtual environment: `C:\StabilityMatrix-win-x64\Data\Packages\ComfyUI\venv\Scripts\python.exe`.

and must have seperate folder C:\ZImage\run_sdnq_ui.py"  ====--   THIS is to manage your output folder where script writes generated image.

@echo off
"C:\StabilityMatrix-win-x64\Data\Packages\ComfyUI\venv\Scripts\python.exe" "C:\ZImage\run_sdnq_ui.py"

$$$ load profile working 1 from drop down


================NOTHING IS OWNED BY ME.===================
