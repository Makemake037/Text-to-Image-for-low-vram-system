Here is the updated version, seamlessly integrating the **Hardware Target Toggle** feature at both the workflow and usage levels.

---

# 🚀 Low-VRAM Text-to-Image Generation

Welcome! This project explores AI-driven image generation with a specific focus on optimization for budget and limited hardware environments.

## 📌 About the Project

This project was born out of a desire to deeply understand the inner mechanics of AI models. Beyond simply running inference, the goal was to investigate how hardware constraints impact performance and how to maximize efficiency through intelligent resource management. It has been an extensive hands-on journey into balancing the interplay between software architecture and hardware limitations.

## ⚙️ How It Works

To ensure smooth generation on hardware with limited VRAM, this project utilizes a **dynamic hotswapping architecture**:

* **Hardware Target Control:** Features a prominent toggle at the top of the interface to completely switch execution between **CPU** and **GPU** mode instantly depending on your immediate resource needs.
* **Memory Efficiency:** The system dynamically swaps assets between the GPU and System RAM on the fly.
* **Space Optimization:** It clears space for new model components by actively managing AI files in real-time.
* **Persistent Performance:** Once loaded into RAM, components remain cached throughout your session to ensure faster subsequent generations, unless you explicitly exit the application.

## 🛠️ Implementation Details

* **User Interface:** Features a clean, user-friendly web interface built with **Gradio**, including quick-access hardware configuration tools at the very top of the page.
* **Model:** Utilizes `Z-Image-Turbo-SDNQ-uint4-svd-r32`. This is a 4-bit (UINT4 with SVD rank 32) quantization of the `Tongyi-MAI/Z-Image-Turbo` model, processed via SDNQ to provide high-quality output while drastically reducing the VRAM footprint.
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

```


3. Once running, open your browser and navigate to the local host address: **`[http://127.0.0.1:7860/](http://127.0.0.1:7860/)`**
4. **Configuration Steps in the UI:**
* Select **`working 1`** from the profile dropdown menu to load the correct configuration.
* Use the **Hardware Target Button** located at the top of the page to easily toggle the execution mode completely between CPU and GPU.



---

> ⚠️ **Disclaimer:** This project is an optimization and implementation showcase. I do not own the underlying models or core architectures utilized in this repository; all credits go to their respective creators.
