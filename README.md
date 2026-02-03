# RoboRefer-FineTuning: Spatial-Aware LLM for Embodied AI

> **A customized implementation of RoboRefer, enhanced with Supervised Fine-Tuning (SFT) on ScanNet datasets to improve 3D spatial reasoning and instruction following.**

## 📌 Project Overview
This repository contains the implementation of a **Multimodal Large Language Model (MLLM)** pipeline designed for Embodied AI agents. The goal is to enable robots to understand complex human instructions involving 3D spatial relationships (e.g., "Go to the chair *behind* the table").

This project builds upon the [RoboRefer] framework and introduces a novel **SFT strategy** and **Data Engine** to handle complex 3D-to-2D correspondences.

## 🚀 Key Features & Technical Contributions

### 1. Large-Scale Data Engine (ETL)
* **Pipeline:** Built an end-to-end data processing pipeline handling **1,006 ScanNet scenes** (~500k+ frames).
* **Geometry Filtering:** Implemented custom geometric logic to automatically cull immobile objects, occlusions, and sensor noise, ensuring high-fidelity training data.
* **Scale:** Generated **220,000+ high-quality instruction-view pairs** for model training.

### 2. Supervised Fine-Tuning (SFT) with Novel Prompting
* **Method:** Fine-tuned the LLM using a custom **"Visual Grid Coordinate"** prompt structure.
* **Innovation:** Unlike standard text prompts, this method overlays spatial grid coordinates onto visual inputs, significantly enhancing the model's ability to reason about relative positions.
* **Robustness:** Incorporated **Negative Sampling** in the training data to improve the model's rejection capabilities (reducing hallucinations).

### 3. Automated Validation & Infrastructure
* **Tooling:** Developed a JSON-based validation harness to visualize and debug alignment issues between 3D meshes and 2D projections.
* **Architecture:** Scripts are optimized for **HPC Clusters (Slurm)**, supporting distributed data processing and training.

## 📊 Results (Metrics)
* **Precision/Recall:** Achieved significantly higher spatial grounding accuracy compared to the baseline zero-shot model.
* **Data Quality:** The automated cleaning pipeline reduced dataset noise by **~40%**, directly contributing to stable convergence during fine-tuning.

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.8+
* PyTorch 2.0+ (with CUDA support)
* Open3D

### Quick Start
```bash
# Clone the repository
git clone [https://github.com/KevinYang/RoboRefer-FineTuning.git](https://github.com/KevinYang/RoboRefer-FineTuning.git)
cd RoboRefer-FineTuning

# Install dependencies
pip install -r requirements.txt

# Run the data validation tool
python tools/check_depth_pipeline.py --scene_id scene0000_00
