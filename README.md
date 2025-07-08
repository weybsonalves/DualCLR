# DualCLR: Enhancing Weight Predictor Networks with Dual-Stage Contrastive Pretraining

**Federal University of Pernambuco (UFPE)** - Center of Informatics  
**Course**: Deep Learning (IN1164)  
**Master's Degree in Computer Science**

# Overview

This repository implements and extends the Weight Predictor Network with Feature Selection (WPFS) for high-dimensional, low-sample-size (HDLSS) biomedical datasets. It introduces C-WPFS, a novel architecture that integrates a two-stage contrastive representation learning approach. The model first utilizes unsupervised contrastive learning via SimCLR's NT-Xent loss to capture data structure, then applies supervised contrastive learning (SupCon) to enhance class-specific clustering.

# Installation

**Requirement:** All project dependencies are included in `requirements.txt`. We assume you have **conda** installed.


**Installing WPFS**
```
conda create python=3.7.9 --name WPFS
conda activate WPFS
pip install -r requirements.txt
```
**Optional:** Change `BASE_DIR` from `/src/_config.py` to point to the project directory on your machine.


# Running an experiment

**Step 1:** Run the script `run_experiment.sh`

**Step 2:** Analyze the results in the notebook `analyze_experiments.ipynb`

**Adding a new dataset is straightforward:**. Search `your_custom_dataset` in the codebase and replace it with your dataset.
