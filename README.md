# edema-analysis
<<<<<<< HEAD
Image analysis scripts for detecting and quantifying cardiac edema in zebrafish embryos.
=======

Python-based image analysis pipeline for detecting and quantifying **cardiac edema in zebrafish embryos** from microscopy images.

## Overview
This repository contains image analysis scripts designed to identify and quantify cardiac edema in zebrafish embryos. The pipeline prioritizes **cardiac edema measurement**, using **eye object detection as an anatomical reference** when direct edema segmentation is challenging.

The goal is to extract **reproducible diameter and area measurements** from brightfield or fluorescence microscopy images.

## Input Data
- Brightfield or fluorescence microscopy images of zebrafish embryos
- Images may include severe cardiac and yolk sac edema  
- Analysis focuses primarily on **cardiac edema regions**

## Analysis Pipeline
1. **Image preprocessing**  
   Normalization, denoising, and thresholding

2. **Eye detection**  
   Eye object detection used as a stable anatomical landmark

3. **Cardiac region localization**  
   Spatial mapping from eye position to cardiac region

4. **Edema segmentation**  
   Identification of edema boundaries in the cardiac region

5. **Quantification**  
   Extraction of edema diameter and area measurements

## Scripts
- `eye_detection.py` — Detects eye objects for anatomical reference
- `cardiac_region_localization.py` — Maps cardiac region relative to eyes
- `edema_segmentation.py` — Segments cardiac edema
- `diameter_analysis.py` — Computes diameter and area metrics

## Tools & Libraries
- Python  
- OpenCV  
- NumPy  
- scikit-image  
- matplotlib  

## Author
Yashita Kaku

>>>>>>> 048f180 (Add project README)
