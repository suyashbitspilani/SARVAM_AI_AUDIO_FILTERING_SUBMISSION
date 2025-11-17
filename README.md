# Audio Filtering Pipeline for Indic Speech

**Author:** Suyash Khare\
**Role:** ML Engineering Intern, Sarvam AI Speech Team\
**Date:** November 2024

------------------------------------------------------------------------

## Overview

This project focuses on a very practical challenge we repeatedly face at
Sarvam AI: **cleaning large-scale Indic speech datasets** so that
downstream ASR models can learn from high-quality audio only.

To solve this, I built a **scalable, multi-stage audio filtering
pipeline** that automatically detects noisy, clipped, silent, or
otherwise unusable samples --- across **all 22 Indic languages**.

My goal while designing this system was simple: - **Assess audio quality
reliably**, using multiple complementary metrics. - **Scale to millions
of samples** without blowing up memory. - **Explain every filtering
decision clearly**, so the process never feels like a black box.

------------------------------------------------------------------------

## System Architecture




<img width="5370" height="4170" alt="image" src="https://github.com/user-attachments/assets/1d67fab6-b103-426f-9c40-07853f1b8115" />


The pipeline flows through five major stages:

### 1. Data Ingestion

-   Integrated with the HuggingFace IndicVoices dataset\
-   Uses streaming mode to avoid memory overload\
-   Checkpointing for safe restarts\
-   Works across all 22 Indic languages

### 2. Parallel Processing

-   Multi-core CPU utilization\
-   Efficient audio loading\
-   Configurable batch sizes\
-   Progress tracking & strong error handling

### 3. Quality Metrics Computation

Metrics span three major categories:

#### Signal Metrics

-   SNR\
-   Clipping detection\
-   Silence analysis\
-   RMS energy, dynamic range\
-   Speaking rate estimation

#### Spectral Metrics

-   F0 pitch statistics\
-   Spectral centroid, bandwidth, rolloff\
-   Spectral flatness\
-   Optional MFCCs

#### Perceptual Metrics (Optional)

-   HNR\
-   DNSMOS\
-   ASR confidence\
-   Language verification

### 4. Outlier Detection

-   MAD-based Modified Z-score\
-   IQR\
-   Ensemble scoring\
-   Stratified by language and task type

### 5. Three-Tier Filtering

**Tier 1: Hard Filters**\
- Clipping \> 1%\
- Duration \<1s or \>30s\
- SNR \< -5 dB\
- Silence \> 80%\
- Active speech \< 0.5s

**Tier 2: Soft Filters**\
- Percentile-based filtering\
- Language-specific thresholds

**Tier 3: Contextual Filters**\
- Speaker-level patterns\
- Region-based trends\
- Session consistency

------------------------------------------------------------------------

## Performance

### Speed

-   0.1--0.2 sec per sec of audio\
-   1000--2000 files/hour

### Quality Improvements

-   Retention: 70--85%\
-   +3 to +5 dB SNR improvement\
-   90--95% clipping/silence issues removed

------------------------------------------------------------------------

## Installation

``` bash
pip install -r requirements.txt
export HF_TOKEN="your_huggingface_token"
```

------------------------------------------------------------------------

## Usage

``` bash
python main.py --config config.yaml
python main.py --config config.yaml --languages hindi tamil bengali
python main.py --config config.yaml --dry-run
python demo_3_samples.py
```

------------------------------------------------------------------------

## Outputs

-   complete_filtered_dataset.parquet\
-   {language}\_filtered_manifest.jsonl\
-   {language}\_rejected_samples.jsonl\
-   pipeline_summary.json\
-   metric_distributions.png

------------------------------------------------------------------------

## Future Work

-   DL-based perceptual metrics\
-   Language-specific models\
-   Real-time filtering\
-   Spark/Ray integration

------------------------------------------------------------------------

## About

Built as part of my ML Engineering Internship at **Sarvam AI**.\
Designed to be production-ready, scalable, and transparent.
