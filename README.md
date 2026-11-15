# DART Motion Studio

A text-to-motion application built using the pretrained DART model.

## Overview

Users enter a natural-language motion prompt through a Gradio
interface. The application runs pretrained DART inference and
returns a skeleton animation and downloadable motion data.

## Features

- Natural-language motion prompts
- Pretrained DART motion generation
- Adjustable motion length
- Animated 3D skeleton preview
- Downloadable motion data

## Project Architecture

Gradio GUI -> Python inference wrapper -> DART model
-> Generated joint coordinates -> MP4 visualization

## Requirements

- Google Colab with a GPU
- Compatible Python/PyTorch environment
- DART source code and pretrained checkpoints
- SMPL-X assets for full-body visualization (optional)

## Running the Application

This prototype assumes DART and its dependencies have already
been installed in:

    /content/DART
    /content/dart-env

Start the application:

    /content/dart-env/bin/python /content/DART/motion_service/frontend/app.py

Then open the Gradio URL.

## Current Limitations

- The model must be installed and running in a compatible environment.
- Inference reloads model checkpoints for each request.
- The Gradio interface currently runs alongside the inference backend.
- Cloud deployment as a separate service is planned.
- Full-body Three.js visualization is still being developed.

## Original Research

DART: A Diffusion-Based Autoregressive Motion Model
for Real-Time Text-Driven Motion Control

Original repository:
https://github.com/zkf1997/DART

This project uses pretrained DART checkpoints.
It does not retrain the original model.
