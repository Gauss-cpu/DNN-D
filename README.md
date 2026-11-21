# DNN-D: Delta Dynamics Network

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **DNN-D (Delta Dynamics Network)**, a dimension-agnostic framework designed to bridge noisy visual kinematics and precise analytical rigid-body dynamics.

## Overview

Extracting kinematic data from monocular videos is a prevalent approach in Embodied AI. However, visual data is inherently noisy. A minor perturbation in position is catastrophically amplified when computing the second derivative (acceleration), leading to severe torque oscillations and system divergence in downstream physics engines.

**DNN-D** fundamentally bypasses this issue by predicting **inter-frame kinematic increments (\(\Delta q\))** rather than absolute states. By utilizing a Transformer-based Encoder-Decoder and a **Visual Anchoring** mechanism, DNN-D achieves highly stable, noise-resilient, and physically consistent trajectory reconstruction.

## Key Features

- **Bypass Second-Order Noise Amplification**: Predicts \(\Delta q\) to entirely avoid the catastrophic noise amplification of numerical differentiation.
- **Visual Anchoring**: Introduces a lightweight complementary filter during autoregressive inference to prevent long-term drift.
- **Any-DOF Generalization**: A dimension-agnostic architecture. The same codebase can seamlessly adapt to a 2-DOF leg or an 8-DOF arm by simply passing the `dim` parameter.
- **High Sample Efficiency**: Achieves identical convergence with only **5% of the raw training data**, effectively eliminating temporal data redundancy.

## File Structure

```text
dnnd_core/
├── dnnd_module.py    # Core model architecture & inference API
├── train.py          # Universal training script (supports any DOF)
├── train_leg_dnnd.py # Specialized fast-training script for 2-DOF leg
└── requirements.txt  # Python dependencies
