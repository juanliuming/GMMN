# GMMN
Graph mapping Mamba network for automated macular edema diagnosis from fundus images

## 🌟 Introduction

This project explores the combination of **Mamba** and **Graph Convolutional Networks (GCN)** for image analysis tasks. The integration of these two architectures represents a mutually beneficial design: Mamba contributes its strong capability for capturing **long-range dependencies**, which GCNs typically lack, while GCNs complement Mamba by enhancing **local connectivity modeling**, which Mamba may underexploit in visual tasks.

Moreover, both Mamba and GCN are inherently designed to process **1D sequential data**, which enables image representation and reasoning without the need for excessive 2D unfolding or complex restructuring. This synergy not only simplifies the data pipeline but also improves model efficiency and interpretability for image understanding tasks.

## 🧠 Framework Overview

<p align="center">
  <img src="assets/framework.png" width="700"/>
</p>

<p align="center">
  <em>Figure: Overall architecture of the proposed GMMN framework combining Mamba and GCN for retinal image analysis.</em>
</p>

## <a name="installation"></a> :wrench: Installation

This codebase was tested with the following environment settings, though it may work with other compatible versions:

- **OS**: Ubuntu 22.04  
- **CUDA**: 11.8+  
- **Python**: 3.11+  
- **PyTorch**: 2.6+

To enable selective scan with efficient hardware design, install the following libraries:

```
pip install causal_conv1d
pip install mamba_ssm
```

You can also create a clean conda environment and install all required packages with a
```
conda install --yes --file requirements.txt
```
