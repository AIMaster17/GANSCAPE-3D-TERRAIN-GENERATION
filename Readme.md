# 🌄 3D Terrain GAN Generator

## 🚀 Project Overview

A GAN‑based pipeline that learns to generate realistic heightmaps from segmentation maps and visualizes them in 3D via Matplotlib \& Streamlit.

## 🗂️ Repo Structure

```
GAN/
├─ dataset/                # Kaggle Earth Terrain maps
├─ Train.py                # Training script
├─ Test.py                 # Metrics & post‑processing
├─ Validation.py           # Validation loop
├─ terrain_3d.py           # Static 3D visualization
├─ terrain_3d1.py          # Enhanced colormap & hillshade
├─ inference.py            # CLI‑style inference script
├─ Checkpoints/            # Saved model checkpoints
├─ gen_images/             # GAN outputs
├─ real_images/            # Ground‑truth heightmaps
├─ Generated_3D/           # 3D render outputs
├─ Inference_3d Plots/     # Saved inference plots
└─ app.py                  # Streamlit app
```


## 📦 Dataset

- **Source**: Kaggle Earth Terrain: Height \& Segmentation Maps
- **Contents**:
    - `_i2.png`: RGB segmentation
    - `_h.png`: Grayscale heightmap


## 🏗️ Model Architectures

- **Generator**: Attention‑UNet with residual blocks
    - Down/Up sampling at 4 scales
    - Bottleneck with two `ResidualBlock`s
    - Tanh output → height in [–1,1]
- **Discriminator**: PatchGAN
    - Spectral normalization
    - Hinge loss


## ⚙️ Training \& Validation

1. **Preprocessing**
    - Resize to 512×512, random flips
    - Height: clamp → log → normalize
2. **Training** (`Train.py`)
    - Optimizers: Adam (G: 2e‑4, D: 1e‑7)
    - Loss: L1 (λ=40) + adversarial (λ=1)
    - Mixed-precision and gradient clipping
3. **Validation** (`Validation.py`)
    - Hinge GAN losses + L1 (λ=50)
    - Track avg G/D loss per epoch

## 🔍 Evaluation Metrics

- **MSE / MAE**
- **PSNR**
- **SSIM**

Computed over matched real vs. generated `.png` files in `gen_images/` vs. `real_images/` using `Test.py`.

## 🎨 3D Visualization

- **terrain_3d.py**: basic Matplotlib surface
- **terrain_3d1.py**:
    - Natural colormap
    - Hillshading (Matplotlib LightSource)
    - Output saved in `Generated_3D/`


## 🎯 Inference

- **inference.py**

1. Point `INPUT_DIR` to your generated maps
2. Configure parameters at top (σ, octaves, etc.)
3. Run:

```bash
python inference.py
```

4. Outputs are saved to `Inference_3d Plots/`


## 📱 Streamlit App

Launch an interactive UI via `app.py`:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload your `.png` maps to generate and view 3D terrains inline.

## 🛠️ Tech Stack

- Python 3.10
- PyTorch 1.13
- TorchVision, Streamlit
- NumPy, SciPy, Pillow
- Matplotlib, PyVista
- OpenSimplex


## ✨ Author

AIMaster17

Feel free to ⭐ the project and open issues!

<div style="text-align: center">⁂</div>

[^1]: https://img.shields.io/badge/Streamlit-App-orange

[^2]: https://img.shields.io/badge/PyTorch-1.13.1-red

[^3]: https://img.shields.io/badge/Python-3.10-blue

