# EPICS ROP Classification (RetCam + Neo)

This repository contains a custom GPU-based CNN pipeline for binary classification of retinal images:

- `ROP` (Retinopathy of Prematurity)
- `Normal`

The current training script uses CuPy-based layers and trains a simple CNN, then saves final classifier weights to `.npy` files.

## Project Structure

- `train.py`: main training + validation loop and sample visualization
- `models.py`: CNN architecture
- `layers.py`: custom CuPy layers (`conv2d`, `relu`, `maxpool`, `linear`)
- `data_loader.py`: loads images from class folders and converts to CuPy tensors
- `utils.py`: softmax, cross-entropy, accuracy
- `Neo_Normal/`, `Neo_ROP/`, `RetCam_Normal/`, `RetCam_ROP/`: image data folders
- `W4.npy`, `b4.npy`: saved FC-layer weights

## Requirements

- Python 3.10+ recommended
- NVIDIA GPU with CUDA support
- Installed dependencies:

```bash
pip install cupy-cuda12x opencv-python matplotlib
```

If your CUDA version differs, install the matching CuPy build from the official CuPy docs.

## Dataset Layout

`data_loader.py` expects class folders directly under the dataset root:

```text
HVDROPDB_RetCam_Neo_Classification/
  Neo_Normal/
  Neo_ROP/
  RetCam_Normal/
  RetCam_ROP/
```

Any folder name containing `ROP` is labeled as class `1`, otherwise class `0`.

## Train

Run:

```bash
python train.py
```

Important: `train.py` currently has a hardcoded dataset path:

```python
X, y = load_dataset("C:/Users/vishw/Downloads/EPICS/HVDROPDB_RetCam_Neo_Classification")
```

Update this path if you clone the repository to a different location.

## Output

After training, the script saves:

- `W4.npy`
- `b4.npy`

It also plots a few validation predictions using `matplotlib`.

## Notes

- Current script updates only the final fully connected layer weights in backprop.
- Default training settings in `train.py`: `batch_size=1`, `lr=0.001`, `epochs=5`.
