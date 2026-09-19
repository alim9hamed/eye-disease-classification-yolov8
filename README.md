# 🔬 Retina & Eye Disease Classification (YOLOv8 & Deep Learning)

An image-classification project that detects eye diseases from fundus images. It trains and compares several deep-learning models (**YOLOv8-cls**, **YOLOv5-cls**, **EfficientNetB3**, **MobileNetV2**, **Xception**) and ships the best one, **YOLOv8s-cls (96.2% test accuracy)**, through a simple **Gradio** interface.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19GTMPA_UiDIw69RAfTRN4kA237hE08Nb?usp=sharing)

> ⚠️ **Medical disclaimer:** This project is for educational and research purposes only. It is **not** a diagnostic tool and must not replace examination by a licensed ophthalmologist. For any eye problem, consult a doctor.

---

## 📌 Overview

The goal is to classify a retinal image into one of **4 categories**: `normal`, `cataract`, `glaucoma`, or `retina_disease`. The notebook covers the full workflow:

1. Downloading the dataset from Roboflow
2. Data exploration and augmentation
3. Training and evaluating multiple architectures
4. Comparing results with classification reports
5. Serving the trained model with Gradio

This project is the computer-vision component of my graduation project, alongside the [Agenatic Assistant Doctor](https://github.com/alim9hamed/Agenatic_Assistant_Doctor) eye-disease chatbot.
## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | [Roboflow Universe](https://universe.roboflow.com/): `student-sqzes / eye-disease-fvopu` (version 5) |
| Format | Folder-per-class (`train/`, `valid/`, `test/`) |
| Image size | 224 × 224 |
| Classes | 4: `1_normal`, `2_cataract`, `2_glaucoma`, `3_retina_disease` |
| Total images | 6,117 |
| License | <!-- TODO: check the license on the Roboflow dataset page --> |

### Class distribution

| Class | Train | Valid | Test | Total |
|-------|------:|------:|-----:|------:|
| 1_normal | 1,270 | 161 | 191 | 1,622 |
| 2_cataract | 1,484 | 141 | 208 | 1,833 |
| 2_glaucoma | 1,320 | 146 | 202 | 1,668 |
| 3_retina_disease | 788 | 90 | 116 | 994 |
| **Total** | **4,862** | **538** | **717** | **6,117** |

The split is roughly 79% train / 9% validation / 12% test. `3_retina_disease` is the minority class (about 16% of the training set).

## 🧠 Models

| Model | Framework | Parameters | Setup |
|-------|-----------|-----------:|-------|
| **YOLOv8s-cls** | Ultralytics 8.3.115 | 5.09 M | Pretrained, 10 epochs, batch 32, img 224, AdamW (auto, lr = 0.00125) |
| **YOLOv5x-cls** | Ultralytics YOLOv5 v7.0 | 46.8 M | Pretrained, 20 epochs, batch 16, img 224 |
| **EfficientNetB3** | TensorFlow / Keras | 30.08 M | ImageNet weights, Flatten → Dense(256) → Dense(128) → Dense(4) |
| **MobileNetV2** | TensorFlow / Keras | 2.26 M | ImageNet weights, GAP → Dense(4) |
| **Xception** | TensorFlow / Keras | 20.87 M | ImageNet weights, GAP → Dense(4) |

**Keras training details:** all layers trainable, SGD (lr = 0.001, momentum = 0.9), categorical cross-entropy, up to 10 epochs, `EarlyStopping` on `val_loss` (`patience=0`) with best-weights restore.

**Augmentation (Keras):** rescale (1/255), horizontal flip, rotation (±20°), zoom (±20%). YOLOv8 uses the Ultralytics default augmentation pipeline.

**Hardware:** Google Colab, NVIDIA Tesla T4 (15 GB), Python 3.11, PyTorch 2.6.0 (CUDA 12.4).

## 📈 Results

### Model comparison

| Model | Epochs run | Test accuracy | Macro F1 | Notes |
|-------|-----------:|--------------:|---------:|-------|
| 🏆 **YOLOv8s-cls** | 10 | **96.2%** | **0.96** | Best model; top-5 accuracy 100% |
| Xception | 10 | see note ¹ | see note ¹ | Final validation accuracy 93.9% (val loss 0.205) |
| EfficientNetB3 | 3 (early stopped) | 29% | 0.11 | Did not converge; predicted almost only `cataract` |
| MobileNetV2 | 2 (early stopped) | 29% | 0.11 | Did not converge; predicted almost only `cataract` |
| YOLOv5x-cls | 20 | not comparable ² | n/a | 84.0% top-1 on an earlier dataset version |

¹ The Xception test report (26% accuracy) is inconsistent with its 93.9% validation accuracy, which points to a label/prediction ordering problem in the evaluation code (the test generator was not created with `shuffle=False`). It should be re-run before drawing conclusions.
² YOLOv5 was validated on a different dataset (`multi-retinal-disease-classifica-2`, 688 images, classes: cataract, diabetic_retinopathy, glaucoma, normal), so its numbers cannot be compared directly with the other models.

### 🏆 YOLOv8s-cls: detailed results (test set, 717 images)

| Class | Precision | Recall | F1-score | Support |
|-------|----------:|-------:|---------:|--------:|
| 1_normal | 0.95 | 0.94 | 0.94 | 191 |
| 2_cataract | 0.96 | 0.99 | 0.97 | 208 |
| 2_glaucoma | 0.98 | 0.98 | 0.98 | 202 |
| 3_retina_disease | 0.96 | 0.93 | 0.95 | 116 |
| **Accuracy** | | | **0.96** | 717 |
| Macro avg | 0.96 | 0.96 | 0.96 | 717 |
| Weighted avg | 0.96 | 0.96 | 0.96 | 717 |

| Metric | Value |
|--------|-------|
| Top-1 accuracy | 96.23% |
| Top-5 accuracy | 100% |
| Model size | 10.3 MB |
| Parameters | 5,085,860 (5,080,324 fused) |
| Compute | 12.5 GFLOPs |
| Inference speed | ≈ 0.5 ms / image (Tesla T4) |
| Training time | ≈ 4.7 minutes (0.078 h) |

### YOLOv8 training progress

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|-------|---|---|---|---|---|---|---|---|---|----|
| Train loss | 1.079 | 0.799 | 0.659 | 0.557 | 0.426 | 0.346 | 0.266 | 0.223 | 0.171 | 0.150 |
| Top-1 accuracy | 0.679 | 0.654 | 0.711 | 0.827 | 0.893 | 0.921 | 0.921 | 0.943 | 0.953 | 0.961 |

### Xception training progress (validation set, 538 images)

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|-------|---|---|---|---|---|---|---|---|---|----|
| Train acc | 0.389 | 0.612 | 0.719 | 0.763 | 0.813 | 0.841 | 0.875 | 0.904 | 0.916 | 0.949 |
| Val acc | 0.545 | 0.580 | 0.647 | 0.745 | 0.786 | 0.820 | 0.876 | 0.876 | 0.916 | 0.939 |
| Val loss | 1.086 | 0.976 | 0.878 | 0.623 | 0.527 | 0.433 | 0.345 | 0.313 | 0.250 | 0.205 |

### YOLOv5x-cls (earlier dataset version, 688 validation images)

| Class | Top-1 accuracy | Images |
|-------|---------------:|-------:|
| cataract | 0.000 | 3 |
| diabetic_retinopathy | 1.000 | 198 |
| glaucoma | 0.641 | 209 |
| normal | 0.885 | 278 |
| **All** | **0.840** | 688 |

Speed: 2.2 ms / image, 46.8 M parameters, 128.9 GFLOPs.

## 🧪 Evaluation Notes

- **YOLOv8 validation split:** Ultralytics looks for a folder named `val`, but Roboflow exports `valid`. As a result, the log shows `Dataset 'split=val' not found, using 'split=test'`, so the per-epoch accuracy and best-checkpoint selection used the **test set**. The reported 96.2% may therefore be slightly optimistic. Renaming `valid` to `val` gives a clean, independent test evaluation.
- **EfficientNetB3 and MobileNetV2** stopped after 2-3 epochs because `EarlyStopping(patience=0)` halts at the first epoch without improvement, and their validation accuracy stayed at 26.2% (the share of `cataract` in the validation set).
- **Xception** trained well (93.9% validation accuracy), but its test report needs to be regenerated with `shuffle=False` on the test generator.

## 🧰 Tech Stack

- **Python 3**
- **Ultralytics** (YOLOv8 / YOLOv5)
- **TensorFlow / Keras**
- **PyTorch**
- **scikit-learn**: metrics and reports
- **Roboflow**: dataset management
- **Gradio**: demo interface
- **OpenCV, Pillow, NumPy, Pandas, Matplotlib**

## 🚀 Getting Started

### Option 1: Google Colab (easiest)

1. Click the **Open in Colab** badge above.
2. Set `ROBOFLOW_API_KEY` in Colab **Secrets** (🔑 icon).
3. Choose a GPU runtime (`Runtime → Change runtime type → T4 GPU`).
4. Run the cells in order.

### Option 2: Run locally

```bash
git clone https://github.com/alim9hamed/<repo-name>.git
cd <repo-name>

pip install ultralytics roboflow tensorflow scikit-learn gradio opencv-python pandas matplotlib
```

### Download the dataset

```python
import os
from roboflow import Roboflow

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("student-sqzes").project("eye-disease-fvopu")
dataset = project.version(5).download("folder")
```

> Never hard-code API keys in notebooks or commit them to the repository.

### Train YOLOv8

```python
import os
from ultralytics import YOLO

# Ultralytics expects a "val" folder; Roboflow exports "valid"
if os.path.isdir(f"{dataset.location}/valid") and not os.path.isdir(f"{dataset.location}/val"):
    os.rename(f"{dataset.location}/valid", f"{dataset.location}/val")

model = YOLO("yolov8s-cls.pt")
model.train(data=dataset.location, imgsz=224, epochs=10, batch=32)
metrics = model.val(data=dataset.location, imgsz=224, split="test")
```

The best weights are saved to `runs/classify/train/weights/best.pt`.

### Run inference

```python
from ultralytics import YOLO

model = YOLO("runs/classify/train/weights/best.pt")
result = model.predict("retina_image.jpg", imgsz=224)[0]

top1 = result.probs.top1
print(model.names[top1], float(result.probs.data[top1]))
```

### Launch the Gradio demo

```python
import gradio as gr
from ultralytics import YOLO

model = YOLO("runs/classify/train/weights/best.pt")

def classify_image(img):
    probs = model.predict(img)[0].probs
    return f"Class: {model.names[probs.top1]}\nConfidence: {probs.data[probs.top1].item():.2f}"

gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload a retina image"),
    outputs=gr.Textbox(label="Prediction"),
    title="Retina Disease Classifier (YOLOv8)",
).launch()
```


<!-- Update the file names above to match your repository. -->

## ⚠️ Limitations

- The dataset comes from a public Roboflow project and may not generalize to real clinical images from other devices or populations.
- `3_retina_disease` is under-represented (16% of training data), and its recall (0.93) is the lowest of the four classes.
- The model outputs a class prediction only; it is not a substitute for clinical evaluation.
- Results come from a single train/validation/test split without cross-validation.


## 🤝 Contributing

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m "Add your feature"`
4. Push and open a Pull Request

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE). Check the dataset's own license before reuse.

---

⭐ If you find this project useful, please give it a star!
