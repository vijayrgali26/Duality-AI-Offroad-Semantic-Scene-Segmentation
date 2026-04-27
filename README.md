# 🚀 Duality AI — Offroad Semantic Scene Segmentation

### BigRock Exchange Hackathon Submission

---

## 📌 Overview

This project implements a **semantic segmentation pipeline** for off-road environments using synthetic data generated from Duality AI’s Falcon platform.

We trained deep learning models to classify **10 terrain classes** (trees, bushes, rocks, sky, etc.) to support **UGV autonomy**.

---

## 🧠 Models Used

- **DeepLabV3+ (ResNet-50 backbone)** ✅ _(Final Model)_
- U-Net (baseline)
- FCN (initial experimentation)

---

## 📂 Project Structure

```
hackathon/
├── train.py
├── test.py
├── visualize.py
├── evaluation_metrics.txt   ← ⭐ FINAL RESULTS FILE
├── README.md
├── ENV_SETUP/
├── Offroad_Segmentation_Training_Dataset/
├── Offroad_Segmentation_testImages/
└── runs/
```

---

## ⚙️ Setup & Installation

### Step 1 — Environment

**Windows**

```
cd ENV_SETUP
setup_env.bat
conda activate EDU
```

**Mac/Linux**

```
cd ENV_SETUP
bash setup_env.sh
conda activate EDU
```

---

### Step 2 — Dataset Placement

```
Offroad_Segmentation_Training_Dataset/
├── train/
├── val/

Offroad_Segmentation_testImages/
├── Color_Images/
├── Segmentation/ (optional)
```

---

## 🏋️ Training

Run:

```
python train.py \
  --data_dir Offroad_Segmentation_Training_Dataset \
  --run_dir runs/exp1 \
  --epochs 30 \
  --batch_size 4 \
  --lr 0.0001
```

### 📊 Training Results

```
Final Val Loss:     1.0450
Final Val IoU:      0.6145
Final Val Dice:     0.3598
Final Val Accuracy: 0.6294
```

### Outputs

- Model weights
- Loss curve
- IoU curve
- Training logs

---

## 🧪 Testing / Evaluation

Run:

```
python test.py \
  --weights runs/exp1/best_model.pth \
  --test_dir Offroad_Segmentation_testImages \
  --out_dir runs/test_out
```

### 📊 Test Results

```
Mean IoU: 0.6055
```

---

## 📊 Final Results File

👉 All evaluation metrics are stored in:

```
evaluation_metrics.txt
```

---

## 🎨 Visualization

```
python visualize.py \
  --rgb_dir Offroad_Segmentation_testImages \
  --pred_dir runs/test_out/predictions \
  --out_dir runs/test_out/viz_hc
```

---

## 📈 Key Performance

- ✅ Mean IoU: **0.60+**
- ⚡ Inference Speed: **< 50 ms/image (GPU)**
- 📊 Accuracy: **~63% validation accuracy**

---

## 🧩 Techniques Used

- Transfer Learning (ResNet-50 pretrained)
- Data Augmentation
- Mixed Precision Training (AMP)
- Cosine Learning Rate Scheduler
- DeepLabV3+ architecture

---

## 🔮 Future Improvements

- Domain adaptation
- Larger backbone (ResNet-101)
- Class-balanced sampling
- Self-supervised learning

---

## 🙌 Acknowledgment

Duality AI × BigRock Exchange Hackathon
Segmentation Track

---
