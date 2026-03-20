# Model Comparison Report

**Timestamp:** 20260125_134327  
**Validation Parameters:** conf=0.25, iou=0.5  
**Dataset:** data_420

---

## 🏆 Overall Ranking (by F1-Score - Mask)

| Rank | Model | Experiment | F1 | Precision | Recall | mAP50 | mAP50-95 |
|------|-------|------------|-----|-----------|--------|-------|----------|
| 1 | yolo11n-seg | strong_geom | **0.985** | 0.971 | 1.000 | 0.995 | **0.995** |
| 2 | yolo26n-seg | strong_geom | **0.985** | 0.970 | 1.000 | 0.983 | 0.960 |
| 3 | yolo26n-seg | moderate_geom | **0.985** | 0.970 | 1.000 | 0.995 | 0.987 |
| 4 | yolo26n-seg | baseline | 0.984 | 0.969 | 1.000 | 0.995 | 0.974 |
| 5 | yolo11n-seg | baseline | 0.977 | 0.955 | 1.000 | 0.981 | 0.981 |
| 6 | yolo26n-seg | no_augmentation | 0.977 | 0.955 | 1.000 | 0.980 | 0.946 |
| 7 | yolo11n-seg | moderate_geom | 0.976 | 0.954 | 1.000 | 0.995 | 0.993 |
| 8 | yolo11n-seg | no_augmentation | 0.974 | 0.949 | 1.000 | 0.978 | 0.972 |

---

## 📈 Comparison by Experiment

### No Augmentation

| Model | F1 | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----|-----------|--------|-------|----------|
| **yolo26n-seg** | **0.977** | 0.955 | 1.000 | 0.980 | 0.946 |
| yolo11n-seg | 0.974 | 0.949 | 1.000 | 0.978 | 0.972 |

**Winner:** yolo26n-seg (F1: 0.977)

---

### Baseline

| Model | F1 | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----|-----------|--------|-------|----------|
| **yolo26n-seg** | **0.984** | 0.969 | 1.000 | 0.995 | 0.974 |
| yolo11n-seg | 0.977 | 0.955 | 1.000 | 0.981 | 0.981 |

**Winner:** yolo26n-seg (F1: 0.984)

---

### Moderate Geometric Augmentation

| Model | F1 | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----|-----------|--------|-------|----------|
| **yolo26n-seg** | **0.985** | 0.970 | 1.000 | 0.995 | 0.987 |
| yolo11n-seg | 0.976 | 0.954 | 1.000 | 0.995 | 0.993 |

**Winner:** yolo26n-seg (F1: 0.985)

---

### Strong Geometric Augmentation

| Model | F1 | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----|-----------|--------|-------|----------|
| **yolo11n-seg** | **0.985** | **0.971** | 1.000 | 0.995 | **0.995** |
| yolo26n-seg | 0.985 | 0.970 | 1.000 | 0.983 | 0.960 |

**Winner:** yolo11n-seg (F1: 0.985, better mAP50-95: 0.995 vs 0.960)

---

## 🤖 Comparison by Model

### YOLO11n-seg

| Experiment | F1 | Precision | Recall | mAP50 | mAP50-95 |
|------------|-----|-----------|--------|-------|----------|
| **strong_geom** | **0.985** | **0.971** | 1.000 | 0.995 | **0.995** |
| baseline | 0.977 | 0.955 | 1.000 | 0.981 | 0.981 |
| moderate_geom | 0.976 | 0.954 | 1.000 | 0.995 | 0.993 |
| no_augmentation | 0.974 | 0.949 | 1.000 | 0.978 | 0.972 |

**Best Configuration:** strong_geom  
**Key Insight:** Strong geometric augmentation provides the best overall performance with highest mAP50-95 (0.995)

---

### YOLO26n-seg

| Experiment | F1 | Precision | Recall | mAP50 | mAP50-95 |
|------------|-----|-----------|--------|-------|----------|
| **strong_geom** | **0.985** | 0.970 | 1.000 | 0.983 | 0.960 |
| **moderate_geom** | **0.985** | 0.970 | 1.000 | 0.995 | 0.987 |
| baseline | 0.984 | 0.969 | 1.000 | 0.995 | 0.974 |
| no_augmentation | 0.977 | 0.955 | 1.000 | 0.980 | 0.946 |

**Best Configuration:** moderate_geom (better mAP50-95: 0.987 vs 0.960)  
**Key Insight:** Moderate geometric augmentation provides best balance for YOLO26n-seg

---

## 🏆 Best Model Overall

**Model:** yolo11n-seg  
**Experiment:** strong_geom  
**Model Path:** `runs/segment/yolo11n-seg/20260125_134327/strong_geom/weights/best.pt`

### Metrics

| Metric | Value |
|--------|-------|
| **F1-Score (Mask)** | **0.985** |
| **Precision (Mask)** | **0.971** |
| **Recall (Mask)** | **1.000** |
| **mAP50 (Mask)** | **0.995** |
| **mAP50-95 (Mask)** | **0.995** |

---

## 📊 Detailed Metrics (All Models)

### Mask Metrics (Segmentation)

| Model | Experiment | Precision | Recall | F1 | mAP50 | mAP50-95 |
|-------|------------|-----------|--------|-----|-------|----------|
| yolo11n-seg | no_augmentation | 0.949 | 1.000 | 0.974 | 0.978 | 0.972 |
| yolo11n-seg | baseline | 0.955 | 1.000 | 0.977 | 0.981 | 0.981 |
| yolo11n-seg | moderate_geom | 0.954 | 1.000 | 0.976 | 0.995 | 0.993 |
| yolo11n-seg | strong_geom | **0.971** | 1.000 | **0.985** | 0.995 | **0.995** |
| yolo26n-seg | no_augmentation | 0.955 | 1.000 | 0.977 | 0.980 | 0.946 |
| yolo26n-seg | baseline | 0.969 | 1.000 | 0.984 | 0.995 | 0.974 |
| yolo26n-seg | moderate_geom | 0.970 | 1.000 | 0.985 | 0.995 | 0.987 |
| yolo26n-seg | strong_geom | 0.970 | 1.000 | 0.985 | 0.983 | 0.960 |

### Box Metrics (Detection)

| Model | Experiment | Precision | Recall | F1 | mAP50 | mAP50-95 |
|-------|------------|-----------|--------|-----|-------|----------|
| yolo11n-seg | no_augmentation | 0.949 | 1.000 | 0.974 | 0.978 | 0.972 |
| yolo11n-seg | baseline | 0.955 | 1.000 | 0.977 | 0.981 | 0.981 |
| yolo11n-seg | moderate_geom | 0.954 | 1.000 | 0.976 | 0.995 | 0.993 |
| yolo11n-seg | strong_geom | 0.971 | 1.000 | 0.985 | 0.995 | 0.995 |
| yolo26n-seg | no_augmentation | 0.955 | 1.000 | 0.977 | 0.980 | 0.946 |
| yolo26n-seg | baseline | 0.969 | 1.000 | 0.984 | 0.995 | 0.974 |
| yolo26n-seg | moderate_geom | 0.970 | 1.000 | 0.985 | 0.995 | 0.987 |
| yolo26n-seg | strong_geom | 0.970 | 1.000 | 0.985 | 0.983 | 0.960 |

---

## 💡 Key Insights

1. **All models achieve perfect recall (1.000)** - No false negatives in the validation set
2. **Strong geometric augmentation** performs best for yolo11n-seg (F1: 0.985, mAP50-95: 0.995)
3. **Moderate geometric augmentation** performs best for yolo26n-seg (F1: 0.985, mAP50-95: 0.987)
4. **YOLO11n-seg with strong_geom** is the overall winner with the highest mAP50-95 (0.995)
5. **Augmentation improves performance** - All augmented models outperform no_augmentation
6. **Geometric augmentation** (moderate/strong) generally provides better results than baseline

---

## 📝 Notes

- **Mask metrics** are more important for segmentation tasks than box metrics
- **mAP50-95** is a stricter metric than mAP50, averaging over IoU thresholds from 0.5 to 0.95
- **F1-Score** balances precision and recall (both are important)
- All models show excellent performance with F1-scores above 0.974

---

*Generated from: model_comparison_20260125_134327.csv*
