# Infrastructure Defect Detection

**Binary crack classification for civil infrastructure using deep learning,
georeferenced against 620,821 real US bridges from the FHWA National Bridge Inventory.**

## 🌐 Live Dashboard
[![ArcGIS Dashboard](https://img.shields.io/badge/ArcGIS-Live%20Dashboard-0079C1?style=for-the-badge&logo=arcgis&logoColor=white)](https://rohith.maps.arcgis.com/apps/dashboards/5e9a06121d9e4f5ba7265cc7b1daff2a)

**Live Demo:** https://rohith.maps.arcgis.com/apps/dashboards/5e9a06121d9e4f5ba7265cc7b1daff2a

---

## Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │              RAW DATA (40,000 images)           │
                    │   data/raw/Positive/  ·  data/raw/Negative/     │
                    └───────────────────┬─────────────────────────────┘
                                        │
                              src/preprocessing.py
                          Stratified 70 / 15 / 15 split
                                        │
                    ┌───────────────────▼─────────────────────────────┐
                    │         data/processed/{train,val,test}.csv     │
                    └───────────────────┬─────────────────────────────┘
                                        │
                               src/train.py
                     ┌──────────────────┴──────────────────┐
              Phase 1 (2 epochs)                  Phase 2 (10 epochs)
           Backbone frozen, LR=1e-3          Top-20 layers unfrozen, LR=1e-5
                     └──────────────────┬──────────────────┘
                                        │
                    ┌───────────────────▼─────────────────────────────┐
                    │    models/crack_detector_final.keras             │
                    │    EfficientNetB0 · 4.06M params · 15.47 MB     │
                    └──────┬────────────────────────────┬─────────────┘
                           │                            │
               src/predict.py                  streamlit_app/app.py
           Batch inference (CSV)              Field inspector QA tool
                           │
               src/georeference.py
         Assign real FHWA bridge GPS coords
           EXIF fallback · EPSG:4326
                           │
                    ┌──────▼──────────────────────────────────────────┐
                    │  outputs/defect_map.shp  ·  20,000 point layer  │
                    │  outputs/inspection_summary.csv  (52 states)    │
                    └─────────────────────────────────────────────────┘
```

---

## Model Performance

Evaluated against the full 40,000-image dataset (not used during training):

| Metric | Value |
|--------|-------|
| **Overall accuracy** | **99.59%** |
| **Recall (crack detection rate)** | **99.53%** |
| **Specificity (no-crack rate)** | **99.64%** |
| Val AUC (best, Phase 2) | 0.9996 |
| Val precision (best, Phase 2) | 99.66% |
| Val loss (best, Phase 2) | 0.0186 |
| False negatives (missed cracks) | 94 / 20,000 |
| False positives (false alarms) | 71 / 20,000 |

Training converged in **12 total epochs** (2 Phase 1 + 10 Phase 2).

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep learning | TensorFlow 2.20 / Keras |
| Backbone | EfficientNetB0 (ImageNet weights) |
| Image processing | OpenCV 4.13, Pillow 11.3 |
| Data wrangling | NumPy 2.2, Pandas 2.3 |
| Geospatial | GeoPandas 1.1, Shapely 2.1, PyProj 3.7 |
| ML utilities | scikit-learn 1.8 |
| Visualisation | Matplotlib 3.9 |
| Web app | Streamlit 1.54 |
| GIS integration | ArcGIS Pro / QGIS (via .shp) |

---

## Dataset

- **Source:** [Concrete Crack Images for Classification](https://www.kaggle.com/datasets/arnavr10880/concrete-crack-images-for-classification) (Kaggle)
- **Size:** 40,000 JPEG images — 20,000 Positive (crack) + 20,000 Negative (no crack)
- **Resolution:** 227 × 227 px (pre-resized to model input size)
- **Split:** 70% train / 15% val / 15% test (stratified)
- **Class balance:** Perfectly balanced (1:1 crack : no-crack)

---

## FHWA National Bridge Inventory (NBI) Integration

Real GPS coordinates are sourced from the 2023 FHWA NBI public dataset:

| Stat | Value |
|------|-------|
| Source | FHWA 2023 All-States Delimited ASCII |
| Total records parsed | 621,581 |
| Valid bridges (with coordinates) | **620,821** |
| States / territories | 53 |
| Lat range | 18.0° N -- 67.4° N |
| Lon range | 162.7° W -- 65.7° W |
| Condition: Good | 274,877 (44.3%) |
| Condition: Fair | 303,587 (48.9%) |
| Condition: Poor | 42,357 (6.8%) |

Coordinates are stored in NBI DDMMSSHS integer format and converted to
decimal degrees.  `SUFFICIENCY_RATING` was removed in the 2023 format;
`BRIDGE_CONDITION` (G/F/P) is used instead.

---

## Project Structure

```
infrastructure-defect-detection/
├── data/
│   ├── raw/
│   │   ├── Positive/          # 20,000 crack images   (git-ignored)
│   │   └── Negative/          # 20,000 no-crack images (git-ignored)
│   ├── processed/
│   │   ├── train.csv          # 70% split (stratified)
│   │   ├── val.csv            # 15% split
│   │   ├── test.csv           # 15% split
│   │   └── nbi_bridges.csv    # 620,821 parsed FHWA bridges
│   └── patches/               # reserved for patch-based approach
│
├── models/
│   ├── best_phase1.keras      # best checkpoint Phase 1  (git-ignored)
│   ├── best_phase2.keras      # best checkpoint Phase 2  (git-ignored)
│   └── crack_detector_final.keras  # production model   (git-ignored)
│
├── outputs/
│   ├── defect_map.shp         # georeferenced prediction map
│   ├── inspection_summary.csv # state-level crack statistics
│   ├── predictions_positive.csv
│   ├── predictions_negative.csv
│   ├── validated_results.csv  # inspector-validated QA results
│   ├── history_phase1.csv     # training curves Phase 1
│   ├── history_phase2.csv     # training curves Phase 2
│   ├── model_summary.txt
│   └── *.png                  # training plots & sample grids
│
├── src/
│   ├── data_exploration.py    # EDA: class balance, sample grid
│   ├── preprocessing.py       # split CSVs + tf.data pipeline
│   ├── model.py               # EfficientNetB0 builder
│   ├── train.py               # two-phase training loop
│   ├── predict.py             # batch inference CLI
│   └── georeference.py        # NBI GPS assignment + shapefile export
│
├── streamlit_app/
│   └── app.py                 # field inspector QA web tool
│
├── arcgis/                    # reserved for ArcGIS Pro integration
├── notebooks/                 # reserved for Jupyter experimentation
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and enter the project

```bash
git clone <repo-url>
cd infrastructure-defect-detection
```

### 2. Create conda environment

```bash
conda create -n crack-detect python=3.11 -y
conda activate crack-detect
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download dataset

Download the Concrete Crack Images dataset from Kaggle and place images in:

```
data/raw/Positive/   # crack images
data/raw/Negative/   # no-crack images
```

### 5. Download FHWA NBI data (for georeferencing)

```bash
# Runs automatically inside georeference.py on first run
# Or manually:
curl -L -o data/raw/2023hwybronefiledel.zip \
  https://www.fhwa.dot.gov/bridge/nbi/2023hwybronefiledel.zip
cd data/raw && unzip 2023hwybronefiledel.zip
```

---

## Running Each Component

### Data Exploration

```bash
python src/data_exploration.py
# Outputs: outputs/sample_images.png, class balance stats
```

### Preprocessing — build train/val/test splits

```bash
python src/preprocessing.py
# Outputs: data/processed/train.csv, val.csv, test.csv
#          outputs/augmented_samples.png
```

### Model architecture preview

```bash
python src/model.py
# Outputs: outputs/model_summary.txt
```

### Training (two-phase)

```bash
python src/train.py
# Phase 1 — backbone frozen,      LR=1e-3, up to 10 epochs
# Phase 2 — top-20 layers thawed, LR=1e-5, up to 10 epochs
# Outputs: models/crack_detector_final.keras
#          outputs/history_phase{1,2}.csv
#          outputs/history_{loss,accuracy,auc}.png
```

### Batch Prediction

```bash
# Single class
python src/predict.py --input-dir data/raw/Positive \
                      --output outputs/predictions_positive.csv

python src/predict.py --input-dir data/raw/Negative \
                      --output outputs/predictions_negative.csv

# Custom threshold
python src/predict.py --input-dir /path/to/images --threshold 0.6
```

### Georeferencing — map predictions to real bridges

```bash
python src/georeference.py
# Requires: outputs/predictions_positive.csv
#           outputs/predictions_negative.csv
#           data/processed/nbi_bridges.csv
# Outputs: outputs/defect_map.shp   (EPSG:4326)
#          outputs/inspection_summary.csv
```

### Streamlit QA Tool

```bash
streamlit run streamlit_app/app.py
# Opens at http://localhost:8501
```

---

## Screenshots

> _Add screenshots of the Streamlit QA tool and ArcGIS map here._

| Streamlit upload + prediction | Sidebar dashboard | ArcGIS defect map |
|---|---|---|
| _(screenshot)_ | _(screenshot)_ | _(screenshot)_ |

---

## Future Enhancements

| Enhancement | Description |
|-------------|-------------|
| **Mapillary integration** | Pull street-level and drone imagery directly from the Mapillary API using bridge GPS coordinates from NBI, replacing simulated assignments with real inspection photos |
| **Real drone imagery** | Replace the surface-crack dataset with actual UAV/drone bridge inspection footage; retrain on bridge-specific defect taxonomy (spalling, delamination, rebar exposure) |
| **Real-time monitoring** | Stream predictions from an edge device (Jetson Nano / RPi + camera) over MQTT; live update the defect map in ArcGIS |
| **Multi-class defects** | Extend from binary crack/no-crack to 5-class classification: crack, spalling, corrosion, joint failure, surface wear |
| **Segmentation model** | Add a U-Net segmentation head to localise crack pixels and compute crack width/length metrics |
| **ArcGIS Pro layer** | Build a full ArcGIS Pro layer package (.lpkx) with symbology, pop-ups, and field-calculated severity scores from the NBI condition rating |
| **Inspector mobile app** | Convert the Streamlit tool to a Progressive Web App optimised for phones and tablets in the field |
| **Automated reporting** | Generate PDF inspection reports per bridge from validated results, keyed to FHWA asset IDs |
