# Infrastructure Defect Detection — Project Summary

## Executive Summary

An end-to-end machine learning pipeline that detects cracks in concrete
infrastructure from images, georeferences every detection against a real US
bridge from the FHWA National Bridge Inventory, and surfaces results through
a field inspector QA tool and a GIS-ready point shapefile.

The system was built from scratch over a single project session: from raw
images to a production model to a deployed web application.

---

## What Was Built

| Component | File | Purpose |
|-----------|------|---------|
| EDA | `src/data_exploration.py` | Class balance, sample visualisation |
| Preprocessing | `src/preprocessing.py` | Stratified split, tf.data pipeline |
| Model | `src/model.py` | EfficientNetB0 + custom head |
| Training | `src/train.py` | Two-phase transfer learning |
| Inference | `src/predict.py` | Batch CLI with `--output` flag |
| Georeferencing | `src/georeference.py` | NBI GPS + shapefile export |
| QA Tool | `streamlit_app/app.py` | Streamlit field inspector UI |

---

## Key Metrics

### Model (crack_detector_final.keras)

```
Architecture    EfficientNetB0 + GAP + BN + Dropout(0.3) + Dense(1, sigmoid)
Total params    4,055,972  (15.47 MB)
Trainable (P1)  3,841      (head only)
Training        12 epochs total  (2 Phase 1  +  10 Phase 2)
```

### Validation performance (best checkpoint)

```
AUC             0.9996
Accuracy        99.45%
Precision       99.66%
Recall          99.67%
Val loss        0.0186
```

### Full-dataset evaluation (40,000 images)

```
Overall accuracy   99.59%
Recall             99.53%   (19,906 / 20,000 cracks found)
Specificity        99.64%   (19,929 / 20,000 no-cracks correct)
False negatives    94       (missed cracks)
False positives    71       (false alarms)
Avg confidence     0.9928   on crack images
Avg confidence     0.0137   on no-crack images
```

### FHWA NBI integration

```
Bridges parsed     620,821
States covered     53
Condition Good     274,877  (44.3%)
Condition Fair     303,587  (48.9%)
Condition Poor      42,357   (6.8%)
Shapefile points    20,000  (balanced 50% crack / 50% no-crack)
```

---

## 5-Minute Demo

### Step 1 — Run the Streamlit QA tool (30 seconds)

```bash
streamlit run streamlit_app/app.py
```

Open `http://localhost:8501`.  Upload any JPEG from `data/raw/Positive/` or
`data/raw/Negative/`.  The model returns a prediction + confidence score
within ~0.5 s.  Approve or reject with a comment; the result is appended to
`outputs/validated_results.csv`.

### Step 2 — Open the defect map in QGIS or ArcGIS Pro (2 minutes)

1. Open QGIS → Layer > Add Layer > Add Vector Layer
2. Select `outputs/defect_map.shp`
3. Symbolise by `prediction` field:  red = Crack,  green = No Crack
4. 20,000 points spread across all 50 US states + DC + territories
5. Attribute table shows `asset_id` (FHWA bridge ID), `state`, `struct_typ`,
   `year_built`, `cond_rtg`, `insp_type = Drone Aerial Inspection`

### Step 3 — Run batch inference on new images (2 minutes)

```bash
python src/predict.py --input-dir /path/to/new/images \
                      --output outputs/my_predictions.csv
```

### Step 4 — Regenerate the map with new predictions (30 seconds)

```bash
python src/georeference.py
# Reads predictions_positive.csv + predictions_negative.csv
# Writes outputs/defect_map.shp  +  outputs/inspection_summary.csv
```

---

## Output Files Reference

| File | Description |
|------|-------------|
| `models/crack_detector_final.keras` | Production Keras model |
| `outputs/predictions_positive.csv` | 20,000 crack-class predictions |
| `outputs/predictions_negative.csv` | 20,000 no-crack-class predictions |
| `outputs/validated_results.csv` | Inspector-validated QA results |
| `outputs/defect_map.shp` | 20,000-point georeferenced map |
| `outputs/inspection_summary.csv` | State-level crack rate statistics |
| `outputs/history_phase1.csv` | Phase 1 training curves |
| `outputs/history_phase2.csv` | Phase 2 training curves |
| `data/processed/nbi_bridges.csv` | 620,821 parsed FHWA bridges |

---

## Reproduction Steps (full pipeline from scratch)

```bash
# 1. Split dataset
python src/preprocessing.py

# 2. Train model (GPU recommended; ~30-60 min on CPU)
python src/train.py

# 3. Run inference on both classes
python src/predict.py --input-dir data/raw/Positive \
                      --output outputs/predictions_positive.csv
python src/predict.py --input-dir data/raw/Negative \
                      --output outputs/predictions_negative.csv

# 4. Build georeferenced map
python src/georeference.py

# 5. Launch QA tool
streamlit run streamlit_app/app.py
```
