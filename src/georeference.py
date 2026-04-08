"""
Drone inspection georeferencing pipeline.

Simulated scenario
------------------
Each image represents a geotagged drone photo taken during a real US bridge
inspection.  Predictions from both image classes (Positive = crack, Negative =
no crack) are sampled equally so the map shows ~50 % red (crack) and ~50 %
green (no crack) spread across real FHWA bridge locations nationwide.

Inputs
------
  outputs/predictions_positive.csv   — crack class predictions  (20 k rows)
  outputs/predictions_negative.csv   — no-crack class predictions (20 k rows)
  data/processed/nbi_bridges.csv     — parsed FHWA NBI bridges (~621 k rows)

Outputs
-------
  outputs/defect_map.shp             — point shapefile, one point per image
  outputs/inspection_summary.csv     — state-level crack statistics

GPS coordinate priority
-----------------------
1. EXIF GPS tags embedded in the JPEG (extracted with Pillow).
2. FHWA bridge coordinates from NBI (fallback for all images without EXIF).

Dataset images from the public SDNET / surface-crack collections typically
carry no GPS EXIF, so FHWA coordinates will be used for the full run.

Shapefile field names are capped at 10 chars (.dbf format); full names are
kept in predictions CSVs and inspection_summary.csv.

Usage
-----
    python src/georeference.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR    = Path(r"C:\Projects\infrastructure-defect-detection")
RAW_DIR        = PROJECT_DIR / "data" / "raw"
NBI_CSV        = PROJECT_DIR / "data" / "processed" / "nbi_bridges.csv"
PRED_POS       = PROJECT_DIR / "outputs" / "predictions_positive.csv"
PRED_NEG       = PROJECT_DIR / "outputs" / "predictions_negative.csv"
OUTPUT_SHP        = PROJECT_DIR / "outputs" / "defect_map.shp"
OUTPUT_SHP_ONLINE = PROJECT_DIR / "outputs" / "defect_map_online.shp"
OUTPUT_SUMMARY    = PROJECT_DIR / "outputs" / "inspection_summary.csv"

SAMPLE_PER_CLASS = 10_000
SEED             = 42
INSPECTION_TYPE  = "Drone Aerial Inspection"

# FHWA state code (integer) -> (abbreviation, full name)
STATE_NAMES = {
    1: ("AL", "Alabama"),         2: ("AK", "Alaska"),
    4: ("AZ", "Arizona"),         5: ("AR", "Arkansas"),
    6: ("CA", "California"),      8: ("CO", "Colorado"),
    9: ("CT", "Connecticut"),    10: ("DE", "Delaware"),
   11: ("DC", "Dist. of Columbia"), 12: ("FL", "Florida"),
   13: ("GA", "Georgia"),        15: ("HI", "Hawaii"),
   16: ("ID", "Idaho"),          17: ("IL", "Illinois"),
   18: ("IN", "Indiana"),        19: ("IA", "Iowa"),
   20: ("KS", "Kansas"),         21: ("KY", "Kentucky"),
   22: ("LA", "Louisiana"),      23: ("ME", "Maine"),
   24: ("MD", "Maryland"),       25: ("MA", "Massachusetts"),
   26: ("MI", "Michigan"),       27: ("MN", "Minnesota"),
   28: ("MS", "Mississippi"),    29: ("MO", "Missouri"),
   30: ("MT", "Montana"),        31: ("NE", "Nebraska"),
   32: ("NV", "Nevada"),         33: ("NH", "New Hampshire"),
   34: ("NJ", "New Jersey"),     35: ("NM", "New Mexico"),
   36: ("NY", "New York"),       37: ("NC", "North Carolina"),
   38: ("ND", "North Dakota"),   39: ("OH", "Ohio"),
   40: ("OK", "Oklahoma"),       41: ("OR", "Oregon"),
   42: ("PA", "Pennsylvania"),   44: ("RI", "Rhode Island"),
   45: ("SC", "South Carolina"), 46: ("SD", "South Dakota"),
   47: ("TN", "Tennessee"),      48: ("TX", "Texas"),
   49: ("UT", "Utah"),           50: ("VT", "Vermont"),
   51: ("VA", "Virginia"),       53: ("WA", "Washington"),
   54: ("WV", "West Virginia"),  55: ("WI", "Wisconsin"),
   56: ("WY", "Wyoming"),
   60: ("AS", "American Samoa"), 66: ("GU", "Guam"),
   69: ("MP", "N. Mariana Islands"), 72: ("PR", "Puerto Rico"),
   78: ("VI", "Virgin Islands"),
}

# Shapefile .dbf column name limit: 10 chars
SHP_RENAME = {
    "inspection_type":  "insp_type",   # 9
    "image_source":     "img_source",  # 10
    "structure_type":   "struct_typ",  # 10
    "condition_rating": "cond_rtg",    # 8
    "gps_source":       "gps_source",  # 10
    "asset_id":         "asset_id",    # 8
}


# ── EXIF GPS extraction ───────────────────────────────────────────────────────
_GPS_IFD_TAG = 34853   # Exif tag pointing to the GPS sub-IFD


def _dms_to_decimal(dms_tuple, ref: str) -> float:
    """Convert EXIF (degrees, minutes, seconds) IFDRational tuple to decimal."""
    d, m, s = (float(x) for x in dms_tuple)
    dec = d + m / 60.0 + s / 3600.0
    return -dec if ref in ("S", "W") else dec


def extract_exif_gps(image_path: Path):
    """
    Return (latitude, longitude) from JPEG EXIF GPS tags, or (None, None).

    Uses Pillow's _getexif(); reads only the image header so it is fast even
    for large files.  Any failure — missing tags, corrupt EXIF, IO error —
    silently returns (None, None).
    """
    try:
        with Image.open(image_path) as img:
            exif = img._getexif()
        if exif is None:
            return None, None
        gps = exif.get(_GPS_IFD_TAG)
        if not gps:
            return None, None
        lat = _dms_to_decimal(gps[2], gps.get(1, "N"))
        lon = _dms_to_decimal(gps[4], gps.get(3, "E"))
        return round(lat, 6), round(lon, 6)
    except Exception:
        return None, None


# ── Helpers ───────────────────────────────────────────────────────────────────
def _check_inputs() -> None:
    missing = [p for p in (PRED_POS, PRED_NEG, NBI_CSV) if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: required file not found: {p}")
        print("\nRun the following first:")
        if not PRED_POS.exists():
            print("  python src/predict.py --input-dir data/raw/Positive "
                  "--output outputs/predictions_positive.csv")
        if not PRED_NEG.exists():
            print("  python src/predict.py --input-dir data/raw/Negative "
                  "--output outputs/predictions_negative.csv")
        if not NBI_CSV.exists():
            print("  python src/georeference.py  (first run parses NBI automatically)")
        sys.exit(1)


def _load_nbi() -> pd.DataFrame:
    return pd.read_csv(
        NBI_CSV,
        dtype={"bridge_id": str, "state": str, "county": str, "place_code": str},
    )


def _build_summary(combined: pd.DataFrame) -> pd.DataFrame:
    """Build state-level crack statistics table."""
    agg = (
        combined
        .groupby("state_code")
        .agg(
            total_images   = ("label", "count"),
            crack_count    = ("label", "sum"),
            avg_confidence = ("confidence", "mean"),
        )
        .reset_index()
    )
    agg["no_crack_count"] = agg["total_images"] - agg["crack_count"]
    agg["crack_pct"]      = (agg["crack_count"] / agg["total_images"] * 100).round(1)
    agg["avg_confidence"] = agg["avg_confidence"].round(4)

    # Bridge condition breakdown per state
    cond = (
        combined
        .groupby(["state_code", "condition_rating"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    cond.columns.name = None
    for col, alias in (("G", "cond_good"), ("F", "cond_fair"), ("P", "cond_poor")):
        cond[alias] = cond.get(col, 0)
    cond = cond[["state_code", "cond_good", "cond_fair", "cond_poor"]]

    summary = agg.merge(cond, on="state_code", how="left")

    # Add human-readable state name
    summary["state_code"] = pd.to_numeric(summary["state_code"], errors="coerce")
    summary["state_abbr"] = summary["state_code"].map(
        lambda c: STATE_NAMES.get(int(c), ("??", "Unknown"))[0]
        if pd.notna(c) else "??"
    )
    summary["state_name"] = summary["state_code"].map(
        lambda c: STATE_NAMES.get(int(c), ("??", "Unknown"))[1]
        if pd.notna(c) else "Unknown"
    )

    return summary.sort_values("crack_count", ascending=False).reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    _check_inputs()

    # ── 1. Load and sample predictions ───────────────────────────────────────
    print("Loading predictions ...")
    pos_df = pd.read_csv(PRED_POS)
    neg_df = pd.read_csv(PRED_NEG)

    pos_sample = pos_df.sample(
        n=min(SAMPLE_PER_CLASS, len(pos_df)), random_state=SEED
    ).copy()
    neg_sample = neg_df.sample(
        n=min(SAMPLE_PER_CLASS, len(neg_df)), random_state=SEED
    ).copy()

    pos_sample["class_dir"] = "Positive"
    neg_sample["class_dir"] = "Negative"

    combined = pd.concat([pos_sample, neg_sample], ignore_index=True)
    n = len(combined)

    crack_total = int(combined["label"].sum())
    print(f"  Positive sample : {len(pos_sample):,}  "
          f"| crack: {int(pos_sample['label'].sum()):,} "
          f"({pos_sample['label'].mean()*100:.1f} %)")
    print(f"  Negative sample : {len(neg_sample):,}  "
          f"| crack: {int(neg_sample['label'].sum()):,} "
          f"({neg_sample['label'].mean()*100:.1f} %)")
    print(f"  Combined total  : {n:,}  "
          f"| crack: {crack_total:,} "
          f"({crack_total / n * 100:.1f} %)")

    # ── 2. Load NBI bridges ───────────────────────────────────────────────────
    print(f"\nLoading NBI bridge data from {NBI_CSV} ...")
    nbi = _load_nbi()
    print(f"  {len(nbi):,} bridges available.")

    # ── 3. Assign one unique bridge per image ─────────────────────────────────
    rng     = np.random.default_rng(SEED)
    replace = n > len(nbi)
    indices = rng.choice(len(nbi), size=n, replace=replace)
    bridges = nbi.iloc[indices].reset_index(drop=True)
    print(f"  Assigned {n:,} images to bridges "
          f"({'with' if replace else 'without'} replacement).")

    # ── 4. EXIF GPS check (falls back to FHWA for every image without tags) ──
    print(f"\nChecking EXIF GPS in {n:,} images ...")
    exif_lats:  list = []
    exif_lons:  list = []
    exif_valid: list = []

    for idx, row in enumerate(combined.itertuples(index=False), 1):
        img_path = RAW_DIR / row.class_dir / row.filename
        lat, lon = extract_exif_gps(img_path)
        exif_lats.append(lat)
        exif_lons.append(lon)
        exif_valid.append(lat is not None)
        if idx % 2000 == 0:
            print(f"  Checked {idx:,} / {n:,} ...", end="\r")

    exif_found = sum(exif_valid)
    print(f"  EXIF GPS found  : {exif_found:,} / {n:,}  "
          f"({'none — using FHWA coords for all' if exif_found == 0 else str(exif_found) + ' images'})")

    # ── 5. Resolve final coordinates (EXIF > FHWA) ───────────────────────────
    nbi_lats = bridges["latitude"].tolist()
    nbi_lons = bridges["longitude"].tolist()

    combined["latitude"]   = [
        el if v else nb for el, v, nb in zip(exif_lats, exif_valid, nbi_lats)
    ]
    combined["longitude"]  = [
        el if v else nb for el, v, nb in zip(exif_lons, exif_valid, nbi_lons)
    ]
    combined["gps_source"] = ["EXIF" if v else "FHWA" for v in exif_valid]

    # ── 6. Attach bridge attribute columns ───────────────────────────────────
    combined["asset_id"]         = bridges["bridge_id"].values
    combined["state_code"]       = bridges["state"].values
    combined["state_abbr"]       = (
        pd.to_numeric(combined["state_code"], errors="coerce")
        .map(lambda c: STATE_NAMES.get(int(c), ("??", "Unknown"))[0]
             if pd.notna(c) else "??")
    )
    combined["state_name"]       = (
        pd.to_numeric(combined["state_code"], errors="coerce")
        .map(lambda c: STATE_NAMES.get(int(c), ("??", "Unknown"))[1]
             if pd.notna(c) else "Unknown")
    )
    combined["county"]           = bridges["county"].values
    combined["place_code"]       = bridges["place_code"].values
    combined["structure_type"]   = bridges["structure_type"].values
    combined["year_built"]       = bridges["year_built"].values
    combined["condition_rating"] = bridges["condition_rating"].values

    # ── 7. Inspection metadata ────────────────────────────────────────────────
    combined["inspection_type"] = INSPECTION_TYPE
    combined["image_source"]    = combined["filename"]

    # ── 8. Build GeoDataFrame ─────────────────────────────────────────────────
    geometry = [
        Point(lon, lat)
        for lon, lat in zip(combined["longitude"], combined["latitude"])
    ]

    out_cols = [
        "filename", "prediction", "confidence", "label",
        "latitude", "longitude", "gps_source",
        "asset_id", "state_code", "state_abbr", "state_name",
        "county", "place_code",
        "structure_type", "year_built", "condition_rating",
        "inspection_type", "image_source",
    ]
    gdf = gpd.GeoDataFrame(combined[out_cols], geometry=geometry, crs="EPSG:4326")
    gdf = gdf.rename(columns={**SHP_RENAME, "state_code": "state"})

    # ── 9a. Save WGS-84 shapefile ─────────────────────────────────────────────
    OUTPUT_SHP.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(OUTPUT_SHP))
    fields = [c for c in gdf.columns if c != "geometry"]
    print(f"\nShapefile saved -> {OUTPUT_SHP}")
    print(f"  Features  : {len(gdf):,}")
    print(f"  CRS       : EPSG:4326 (WGS-84)")
    print(f"  Fields    : {', '.join(fields)}")

    # ── 9b. Save Web Mercator shapefile for ArcGIS Online ────────────────────
    gdf_online = gdf.to_crs("EPSG:3857")
    gdf_online.to_file(str(OUTPUT_SHP_ONLINE))
    print(f"\nOnline shapefile saved -> {OUTPUT_SHP_ONLINE}")
    print(f"  Features  : {len(gdf_online):,}")
    print(f"  CRS       : EPSG:3857 (WGS 1984 Web Mercator)")

    # ── 10. State-level inspection summary ────────────────────────────────────
    summary = _build_summary(combined)
    summary.to_csv(OUTPUT_SUMMARY, index=False)
    print(f"\nInspection summary saved -> {OUTPUT_SUMMARY}")

    # ── 11. Console report ────────────────────────────────────────────────────
    total    = len(combined)
    cracks   = int(combined["label"].sum())
    no_crack = total - cracks
    b        = gdf.total_bounds

    print()
    print("-" * 54)
    print("  DRONE INSPECTION SUMMARY")
    print("-" * 54)
    print(f"  Total images inspected  : {total:,}")
    print(f"  Crack detections        : {cracks:,}  ({cracks / total * 100:.1f} %)")
    print(f"  No-crack                : {no_crack:,}  ({no_crack / total * 100:.1f} %)")
    print(f"  EXIF GPS used           : {exif_found:,}")
    print(f"  FHWA coords used        : {total - exif_found:,}")
    print(f"  Unique bridges          : {combined['asset_id'].nunique():,}")
    print(f"  States / territories    : {combined['state_code'].nunique()}")
    print(f"  Inspection type         : {INSPECTION_TYPE}")
    print(f"  Lat range               : {b[1]:.4f} N -- {b[3]:.4f} N")
    print(f"  Lon range               : {b[0]:.4f} -- {b[2]:.4f}")
    print("-" * 54)
    print()
    print(f"  {'Abbr':<5}  {'State':<22}  {'Total':>6}  {'Cracks':>6}  {'Crack%':>6}")
    print("  " + "-" * 48)
    for _, row in summary.head(15).iterrows():
        print(f"  {row['state_abbr']:<5}  {row['state_name']:<22}  "
              f"{row['total_images']:>6,}  {row['crack_count']:>6,}  "
              f"{row['crack_pct']:>5.1f} %")
    print("-" * 54)


if __name__ == "__main__":
    main()
