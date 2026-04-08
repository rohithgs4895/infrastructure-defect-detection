"""
ArcGIS Pro full automation — infrastructure crack detection spatial analysis.

Pipeline
--------
1. Import defect_map.shp into an in-memory feature class
2. Run Kernel Density on crack-only points (weighted by confidence)
3. Run Optimized Hot Spot Analysis on crack-only points
4. Join inspection_summary.csv to US states, compute condition scores,
   classify states as Critical / At Risk / Good, export shapefile
5. Print a national summary report

Inputs
------
  outputs/defect_map.shp          20,000 georeferenced predictions (WGS 1984)
  outputs/inspection_summary.csv  State-level crack statistics

Outputs
-------
  outputs/crack_density.tif            Kernel Density raster (crack points only)
  outputs/hotspot_analysis.shp         Gi* hot/cold spot classification
  outputs/state_condition_scores.shp   US states with score + condition fields

Requirements
------------
  ArcGIS Pro >= 3.x  |  Spatial Analyst extension  |  ArcGIS.com portal sign-in

Usage
-----
    python src/arcgis_analysis.py
"""

import csv
import statistics
import sys
from collections import Counter
from pathlib import Path

import arcpy
from arcpy.sa import KernelDensity

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR      = Path(r"C:\Projects\infrastructure-defect-detection")
DEFECT_MAP       = str(PROJECT_DIR / "outputs" / "defect_map.shp")
INSPECTION_CSV   = str(PROJECT_DIR / "outputs" / "inspection_summary.csv")
CRACK_DENSITY_OUT = str(PROJECT_DIR / "outputs" / "crack_density.tif")
HOTSPOT_OUT      = str(PROJECT_DIR / "outputs" / "hotspot_analysis.shp")
STATE_SCORES_OUT = str(PROJECT_DIR / "outputs" / "state_condition_scores.shp")

# Living Atlas — USA States Generalized Boundaries (public)
USA_STATES_URL = (
    "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services"
    "/USA_States_Generalized_Boundaries/FeatureServer/0"
)

# USA Contiguous Albers Equal Area Conic (WKID 102003) — metric area units
ALBERS_SR = arcpy.SpatialReference(102003)

# Kernel Density parameters for ~10,000 nationwide crack points
CELL_SIZE_M   = 50_000    # 50 km cells
SEARCH_RADIUS = 500_000   # 500 km bandwidth

# Condition score thresholds  (score = crack_pct, already 0-100)
CRITICAL_THRESHOLD = 60.0   # >60%  -> Critical
AT_RISK_THRESHOLD  = 40.0   # 40-60% -> At Risk  |  <40% -> Good


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _divider(char: str = "-", width: int = 60) -> None:
    print(char * width)


def _check_spatial_analyst() -> None:
    status = arcpy.CheckExtension("Spatial")
    if status != "Available":
        print(f"ERROR: Spatial Analyst extension is '{status}'. Cannot continue.")
        sys.exit(1)
    arcpy.CheckOutExtension("Spatial")
    print("  Spatial Analyst extension: checked out.")


def _classify(pct: float) -> str:
    if pct > CRITICAL_THRESHOLD:
        return "Critical"
    if pct >= AT_RISK_THRESHOLD:
        return "At Risk"
    return "Good"


def _delete_if_exists(path: str) -> None:
    """Delete an existing dataset before overwriting it.

    arcpy.env.overwriteOutput = True does not release a schema lock held by
    ArcGIS Pro when the layer is open in a map.  Explicit deletion with a
    clear error message is the reliable alternative.
    """
    if not arcpy.Exists(path):
        return
    try:
        arcpy.management.Delete(path)
    except arcpy.ExecuteError:
        msgs = arcpy.GetMessages(2)
        if "000464" in msgs or "schema lock" in msgs.lower():
            print(f"\n  LOCK ERROR: '{path}' is open in ArcGIS Pro.")
            print("  Remove or close that layer in the map, then re-run.")
            sys.exit(1)
        raise


# ---------------------------------------------------------------------------
# Step 1 — Import predictions to in-memory feature class
# ---------------------------------------------------------------------------
def step1_import_to_feature_class() -> tuple[str, int, int]:
    """
    Copy defect_map.shp into an in-memory feature class so all downstream
    tools operate on a stable, GDB-native dataset rather than a shapefile.
    """
    print()
    _divider()
    print("  STEP 1 - Import Predictions to Feature Class")
    _divider()

    arcpy.env.overwriteOutput = True

    fc_all = r"memory\defect_points"
    arcpy.management.CopyFeatures(DEFECT_MAP, fc_all)

    n_total = int(arcpy.GetCount_management(fc_all)[0])
    crack_n  = sum(
        1 for row in arcpy.da.SearchCursor(fc_all, ["prediction"])
        if row[0] == "Crack"
    )
    no_crack_n = n_total - crack_n

    print(f"  Source         : {DEFECT_MAP}")
    print(f"  Target FC      : {fc_all}")
    print(f"  Total features : {n_total:,}")
    print(f"  Crack          : {crack_n:,}  ({crack_n / n_total * 100:.1f}%)")
    print(f"  No Crack       : {no_crack_n:,}  ({no_crack_n / n_total * 100:.1f}%)")
    print()
    print(f"  SUCCESS: Predictions imported -> {fc_all}")

    return fc_all, crack_n, no_crack_n


# ---------------------------------------------------------------------------
# Step 2 — Kernel Density (crack points only)
# ---------------------------------------------------------------------------
def step2_kernel_density(fc_all: str) -> None:
    """
    Filter the feature class to crack predictions, then run Kernel Density
    weighted by model confidence.  Output is a GeoTIFF in Albers Equal Area.
    """
    print()
    _divider()
    print("  STEP 2 - Kernel Density (Crack Points Only)")
    _divider()

    # Temporary layer with only Crack features
    arcpy.management.MakeFeatureLayer(fc_all, "crack_lyr", "prediction = 'Crack'")
    n_crack = int(arcpy.GetCount_management("crack_lyr")[0])

    print(f"  Crack features : {n_crack:,}")
    print(f"  Weight field   : confidence  (model score 0-1)")
    print(f"  Cell size      : {CELL_SIZE_M / 1_000:.0f} km")
    print(f"  Search radius  : {SEARCH_RADIUS / 1_000:.0f} km")
    print(f"  Output SR      : USA Contiguous Albers Equal Area (WKID 102003)")
    print(f"  Output         : {CRACK_DENSITY_OUT}")
    print()

    _delete_if_exists(CRACK_DENSITY_OUT)
    arcpy.env.outputCoordinateSystem = ALBERS_SR
    arcpy.env.overwriteOutput = True

    density = KernelDensity(
        in_features           = "crack_lyr",
        population_field      = "confidence",
        cell_size             = CELL_SIZE_M,
        search_radius         = SEARCH_RADIUS,
        area_unit_scale_factor= "SQUARE_KILOMETERS",
        out_cell_values       = "DENSITIES",
        method                = "PLANAR",
    )
    density.save(CRACK_DENSITY_OUT)

    arcpy.management.Delete("crack_lyr")

    r = arcpy.Describe(CRACK_DENSITY_OUT)
    print(f"  Raster extent  : {r.extent}")
    print(f"  Cell size      : {r.meanCellWidth:.0f} x {r.meanCellHeight:.0f} m")
    print()
    print(f"  SUCCESS: Crack density raster saved -> {CRACK_DENSITY_OUT}")


# ---------------------------------------------------------------------------
# Step 3 — Optimized Hot Spot Analysis (crack points only)
# ---------------------------------------------------------------------------
def step3_hotspot_analysis(fc_all: str) -> None:
    """
    Export crack-only points to a temporary feature class, then run
    Optimized Hot Spot Analysis on the confidence field.
    """
    print()
    _divider()
    print("  STEP 3 - Optimized Hot Spot Analysis (Crack Points Only)")
    _divider()
    print(f"  Analysis field : confidence")
    print(f"  Output         : {HOTSPOT_OUT}")
    print()

    # Export crack points to temp FC — OHSA needs a persisted dataset
    crack_fc = r"memory\crack_points_hs"
    arcpy.conversion.ExportFeatures(
        in_features  = fc_all,
        out_features = crack_fc,
        where_clause = "prediction = 'Crack'",
    )
    n_in = int(arcpy.GetCount_management(crack_fc)[0])
    print(f"  Input features : {n_in:,} crack points")

    _delete_if_exists(HOTSPOT_OUT)
    arcpy.env.outputCoordinateSystem = None   # let tool project internally
    arcpy.env.overwriteOutput = True

    print("  Running Optimized Hot Spot Analysis (1-3 min) ...")
    arcpy.stats.OptimizedHotSpotAnalysis(
        Input_Features = crack_fc,
        Output_Features= HOTSPOT_OUT,
        Analysis_Field = "confidence",
    )

    arcpy.management.Delete(crack_fc)

    n_out = int(arcpy.GetCount_management(HOTSPOT_OUT)[0])
    print(f"  Output features: {n_out:,}")

    out_fields = [f.name for f in arcpy.ListFields(HOTSPOT_OUT)]

    if "GiZScore" in out_fields:
        zi = [
            row[0]
            for row in arcpy.da.SearchCursor(HOTSPOT_OUT, ["GiZScore"])
            if row[0] is not None
        ]
        if zi:
            print(
                f"  Gi* Z-score    : min {min(zi):.3f}  "
                f"max {max(zi):.3f}  "
                f"mean {statistics.mean(zi):.3f}"
            )

    if "Gi_Bin" in out_fields:
        bins = Counter(
            row[0] for row in arcpy.da.SearchCursor(HOTSPOT_OUT, ["Gi_Bin"])
        )
        bin_labels = {
             3: "Hot spot 99%",   2: "Hot spot 95%",   1: "Hot spot 90%",
             0: "Not significant",
            -1: "Cold spot 90%", -2: "Cold spot 95%", -3: "Cold spot 99%",
        }
        print()
        print("  Gi* Bin breakdown:")
        for b in sorted(bins, reverse=True):
            print(f"    {bin_labels.get(b, str(b)):22s}: {bins[b]:,}")

    print()
    print(f"  SUCCESS: Hot Spot Analysis saved -> {HOTSPOT_OUT}")


# ---------------------------------------------------------------------------
# Step 4 — State condition scores
# ---------------------------------------------------------------------------
def step4_state_condition_scores() -> dict:
    """
    Load inspection_summary.csv, join to US state polygons from Living Atlas,
    compute score = crack_pct, classify each state, export shapefile.

    Returns the summary dict (state_abbr -> stats) for the report step.
    """
    print()
    _divider()
    print("  STEP 4 - State Condition Scores")
    _divider()
    print(f"  Summary CSV    : {INSPECTION_CSV}")
    print(f"  Join key       : state_abbr -> STATE_ABBR")
    print(f"  Score formula  : (crack_count / total_inspections) * 100")
    print(f"  Critical       : score > {CRITICAL_THRESHOLD:.0f}%")
    print(f"  At Risk        : {AT_RISK_THRESHOLD:.0f}% - {CRITICAL_THRESHOLD:.0f}%")
    print(f"  Good           : score < {AT_RISK_THRESHOLD:.0f}%")
    print(f"  Output         : {STATE_SCORES_OUT}")
    print()

    # -- Load inspection summary ----------------------------------------
    summary: dict = {}
    with open(INSPECTION_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            abbr = row["state_abbr"].strip()
            summary[abbr] = {
                "state_name":  row["state_name"].strip(),
                "total":       int(row["total_images"]),
                "crack_count": int(row["crack_count"]),
                "crack_pct":   float(row["crack_pct"]),
            }
    print(f"  States in CSV  : {len(summary)}")

    # -- Fetch US states from Living Atlas ---------------------------------
    print("  Fetching US states from ArcGIS Living Atlas ...")
    arcpy.management.MakeFeatureLayer(USA_STATES_URL, "states_src_lyr")
    states_fc = r"memory\states_scored"
    arcpy.management.CopyFeatures("states_src_lyr", states_fc)
    arcpy.management.Delete("states_src_lyr")
    n_states = int(arcpy.GetCount_management(states_fc)[0])
    print(f"  State polygons : {n_states}")

    # -- Add score fields --------------------------------------------------
    arcpy.management.AddField(states_fc, "inspectns",  "LONG")
    arcpy.management.AddField(states_fc, "crack_cnt",  "LONG")
    arcpy.management.AddField(states_fc, "score",      "DOUBLE")
    arcpy.management.AddField(states_fc, "condition",  "TEXT", field_length=12)

    # -- Populate via UpdateCursor -----------------------------------------
    edit_fields = ["STATE_ABBR", "inspectns", "crack_cnt", "score", "condition"]
    matched = 0
    with arcpy.da.UpdateCursor(states_fc, edit_fields) as cur:
        for row in cur:
            abbr = row[0]
            if abbr in summary:
                d      = summary[abbr]
                row[1] = d["total"]
                row[2] = d["crack_count"]
                row[3] = round(d["crack_pct"], 2)
                row[4] = _classify(d["crack_pct"])
                matched += 1
            else:
                row[1] = 0
                row[2] = 0
                row[3] = 0.0
                row[4] = "No Data"
            cur.updateRow(row)

    print(f"  States matched : {matched} / {len(summary)}")

    # -- Tally classifications ---------------------------------------------
    tally = Counter(
        row[0]
        for row in arcpy.da.SearchCursor(states_fc, ["condition"])
    )
    print()
    print("  Condition breakdown:")
    for label in ("Critical", "At Risk", "Good", "No Data"):
        print(f"    {label:<10}: {tally.get(label, 0)} states")

    # -- Export shapefile --------------------------------------------------
    _delete_if_exists(STATE_SCORES_OUT)
    arcpy.env.overwriteOutput = True
    arcpy.conversion.ExportFeatures(states_fc, STATE_SCORES_OUT)
    arcpy.management.Delete(states_fc)

    print()
    print(f"  SUCCESS: State condition scores saved -> {STATE_SCORES_OUT}")

    return summary


# ---------------------------------------------------------------------------
# Step 5 — Summary report
# ---------------------------------------------------------------------------
def step5_summary_report(summary: dict) -> None:
    """Print national statistics and top/bottom state tables."""
    total_inspected = sum(d["total"]       for d in summary.values())
    total_cracks    = sum(d["crack_count"] for d in summary.values())
    national_rate   = total_cracks / total_inspected * 100

    ranked = sorted(summary.items(), key=lambda x: x[1]["crack_pct"], reverse=True)

    print()
    _divider("=")
    print("  INFRASTRUCTURE DEFECT DETECTION - NATIONAL SUMMARY")
    _divider("=")
    print(f"  Total bridges inspected   : {total_inspected:,}")
    print(f"  Total crack detections    : {total_cracks:,}")
    print(f"  National crack rate       : {national_rate:.1f}%")
    print()

    header = f"  {'State':<22} {'Inspected':>10} {'Cracks':>8} {'Score':>7}   Condition"
    row_div = "  " + "-" * 60

    print("  TOP 5 CRITICAL STATES  (highest crack rate)")
    print(header)
    print(row_div)
    for abbr, d in ranked[:5]:
        cond = _classify(d["crack_pct"])
        print(
            f"  {d['state_name']:<22} {d['total']:>10,} {d['crack_count']:>8,} "
            f"{d['crack_pct']:>6.1f}%   {cond}"
        )

    print()
    print("  TOP 5 BEST CONDITION STATES  (lowest crack rate)")
    print(header)
    print(row_div)
    for abbr, d in reversed(ranked[-5:]):
        cond = _classify(d["crack_pct"])
        print(
            f"  {d['state_name']:<22} {d['total']:>10,} {d['crack_count']:>8,} "
            f"{d['crack_pct']:>6.1f}%   {cond}"
        )

    print()
    _divider("=")
    print("  OUTPUT FILES")
    _divider("-")
    print(f"  Crack density raster   : outputs/crack_density.tif")
    print(f"  Hot spot layer         : outputs/hotspot_analysis.shp")
    print(f"  State scores layer     : outputs/state_condition_scores.shp")
    _divider("=")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _divider("=")
    print("  Infrastructure Defect Detection - ArcGIS Full Automation")
    _divider("=")
    print(f"  ArcGIS Pro : {arcpy.GetInstallInfo()['Version']}")
    print(f"  License    : {arcpy.ProductInfo()}")
    print(f"  Portal     : {arcpy.GetActivePortalURL()}")
    _divider()

    _check_spatial_analyst()
    arcpy.env.overwriteOutput = True

    try:
        fc_all, crack_n, no_crack_n = step1_import_to_feature_class()
        step2_kernel_density(fc_all)
        step3_hotspot_analysis(fc_all)
        summary = step4_state_condition_scores()
        step5_summary_report(summary)

    except arcpy.ExecuteError:
        print()
        print("ArcPy execution error:")
        print(arcpy.GetMessages(2))
        sys.exit(1)
    except Exception as exc:
        print()
        print(f"Unexpected error: {exc}")
        raise
    finally:
        arcpy.management.Delete(r"memory\defect_points")
        arcpy.CheckInExtension("Spatial")
        print("\n  Spatial Analyst extension returned.")


if __name__ == "__main__":
    main()
