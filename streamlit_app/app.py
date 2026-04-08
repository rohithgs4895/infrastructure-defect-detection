"""
Crack Detection QA Tool — Field Inspector Interface.

Model  : models/crack_detector_final.keras  (EfficientNetB0, binary)
Inputs : any JPEG / PNG image uploaded by the inspector
Outputs: outputs/validated_results.csv  (appended on each validation)

Run with:
    streamlit run streamlit_app/app.py
"""

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR   = Path(r"C:\Projects\infrastructure-defect-detection")
MODEL_PATH    = PROJECT_DIR / "models" / "crack_detector_final.keras"
VALIDATED_CSV = PROJECT_DIR / "outputs" / "validated_results.csv"

IMG_SIZE  = (227, 227)
THRESHOLD = 0.5

CSV_COLUMNS = [
    "filename", "prediction", "confidence",
    "inspector_decision", "comment", "timestamp",
]

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crack Detection QA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Page background and font */
    .stApp { background-color: #f4f6f9; }

    /* App header */
    .app-title {
        font-size: 1.9rem;
        font-weight: 800;
        color: #1c2b3a;
        letter-spacing: -0.5px;
        margin-bottom: 0.1rem;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #6b7a8d;
        margin-bottom: 0;
    }

    /* Prediction result card */
    .result-card {
        border-radius: 12px;
        padding: 1.6rem 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-crack    { background: #fff0f0; border: 2.5px solid #e53935; }
    .result-nocrack  { background: #f0fff4; border: 2.5px solid #2e7d32; }

    .result-label {
        font-size: 2rem;
        font-weight: 900;
        letter-spacing: 1px;
        margin: 0 0 0.4rem 0;
    }
    .label-crack   { color: #c62828; }
    .label-nocrack { color: #1b5e20; }

    .result-tier {
        font-size: 0.95rem;
        color: #555;
        margin: 0;
    }

    /* Section headings */
    .section-heading {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1c2b3a;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.6rem;
    }

    /* Subtle detail table */
    .detail-row {
        display: flex;
        justify-content: space-between;
        padding: 0.3rem 0;
        border-bottom: 1px solid #e0e4ea;
        font-size: 0.9rem;
        color: #444;
    }
    .detail-key   { font-weight: 600; }
    .detail-value { font-family: monospace; color: #1c2b3a; }

    /* Sidebar */
    .sidebar-stat-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #6b7a8d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Empty-state prompt */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        color: #9aa5b4;
    }
    .empty-state-icon  { font-size: 3.5rem; margin-bottom: 0.5rem; }
    .empty-state-title { font-size: 1.2rem; font-weight: 700; color: #6b7a8d; }

    /* Form submit button full width */
    .stForm [data-testid="stFormSubmitButton"] > button {
        width: 100%;
        font-size: 1rem;
        font-weight: 700;
        padding: 0.6rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Model (cached — loads once per session) ───────────────────────────────────
@st.cache_resource(show_spinner="Loading crack detection model…")
def load_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        st.error(f"Model not found: `{MODEL_PATH}`")
        st.stop()
    return tf.keras.models.load_model(str(MODEL_PATH))


# ── Inference helpers ─────────────────────────────────────────────────────────
def preprocess(pil_img: Image.Image) -> np.ndarray:
    """PIL Image → normalised float32 batch tensor (1, 227, 227, 3) in [0, 1]."""
    arr = np.array(pil_img.convert("RGB").resize(IMG_SIZE, Image.BILINEAR), dtype=np.float32)
    return arr[np.newaxis] / 255.0


def run_inference(model: tf.keras.Model, pil_img: Image.Image) -> tuple[str, float]:
    """Return (label, raw_sigmoid_score)."""
    score = float(model.predict(preprocess(pil_img), verbose=0)[0, 0])
    return ("Crack" if score >= THRESHOLD else "No Crack"), round(score, 6)


def confidence_in_prediction(label: str, score: float) -> float:
    """Confidence that the prediction is correct (always 0.5–1.0)."""
    return score if label == "Crack" else 1.0 - score


def confidence_tier(conf: float) -> str:
    if conf >= 0.95:
        return "Very high confidence"
    if conf >= 0.80:
        return "High confidence"
    if conf >= 0.65:
        return "Moderate confidence"
    return "Low confidence — review carefully"


# ── Validated CSV helpers ─────────────────────────────────────────────────────
@st.cache_data(ttl=1)
def load_validated() -> pd.DataFrame:
    if VALIDATED_CSV.exists() and VALIDATED_CSV.stat().st_size > 0:
        return pd.read_csv(VALIDATED_CSV, parse_dates=["timestamp"])
    return pd.DataFrame(columns=CSV_COLUMNS)


def append_row(row: dict) -> None:
    VALIDATED_CSV.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not VALIDATED_CSV.exists() or VALIDATED_CSV.stat().st_size == 0
    with open(VALIDATED_CSV, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
    load_validated.clear()   # invalidate cache so sidebar refreshes immediately


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    df = load_validated()
    total    = len(df)
    approved = int((df["inspector_decision"] == "Approve").sum()) if total else 0
    rejected = total - approved

    with st.sidebar:
        st.markdown("## 📋 Inspector Dashboard")
        st.divider()

        # ── Key metrics ──────────────────────────────────────────────────────
        m1, m2 = st.columns(2)
        m1.metric("Validated", f"{total:,}")
        m2.metric("Approved",  f"{approved:,}")

        m3, m4 = st.columns(2)
        m3.metric("Rejected", f"{rejected:,}")

        if approved:
            crack_rate = (
                df.loc[df["inspector_decision"] == "Approve", "prediction"]
                  .eq("Crack").mean() * 100
            )
            m4.metric("Crack Rate", f"{crack_rate:.1f}%")
        else:
            m4.metric("Crack Rate", "—")

        # ── Agreement bar ─────────────────────────────────────────────────────
        if total:
            st.markdown(
                f"<p class='sidebar-stat-label'>Model agreement</p>",
                unsafe_allow_html=True,
            )
            st.progress(approved / total, text=f"{approved / total * 100:.0f}% approved")

        st.divider()

        # ── Recent validations ────────────────────────────────────────────────
        st.markdown("**Recent validations**")
        if total:
            recent = (
                df[["filename", "prediction", "inspector_decision"]]
                  .tail(8)
                  .iloc[::-1]
                  .copy()
            )
            recent.columns = ["File", "Model", "Decision"]

            # Colour-code the Decision column
            def _colour_decision(val):
                colour = "#2e7d32" if val == "Approve" else "#c62828"
                return f"color: {colour}; font-weight: 600"

            st.dataframe(
                recent.style.map(_colour_decision, subset=["Decision"]),
                use_container_width=True,
                hide_index=True,
                height=min(8, total) * 35 + 38,
            )
        else:
            st.caption("No validations yet — submit the first one.")

        st.divider()
        st.caption(f"Model · `crack_detector_final.keras`")
        st.caption(f"Decision threshold · `{THRESHOLD}`")
        st.caption(f"Output · `outputs/validated_results.csv`")


# ── Prediction card ───────────────────────────────────────────────────────────
def render_prediction_card(label: str, score: float) -> None:
    is_crack = label == "Crack"
    conf     = confidence_in_prediction(label, score)
    tier     = confidence_tier(conf)
    icon     = "⚠️" if is_crack else "✅"

    card_cls  = "result-crack"   if is_crack else "result-nocrack"
    label_cls = "label-crack"    if is_crack else "label-nocrack"
    bar_col   = "#e53935"        if is_crack else "#2e7d32"

    st.markdown(
        f"""
        <div class="result-card {card_cls}">
            <p class="result-label {label_cls}">{icon}&nbsp;&nbsp;{label.upper()}</p>
            <p class="result-tier">{tier} &nbsp;·&nbsp; raw score&nbsp;{score:.6f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<p class='section-heading'>Confidence in prediction — {conf*100:.1f}%</p>",
        unsafe_allow_html=True,
    )
    st.progress(conf)


# ── Detail table ──────────────────────────────────────────────────────────────
def render_image_details(filename: str, img: Image.Image, score: float) -> None:
    st.markdown("<p class='section-heading'>Image details</p>", unsafe_allow_html=True)
    rows = [
        ("Filename",   filename),
        ("Dimensions", f"{img.width} × {img.height} px"),
        ("Mode",       img.mode),
        ("Raw score",  f"{score:.6f}"),
        ("Threshold",  str(THRESHOLD)),
    ]
    html = "".join(
        f"<div class='detail-row'>"
        f"<span class='detail-key'>{k}</span>"
        f"<span class='detail-value'>{v}</span>"
        f"</div>"
        for k, v in rows
    )
    st.markdown(html, unsafe_allow_html=True)


# ── Validation form ───────────────────────────────────────────────────────────
def render_validation_form(filename: str, label: str, score: float) -> None:
    st.markdown("<p class='section-heading'>Inspector Validation</p>", unsafe_allow_html=True)

    if st.session_state.get("validated"):
        st.success(
            f"✅ Validation saved for **{filename}**. Upload a new image to continue.",
        )
        return

    with st.form("validation_form"):
        st.markdown(
            f"Model predicts **{label}** with {confidence_in_prediction(label, score)*100:.1f}% "
            f"confidence. Do you agree with this assessment?"
        )

        dec_col, cmt_col = st.columns([1, 2], gap="medium")

        with dec_col:
            decision = st.radio(
                "Inspector decision",
                options=["Approve", "Reject"],
                captions=[
                    "Model is correct",
                    "Model is wrong",
                ],
                index=0,
            )

        with cmt_col:
            comment = st.text_area(
                "Comment (optional)",
                placeholder=(
                    "e.g. 'Hairline crack visible at top-left corner'\n"
                    "      'False positive — shadow artifact'\n"
                    "      'Confirmed crack near expansion joint'"
                ),
                height=110,
            )

        submitted = st.form_submit_button(
            "Submit Validation",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        append_row({
            "filename":           filename,
            "prediction":         label,
            "confidence":         score,
            "inspector_decision": decision,
            "comment":            comment.strip(),
            "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        st.session_state["validated"] = True
        st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    render_sidebar()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        "<p class='app-title'>🔍 Crack Detection QA Tool</p>"
        "<p class='app-subtitle'>Infrastructure Defect Detection — Field Inspector Interface</p>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    model = load_model()

    # ── File uploader ──────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload inspection image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Drag and drop or click to upload. Supports JPEG and PNG.",
        key="uploader",
    )

    # ── Session state: reset on new upload ────────────────────────────────────
    if uploaded is not None:
        file_key = f"{uploaded.name}::{uploaded.size}"
        if st.session_state.get("_file_key") != file_key:
            st.session_state["_file_key"]   = file_key
            st.session_state["label"]       = None
            st.session_state["score"]       = None
            st.session_state["validated"]   = False

        # Run inference only once per image
        if st.session_state.get("label") is None:
            with st.spinner("Running inference…"):
                pil_img = Image.open(uploaded)
                label, score = run_inference(model, pil_img)
                st.session_state["label"] = label
                st.session_state["score"] = score

    # ── Results ────────────────────────────────────────────────────────────────
    if uploaded is not None and st.session_state.get("label") is not None:
        label = st.session_state["label"]
        score = st.session_state["score"]
        pil_img = Image.open(uploaded)

        st.divider()
        img_col, pred_col = st.columns([1, 1], gap="large")

        with img_col:
            st.markdown("<p class='section-heading'>Uploaded Image</p>", unsafe_allow_html=True)
            st.image(pil_img, caption=uploaded.name, use_container_width=True)

        with pred_col:
            st.markdown("<p class='section-heading'>Model Prediction</p>", unsafe_allow_html=True)
            render_prediction_card(label, score)
            st.markdown("")
            render_image_details(uploaded.name, pil_img, score)

        st.divider()
        render_validation_form(uploaded.name, label, score)

    else:
        # ── Empty state ───────────────────────────────────────────────────────
        st.divider()
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-icon">📷</div>
                <p class="empty-state-title">Upload an image to begin inspection</p>
                <p>Drag and drop a JPEG or PNG above, or click to browse</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
