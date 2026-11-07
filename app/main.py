
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Injection Molding | Predictive Quality Control",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "random_forest_model.pkl"
VALIDATION_PATH = ROOT / "data" / "validation.csv"

FEATURES = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]

R_THRESHOLD = 0.30
Y_THRESHOLD = 0.10

BLUE = "#2563EB"
NAVY = "#172B4D"
MUTED = "#64748B"
GREEN = "#168B65"
RED = "#D74E57"
AMBER = "#E6A13B"

COLORS = {
    "G": GREEN,
    "R": RED,
    "Y": AMBER,
}

CLASS_NAMES = {
    "G": "Feasible Process",
    "R": "Infeasible Process",
    "Y": "Feasible Process with Short Shots",
}

# label, minimum, maximum, default, step, unit
SETTINGS = {
    "Tinj": (
        "Injection Temperature",
        180.0, 240.0, 200.0, 5.0, "°C"
    ),
    "tinj": (
        "Injection Time",
        0.5, 2.0, 1.0, 0.1, "s"
    ),
    "Pinj": (
        "Injection Pressure",
        5.0, 35.0, 20.0, 1.0, "bar"
    ),
    "Ph": (
        "Holding Pressure",
        5.0, 35.0, 15.0, 1.0, "bar"
    ),
    "Bp": (
        "Back Pressure",
        15.0, 30.0, 20.0, 1.0, "bar"
    ),
    "th": (
        "Holding Time",
        4.0, 6.0, 5.0, 0.1, "s"
    ),
}


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    :root {
        --primary: #2563EB;
        --navy: #172B4D;
        --muted: #64748B;
        --border: #DCE4EE;
        --background: #F7F9FC;
        --green: #168B65;
        --red: #D74E57;
        --amber: #E6A13B;
    }

    .stApp {
        background: var(--background);
        color: var(--navy);
    }

    .block-container {
        max-width: 1600px;
        padding-top: 1rem;
        padding-bottom: 1.5rem;
    }

    h1, h2, h3 {
        color: var(--navy);
        letter-spacing: -0.02em;
    }

    h1 {
        font-size: 1.8rem !important;
        font-weight: 750 !important;
    }

    h2 {
        font-size: 1.3rem !important;
    }

    h3 {
        font-size: 1.05rem !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
    }

    div[data-testid="stMetric"] {
        background: white;
        padding: 12px;
        border: 1px solid var(--border);
        border-radius: 12px;
    }

    .stButton button[kind="primary"],
    div[data-testid="stFormSubmitButton"]
    button[kind="primary"] {
        background: #2563EB;
        border-color: #2563EB;
        color: white;
        border-radius: 9px;
        font-weight: 650;
    }

    .stButton button[kind="primary"]:hover,
    div[data-testid="stFormSubmitButton"]
    button[kind="primary"]:hover {
        background: #1D4ED8;
        border-color: #1D4ED8;
    }

    div[data-testid="stSlider"] {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }

    div[data-testid="stSlider"] label {
        font-size: 0.86rem;
        font-weight: 600;
    }

    div[data-testid="stForm"]
    div[data-testid="stVerticalBlock"] {
        gap: 0.55rem;
    }

    div[data-testid="stTabs"] button {
        font-weight: 650;
    }

    .section-heading {
        font-size: 1.05rem;
        font-weight: 700;
        color: #172B4D;
        margin-bottom: 0.3rem;
    }

    .decision-card {
        border-radius: 10px;
        padding: 12px 15px;
        margin: 6px 0;
        font-size: 0.9rem;
        font-weight: 650;
    }

    .decision-safe {
        background: #E5F5ED;
        border-left: 4px solid #168B65;
        color: #116347;
    }

    .decision-danger {
        background: #FBE9EA;
        border-left: 4px solid #D74E57;
        color: #9B3039;
    }

    .decision-warning {
        background: #FFF3DC;
        border-left: 4px solid #E6A13B;
        color: #8B5E14;
    }

    .about-intro {
        background: #EDF4FF;
        border: 1px solid #D6E5FF;
        border-left: 4px solid #2563EB;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 14px;
        line-height: 1.6;
    }

    .workflow-step {
        background: #F0F5FF;
        border: 1px solid #D6E5FF;
        border-radius: 9px;
        padding: 11px;
        text-align: center;
        color: #172B4D;
        font-size: 0.88rem;
        font-weight: 650;
    }

    .workflow-arrow {
        text-align: center;
        color: #2563EB;
        font-size: 1.3rem;
        padding: 3px;
    }

    .small-note {
        color: #64748B;
        font-size: 0.82rem;
        line-height: 1.5;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. LOAD MODEL AND SHAP EXPLAINER
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Random Forest model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_explainer():
    return shap.TreeExplainer(load_model())


@st.cache_data
def load_global_importance():
    validation_data = pd.read_csv(VALIDATION_PATH)

    X_validation = validation_data[FEATURES]

    explanation = load_explainer()(X_validation)

    mean_abs_shap = np.abs(
        explanation.values
    ).mean(axis=(0, 2))

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Mean Absolute SHAP": mean_abs_shap,
    })

    return importance.sort_values(
        "Mean Absolute SHAP",
        ascending=False,
    )


try:
    model = load_model()
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


# ============================================================
# 4. PREDICTION FUNCTIONS
# ============================================================

def make_input_frame(values):
    return pd.DataFrame(
        [{
            feature: values[feature]
            for feature in FEATURES
        }],
        columns=FEATURES,
    )


def predict_configuration(input_frame):
    predicted_class = model.predict(input_frame)[0]

    probability_array = model.predict_proba(
        input_frame
    )[0]

    probabilities = {
        str(label): float(probability)
        for label, probability in zip(
            model.classes_,
            probability_array,
        )
    }

    return predicted_class, probabilities


def review_required(probabilities):
    return (
        probabilities["R"] >= R_THRESHOLD
        or probabilities["Y"] >= Y_THRESHOLD
    )


# ============================================================
# 5. CIRCULAR PROBABILITY CHART
# ============================================================

def create_donut_chart(class_label, probability):
    percentage = probability * 100

    fig = go.Figure()

    fig.add_trace(
        go.Pie(
            values=[
                percentage,
                100 - percentage,
            ],
            hole=0.76,
            sort=False,
            direction="clockwise",
            rotation=90,
            marker=dict(
                colors=[
                    COLORS[class_label],
                    "#E4EAF1",
                ],
                line=dict(width=0),
            ),
            textinfo="none",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_annotation(
        x=0.5,
        y=0.59,
        text=f"<b>{class_label}</b>",
        showarrow=False,
        font=dict(
            size=25,
            color=NAVY,
        ),
    )

    fig.add_annotation(
        x=0.5,
        y=0.39,
        text=f"<b>{percentage:.1f}%</b>",
        showarrow=False,
        font=dict(
            size=19,
            color=COLORS[class_label],
        ),
    )

    fig.update_layout(
        height=175,
        margin=dict(
            l=0, r=0, t=0, b=0
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )

    return fig


# ============================================================
# 6. AUTOMATIC ENGINEERING INSIGHTS
# ============================================================

def generate_alternative_settings(original_input):
    """
    Explore one-parameter-at-a-time alternatives.

    Results are model-based what-if scenarios,
    not verified manufacturing corrections.
    """

    candidates = []

    _, original_probs = predict_configuration(
        original_input
    )

    original_risk = (
        original_probs["R"] +
        original_probs["Y"]
    )

    for feature in FEATURES:

        (
            label,
            minimum,
            maximum,
            _,
            _,
            unit,
        ) = SETTINGS[feature]

        trial_values = np.linspace(
            minimum,
            maximum,
            7,
        )

        current_value = float(
            original_input[feature].iloc[0]
        )

        for value in trial_values:

            value = round(float(value), 2)

            if np.isclose(
                value,
                current_value,
            ):
                continue

            candidate = original_input.copy()

            candidate.loc[:, feature] = value

            prediction, probabilities = (
                predict_configuration(candidate)
            )

            combined_risk = (
                probabilities["R"] +
                probabilities["Y"]
            )

            candidates.append({
                "Parameter": label,
                "Current": current_value,
                "Suggested": value,
                "Unit": unit,
                "Predicted": prediction,
                "P(G)": probabilities["G"],
                "P(R)": probabilities["R"],
                "P(Y)": probabilities["Y"],
                "Review": review_required(
                    probabilities
                ),
                "Combined Risk": combined_risk,
                "Risk Reduction": (
                    original_risk -
                    combined_risk
                ),
            })

    candidates = pd.DataFrame(candidates)

    candidates = candidates[
        candidates["Risk Reduction"] > 1e-9
    ].copy()

    if candidates.empty:
        return candidates

    candidates = candidates.sort_values(
        by=[
            "Review",
            "Combined Risk",
            "P(G)",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )

    return candidates.head(5)


# ============================================================
# 7. SESSION STATE
# ============================================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "probabilities" not in st.session_state:
    st.session_state.probabilities = None

if "analyzed_input" not in st.session_state:
    st.session_state.analyzed_input = None


# ============================================================
# 8. HEADER AND NAVIGATION
# ============================================================

st.title("🏭 Injection Molding Quality Control")

st.markdown(
    """
    <div style="margin-top:-8px; margin-bottom:5px;
                font-size:15px; font-weight:500; color:#526581;">
        AI-Based Predictive Quality Control for Polypropylene Injection Molding
    </div>
    <div style="font-size:12px; color:#8290A5; margin-bottom:18px;">
        Random Forest &nbsp;•&nbsp;
        SHAP Explainability &nbsp;•&nbsp;
        Industrial Decision Support
    </div>
    """,
    unsafe_allow_html=True,
)

(
    tab_prediction,
    tab_explainability,
    tab_performance,
    tab_about,
) = st.tabs([
    "Prediction",
    "Explainability",
    "Model Performance",
    "About Project",
])


# ============================================================
# 9. PREDICTION TAB
# ============================================================

with tab_prediction:

    left_column, right_column = st.columns(
        [1, 1.25],
        gap="medium",
    )

    # --------------------------------------------------------
    # MACHINE CONFIGURATION
    # --------------------------------------------------------

    with left_column:

        with st.container(border=True):

            st.markdown(
                '<div class="section-heading">'
                'Machine Configuration'
                '</div>',
                unsafe_allow_html=True,
            )

            st.caption(
                "Select the six pre-cycle machine settings."
            )

            with st.form("machine_settings_form"):

                input_values = {}

                for feature in FEATURES:

                    (
                        label,
                        minimum,
                        maximum,
                        default,
                        step,
                        unit,
                    ) = SETTINGS[feature]

                    input_values[feature] = st.slider(
                        f"{label} ({unit})",
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        step=step,
                        key=f"slider_{feature}",
                    )

                analyze_clicked = (
                    st.form_submit_button(
                        "Analyze Process Configuration",
                        type="primary",
                        use_container_width=True,
                    )
                )

            if analyze_clicked:

                input_frame = make_input_frame(
                    input_values
                )

                prediction, probabilities = (
                    predict_configuration(
                        input_frame
                    )
                )

                st.session_state.prediction = prediction
                st.session_state.probabilities = probabilities
                st.session_state.analyzed_input = input_frame

    # --------------------------------------------------------
    # PREDICTION RESULTS
    # --------------------------------------------------------

    with right_column:

        with st.container(border=True):

            st.markdown(
                '<div class="section-heading">'
                'Prediction Results'
                '</div>',
                unsafe_allow_html=True,
            )

            if st.session_state.prediction is None:

                st.info(
                    "Select machine settings and click "
                    "Analyze Process Configuration."
                )

            else:

                prediction = st.session_state.prediction

                probabilities = (
                    st.session_state.probabilities
                )

                analyzed_input = (
                    st.session_state.analyzed_input
                )

                # Prediction class

                if prediction == "G":
                    css_class = "decision-safe"
                elif prediction == "R":
                    css_class = "decision-danger"
                else:
                    css_class = "decision-warning"

                st.markdown(
                    f'<div class="decision-card '
                    f'{css_class}">'
                    f'{prediction} — '
                    f'{CLASS_NAMES[prediction]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Circular probabilities

                st.markdown(
                    '<div class="section-heading">'
                    'Class Probability Distribution'
                    '</div>',
                    unsafe_allow_html=True,
                )

                probability_columns = st.columns(
                    3,
                    gap="small",
                )

                for column, label in zip(
                    probability_columns,
                    ["G", "R", "Y"],
                ):

                    with column:

                        st.plotly_chart(
                            create_donut_chart(
                                label,
                                probabilities[label],
                            ),
                            use_container_width=True,
                            config={
                                "displayModeBar": False,
                                "staticPlot": True,
                            },
                            key=f"donut_{label}",
                        )

                        st.caption(
                            CLASS_NAMES[label]
                        )

                # Industrial review

                st.markdown(
                    '<div class="section-heading">'
                    'Industrial Review Decision'
                    '</div>',
                    unsafe_allow_html=True,
                )

                if review_required(probabilities):

                    st.markdown(
                        '<div class="decision-card '
                        'decision-danger">'
                        '⚠ Review Recommended'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    triggered = []

                    if probabilities["R"] >= R_THRESHOLD:
                        triggered.append(
                            f"R = {probabilities['R']:.1%} "
                            f"(threshold {R_THRESHOLD:.0%})"
                        )

                    if probabilities["Y"] >= Y_THRESHOLD:
                        triggered.append(
                            f"Y = {probabilities['Y']:.1%} "
                            f"(threshold {Y_THRESHOLD:.0%})"
                        )

                    st.caption(
                        " | ".join(triggered)
                    )

                else:

                    st.markdown(
                        '<div class="decision-card '
                        'decision-safe">'
                        '✓ No Review Flag'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    st.caption(
                        "Both problem probabilities "
                        "are below their review thresholds."
                    )

                # Automatic insights for R/Y

                if prediction in ["R", "Y"]:

                    st.divider()

                    st.markdown(
                        '<div class="section-heading">'
                        'Engineering Insights'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    st.caption(
                        "One-parameter alternatives "
                        "ranked by estimated R/Y risk."
                    )

                    alternatives = (
                        generate_alternative_settings(
                            analyzed_input
                        )
                    )

                    if alternatives.empty:

                        st.info(
                            "No lower-risk alternative "
                            "was found among the tested "
                            "configurations."
                        )

                    else:

                        display_columns = [
                            "Parameter",
                            "Current",
                            "Suggested",
                            "Unit",
                            "Predicted",
                            "P(G)",
                            "P(R)",
                            "P(Y)",
                            "Review",
                        ]

                        st.dataframe(
                            alternatives[
                                display_columns
                            ],
                            hide_index=True,
                            use_container_width=True,
                            column_config={
                                "P(G)": (
                                    st.column_config.NumberColumn(
                                        "P(G)",
                                        format="percent",
                                    )
                                ),
                                "P(R)": (
                                    st.column_config.NumberColumn(
                                        "P(R)",
                                        format="percent",
                                    )
                                ),
                                "P(Y)": (
                                    st.column_config.NumberColumn(
                                        "P(Y)",
                                        format="percent",
                                    )
                                ),
                                "Review": (
                                    st.column_config.CheckboxColumn(
                                        "Review",
                                        disabled=True,
                                    )
                                ),
                            },
                        )

                        best = alternatives.iloc[0]

                        st.markdown(
                            "**Lowest-risk tested alternative:** "
                            f"Change **{best['Parameter']}** "
                            f"from **{best['Current']:g} "
                            f"{best['Unit']}** to "
                            f"**{best['Suggested']:g} "
                            f"{best['Unit']}**."
                        )

                        st.caption(
                            "Estimated feasible-process "
                            "probability: "
                            f"{best['P(G)']:.1%}"
                        )

                    st.warning(
                        "Suggestions are model-based "
                        "what-if scenarios, not verified "
                        "manufacturing corrections."
                    )

                st.caption(
                    "Predictions describe process "
                    "viability, not guaranteed part "
                    "quality. Probabilities have not "
                    "been independently calibrated."
                )


# ============================================================
# 10. EXPLAINABILITY TAB
# ============================================================

with tab_explainability:

    local_column, global_column = st.columns(
        [1.1, 1],
        gap="medium",
    )

    # LOCAL SHAP

    with local_column:

        with st.container(border=True):

            st.markdown(
                '<div class="section-heading">'
                'Local SHAP Explanation'
                '</div>',
                unsafe_allow_html=True,
            )

            if st.session_state.analyzed_input is None:

                st.info(
                    "Analyze a configuration first "
                    "to display its SHAP explanation."
                )

            else:

                selected_class = st.selectbox(
                    "Class to explain",
                    options=["G", "R", "Y"],
                    index=["G", "R", "Y"].index(
                        st.session_state.prediction
                    ),
                )

                try:

                    explanation = load_explainer()(
                        st.session_state.analyzed_input
                    )

                    class_index = list(
                        model.classes_
                    ).index(selected_class)

                    plt.close("all")

                    shap.plots.waterfall(
                        explanation[
                            0, :, class_index
                        ],
                        show=False,
                    )

                    fig = plt.gcf()
                    fig.set_size_inches(7, 4.5)

                    st.pyplot(
                        fig,
                        clear_figure=True,
                        use_container_width=True,
                    )

                    st.caption(
                        f"Positive SHAP values increase "
                        f"the model's estimated "
                        f"P({selected_class}); negative "
                        "values decrease it."
                    )

                except Exception as error:

                    st.error(
                        f"SHAP explanation error: {error}"
                    )

    # GLOBAL SHAP

    with global_column:

        with st.container(border=True):

            st.markdown(
                '<div class="section-heading">'
                'Global Feature Importance'
                '</div>',
                unsafe_allow_html=True,
            )

            try:

                importance = (
                    load_global_importance()
                )

                fig = go.Figure()

                fig.add_trace(
                    go.Bar(
                        x=importance[
                            "Mean Absolute SHAP"
                        ],
                        y=importance["Feature"],
                        orientation="h",
                        marker_color=BLUE,
                        hovertemplate=(
                            "%{y}: %{x:.4f}"
                            "<extra></extra>"
                        ),
                    )
                )

                fig.update_layout(
                    height=340,
                    margin=dict(
                        l=5, r=5, t=10, b=35
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Mean Absolute SHAP",
                    yaxis=dict(
                        autorange="reversed"
                    ),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    },
                )

                st.caption(
                    "Calculated using the "
                    "validation dataset."
                )

            except Exception as error:

                st.error(
                    f"Global SHAP error: {error}"
                )

    st.caption(
        "SHAP explains model predictions, "
        "not physical causality."
    )


# ============================================================
# 11. MODEL PERFORMANCE TAB
# ============================================================

with tab_performance:

    st.subheader(
        "Random Forest — Model Evaluation Results"
    )

    metrics = st.columns(4)

    metrics[0].metric(
        "Test Accuracy",
        "98.67%",
    )

    metrics[1].metric(
        "Test Macro F1",
        "0.9466",
    )

    metrics[2].metric(
        "R Recall",
        "100%",
    )

    metrics[3].metric(
        "Y Recall",
        "85.71%",
    )

    st.divider()

    matrix_column, policy_column = st.columns(
        2,
        gap="medium",
    )

    with matrix_column:

        st.markdown(
            "#### Held-Out Test Confusion Matrix"
        )

        confusion_matrix = np.array([
            [103, 0, 1],
            [0, 39, 0],
            [0, 1, 6],
        ])

        fig = go.Figure(
            go.Heatmap(
                z=confusion_matrix,
                x=["G", "R", "Y"],
                y=["G", "R", "Y"],
                text=confusion_matrix,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            )
        )

        fig.update_yaxes(
            autorange="reversed"
        )

        fig.update_layout(
            height=330,
            xaxis_title="Predicted Class",
            yaxis_title="Actual Class",
            margin=dict(
                l=20, r=20, t=20, b=20
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with policy_column:

        st.markdown(
            "#### Combined Review Policy"
        )

        st.metric(
            "Problem Detection Recall",
            "100%",
        )

        st.metric(
            "Review Precision",
            "79.31%",
        )

        st.metric(
            "False Review Flags",
            "12",
        )

        st.metric(
            "Configurations Flagged",
            "58 / 150",
        )

        st.caption(
            "Frozen thresholds: "
            "P(R) ≥ 30% or P(Y) ≥ 10%."
        )

    st.divider()
    st.markdown("#### Experimental-Stage Transfer Audit")
    st.caption(
        "Separate robustness audit: A new Random Forest was trained "
        "on Stage 1 observations and evaluated on 232 Stage 2 observations. "
        "This uses a different training and evaluation strategy from the "
        "original random held-out test."
    )

    stage_metrics = st.columns(4)
    stage_metrics[0].metric("Stage 2 Accuracy", "84.48%")
    stage_metrics[1].metric("Stage 2 Macro F1", "0.54")
    stage_metrics[2].metric("Stage 2 R Recall", "60.0%")
    stage_metrics[3].metric("Stage 2 Y Recall", "0.0% (0/5)")

    stage_left, stage_right = st.columns([1, 1.1], gap="medium")
    with stage_left:
        stage_matrix = np.array([
            [166, 3, 8],
            [15, 30, 5],
            [4, 1, 0],
        ])
        stage_fig = go.Figure(
            go.Heatmap(
                z=stage_matrix,
                x=["G", "R", "Y"],
                y=["G", "R", "Y"],
                text=stage_matrix,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            )
        )
        stage_fig.update_yaxes(autorange="reversed")
        stage_fig.update_layout(
            height=285,
            xaxis_title="Predicted Class",
            yaxis_title="Actual Class",
            margin=dict(l=20, r=20, t=10, b=20),
        )
        st.plotly_chart(stage_fig, use_container_width=True)

    with stage_right:
        st.markdown("**What this audit shows**")
        st.write(
            "The model correctly classified **196 of 232** Stage 2 "
            "observations. It identified **30 of 50** R cases and "
            "**0 of 5** Y cases."
        )
        st.write(
            "All 232 Stage 2 observations contain at least one feature "
            "value not present in Stage 1. Their values still fall within "
            "the overall Stage 1 feature ranges."
        )
        st.warning(
            "Random-split test performance does not establish reliable "
            "generalization to a different experimental design. "
            "The Stage 2 Y sample contains only five observations."
        )

    st.warning(
        "This model was evaluated on one experimental dataset. "
        "Generalization to other machines, materials, and production "
        "conditions has not been established."
    )


# ============================================================
# 12. ABOUT PROJECT — ENGINEERING CASE STUDY
# ============================================================

with tab_about:

    # --------------------------------------------------------
    # INTRODUCTION
    # --------------------------------------------------------

    st.markdown("### About This Project")

    st.markdown(
        """
        <div class="about-intro">
        <strong>AI-Based Predictive Quality Control
        for Polypropylene Injection Molding</strong>

        <p style="margin-bottom:0;">
        This project investigates whether machine settings
        can be used to predict injection-molding process
        viability before a manufacturing cycle begins.
        A Random Forest classifier estimates the likelihood
        of feasible, infeasible, and short-shot conditions.
        The results are presented through an interactive
        engineering dashboard with SHAP explanations
        and model-based review recommendations.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_about, right_about = st.columns(
        [1, 1],
        gap="medium",
    )

    # --------------------------------------------------------
    # LEFT COLUMN
    # --------------------------------------------------------

    with left_about:

        # INDUSTRIAL PROBLEM

        with st.container(border=True):

            st.markdown(
                "#### 1. Industrial Problem"
            )

            st.write(
                "Injection molding requires suitable "
                "combinations of temperature, pressure, "
                "and timing parameters."
            )

            st.write(
                "Unsuitable settings can lead to "
                "infeasible processes or short shots, "
                "potentially increasing material waste, "
                "machine time, and inspection effort."
            )

            st.info(
                "Project objective: Estimate process "
                "viability from six pre-cycle settings "
                "and flag potentially problematic "
                "configurations for engineering review."
            )

        # DATASET

        with st.container(border=True):

            st.markdown(
                "#### 2. Dataset & Methodology"
            )

            dataset_metrics = st.columns(3)

            dataset_metrics[0].metric(
                "Observations",
                "1,000",
            )

            dataset_metrics[1].metric(
                "Input Features",
                "6",
            )

            dataset_metrics[2].metric(
                "Output Classes",
                "3",
            )

            st.markdown(
                "**Train / Validation / Test Split**"
            )

            split_data = pd.DataFrame({
                "Dataset": [
                    "Training",
                    "Validation",
                    "Held-Out Test",
                ],
                "Observations": [
                    700, 150, 150
                ],
                "Percentage": [
                    "70%", "15%", "15%"
                ],
            })

            st.dataframe(
                split_data,
                hide_index=True,
                use_container_width=True,
            )

            st.caption(
                "Stratified splitting preserves "
                "approximately the same G/R/Y "
                "class proportions in each subset."
            )

            st.markdown("**Prediction Inputs**")

            input_table = pd.DataFrame({
                "Feature": FEATURES,
                "Description": [
                    SETTINGS[f][0]
                    for f in FEATURES
                ],
                "Unit": [
                    SETTINGS[f][5]
                    for f in FEATURES
                ],
            })

            st.dataframe(
                input_table,
                hide_index=True,
                use_container_width=True,
            )

            st.caption(
                "Post-cycle measurements such as "
                "part weight, energy consumption, "
                "and cycle time were excluded "
                "because they would not be available "
                "before the cycle."
            )

        # WORKFLOW

        with st.container(border=True):

            st.markdown(
                "#### 3. Engineering Workflow"
            )

            st.markdown(
                """
                <div class="workflow-step">
                Six Pre-Cycle Machine Settings
                </div>

                <div class="workflow-arrow">↓</div>

                <div class="workflow-step">
                Data Preparation & Trained Random Forest
                </div>

                <div class="workflow-arrow">↓</div>

                <div class="workflow-step">
                G / R / Y Classification
                </div>

                <div class="workflow-arrow">↓</div>

                <div class="workflow-step">
                Risk-Based Review Decision
                </div>

                <div class="workflow-arrow">↓</div>

                <div class="workflow-step">
                SHAP Explanation & Engineering Insights
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                "Engineering Insights evaluate "
                "one-parameter-at-a-time alternatives "
                "using the trained model."
            )

    # --------------------------------------------------------
    # RIGHT COLUMN
    # --------------------------------------------------------

    with right_about:

        # VERIFIED RESULTS

        with st.container(border=True):

            st.markdown(
                "#### 4. Verified Results"
            )

            result_columns = st.columns(2)

            result_columns[0].metric(
                "Test Accuracy",
                "98.67%",
            )

            result_columns[1].metric(
                "Test Macro F1",
                "0.9466",
            )

            result_columns[0].metric(
                "R Recall",
                "100%",
            )

            result_columns[1].metric(
                "Y Recall",
                "85.71%",
            )

            st.markdown(
                "**Held-Out Test Performance**"
            )

            st.write(
                "The Random Forest correctly "
                "classified **148 of 150** "
                "held-out test observations."
            )

            st.write(
                "The combined review policy "
                "flagged **46 of 46** problematic "
                "conditions (R or Y), with "
                "**12 additional review flags**."
            )

            st.write(
                "A separate experimental-stage transfer audit achieved "
                "**84.48% accuracy**, with **60% R recall** and "
                "**0/5 Y cases detected**. This exposes an important "
                "generalization limitation."
            )

            st.caption(
                "Both confusion matrices and the review-policy metrics "
                "are available under Model Performance."
            )

        # OUTPUT CLASSES

        with st.container(border=True):

            st.markdown(
                "#### 5. Predicted Process Classes"
            )

            st.success(
                "G — Feasible Process"
            )

            st.error(
                "R — Infeasible Process"
            )

            st.warning(
                "Y — Feasible Process with Short Shots"
            )

            st.caption(
                "G indicates process viability, "
                "not guaranteed defect-free parts."
            )

        # TECHNICAL CONTRIBUTIONS

        with st.container(border=True):

            st.markdown(
                "#### 6. Technical Contributions"
            )

            st.markdown(
                """
                - Exploratory data analysis and
                  class-imbalance assessment
                - Leakage-aware feature selection
                  and preprocessing
                - Baseline model comparison:
                  Logistic Regression,
                  Random Forest, and XGBoost
                - Hyperparameter tuning and
                  held-out model evaluation
                - Risk-based engineering review rules
                - Global and local SHAP explainability
                - Interactive Streamlit dashboard
                - Model-based alternative-setting exploration
                """
            )

        # TECHNOLOGY STACK

        with st.container(border=True):

            st.markdown(
                "#### 7. Tools & Technologies"
            )

            st.markdown(
                """
                **Programming:** Python

                **Data Analysis:** pandas, NumPy

                **Machine Learning:** scikit-learn,
                XGBoost

                **Explainable AI:** SHAP

                **Visualization:** Matplotlib, Plotly

                **Application:** Streamlit

                **Development:** VS Code
                """
            )

    # --------------------------------------------------------
    # LIMITATIONS & FUTURE WORK
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        "### 8. Limitations & Future Work"
    )

    limitation_column, future_column = st.columns(
        2,
        gap="medium",
    )

    with limitation_column:

        with st.container(border=True):

            st.markdown(
                "#### Current Limitations"
            )

            st.markdown(
                """
                - Model developed using one
                  experimental dataset
                - Only one held-out test split
                - Rare Y class has limited examples
                - Random Forest probabilities
                  have not been independently calibrated
                - Model-based setting suggestions
                  are not physical causal conclusions
                - Stage-transfer audit showed reduced
                  performance: 84.48% accuracy, 60% R recall,
                  and 0/5 Y cases detected
                - No production-machine validation
                  has been performed
                """
            )

    with future_column:

        with st.container(border=True):

            st.markdown(
                "#### Potential Future Improvements"
            )

            st.markdown(
                """
                - Test generalization on additional
                  experimental or production datasets
                - Evaluate probability calibration
                - Assess performance across different
                  machines and material grades
                - Experimentally validate
                  suggested process settings
                - Investigate model monitoring
                  and production integration
                """
            )

    st.warning(
        "Research and engineering portfolio "
        "demonstration only. The system has "
        "not been validated or approved "
        "for industrial production control."
    )
