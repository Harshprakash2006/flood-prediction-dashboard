"""Interactive Streamlit dashboard for the trained flood prediction model."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH, DATA_PATH, OUTPUTS = ROOT / "models" / "best_flood_model.joblib", ROOT / "data" / "flood_risk_dataset_india.csv", ROOT / "outputs"
TARGET = "Flood Occurred"

def display_name(name: str) -> str:
    return name.replace("Â", "").replace("�", "")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python src/train.py --data data/flood_risk_dataset_india.csv")
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_reference_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

def widget_key(column: str) -> str:
    return f"input_{column}"

def initialise_inputs(data: pd.DataFrame, features: list[str]) -> None:
    numeric = data[features].select_dtypes(include="number").columns.tolist()
    for feature in features:
        key = widget_key(feature)
        if key not in st.session_state:
            st.session_state[key] = float(data[feature].median()) if feature in numeric else sorted(data[feature].dropna().astype(str).unique())[0]

def set_scenario(data: pd.DataFrame, scenario: str, features: list[str]) -> None:
    numeric = data[features].select_dtypes(include="number").columns.tolist()
    categorical = [f for f in features if f not in numeric]
    quantile = {"Lower-risk scenario": .25, "Typical conditions": .50, "Higher-risk scenario": .75}[scenario]
    for feature in numeric:
        st.session_state[widget_key(feature)] = float(data[feature].quantile(quantile))
    for feature in categorical:
        st.session_state[widget_key(feature)] = sorted(data[feature].dropna().astype(str).unique())[0]
    if scenario == "Higher-risk scenario":
        for hint in ("Rainfall", "Humidity", "River Discharge", "Water Level", "Historical Floods"):
            found = next((f for f in numeric if hint.lower() in f.lower()), None)
            if found: st.session_state[widget_key(found)] = float(data[found].quantile(.90))
        elevation = next((f for f in numeric if "elevation" in f.lower()), None)
        if elevation: st.session_state[widget_key(elevation)] = float(data[elevation].quantile(.10))

def predictor_tab(model, data: pd.DataFrame, features: list[str]) -> None:
    numeric = data[features].select_dtypes(include="number").columns.tolist()
    categorical = [f for f in features if f not in numeric]
    initialise_inputs(data, features)
    st.subheader("Try a scenario")
    for col, scenario in zip(st.columns(3), ["Lower-risk scenario", "Typical conditions", "Higher-risk scenario"]):
        col.button(scenario, width="stretch", on_click=set_scenario, args=(data, scenario, features))
    st.caption("Scenario buttons use observed dataset percentiles. You can edit every value below.")
    with st.form("prediction_form"):
        st.subheader("Environmental and location inputs")
        columns, values = st.columns(2), {}
        for index, feature in enumerate(numeric):
            series = data[feature].dropna(); minimum, maximum = float(series.min()), float(series.max())
            with columns[index % 2]:
                values[feature] = st.number_input(display_name(feature), min_value=minimum, max_value=maximum, step=max((maximum-minimum)/100, .01), key=widget_key(feature), help=f"Dataset range: {minimum:.2f} to {maximum:.2f}")
        for index, feature in enumerate(categorical):
            with columns[(index + len(numeric)) % 2]:
                values[feature] = st.selectbox(display_name(feature), sorted(data[feature].dropna().astype(str).unique()), key=widget_key(feature))
        submitted = st.form_submit_button("Predict flood risk", type="primary", width="stretch")
    if submitted:
        input_frame = pd.DataFrame([[values[f] for f in features]], columns=features)
        prediction, probability = int(model.predict(input_frame)[0]), float(model.predict_proba(input_frame)[0][1])
        st.divider(); result, gauge = st.columns(2)
        with result:
            (st.error if prediction else st.success)("Prediction: Flood likely" if prediction else "Prediction: No flood predicted")
            st.metric("Estimated flood probability", f"{probability:.1%}")
        with gauge:
            st.write("Flood-probability indicator"); st.progress(probability)
            st.caption("Educational baseline only—not an emergency warning.")
        percentiles = pd.DataFrame({"Feature": [display_name(f) for f in numeric], "Your value percentile": [round((data[f] <= float(values[f])).mean()*100, 1) for f in numeric]}).sort_values("Your value percentile", ascending=False)
        st.subheader("Where your numeric inputs sit in this dataset")
        st.bar_chart(percentiles.set_index("Feature"), horizontal=True)
        with st.expander("View submitted values"):
            st.dataframe(input_frame.rename(columns=display_name), hide_index=True, width="stretch")

def explorer_tab(data: pd.DataFrame, features: list[str]) -> None:
    st.subheader("Explore the training data")
    st.caption(f"{len(data):,} observations · {len(features)} predictors · target: {TARGET}")
    numeric = data[features].select_dtypes(include="number").columns.tolist()
    categorical = [f for f in features if f not in numeric]
    left, right = st.columns(2)
    with left:
        feature = st.selectbox("Distribution feature", numeric, format_func=display_name)
        counts, bins = np.histogram(data[feature], bins=25)
        st.bar_chart(pd.DataFrame({"Range start": bins[:-1], "Rows": counts}).set_index("Range start"))
    with right:
        feature = st.selectbox("Flood-rate grouping", categorical, format_func=display_name)
        rates = data.groupby(feature, observed=True)[TARGET].mean().mul(100).sort_values(ascending=False)
        st.bar_chart(rates); st.caption("Percentage of rows with flood occurrence = 1.")
    selection = st.radio("Show rows", ["All", "Flood occurred", "No flood"], horizontal=True)
    filtered = data if selection == "All" else data[data[TARGET] == (1 if selection == "Flood occurred" else 0)]
    st.dataframe(filtered.rename(columns=display_name), width="stretch", height=330)

def insights_tab() -> None:
    st.subheader("Model performance")
    comparison_path, importance_path = OUTPUTS / "model_comparison.csv", OUTPUTS / "random_forest_top_10_features.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path); best = comparison.iloc[0]
        one, two, three = st.columns(3); one.metric("Best baseline", best["model"]); two.metric("Best ROC-AUC", f"{best['roc_auc']:.3f}"); three.metric("Best F1-score", f"{best['f1_score']:.3f}")
        st.dataframe(comparison, hide_index=True, width="stretch")
    if importance_path.exists():
        st.subheader("Random Forest: top 10 features")
        importance = pd.read_csv(importance_path)
        importance["feature"] = importance["feature"].str.replace("numeric__", "", regex=False).str.replace("categorical__", "", regex=False).map(display_name)
        st.bar_chart(importance.set_index("feature")["importance"], horizontal=True)
    st.info("The best baseline ROC-AUC is close to 0.50. Use this as a learning demo, not a real flood-warning system.")

def main() -> None:
    st.set_page_config(page_title="Flood Predictor", page_icon="🌊", layout="wide")
    st.title("🌊 Flood Prediction Dashboard")
    st.write("Explore the dataset, try scenarios, and estimate flood occurrence from local conditions.")
    try: model, data = load_model(), load_reference_data()
    except FileNotFoundError as error: st.error(str(error)); st.stop()
    features = [column for column in data.columns if column != TARGET]
    predictor, explorer, insights = st.tabs(["🔮 Predictor", "📊 Data explorer", "🧠 Model insights"])
    with predictor: predictor_tab(model, data, features)
    with explorer: explorer_tab(data, features)
    with insights: insights_tab()

if __name__ == "__main__":
    main()
