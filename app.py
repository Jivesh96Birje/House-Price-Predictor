import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from joblib import load
from pathlib import Path
import sqlite3
import time
from typing import Dict, Tuple

try:
    import folium  # type: ignore
    from streamlit_folium import st_folium  # type: ignore

    _MAP_AVAILABLE = True
except Exception:
    folium = None
    st_folium = None
    _MAP_AVAILABLE = False

# Set page configuration for dark mode
st.set_page_config(
    page_title="Maharashtra House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enable Dark Mode using custom CSS (game-like / neon)
st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(1200px 600px at 20% 10%, rgba(124,77,255,0.18), rgba(18,18,18,1) 55%),
                        radial-gradient(900px 500px at 80% 15%, rgba(0,229,255,0.12), rgba(18,18,18,1) 60%),
                        #0e0f14;
            color: #e8e8f2;
        }
        .stButton>button {
            background: linear-gradient(90deg, rgba(124,77,255,1), rgba(0,229,255,1));
            color: #0b0c10;
            border: 0;
            border-radius: 14px;
            padding: 0.65rem 1rem;
            font-weight: 800;
            letter-spacing: 0.4px;
            box-shadow: 0 0 0 1px rgba(255,255,255,0.06), 0 10px 30px rgba(0,229,255,0.10);
        }
        .stButton>button:hover {
            filter: brightness(1.05);
            transform: translateY(-1px);
        }
        .stSidebar {
            background: linear-gradient(180deg, rgba(18,19,28,1), rgba(10,10,14,1));
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 14px 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        }
        div[data-testid="stMetric"] label {
            font-size: 0.95rem !important;
            letter-spacing: 0.35px;
            opacity: 0.9;
        }
        div[data-testid="stMetricValue"] {
            font-size: 2.0rem !important;
            font-weight: 900 !important;
            color: #00FF00 !important;
        }
        div[data-testid="stMetricLabel"], div[data-testid="stMetric"] label {
            color: #e0e0e0 !important;
        }
        div[data-testid="stMetricDelta"] {
            font-weight: 700 !important;
            color: #e0e0e0 !important;
        }
        div[data-testid="stAlert"] {
            color: #e0e0e0 !important;
        }
        /* The deep-target fix for Streamlit Progress Bars */
         div[data-testid="stProgressBar"] div[role="progressbar"] > div {
         background-color: #FF00FF !important;
        }
        .hud {
            background: linear-gradient(90deg, rgba(124,77,255,0.20), rgba(0,229,255,0.10));
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 16px;
            padding: 12px 14px;
            box-shadow: 0 12px 40px rgba(0,0,0,0.30);
        }
    </style>
""",
    unsafe_allow_html=True,
)

# App title
st.title("🌟 Maharashtra House Price Prediction App 🏡")

DATASET_PATH = Path("maharashtra_house_prices.csv")
MODEL_PATH = Path("rf_model.pkl")
DB_PATH = Path("predictions.db")
FEATURE_COLUMNS = [
    "Size (sqft)",
    "BHK",
    "Bathrooms",
    "Age of Property (years)",
    "Floor Number",
    "Total Floors",
    "Parking",
]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _pct(value: float, min_v: float, max_v: float) -> float:
    if max_v <= min_v:
        return 0.0
    return _clamp01((value - min_v) / (max_v - min_v))


LOCATION_COORDS: Dict[str, Tuple[float, float]] = {
    "Mumbai": (19.0760, 72.8777),
    "Pune": (18.5204, 73.8567),
    "Nagpur": (21.1458, 79.0882),
    "Nashik": (20.0059, 73.7897),
    "Aurangabad": (19.8762, 75.3433),
    "Thane": (19.2183, 72.9781),
    "Navi Mumbai": (19.0330, 73.0297),
    "Kolhapur": (16.7050, 74.2433),
    "Solapur": (17.6599, 75.9064),
    "Amravati": (20.9374, 77.7796),
}

# Load dataset automatically from local disk
@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        st.error(
            "Dataset not found! Please ensure 'maharashtra_house_prices.csv' is in the project directory."
        )
        st.stop()

data = load_data()


@st.cache_resource
def load_model():
    try:
        artifact = load(MODEL_PATH)
    except FileNotFoundError:
        st.error(
            "Model not found! Please run `python train_model.py` to generate 'rf_model.pkl' in the project directory."
        )
        st.stop()

    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"], artifact.get("feature_columns", FEATURE_COLUMNS)

    return artifact, FEATURE_COLUMNS


model, model_feature_columns = load_model()


@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            location TEXT,
            size_sqft INTEGER,
            bhk INTEGER,
            bathrooms INTEGER,
            age_years INTEGER,
            floor_number INTEGER,
            total_floors INTEGER,
            parking INTEGER,
            predicted_price_inr REAL
        )
        """
    )
    conn.commit()


def log_prediction(
    conn: sqlite3.Connection,
    *,
    location: str,
    size_sqft: int,
    bhk: int,
    bathrooms: int,
    age_years: int,
    floor_number: int,
    total_floors: int,
    parking: int,
    predicted_price_inr: float,
) -> None:
    conn.execute(
        """
        INSERT INTO predictions (
            location,
            size_sqft,
            bhk,
            bathrooms,
            age_years,
            floor_number,
            total_floors,
            parking,
            predicted_price_inr
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            location,
            int(size_sqft),
            int(bhk),
            int(bathrooms),
            int(age_years),
            int(floor_number),
            int(total_floors),
            int(parking),
            float(predicted_price_inr),
        ),
    )
    conn.commit()


def fetch_recent_predictions(conn: sqlite3.Connection, limit: int = 5) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT
            created_at,
            location,
            size_sqft AS "Size (sqft)",
            bhk AS "BHK",
            bathrooms AS "Bathrooms",
            age_years AS "Age of Property (years)",
            floor_number AS "Floor Number",
            total_floors AS "Total Floors",
            parking AS "Parking",
            predicted_price_inr AS "Predicted Price (INR)"
        FROM predictions
        ORDER BY datetime(created_at) DESC, id DESC
        LIMIT ?
        """,
        conn,
        params=(int(limit),),
    )


db_conn = get_db_connection()
init_db(db_conn)

# Location selection (map-driven)
st.sidebar.subheader("🗺 Location Selection")
location_options = sorted(data["Location"].unique())

if "selected_locations" not in st.session_state:
    st.session_state.selected_locations = ["Mumbai"] if "Mumbai" in location_options else []

if st.sidebar.button("Clear selected locations"):
    st.session_state.selected_locations = []

if _MAP_AVAILABLE:
    st.sidebar.caption("Tip: click a marker to add that location.")
    m = folium.Map(
        location=[19.2, 73.2],
        zoom_start=6,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    for loc in location_options:
        coords = LOCATION_COORDS.get(loc)
        if not coords:
            continue
        folium.CircleMarker(
            location=list(coords),
            radius=7,
            color="#00e5ff",
            weight=2,
            fill=True,
            fill_color="#7c4dff",
            fill_opacity=0.6,
            popup=loc,
            tooltip=loc,
        ).add_to(m)

    map_out = st_folium(m, height=360, width=None, returned_objects=["last_object_clicked_popup"])
    clicked = None
    if isinstance(map_out, dict):
        clicked = map_out.get("last_object_clicked_popup")

    if clicked and clicked in location_options:
        if clicked not in st.session_state.selected_locations:
            st.session_state.selected_locations.append(clicked)

    locations = st.session_state.selected_locations
else:
    st.sidebar.info("Install `folium` + `streamlit-folium` to enable the interactive map.")
    locations = st.sidebar.multiselect(
        "Select Locations",
        location_options,
        default=st.session_state.selected_locations,
    )
    st.session_state.selected_locations = locations

st.sidebar.markdown(
    f"""
    <div class="hud">
      <div style="font-weight:800; letter-spacing:0.3px;">Selected</div>
      <div style="opacity:0.9;">{", ".join(locations) if locations else "None"}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Filter data based on selected locations
data_filtered = data[data["Location"].isin(locations)] if locations else data.iloc[0:0]

# Data overview
st.subheader("📊 Data Overview")
st.dataframe(data_filtered.head(20), use_container_width=True)

# Visualizations
st.subheader("📈 Price Distribution by Location")
fig = px.histogram(
    data,
    x="Price (INR)",
    color="Location", # This tells Plotly to separate the bars by city
    nbins=50,
    title="Price Distribution by Location",
    # This array assigns a unique, high-contrast neon color to each location
    color_discrete_sequence=[
        "#0047AB", # Deep Blue
        "#FF00FF", # Neon Magenta 
        "#00FF00", # Bright Green
        "#FFC107", # Electric Amber
        "#9D00FF", # Synthwave Purple
        "#FF6F61"  # Vibrant Coral
    ]
)

# Optional: Make the chart background transparent to blend with your dark theme
fig.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e0e0e0")
)

st.plotly_chart(fig)

st.subheader("🔍 Price vs Size (sqft)")
fig, ax = plt.subplots(figsize=(10, 6))

# Define the exact same colors for consistency across graphs
custom_palette = ["#0047AB", "#FF00FF", "#00FF00", "#FFC107", "#9D00FF", "#FF6F61"]

sns.scatterplot(
    data=data_filtered, 
    x='Size (sqft)', 
    y='Price (INR)', 
    hue='Location', 
    palette=custom_palette, # Applies the custom colors here
    ax=ax
)

ax.set_title("Price vs Size (sqft)", color="#e0e0e0", fontsize=16)
ax.tick_params(colors="#e0e0e0")
fig.patch.set_facecolor('#121212') # Matches the dark background
ax.set_facecolor('#121212')

st.pyplot(fig)

st.subheader("🔍 Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(data_filtered.corr(numeric_only=True), annot=True, cmap="YlGnBu", ax=ax)
ax.set_title('Feature Correlation Heatmap', fontsize=16)
st.pyplot(fig)

st.sidebar.subheader("📊 Model Accuracy")
st.sidebar.write(f"Accuracy: 77.27%")

# Prediction (live sliders + game-like progress)
st.subheader("🔮 Predict House Price")
st.markdown(
    """
    <div class="hud">
      <div style="font-weight:900; font-size:1.05rem; letter-spacing:0.5px;">
        Control Panel
      </div>
      <div style="opacity:0.9;">Tune the build parameters. Watch the bars power up as you play.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    size = st.slider("Size (sqft)", min_value=300, max_value=5000, value=1000, key="size")
    st.progress(int(_pct(size, 300, 5000) * 100), text="Size power")
    age = st.slider("Age of Property (years)", min_value=0, max_value=50, value=5, key="age")
    st.progress(int(_pct(age, 0, 50) * 100), text="Age modifier")

with col2:
    bhk = st.slider("BHK", min_value=1, max_value=5, value=3, key="bhk")
    st.progress(int(_pct(bhk, 1, 5) * 100), text="Room capacity")
    floor_number = st.slider("Floor Number", min_value=1, max_value=40, value=2, key="floor_number")
    st.progress(int(_pct(floor_number, 1, 40) * 100), text="Elevation")

with col3:
    bathrooms = st.slider("Bathrooms", min_value=1, max_value=4, value=2, key="bathrooms")
    st.progress(int(_pct(bathrooms, 1, 4) * 100), text="Comfort")
    total_floors = st.slider(
        "Total Floors", min_value=floor_number, max_value=50, value=10, key="total_floors"
    )
    st.progress(int(_pct(total_floors, 1, 50) * 100), text="Building scale")
    parking = st.slider("Parking", min_value=0, max_value=3, value=1, key="parking")
    st.progress(int(_pct(parking, 0, 3) * 100), text="Utility")

# Overall "energy" meter based on interaction (deviation from defaults)
defaults = {
    "size": 1000,
    "age": 5,
    "bhk": 3,
    "floor_number": 2,
    "bathrooms": 2,
    "total_floors": 10,
    "parking": 1,
}
energy = (
    _pct(abs(size - defaults["size"]), 0, 5000 - 300)
    + _pct(abs(age - defaults["age"]), 0, 50)
    + _pct(abs(bhk - defaults["bhk"]), 0, 4)
    + _pct(abs(floor_number - defaults["floor_number"]), 0, 39)
    + _pct(abs(bathrooms - defaults["bathrooms"]), 0, 3)
    + _pct(abs(total_floors - defaults["total_floors"]), 0, 49)
    + _pct(abs(parking - defaults["parking"]), 0, 3)
) / 7.0
st.sidebar.subheader("⚡ Interaction Energy")
st.sidebar.progress(int(energy * 100), text=f"Charge: {int(energy * 100)}%")

reveal = st.button("Reveal Price", use_container_width=True)

if reveal:
    if not locations:
        st.warning("Select at least one location on the map first.")
        st.stop()

    input_data = np.array([[size, bhk, bathrooms, age, floor_number, total_floors, parking]])
    input_df = pd.DataFrame(input_data, columns=model_feature_columns)

    # Small reveal animation
    bar = st.progress(0, text="Calibrating model…")
    for i in range(1, 101, 5):
        bar.progress(i, text="Calibrating model…" if i < 60 else "Generating prediction…")
        time.sleep(0.02)
    bar.empty()

    predicted_price = float(model.predict(input_df)[0])

    # Log to SQLite
    log_prediction(
        db_conn,
        location=", ".join(locations),
        size_sqft=size,
        bhk=bhk,
        bathrooms=bathrooms,
        age_years=age,
        floor_number=floor_number,
        total_floors=total_floors,
        parking=parking,
        predicted_price_inr=predicted_price,
    )

    # Satisfying reveal using metric cards
    avg_price = float(data_filtered["Price (INR)"].mean()) if not data_filtered.empty else 0.0
    delta_vs_avg = predicted_price - avg_price if avg_price else None

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Predicted Price (INR)",
        f"₹ {predicted_price:,.0f}",
        delta=(f"₹ {delta_vs_avg:,.0f} vs selected avg" if delta_vs_avg is not None else None),
    )
    m2.metric(
        "Locations",
        f"{len(locations)}",
        delta=", ".join(locations[:2]) + ("…" if len(locations) > 2 else ""),
    )
    m3.metric("Build Score", f"{int(energy * 100)}%", delta="Interaction energy")

    # Save result to CSV
    result_df = pd.DataFrame(
        {
            "Location": [", ".join(locations)],
            "Size (sqft)": [size],
            "BHK": [bhk],
            "Bathrooms": [bathrooms],
            "Age of Property (years)": [age],
            "Floor Number": [floor_number],
            "Total Floors": [total_floors],
            "Parking": [parking],
            "Predicted Price (INR)": [predicted_price],
        }
    )

    result_df.to_csv("prediction_result.csv", index=False)

    with open("prediction_result.csv", "rb") as file:
        st.download_button(
            label="Download Prediction Result as CSV",
            data=file,
            file_name="prediction_result.csv",
            mime="text/csv",
        )

# Reset form button
reset_btn = st.button('Reset Form 🔄')
if reset_btn:
    st.session_state.size = 1000
    st.session_state.age = 5
    st.session_state.bhk = 3
    st.session_state.floor_number = 2
    st.session_state.bathrooms = 2
    st.session_state.total_floors = 10
    st.session_state.parking = 1
    st.experimental_rerun()

st.divider()
st.subheader("🗄 5 Most Recent Predictions")
recent_df = fetch_recent_predictions(db_conn, limit=5)
if recent_df.empty:
    st.info("No predictions logged yet. Make a prediction to see it here.")
else:
    st.dataframe(recent_df, use_container_width=True)