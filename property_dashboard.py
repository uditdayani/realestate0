import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

# Folium imports
import folium
from streamlit_folium import st_folium

# -----------------------------
# Load data & model
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("processed_properties.csv")

@st.cache_data
def load_suburb_coords():
    return pd.read_csv("suburb_agg.csv")[["suburb", "latitude", "longitude"]]

@st.cache_resource
def load_model():
    return joblib.load("xgb_price_per_sqft_model.pkl")

df = load_data()
suburb_coords = load_suburb_coords()
model = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="🏠 Property Value Dashboard", layout="wide")

st.title("🏠 Property Price per Sqft Dashboard")
st.markdown("Analyze **market prices**, identify **underpriced/overpriced listings**, predict new property prices, and explore properties on the map.")

# Sidebar filters
st.sidebar.header("🔍 Filters")
selected_suburb = st.sidebar.selectbox("Select Suburb", sorted(df["suburb"].unique()))
tolerance = st.sidebar.slider("Fair Price Tolerance (%)", min_value=5, max_value=30, value=15, step=1)

# -----------------------------
# Filtered data & remove duplicates
# -----------------------------
suburb_df = df[df["suburb"] == selected_suburb].copy()

# Columns for identifying duplicates
duplicate_cols = ["suburb", "bhk", "bathroom", "balcony", "price_per_sqft"]

if "scrape_date" in suburb_df.columns:
    suburb_df["scrape_date"] = pd.to_datetime(suburb_df["scrape_date"])
    suburb_df.sort_values("scrape_date", ascending=False, inplace=True)

# Drop duplicates
suburb_df = suburb_df.drop_duplicates(subset=duplicate_cols, keep="first")

# -----------------------------
# Merge suburb coordinates for map
# -----------------------------
suburb_df = pd.merge(suburb_df, suburb_coords, on="suburb", how="left")

# Display market overview
st.subheader(f"📊 Market Overview: {selected_suburb}")
st.write(f"Number of listings: {len(suburb_df)}")
st.write(f"Average price per sqft: ₹{suburb_df['price_per_sqft'].mean():.0f}")



# -----------------------------
# Model Prediction & Flagging
# -----------------------------
features = [
    "bhk", "bathroom", "balcony",
    "school_density", "park_density", "mall_density",
    "hospital_density", "historical_price"
]

suburb_df["predicted_price"] = model.predict(suburb_df[features])
suburb_df["diff_percent"] = ((suburb_df["price_per_sqft"] - suburb_df["predicted_price"]) / suburb_df["predicted_price"]) * 100


def price_flag(diff):
    if diff > tolerance:
        return "🔴 Overpriced"
    elif diff < -tolerance:
        return "🟢 Underpriced"
    else:
        return "⚪ Fair Price"

suburb_df["price_flag"] = suburb_df["diff_percent"].apply(price_flag)

# -----------------------------
# Display summary metrics
# -----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Median Predicted Price", f"₹{suburb_df['predicted_price'].median():,.0f}")
col2.metric("Median Actual Price", f"₹{suburb_df['price_per_sqft'].median():,.0f}")
col3.metric("Underpriced Listings", f"{(suburb_df['price_flag']=='🟢 Underpriced').mean()*100:.1f}%")

# -----------------------------
# Table display
# -----------------------------
st.subheader("🏡 Property-Level Price Analysis")
st.dataframe(
    suburb_df[["title", "location", "bhk", "bathroom", "balcony", "price_per_sqft", "predicted_price", "diff_percent", "price_flag"]]
    .sort_values("diff_percent")
    .reset_index(drop=True)
)

# -----------------------------
# Plot Predicted vs Actual with 45° line
# -----------------------------
st.subheader("📈 Predicted vs Actual Price per Sqft")
fig, ax = plt.subplots(figsize=(7,7))  # square for 45° line
min_val = min(suburb_df["predicted_price"].min(), suburb_df["price_per_sqft"].min())
max_val = max(suburb_df["predicted_price"].max(), suburb_df["price_per_sqft"].max())
ax.scatter(suburb_df["predicted_price"], suburb_df["price_per_sqft"], alpha=0.6)
ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2)
ax.set_xlabel("Predicted Price per Sqft (₹)")
ax.set_ylabel("Actual Price per Sqft (₹)")
ax.set_title(f"Predicted vs Actual Prices ({selected_suburb})")
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)
ax.set_aspect('equal', adjustable='box')
ax.grid(True)
st.pyplot(fig)

# -----------------------------
# User input section for new property
# -----------------------------
st.subheader("🧮 Predict New Property Price & Flag")
col_a, col_b, col_c = st.columns(3)
bhk = col_a.number_input("BHK", min_value=1, max_value=10, value=2)
bathroom = col_b.number_input("Bathrooms", min_value=1, max_value=10, value=2)
balcony = col_c.number_input("Balconies", min_value=0, max_value=7, value=1)

if st.button("Predict Price per Sqft"):
    sample = pd.DataFrame({
        "bhk": [bhk],
        "bathroom": [bathroom],
        "balcony": [balcony],
        "school_density": [suburb_df["school_density"].mean()],
        "park_density": [suburb_df["park_density"].mean()],
        "mall_density": [suburb_df["mall_density"].mean()],
        "hospital_density": [suburb_df["hospital_density"].mean()],
        "historical_price": [suburb_df["historical_price"].mean()]
    })
    
    predicted = model.predict(sample)[0]
    median_pred = suburb_df['predicted_price'].median()
    diff_pct = (predicted - median_pred) / median_pred * 100
    if diff_pct > tolerance:
        flag = "🔴 Overpriced"
    elif diff_pct < -tolerance:
        flag = "🟢 Underpriced"
    else:
        flag = "⚪ Fair Price"

    st.success(f"💰 Estimated Price per Sqft: ₹{predicted:,.0f}  |  {flag}")

# -----------------------------
# Interactive Folium Map (stable view)
# -----------------------------
st.subheader("📍 Explore Properties on Map")

# Map filters
bhk_options = sorted(df["bhk"].dropna().unique())
bhk_filter = st.multiselect(
    "Filter by BHK",
    options=bhk_options,
    default=bhk_options
)

# Limit price slider range between ₹1,000 and ₹100,000 per sqft
price_min, price_max = st.slider(
    "Filter by Price per Sqft (₹)",
    min_value=1000,
    max_value=100000,
    value=(10000, 20000),
    step=500
)


map_df = suburb_df[
    (suburb_df["bhk"].isin(bhk_filter)) &
    (suburb_df["price_per_sqft"] >= price_min) &
    (suburb_df["price_per_sqft"] <= price_max)
].copy()

# Cache or persist map between reruns
if "last_map" not in st.session_state or st.session_state.get("last_filters") != (selected_suburb, tuple(bhk_filter), price_min, price_max):
    if not map_df.empty:
        center_lat = map_df["latitude"].mean()
        center_lon = map_df["longitude"].mean()
    else:
        center_lat = suburb_coords[suburb_coords["suburb"]==selected_suburb]["latitude"].values[0]
        center_lon = suburb_coords[suburb_coords["suburb"]==selected_suburb]["longitude"].values[0]

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    # Add jitter to avoid overlap
    np.random.seed(42)
    jitter = 0.0005
    map_df["latitude_jitter"] = map_df["latitude"] + np.random.uniform(-jitter, jitter, len(map_df))
    map_df["longitude_jitter"] = map_df["longitude"] + np.random.uniform(-jitter, jitter, len(map_df))

    for _, row in map_df.iterrows():
        popup_text = f"""
        <b>Price per Sqft:</b> ₹{row['price_per_sqft']:.0f}<br>
        <b>BHK:</b> {row['bhk']}<br>
        <b>Bathrooms:</b> {row['bathroom']}<br>
        <b>Balconies:</b> {row['balcony']}<br>
        <b>Predicted Price:</b> ₹{row['predicted_price']:.0f}<br>
        <b>Flag:</b> {row['price_flag']}
        """
        color = "green" if row["price_flag"] == "🟢 Underpriced" else "red" if row["price_flag"] == "🔴 Overpriced" else "blue"
        folium.CircleMarker(
            location=[row["latitude_jitter"], row["longitude_jitter"]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=popup_text
        ).add_to(m)

    # Save map and filters in session
    st.session_state.last_map = m
    st.session_state.last_filters = (selected_suburb, tuple(bhk_filter), price_min, price_max)

# Display cached map
st_folium(st.session_state.last_map, width=900, height=500)



# -----------------------------
# Feature Importance
# -----------------------------
if st.checkbox("Show Feature Importances"):
    fig, ax = plt.subplots(figsize=(6,4))
    xgb.plot_importance(model, max_num_features=10, importance_type='weight', height=0.5, ax=ax)
    plt.title("Top 10 Feature Importances")
    st.pyplot(fig)
