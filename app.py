import streamlit as st
import pandas as pd
import plotly.express as px
from forecast_utils import (
    preprocess_data, forecast_sales,
    calculate_target_analysis, generate_recommendations,
    plot_forecast, plot_actual_vs_forecast,
    plot_daily_bar_chart, generate_daily_table,
    get_forecast_explanation, detect_pattern,
    forecast_by_region
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="🍦 Hico Ice Cream Sales Forecast", layout="wide")
st.markdown("<h1 style='text-align:center; color:#FF69B4;'>🍨 Hico Ice Cream Sales Forecast & Target Tracker</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Scoop your way into smart sales planning! 🍧📊</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- SESSION INIT ----------------
if 'forecast_ran' not in st.session_state:
    st.session_state.forecast_ran = False
if 'show_charts' not in st.session_state:
    st.session_state.show_charts = False

# ---------------- FILE UPLOAD ----------------
with st.expander("📤 Upload Your Sales File (CSV or Excel)"):
    uploaded_file = st.file_uploader("Upload File Here", type=["csv", "xlsx"])

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file, engine='openpyxl')
    except Exception as e:
        st.error(f"🍓 Error reading file: {e}")
        st.stop()

    df_raw.columns = df_raw.columns.str.lower().str.strip().str.replace(" ", "_").str.replace(r'[^\w\s]', '', regex=True)
    st.session_state.df_raw = df_raw
    st.success("✅ File uploaded successfully! Let's get forecasting 🍦")

    if st.checkbox("👀 Show Preview of Data"):
        st.dataframe(df_raw.head(), use_container_width=True)

    # ---------------- COLUMN SELECTION ----------------
    with st.expander("🔧 Customize Your Forecast Setup", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            date_col = st.selectbox("📅 Date Column", df_raw.select_dtypes(include=["object", "datetime"]).columns)
        with col2:
            target_col = st.selectbox("🍦 Sales/Quantity Column", df_raw.select_dtypes("number").columns)
        with col3:
            filters = st.multiselect("🔍 Optional Filters", [col for col in df_raw.columns if col not in [date_col, target_col]])

        df_clean = preprocess_data(df_raw, date_col, target_col, filters)
        st.session_state.df_clean = df_clean

    # ---------------- FORECAST SETTINGS ----------------
    st.markdown("## 🔮 Choose Your Forecast Settings")
    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.radio("📘 Forecast Method", ["Prophet", "Linear", "Exponential"], horizontal=True)
        st.caption(get_forecast_explanation(model_choice))
    with col2:
        target_mode = st.radio("📆 Target Period", ["Monthly", "Yearly"], horizontal=True)
        target_value = st.number_input("🎯 Enter Sales Target", step=1000)

    st.markdown("### ⏳ Forecast Duration")
    forecast_range = st.selectbox("How far do you want to forecast?", ["Till Month End", "Till Quarter End", "Till Year End", "Custom Days"])
    forecast_until = 'year_end'
    custom_days = None
    if forecast_range == "Till Month End":
        forecast_until = 'month_end'
    elif forecast_range == "Till Quarter End":
        forecast_until = 'quarter_end'
    elif forecast_range == "Custom Days":
        forecast_until = 'custom'
        custom_days = st.number_input("🗓️ Enter custom days", min_value=1, value=30)

    # ---------------- EVENTS ----------------
    with st.expander("🎉 Add Seasonal Events (e.g., Eid, Summer Launch)"):
        include_events = st.radio("Include Special Dates?", ["No", "Yes"], horizontal=True)
        event_dates = st.date_input("📌 Pick Dates", []) if include_events == "Yes" else []

    # ---------------- RUN FORECAST ----------------
    if st.button("🍨 Generate Forecast!", use_container_width=True):
        forecast_df, last_data_date, days_left, full_df = forecast_sales(
            df_clean, model_choice, target_mode, event_dates,
            forecast_until=forecast_until, custom_days=custom_days
        )
        if forecast_df.empty:
            st.warning("⚠️ Not enough data or no remaining days to forecast.")
        else:
            st.session_state.forecast_df = forecast_df
            st.session_state.full_forecast_df = full_df
            st.session_state.last_data_date = last_data_date
            st.session_state.target_value = target_value
            st.session_state.target_mode = target_mode
            st.session_state.forecast_ran = True
            st.session_state.show_charts = False

    # ---------------- FORECAST RESULTS ----------------
    if st.session_state.forecast_ran:
        st.markdown("---")
        st.markdown("## 🍦 Forecast Results & Dashboard")

        with st.expander("📈 Target Achievement Overview"):
            metrics = calculate_target_analysis(
                st.session_state.df_clean,
                st.session_state.forecast_df,
                st.session_state.last_data_date,
                st.session_state.target_value,
                st.session_state.target_mode
            )

            col1, col2, col3 = st.columns(3)
            keys = list(metrics.keys())
            if len(keys) >= 3:
                col1.metric(label=f"🍧 {keys[0]}", value=str(metrics[keys[0]]))
                col2.metric(label=f"🍨 {keys[1]}", value=str(metrics[keys[1]]))
                col3.metric(label=f"🍦 {keys[2]}", value=str(metrics[keys[2]]))

            st.success("📌 " + generate_recommendations(metrics))

        # ---------------- PATTERN DETECTION ----------------
        st.markdown("### 📊 Trend & Seasonality Insights")
        try:
            st.info(detect_pattern(st.session_state.full_forecast_df.dropna(subset=['yhat']).rename(columns={'yhat': 'y'})))
        except Exception as e:
            st.warning(f"⚠️ Pattern detection failed: {e}")

        if st.button("📊 Show All Charts", use_container_width=True):
            st.session_state.show_charts = True

        if st.session_state.show_charts:
            tab1, tab2, tab3 = st.tabs(["📈 Forecast Trend", "📉 Actual vs Forecast", "📋 Daily Forecast Table"])
            with tab1:
                st.plotly_chart(plot_forecast(st.session_state.full_forecast_df), use_container_width=True)
            with tab2:
                st.plotly_chart(plot_actual_vs_forecast(st.session_state.df_clean, st.session_state.full_forecast_df), use_container_width=True)
            with tab3:
                st.plotly_chart(plot_daily_bar_chart(st.session_state.df_clean), use_container_width=True)
                st.subheader("📆 Daily Forecast Table")
                st.dataframe(generate_daily_table(st.session_state.forecast_df), use_container_width=True)

            # ---------------- REGION FORECAST ----------------
            if 'region' in df_raw.columns:
                with st.expander("🌍 Region-Wise Forecast 🍧"):
                    if st.checkbox("Show Region Summary"):
                        df_region = preprocess_data(df_raw, date_col, target_col, ['region'])
                        region_df = forecast_by_region(df_region, model_choice, event_dates, forecast_until, custom_days)
                        if not region_df.empty:
                            st.dataframe(region_df, use_container_width=True)
                            st.plotly_chart(
                                px.pie(region_df, names='Region', values='Forecasted_Volume',
                                       title='📍 Region-wise Forecasted Contribution'),
                                use_container_width=True
                            )
                        else:
                            st.warning("⚠️ No region data available.")
else:
    st.info("📂 Please upload your ice cream sales data to get started!")
