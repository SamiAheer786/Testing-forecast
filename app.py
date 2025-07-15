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

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="📈 Smart Sales Forecast App", layout="wide")
st.markdown("<h1 style='text-align:center; color:#4B8BBE;'>🤖 Smart Sales Forecast & Target Tracker - HICO</h1>", unsafe_allow_html=True)
st.markdown("---")

# -------------------- SESSION INIT --------------------
if 'forecast_ran' not in st.session_state:
    st.session_state.forecast_ran = False
if 'show_charts' not in st.session_state:
    st.session_state.show_charts = False

# -------------------- FILE UPLOADER --------------------
with st.expander("📤 Upload Sales File", expanded=True):
    uploaded_file = st.file_uploader("Upload Sales File (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file, engine='openpyxl')
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    df_raw.columns = df_raw.columns.str.lower().str.strip().str.replace(" ", "_").str.replace(r'[^\w\s]', '', regex=True)
    st.session_state.df_raw = df_raw
    st.success("✅ File uploaded successfully!")

    if st.checkbox("🔍 Show Raw Data"):
        st.dataframe(df_raw.head(), use_container_width=True)

    # -------------------- INPUT SELECTION --------------------
    with st.expander("⚙️ Data Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            date_col = st.selectbox("📅 Select Date Column", df_raw.select_dtypes(include=["object", "datetime"]).columns)
        with col2:
            target_col = st.selectbox("🎯 Select Sales/Quantity Column", df_raw.select_dtypes("number").columns)
        with col3:
            filters = st.multiselect("🔎 Filter Columns (Optional)", [col for col in df_raw.columns if col not in [date_col, target_col]])

        df_clean = preprocess_data(df_raw, date_col, target_col, filters)
        st.session_state.df_clean = df_clean

    # -------------------- FORECAST SETUP --------------------
    st.markdown("## 🔮 Forecast Configuration")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            model_choice = st.radio("📘 Forecasting Method", ["Prophet", "Linear", "Exponential"], horizontal=True)
            st.caption(get_forecast_explanation(model_choice))
        with col2:
            target_mode = st.radio("📆 Target Period", ["Monthly", "Yearly"], horizontal=True)
            target_value = st.number_input("💰 Enter Your Sales Target", step=1000)

        st.markdown("### ⏳ Forecast Horizon")
        forecast_range = st.selectbox("Select Forecast Range", ["Till Month End", "Till Quarter End", "Till Year End", "Custom Days"])
        forecast_until = 'year_end'
        custom_days = None
        if forecast_range == "Till Month End":
            forecast_until = 'month_end'
        elif forecast_range == "Till Quarter End":
            forecast_until = 'quarter_end'
        elif forecast_range == "Custom Days":
            forecast_until = 'custom'
            custom_days = st.number_input("📆 Custom Days", min_value=1, value=30)

        # Special events
        with st.expander("🎉 Add Seasonal/Special Event Dates"):
            include_events = st.radio("Include Event Dates?", ["No", "Yes"], horizontal=True)
            event_dates = st.date_input("📌 Select Special Dates", []) if include_events == "Yes" else []

    # -------------------- RUN FORECAST --------------------
    st.markdown("### 🚀 Run Forecast")
    if st.button("▶️ Run Forecast", use_container_width=True):
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

    # -------------------- POST-FORECAST OUTPUT --------------------
    if st.session_state.forecast_ran:
        st.markdown("---")
        st.markdown("## 📊 Forecast Insights")

        with st.expander("📈 Target Achievement Summary", expanded=True):
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
                col1.metric(label=str(keys[0]), value=str(metrics[keys[0]]))
                col2.metric(label=str(keys[1]), value=str(metrics[keys[1]]))
                col3.metric(label=str(keys[2]), value=str(metrics[keys[2]]))

            st.success(generate_recommendations(metrics))

        st.markdown("### 📉 Pattern Detection")
        try:
            st.info(detect_pattern(st.session_state.full_forecast_df.dropna(subset=['yhat']).rename(columns={'yhat': 'y'})))
        except Exception as e:
            st.warning(f"Pattern detection failed: {e}")

        if st.button("📊 Show Charts & Table", use_container_width=True):
            st.session_state.show_charts = True

        if st.session_state.show_charts:
            tab1, tab2, tab3 = st.tabs(["📈 Forecast Trend", "📊 Actual vs Forecast", "📋 Daily Summary"])
            with tab1:
                st.plotly_chart(plot_forecast(st.session_state.full_forecast_df), use_container_width=True)
            with tab2:
                st.plotly_chart(plot_actual_vs_forecast(st.session_state.df_clean, st.session_state.full_forecast_df), use_container_width=True)
            with tab3:
                st.plotly_chart(plot_daily_bar_chart(st.session_state.df_clean), use_container_width=True)
                st.subheader("📆 Daily Forecast Table")
                st.dataframe(generate_daily_table(st.session_state.forecast_df), use_container_width=True)

            # -------------------- REGION-WISE FORECAST --------------------
            if 'region' in df_raw.columns:
                with st.expander("🌍 Region-wise Forecast Summary"):
                    if st.checkbox("Show Region Summary"):
                        df_region = preprocess_data(df_raw, date_col, target_col, ['region'])
                        region_df = forecast_by_region(df_region, model_choice, event_dates, forecast_until, custom_days)
                        if not region_df.empty:
                            st.dataframe(region_df, use_container_width=True)
                            st.plotly_chart(px.pie(region_df, names='Region', values='Forecasted_Volume', title='📊 Region-wise Forecasted Contribution'), use_container_width=True)
                        else:
                            st.warning("⚠️ No region data available to generate forecast.")
else:
    st.info("📂 Please upload a valid file to begin.")
