import streamlit as st
import pandas as pd
from forecast_utils import (
    preprocess_data, forecast_sales,
    calculate_target_analysis, generate_recommendations,
    plot_forecast, plot_actual_vs_forecast,
    plot_daily_bar_chart, generate_daily_table,
    get_forecast_explanation, detect_pattern,
    forecast_by_region, plot_region_contribution_pie,
    plot_region_current_sales_pie
)

st.set_page_config(page_title="Smart Sales Forecast App", layout="wide")
st.title("Smart Sales Forecast & Target Tracker - HICO")

if 'forecast_ran' not in st.session_state:
    st.session_state.forecast_ran = False
if 'show_charts' not in st.session_state:
    st.session_state.show_charts = False

uploaded_file = st.file_uploader("Upload Sales File (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    df_raw.columns = df_raw.columns.str.lower().str.strip().str.replace(" ", "_").str.replace(r'[^\w\s]', '', regex=True)
    st.session_state.df_raw = df_raw

    st.success("File uploaded successfully!")
    if st.checkbox("Show Data"):
        st.dataframe(df_raw.head())

    date_col = st.selectbox("Select Date Column", df_raw.select_dtypes(include=["object", "datetime"]).columns)
    target_col = st.selectbox("Select Sales/Quantity Column", df_raw.select_dtypes("number").columns)
    filters = st.multiselect("Select Filter Columns (Optional)", [col for col in df_raw.columns if col not in [date_col, target_col]])

    df_clean = preprocess_data(df_raw, date_col, target_col, filters)
    st.session_state.df_clean = df_clean

    st.markdown("## Select Forecasting Method")
    model_choice = st.radio("Choose a method", ["Prophet", "Linear", "Exponential"])
    st.caption(get_forecast_explanation(model_choice))

    target_mode = st.radio("Target Period", ["Monthly", "Yearly"], horizontal=True)
    target_value = st.number_input("Enter Your Sales Target", step=1000)

    st.markdown("## Select Forecast Horizon")
    forecast_range = st.selectbox("How far do you want to forecast?", ["Till Month End", "Till Quarter End", "Till Year End", "Custom Days"])
    forecast_until = 'year_end'
    custom_days = None
    if forecast_range == "Till Month End":
        forecast_until = 'month_end'
    elif forecast_range == "Till Quarter End":
        forecast_until = 'quarter_end'
    elif forecast_range == "Custom Days":
        forecast_until = 'custom'
        custom_days = st.number_input("Enter custom number of days", min_value=1, value=30)

    st.markdown("### Any Special Events or Seasonal Days")
    include_events = st.radio("Include Special Event Dates?", ["No", "Yes"], horizontal=True)
    event_dates = []
    if include_events == "Yes":
        event_dates = st.date_input("Select One or More Special Dates", [])

    if st.button("Run Forecast"):
        forecast_df, last_data_date, days_left, full_df = forecast_sales(
            df_clean, model_choice, target_mode, event_dates,
            forecast_until=forecast_until, custom_days=custom_days
        )
        if forecast_df.empty:
            st.warning("Not enough data or no remaining days to forecast.")
        else:
            st.session_state.forecast_df = forecast_df
            st.session_state.full_forecast_df = full_df
            st.session_state.last_data_date = last_data_date
            st.session_state.target_value = target_value
            st.session_state.target_mode = target_mode
            st.session_state.forecast_ran = True
            st.session_state.show_charts = False

    if st.session_state.forecast_ran:
        st.subheader("Target Analysis")
        metrics = calculate_target_analysis(
            st.session_state.df_clean,
            st.session_state.forecast_df,
            st.session_state.last_data_date,
            st.session_state.target_value,
            st.session_state.target_mode
        )

        for k, v in metrics.items():
            try:
                st.metric(label=str(k), value=str(v))
            except Exception as e:
                st.text(f"{k}: {v}")

        st.success(generate_recommendations(metrics))

        st.subheader("Trend Pattern Insight")
        st.info(detect_pattern(st.session_state.full_forecast_df.dropna(subset=['yhat']).rename(columns={'yhat': 'y'})))

        if st.button("Show Charts and Table"):
            st.session_state.show_charts = True

        if st.session_state.show_charts:
            st.plotly_chart(plot_forecast(st.session_state.full_forecast_df), use_container_width=True)
            st.plotly_chart(plot_actual_vs_forecast(st.session_state.df_clean, st.session_state.full_forecast_df), use_container_width=True)
            st.plotly_chart(plot_daily_bar_chart(st.session_state.df_clean), use_container_width=True)

            st.subheader("Daily Forecast Table")
            st.dataframe(generate_daily_table(st.session_state.forecast_df))

            if 'region' in df_raw.columns:
                show_region_summary = st.checkbox("Show Region-wise Forecast Summary")
                if show_region_summary:
                    df_region = preprocess_data(df_raw, date_col, target_col, ['region'])
                    region_df = forecast_by_region(df_region, model_choice, event_dates, forecast_until, custom_days)
                    if not region_df.empty:
                        st.subheader("Region-wise Forecast Summary")
                        st.dataframe(region_df)
                        st.plotly_chart(plot_region_contribution_pie(region_df), use_container_width=True)

                        st.subheader("Region-wise Current Sales Contribution")
                        st.plotly_chart(plot_region_current_sales_pie(df_clean), use_container_width=True)
                    else:
                        st.warning("No region data found to generate forecast.")
else:
    st.info("Upload data file to begin.")
