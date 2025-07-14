import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
from numpy import polyfit
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from calendar import monthrange

def preprocess_data(df, date_col, target_col, filters=[]):
    df = df[[date_col, target_col] + filters].copy()
    df.columns = ['date', 'target'] + filters
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df.dropna(subset=['date', 'target'], inplace=True)
    return df

def forecast_sales(df, model_type, target_mode, event_dates=None, forecast_until='year_end', custom_days=None):
    df_grouped = df.groupby("date")["target"].sum().reset_index()
    df_grouped.columns = ['ds', 'y']
    df_grouped = df_grouped.sort_values("ds")
    last_data_date = pd.to_datetime(df_grouped['ds'].max())

    if forecast_until == 'month_end':
        year, month = last_data_date.year, last_data_date.month
        end_date = datetime(year, month, monthrange(year, month)[1])
    elif forecast_until == 'quarter_end':
        q_month = ((last_data_date.month - 1) // 3 + 1) * 3
        end_date = datetime(last_data_date.year, q_month, monthrange(last_data_date.year, q_month)[1])
    elif forecast_until == 'custom':
        end_date = last_data_date + timedelta(days=custom_days)
    else:
        end_date = datetime(last_data_date.year, 12, 31)

    future_dates = pd.date_range(start=last_data_date + timedelta(days=1), end=end_date)
    forecast_days = len(future_dates)
    if forecast_days <= 0:
        return pd.DataFrame(), last_data_date, 0, df_grouped

    if model_type == "Prophet":
        model = Prophet(daily_seasonality=True)
        model.fit(df_grouped)
        future = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future)[['ds', 'yhat']]
    elif model_type == "Linear":
        df_grouped['ds_ord'] = df_grouped['ds'].map(datetime.toordinal)
        m, b = polyfit(df_grouped['ds_ord'], df_grouped['y'], 1)
        forecast = pd.DataFrame({'ds': future_dates})
        forecast['yhat'] = [m * d.toordinal() + b for d in forecast['ds']]
    elif model_type == "Exponential":
        model = ExponentialSmoothing(df_grouped['y'], trend='add').fit()
        forecast_vals = model.forecast(forecast_days)
        forecast = pd.DataFrame({'ds': future_dates, 'yhat': forecast_vals})

    forecast_full = pd.concat([
        df_grouped[['ds', 'y']].rename(columns={'y': 'yhat'}),
        forecast
    ], ignore_index=True).sort_values('ds')

    forecast_full['yhat_lower'] = forecast_full['yhat'] * 0.95
    forecast_full['yhat_upper'] = forecast_full['yhat'] * 1.05

    return forecast, last_data_date, forecast_days, forecast_full

def forecast_by_region(df, model_type, event_dates=None, forecast_until='year_end', custom_days=None):
    df = df.copy()
    df.columns = df.columns.str.lower()

    if 'region' not in df.columns or 'date' not in df.columns or 'target' not in df.columns:
        return pd.DataFrame(columns=['Region', 'Forecasted_Volume'])

    regions = df['region'].dropna().unique()
    region_forecasts = []

    for region in regions:
        region_df = df[df['region'] == region]
        if region_df.empty:
            continue
        forecast, _, _, _ = forecast_sales(region_df, model_type, 'Yearly', event_dates, forecast_until, custom_days)
        if not forecast.empty:
            total = forecast['yhat'].sum()
            region_forecasts.append({'Region': region, 'Forecasted_Volume': round(total, 2)})

    df_out = pd.DataFrame(region_forecasts)
    if not df_out.empty and 'Forecasted_Volume' in df_out.columns:
        return df_out.sort_values(by="Forecasted_Volume", ascending=False)
    return df_out

def plot_region_contribution_pie(df):
    if 'Region' in df.columns and 'Forecasted_Volume' in df.columns and not df.empty:
        fig = px.pie(df, names='Region', values='Forecasted_Volume', title='📊 Region-wise Forecasted Contribution')
        return fig
    return go.Figure()

def plot_region_current_sales_pie(df):
    if 'region' in df.columns and 'date' in df.columns and 'target' in df.columns:
        region_sales = df.groupby('region')['target'].sum().reset_index()
        region_sales.columns = ['Region', 'Current_Sales']
        fig = px.pie(region_sales, names='Region', values='Current_Sales', title='🏆 Region-wise Current Sales Contribution')
        return fig
    return go.Figure()

# The rest of the forecasting utility functions should follow (not duplicated here due to space)
