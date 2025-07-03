# Save this as forecast_utils.py

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

    # Determine forecast end date
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

    # Prophet Model
    if model_type == "Prophet":
        df_grouped['y'] = df_grouped['y'].rolling(window=3, min_periods=1).mean()
        cap = df_grouped['y'].max() * 1.05
        df_grouped['cap'] = cap

        if event_dates:
            df_grouped['event'] = df_grouped['ds'].isin(event_dates).astype(int)

        model = Prophet(growth='linear', daily_seasonality=True, yearly_seasonality=False,
                        weekly_seasonality=True, changepoint_prior_scale=0.02)

        if event_dates:
            model.add_regressor('event')

        model.fit(df_grouped)

        future = pd.DataFrame({'ds': future_dates})
        future['cap'] = cap
        if event_dates:
            future['event'] = future['ds'].isin(event_dates).astype(int)

        forecast = model.predict(future)
        forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        forecast['yhat'] = forecast['yhat'].clip(lower=0, upper=cap)

    elif model_type == "Linear":
        df_grouped['ds_ord'] = df_grouped['ds'].map(datetime.toordinal)
        m, b = polyfit(df_grouped['ds_ord'], df_grouped['y'], 1)
        forecast = pd.DataFrame({'ds': future_dates})
        forecast['yhat'] = [m * d.toordinal() + b for d in forecast['ds']]
        forecast['yhat_lower'] = forecast['yhat'] * 0.95
        forecast['yhat_upper'] = forecast['yhat'] * 1.05
        forecast['yhat'] = forecast['yhat'].clip(lower=0)

    elif model_type == "Exponential":
        model = ExponentialSmoothing(df_grouped['y'], trend='add').fit()
        forecast_vals = model.forecast(forecast_days)
        forecast = pd.DataFrame({'ds': future_dates, 'yhat': forecast_vals})
        forecast['yhat_lower'] = forecast['yhat'] * 0.9
        forecast['yhat_upper'] = forecast['yhat'] * 1.1
        forecast['yhat'] = forecast['yhat'].clip(lower=0)

    forecast_full = pd.concat([
        df_grouped[['ds', 'y']].rename(columns={'y': 'yhat'}),
        forecast[['ds', 'yhat']]
    ], ignore_index=True).sort_values('ds')

    forecast_full['yhat_lower'] = forecast_full['yhat'] * 0.95
    forecast_full['yhat_upper'] = forecast_full['yhat'] * 1.05

    return forecast, last_data_date, forecast_days, forecast_full

def detect_pattern(df_grouped):
    df_grouped['rolling_mean'] = df_grouped['y'].rolling(window=7).mean()
    slope = polyfit(range(len(df_grouped['rolling_mean'].dropna())), df_grouped['rolling_mean'].dropna(), 1)[0]
    if abs(slope) < 1e-2:
        return "↔️ Stationary or flat trend"
    elif slope > 0:
        return "📈 Upward trend detected"
    else:
        return "📉 Downward trend detected"

def calculate_target_analysis(df, forecast_df, last_date, target, mode):
    if mode == 'Monthly':
        current = df[(df['date'].dt.month == last_date.month) & (df['date'].dt.year == last_date.year)]['target'].sum()
    else:
        current = df[df['date'].dt.year == last_date.year]['target'].sum()

    forecast = forecast_df[forecast_df['ds'] > last_date]['yhat'].sum()
    total = current + forecast
    remaining = max(0, target - current)
    days_left = (forecast_df['ds'].max() - last_date).days
    per_day = round(remaining / days_left, 2) if days_left > 0 else 0
    pct = round((total / target) * 100, 2)

    return {
        "📌 Target": target,
        "🟢 Current Sales": round(current, 2),
        "🔮 Forecasted Sales (Remaining Days)": round(forecast, 2),
        "📊 Total Projected (Actual + Forecast)": round(total, 2),
        "📉 Remaining to Hit Target": round(remaining, 2),
        "📅 Days Left to Forecast": days_left,
        "📈 Required Per Day": per_day,
        "🎯 Projected % of Target": pct
    }

def generate_recommendations(metrics):
    if metrics["🎯 Projected % of Target"] >= 100:
        return "✅ You're on track or exceeding your goal!"
    return f"⚠️ You need to sell {metrics['📈 Required Per Day']} units/day for {metrics['📅 Days Left to Forecast']} days."

def plot_forecast(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['yhat'], name='Forecast'))
    fig.add_trace(go.Scatter(x=df['ds'], y=df['yhat_upper'], name='Upper', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=df['ds'], y=df['yhat_lower'], name='Lower', line=dict(dash='dot')))
    fig.update_layout(title="📈 Forecast with Confidence Bands", xaxis_title="Date", yaxis_title="Sales")
    return fig

def plot_actual_vs_forecast(df, forecast_df):
    actual = df.groupby('date')['target'].sum().reset_index()
    actual.columns = ['ds', 'y']
    merged = pd.merge(forecast_df[['ds', 'yhat']], actual, on='ds', how='left')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged['ds'], y=merged['yhat'], name='Forecast'))
    fig.add_trace(go.Scatter(x=merged['ds'], y=merged['y'], name='Actual'))
    fig.update_layout(title='📊 Actual vs Forecasted', xaxis_title='Date', yaxis_title='Sales')
    return fig

def plot_daily_bar_chart(df):
    daily = df.groupby('date')['target'].sum().reset_index()
    fig = px.bar(daily, x='date', y='target', title="📊 Daily Sales Trend")
    return fig

def generate_daily_table(forecast_df):
    return forecast_df[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Forecasted Sales'}).round(2)

def get_forecast_explanation(method):
    explanations = {
        "Prophet": "Prophet models trends and special events to forecast future sales.",
        "Linear": "Linear regression fits a simple trend line based on past values.",
        "Exponential": "Exponential smoothing weighs recent values more heavily."
    }
    return explanations.get(method, "No explanation available.")
