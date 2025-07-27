import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go

def calculate_target_analysis(df, target):
    current_sales = df[df['Date'] <= pd.Timestamp.today()]['Sales'].sum()
    forecasted_sales = df[df['Date'] > pd.Timestamp.today()]['Sales'].sum()
    return {
        "Target": target,
        "Current Sales": current_sales,
        "Forecasted Sales": forecasted_sales
    }

def generate_daily_table(df):
    future_df = df[df['Date'] > pd.Timestamp.today()].copy()
    future_df["Forecasted Sales"] = future_df["Sales"]
    return future_df[["Date", "Forecasted Sales"]].reset_index(drop=True)

def forecast_by_region(df):
    future_df = df[df['Date'] > pd.Timestamp.today()].copy()
    grouped = future_df.groupby("Region")["Sales"].sum().reset_index()
    grouped.rename(columns={"Sales": "Forecasted Sales"}, inplace=True)
    return grouped

def generate_line_chart(df):
    fig = px.line(df, x="Date", y="Sales", title="Sales Forecast", color_discrete_sequence=['#00A8E8'])
    fig.update_traces(mode="lines+markers")
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Sales",
        title_x=0.5,
        template="plotly_white"
    )
    return fig

def generate_region_chart(df):
    df_future = df[df['Date'] > pd.Timestamp.today()]
    df_grouped = df_future.groupby("Region")["Sales"].sum().reset_index()
    fig = px.bar(df_grouped, x="Region", y="Sales", title="Forecasted Sales by Region", color_discrete_sequence=['#0077B6'])
    fig.update_layout(
        xaxis_title="Region",
        yaxis_title="Forecasted Sales",
        title_x=0.5,
        template="plotly_white"
    )
    return fig
