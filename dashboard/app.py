import os
import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
)


st.set_page_config(
    page_title=(
        "FMCG Demand Intelligence"
    ),
    page_icon="📦",
    layout="wide",
)


st.title(
    "FMCG Demand Intelligence Platform"
)


st.caption(
    "Demand forecasting and "
    "inventory decision support"
)


# ---------------------------------
# Load stores
# ---------------------------------

@st.cache_data
def get_stores():

    response = requests.get(
        f"{API_URL}/stores"
    )

    response.raise_for_status()

    return (
        response
        .json()["stores"]
    )


# ---------------------------------
# Load families
# ---------------------------------

@st.cache_data
def get_families():

    response = requests.get(
        f"{API_URL}/families"
    )

    response.raise_for_status()

    return (
        response
        .json()["families"]
    )


try:

    stores = get_stores()

    families = get_families()

except Exception as exc:

    st.error(
        "Could not connect "
        "to forecasting API."
    )

    st.code(
        str(exc)
    )

    st.stop()


# ---------------------------------
# Sidebar
# ---------------------------------

st.sidebar.header(
    "Forecast Settings"
)


store_nbr = (
    st.sidebar.selectbox(
        "Store",
        stores,
    )
)


family = (
    st.sidebar.selectbox(
        "Product Family",
        families,
    )
)


forecast_days = (
    st.sidebar.slider(
        "Forecast Horizon",
        min_value=1,
        max_value=30,
        value=7,
    )
)


default_promotion = (
    st.sidebar.number_input(
        "Promotion Units",
        min_value=0,
        value=0,
        step=1,
    )
)


current_inventory = (
    st.sidebar.number_input(
        "Current Inventory",
        min_value=0.0,
        value=10000.0,
        step=100.0,
    )
)


safety_stock_percent = (
    st.sidebar.slider(
        "Safety Stock %",
        min_value=0,
        max_value=100,
        value=20,
    )
)


# ---------------------------------
# Forecast button
# ---------------------------------

if st.sidebar.button(
    "Generate Forecast",
    type="primary",
):

    promotions = [
        int(default_promotion)
    ] * forecast_days

    payload = {

        "store_nbr":
            int(store_nbr),

        "family":
            family,

        "forecast_days":
            forecast_days,

        "promotions":
            promotions,
    }

    try:

        response = requests.post(

            f"{API_URL}/forecast/multi",

            json=payload,

            timeout=60,
        )

        response.raise_for_status()

        result = response.json()

    except Exception as exc:

        st.error(
            "Forecast request failed."
        )

        st.code(
            str(exc)
        )

        st.stop()


    # ---------------------------------
    # Convert forecast to DataFrame
    # ---------------------------------

    forecast_df = pd.DataFrame(
        result["forecast"]
    )

    forecast_df["date"] = (
        pd.to_datetime(
            forecast_df["date"]
        )
    )


    # ---------------------------------
    # Business calculations
    # ---------------------------------

    total_demand = (
        forecast_df[
            "predicted_sales"
        ]
        .sum()
    )


    average_daily_demand = (
        forecast_df[
            "predicted_sales"
        ]
        .mean()
    )


    safety_stock = (

        total_demand

        * (
            safety_stock_percent
            / 100
        )
    )


    required_inventory = (
        total_demand
        + safety_stock
    )


    shortage = max(
        0,
        required_inventory
        - current_inventory,
    )


    recommended_reorder = (
        shortage
    )


    ending_inventory = (
        current_inventory
        - total_demand
    )


    # ---------------------------------
    # Header
    # ---------------------------------

    st.subheader(
        f"{family} — Store {store_nbr}"
    )


    # ---------------------------------
    # KPI cards
    # ---------------------------------

    col1, col2, col3, col4 = (
        st.columns(4)
    )


    col1.metric(
        "Forecast Demand",
        f"{total_demand:,.0f}",
    )


    col2.metric(
        "Average / Day",
        f"{average_daily_demand:,.0f}",
    )


    col3.metric(
        "Current Inventory",
        f"{current_inventory:,.0f}",
    )


    col4.metric(
        "Recommended Reorder",
        f"{recommended_reorder:,.0f}",
    )


    # ---------------------------------
    # Forecast chart
    # ---------------------------------

    st.subheader(
        "Demand Forecast"
    )


    chart_df = (
        forecast_df[
            [
                "date",
                "predicted_sales",
            ]
        ]
        .set_index(
            "date"
        )
    )


    st.line_chart(
        chart_df
    )


    # ---------------------------------
    # Inventory analysis
    # ---------------------------------

    st.subheader(
        "Inventory Analysis"
    )


    inventory_col1, (
        inventory_col2
    ), inventory_col3 = (
        st.columns(3)
    )


    inventory_col1.metric(
        "Safety Stock",
        f"{safety_stock:,.0f}",
    )


    inventory_col2.metric(
        "Required Inventory",
        f"{required_inventory:,.0f}",
    )


    inventory_col3.metric(
        "Expected Ending Inventory",
        f"{ending_inventory:,.0f}",
    )


    if shortage > 0:

        st.warning(
            f"Potential stock shortage. "
            f"Recommended reorder: "
            f"{recommended_reorder:,.0f} units."
        )

    else:

        st.success(
            "Current inventory is sufficient "
            "for the forecast horizon."
        )


    # ---------------------------------
    # Forecast table
    # ---------------------------------

    st.subheader(
        "Forecast Details"
    )


    st.dataframe(

        forecast_df,

        use_container_width=True,

        hide_index=True,
    )


else:

    st.info(
        "Select your forecasting settings "
        "and click Generate Forecast."
    )