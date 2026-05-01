import streamlit as st
import pandas as pd

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag


st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📦",
    layout="wide"
)


st.sidebar.title("🔍 Model Selection")

selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    [
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Business Impact")
st.sidebar.markdown(
    """
- 📉 Improved cost forecasting
- 🧾 Reduced invoice fraud & anomalies
- ⚙️ Faster finance operations
"""
)


st.title("📦 Vendor Invoice Intelligence Portal")

st.header("AI-Driven Freight Cost Prediction & Invoice Risk Flagging")

st.markdown(
    """
This internal analytics portal leverages machine learning to:

- **Forecast freight costs accurately**
- **Detect risky or abnormal vendor invoices**
- **Reduce financial leakage and manual workload**
"""
)

st.markdown("---")


if selected_model == "Freight Cost Prediction":

    st.subheader("🚚 Freight Cost Prediction")

    st.markdown(
        """
**Objective:**  
Predict freight cost for a vendor invoice using **Quantity** and **Invoice Dollars**
to support budgeting, forecasting, and vendor negotiations.
"""
    )

    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            quantity = st.number_input(
                "📦 Quantity",
                min_value=0,
                value=1200,
                step=100
            )

        with col2:
            dollars = st.number_input(
                "💰 Invoice Dollars",
                min_value=0.0,
                value=18500.00,
                step=100.00,
                format="%.2f"
            )

        if st.button("🔮 Predict Freight Cost"):
            input_data = {
                "Quantity": [quantity],
                "Dollars": [dollars]
            }

            result = predict_freight_cost(input_data)

            predicted_freight = result["predicted_freight"].iloc[0]

            st.success("Freight cost prediction completed.")

            st.metric(
                label="Predicted Freight Cost",
                value=f"${predicted_freight:,.2f}"
            )

            st.dataframe(result, use_container_width=True)


elif selected_model == "Invoice Manual Approval Flag":

    st.subheader("🚨 Invoice Manual Approval Prediction")

    st.markdown(
        """
**Objective:**  
Predict whether a vendor invoice should be **flagged for manual approval**
based on abnormal cost, freight, or delivery patterns.
"""
    )

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            invoice_quantity = st.number_input(
                "Invoice Quantity",
                min_value=0,
                value=50,
                step=1
            )

        with col2:
            invoice_amount = st.number_input(
                "Invoice Dollars",
                min_value=0.0,
                value=352.95,
                step=10.00,
                format="%.2f"
            )

        with col3:
            total_amount = st.number_input(
                "Total Item Dollars",
                min_value=0.0,
                value=2476.00,
                step=100.00,
                format="%.2f"
            )

        col4, col5, col6 = st.columns(3)

        with col4:
            freight = st.number_input(
                "Freight Cost",
                min_value=0.0,
                value=1.73,
                step=1.00,
                format="%.2f"
            )

        with col5:
            total_quantity = st.number_input(
                "Total Item Quantity",
                min_value=0,
                value=162,
                step=1
            )

        with col6:
            invoice_to_podate = st.number_input(
                "Invoice to PO Date Difference",
                min_value=0,
                value=7,
                step=1
            )

        avg_receiving_delay = st.number_input(
            "Average Receiving Delay",
            min_value=0.0,
            value=2.5,
            step=0.5,
            format="%.2f"
        )

        if st.button("🧠 Evaluate Invoice Risk"):
            input_data = {
                "invoice_quantity": [invoice_quantity],
                "invoice_amount": [invoice_amount],
                "Freight": [freight],
                "invoice_to_podate": [invoice_to_podate],
                "total_quantity": [total_quantity],
                "total_amount": [total_amount],
                "avg_receiving_delay": [avg_receiving_delay]
            }

            result = predict_invoice_flag(input_data)

            prediction = result["predicted_invoice_flag"].iloc[0]

            probability = None
            if "flag_probability" in result.columns:
                probability = result["flag_probability"].iloc[0]

            if prediction == 1:
                st.error("🚨 Invoice should be flagged for manual approval.")
                prediction_label = "Flagged for Manual Approval"
            else:
                st.success("✅ Invoice does not require manual approval.")
                prediction_label = "Not Flagged"

            col_a, col_b = st.columns(2)

            with col_a:
                st.metric("Prediction", prediction_label)

            with col_b:
                if probability is not None:
                    st.metric("Flag Probability", f"{probability * 100:.2f}%")
                else:
                    st.metric("Flag Probability", "Not Available")

            st.dataframe(result, use_container_width=True)