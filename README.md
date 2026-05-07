# Invoice_Inventory_intelligence

An end-to-end machine learning and Streamlit application that helps finance, procurement, and supply chain teams forecast freight costs and flag vendor invoices that may require manual review.

This project simulates a real-world invoice intelligence workflow where purchase order, invoice, freight, and receiving-delay patterns are used to support better cost forecasting, anomaly detection, and faster finance operations.

---

## Project Overview

Vendor invoice review is often a manual, time-consuming process. Finance and procurement teams frequently need to compare invoice amounts, freight charges, purchase order timing, item quantities, and receiving delays to determine whether an invoice looks normal or should be reviewed more closely.

This project addresses that problem by building an internal analytics portal that supports two key use cases:

1. **Freight Cost Prediction**
   - Predicts expected freight cost based on invoice quantity and invoice dollar amount.
   - Helps teams improve cost forecasting, budgeting, and vendor cost visibility.
<img width="2806" height="1278" alt="image" src="https://github.com/user-attachments/assets/ff263c34-cb58-447a-b6b0-fa3655163dbe" />


2. **Invoice Manual Approval Flagging**
   - Predicts whether a vendor invoice should be flagged for manual approval.
   - Uses invoice, purchase order, freight, and receiving delay patterns to support exception-based review.
<img width="2766" height="1428" alt="image" src="https://github.com/user-attachments/assets/9a2d40e6-ddae-447a-9b97-c77045947af4" />


The final output is a Streamlit application where users can input invoice details and receive real-time machine learning predictions.

---

## Business Problem

In many finance and supply chain environments, invoice review depends heavily on manual checks, spreadsheet comparisons, and historical judgment. This creates several challenges:

- High-volume invoice review can become slow and repetitive.
- Freight costs may be difficult to estimate consistently.
- Abnormal invoices may not be detected until late in the payment cycle.
- Finance teams may spend time reviewing low-risk invoices instead of prioritizing exceptions.
- Procurement and operations teams may lack visibility into cost drivers.

This project demonstrates how machine learning can be used to move from reactive reporting to proactive decision support.

---

## Business Objectives

The main objectives of this project are to:

- Forecast expected freight costs using invoice-level variables.
- Identify invoices that may require manual approval.
- Reduce manual review effort by prioritizing higher-risk invoices.
- Improve visibility into freight and invoice cost patterns.
- Build a simple user interface for business users.
- Demonstrate how machine learning can support finance and supply chain operations.

---

## Application Modules

The Streamlit application contains two prediction modules.

### 1. Freight Cost Prediction

This module predicts the expected freight cost for a vendor invoice.

#### Inputs

- Quantity
- Invoice Dollars

#### Output

- Predicted Freight Cost

#### Business Value

This helps finance, procurement, and supply chain teams estimate freight cost earlier in the invoice review process. It can support budgeting, vendor negotiations, cost monitoring, and variance analysis.

---

### 2. Invoice Manual Approval Flagging

This module predicts whether an invoice should be flagged for manual review.

#### Inputs

- Invoice Quantity
- Invoice Dollars
- Freight Cost
- Invoice-to-PO Date Difference
- Total Item Quantity
- Total Item Dollars
- Average Receiving Delay

#### Output

- Flagged for Manual Approval
- Not Flagged

#### Business Value

This supports exception-based invoice review. Instead of manually reviewing every invoice with the same level of effort, teams can prioritize invoices that show unusual cost, freight, timing, or receiving-delay patterns.

--- 

## Tech Stack

| Category | Tools |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model Serialization | Joblib |
| Database / Local Storage | SQLite |
| Application Layer | Streamlit |
| Development Environment | Jupyter Notebook |

---

## Repository Structure

```text
Invoice_Inventory_intelligence/
│
├── app.py
├── requirements.txt
├── runtime.txt
│
├── cost_prediction/
│   ├── data_preprocess.py
│   ├── model_evaluation.py
│   ├── train.py
│   └── cost_prediction_model/
│
├── invoice_flag/
│   ├── datapreprocess.py
│   ├── model_evaluation.py
│   ├── train.py
│   └── invoice_flag_model/
│
├── inference/
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
│
└── notebooks/
