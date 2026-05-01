import pandas as pd
import sqlite3
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd


def load_invoice_data():
    conn = sqlite3.connect('/Users/navaldhandha/Library/CloudStorage/GoogleDrive-dhandhanaval@gmail.com/My Drive/Invoice_inventory_intelligence/data/inventory.db')
    query = '''
with purchase_agg as (
select 
    PONumber,
    count(distinct Brand) as no_brands,
    sum(Quantity) as total_quantity,
    sum(Dollars) as total_amount,
    avg(julianday(ReceivingDate)-julianday(PODate)) as avg_receiving_delay,
    avg(julianday(InvoiceDate) - julianday(ReceivingDate)) as avg_invoice_raise_delay
from 
    purchases
group by PONumber
)
select
    vi.PONumber,
    vi.Quantity as invoice_quantity,
    vi.Dollars as invoice_amount,
    vi.Freight,
    julianday(vi.InvoiceDate) - julianday(vi.PODate) as invoice_to_podate,
    julianday(vi.PayDate) - julianday(vi.InvoiceDate) as days_to_pay,
    p.no_brands,
    p.total_quantity,
    p.total_amount,
    p.avg_receiving_delay,
    p.avg_invoice_raise_delay
from 
    vendor_invoice vi
left join
    purchase_agg p
on vi.PONumber = p.PONumber    
'''
    df = pd.read_sql_query(query,conn)
    conn.close()
    return df

def create_credit_risk_label(row):
    
    if abs(row['invoice_amount'] - row['total_amount']) > 5:
        return 1

    if row['avg_receiving_delay'] > 10:
        return 1

    return 0

def apply_label(df):
    df['flag_invoice'] = df.apply(create_credit_risk_label, axis = 1)
    return df

def split_data(X,y):
    
    return train_test_split(X, y,test_size = 0.2, random_state = 42)

def feature_scaling(X_train,X_test,scaler_path):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    
    joblib.dump(scaler,scaler_path)
    return X_train_scaled, X_test_scaled