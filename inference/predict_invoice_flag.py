import joblib
import pandas as pd

features = ['invoice_quantity',
 'invoice_amount',
 'Freight',
 'invoice_to_podate',
 'total_quantity',
 'total_amount',
 'avg_receiving_delay']

model_path = 'invoice_flag/invoice_flag_model/predict_invoice_flag.pkl'
scaler_path = 'invoice_flag/invoice_flag_model/scaler.pkl'

def load_model(model_path: str = model_path,scaler_path: str = scaler_path):
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model,scaler

def predict_invoice_flag(input_data):
    
    model,scaler = load_model(model_path,scaler_path)
    
    input_df = pd.DataFrame(input_data)    
    input_df = input_df[features]    
    input_scaled = scaler.transform(input_df)
    input_df['predicted_invoice_flag']= model.predict(input_scaled)
    
    return input_df


if __name__ == '__main__':
    
    sample_data = {
        'invoice_quantity': [6700, 450],
        'invoice_amount': [18650, 9500],
        'Freight': [250.75, 90.50],
        'invoice_to_podate':  [14, 5],
        'total_quantity': [15000, 3000],
        'total_amount': [45000, 12500],
        'avg_receiving_delay': [3.5, 1.2]
    }
    prediction = predict_invoice_flag(sample_data)
    print(prediction)