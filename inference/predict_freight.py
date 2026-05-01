import joblib
import pandas as pd

model_path = 'cost_prediction/cost_prediction_model/predict_freight_cost.pkl'

def load_model(model_path):
    
    with open(model_path,'rb') as f:
        model = joblib.load(f)    
    return model


def predict_freight_cost(input_data):
    
    model = load_model(model_path)
    input_df = pd.DataFrame(input_data)
    input_df['predicted_freight'] = model.predict(input_df).round(2)
    return input_df


if __name__ == '__main__':
    
    sample_data = {
        'Quantity':[6700,450],
        'Dollars':[18650,9500]
    }
    
    prediction = predict_freight_cost(sample_data)
    print(prediction)