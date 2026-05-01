from datapreprocess import (
    load_invoice_data, 
    create_credit_risk_label, 
    apply_label, split_data, 
    feature_scaling
)
from model_evaluation import train_random_forest, model_evaluation
import joblib
from pathlib import Path

features = ['invoice_quantity',
 'invoice_amount',
 'Freight',
 'invoice_to_podate',
 'total_quantity',
 'total_amount',
 'avg_receiving_delay']

target = 'flag_invoice'

def main():
    
    df = load_invoice_data()
    df = apply_label(df)
    
    model_dir = Path('invoice_flag_model')
    model_dir.mkdir(exist_ok = True)
    
    X_train,X_test,y_train,y_test = split_data(df[features],df[target])
    X_train_scaled, X_test_scaled = feature_scaling(X_train,X_test,'invoice_flag_model/scaler.pkl')
    
    grid_search = train_random_forest(X_train_scaled,y_train)
    
    model_evaluation(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        'Random Forest Classifier'
    )
    
    joblib.dump(grid_search.best_estimator_, 'invoice_flag_model/predict_invoice_flag.pkl')
    
    
if __name__ == '__main__':
    main()
        
    