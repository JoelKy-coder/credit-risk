import os
import sys
from flask import Flask, request, jsonify, render_template

# Ensure the root project directory is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prediction import CreditRiskPredictor

app = Flask(__name__)

# Initialize predictor lazily
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
        predictor = CreditRiskPredictor(models_dir=models_dir)
    return predictor

@app.route('/')
def home():
    """Renders the dashboard home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts JSON input, validates, performs feature engineering, 
    and returns loan default risk prediction and probability.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided. Request body must be JSON."}), 400
            
        # Basic input validation
        required_fields = [
            'customer_id', 'lender_id', 'loan_type', 'Total_Amount', 
            'Total_Amount_to_Repay', 'disbursement_date', 'due_date', 
            'duration', 'New_versus_Repeat', 'Amount_Funded_By_Lender', 
            'Lender_portion_Funded', 'Lender_portion_to_be_repaid'
        ]
        
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required parameters: {missing_fields}"}), 400
            
        # Type and logic validations
        try:
            total_amt = float(data['Total_Amount'])
            repay_amt = float(data['Total_Amount_to_Repay'])
            duration = int(data['duration'])
            funded_amt = float(data['Amount_Funded_By_Lender'])
            portion_funded = float(data['Lender_portion_Funded'])
            lender_repay = float(data['Lender_portion_to_be_repaid'])
        except (ValueError, TypeError) as e:
            return jsonify({"error": "Invalid numerical data types provided."}), 400
            
        if total_amt <= 0 or repay_amt <= 0:
            return jsonify({"error": "Loan amounts must be greater than zero."}), 400
        if duration <= 0:
            return jsonify({"error": "Loan duration must be greater than zero."}), 400
        if funded_amt > total_amt:
            return jsonify({"error": "Amount funded by lender cannot exceed the total disbursed loan amount."}), 400
        if repay_amt < total_amt:
            return jsonify({"error": "Total amount to repay cannot be less than the disbursed loan amount."}), 400

        # Run inference
        risk_predictor = get_predictor()
        preds, probs = risk_predictor.predict(data)
        
        prediction = int(preds[0])
        probability = float(probs[0])
        
        return jsonify({
            "status": "success",
            "prediction": prediction,
            "probability": probability,
            "risk_label": "HIGH" if prediction == 1 else "LOW"
        })
        
    except FileNotFoundError as fnf:
        app.logger.error(f"Model artifacts not found: {str(fnf)}")
        return jsonify({"error": "Prediction engine models are currently offline. Artifacts missing."}), 500
    except Exception as e:
        app.logger.error(f"Inference error: {str(e)}")
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # Run locally on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
