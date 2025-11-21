#api_backend/api.py
from flask import Flask, request, jsonify
from predictor import WalmartSalesPredictor
import os
import traceback

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    "MODEL_DIR": "walmart_sales_model_20251121_104049",
    "DATA_PATH": "../data/full_historical_data.csv"
}

# ==============================================================================
# INITIALIZATION
# ==============================================================================
app = Flask(__name__)

try:
    predictor = WalmartSalesPredictor(
        model_dir=CONFIG["MODEL_DIR"],
        data_path=CONFIG["DATA_PATH"]
    )
except Exception as e:
    print(f"FATAL: Could not initialize predictor. {e}")
    traceback.print_exc()
    predictor = None

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    status = "ok" if predictor else "error"
    message = "API is running and model is loaded." if predictor else "API is running, but model failed to load."
    return jsonify({"status": status, "message": message}), 200 if predictor else 500

@app.route('/dashboard_summary', methods=['GET'])
def get_dashboard_summary():
    if not predictor: return jsonify({"error": "Model not loaded"}), 503
    try:
        summary = predictor.get_dashboard_summary()
        return jsonify(summary), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/forecast', methods=['POST'])
def get_forecast():
    if not predictor: return jsonify({"error": "Model not loaded"}), 503
    try:
        data = request.get_json()
        store = int(data['store'])
        dept = int(data['dept'])
        hist_weeks = int(data.get('hist_weeks', 12))
        forecast_weeks = int(data.get('forecast_weeks', 4))
        
        forecast_data = predictor.get_forecast(store, dept, hist_weeks, forecast_weeks)
        return jsonify(forecast_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/simulate', methods=['POST'])
def run_simulation():
    if not predictor: return jsonify({"error": "Model not loaded"}), 503
    try:
        data = request.get_json()
        store = int(data['store'])
        dept = int(data['dept'])
        markdowns = data['markdowns']
        
        result = predictor.run_simulation(store, dept, markdowns)
        return jsonify(result), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/insights/roi', methods=['GET'])
def get_roi():
    if not predictor: return jsonify({"error": "Model not loaded"}), 503
    try:
        roi_data = predictor.get_roi_data()
        return jsonify(roi_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/insights/hotspots', methods=['GET'])
def get_hotspots():
    if not predictor: return jsonify({"error": "Model not loaded"}), 503
    try:
        hotspots_data = predictor.get_hotspots_data()
        return jsonify(hotspots_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================================================================
# MAIN RUNNER
# ==============================================================================
if __name__ == '__main__':
    if predictor:
        print("\n======================================================")
        print(" [API] All systems ready. You can now start the frontend.")
        print("======================================================")
    else:
        print("\n======================================================")
        print(" [API] CRITICAL ERROR: Predictor failed to load.")
        print("   The API is running but will return errors.")
        print("======================================================")
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 