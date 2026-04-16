import json

from inference_utils import load_model_bundle, predict_error_message


MODEL_BUNDLE = None


def init():
    global MODEL_BUNDLE
    MODEL_BUNDLE = load_model_bundle("/var/azureml-app/azureml-models")


def run(raw_data):
    try:
        if isinstance(raw_data, (bytes, bytearray)):
            raw_data = raw_data.decode("utf-8")
        payload = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        if not isinstance(payload, dict):
            return json.dumps({"prediction": ""})
        error_message = str(payload.get("errorMessage", ""))
        prediction = predict_error_message(MODEL_BUNDLE, error_message)
        return json.dumps({"prediction": prediction})
    except Exception:
        return json.dumps({"prediction": ""})
