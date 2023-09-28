import requests
from flask import Flask, jsonify, request
from google.auth.transport import requests as grequests
from google.oauth2 import service_account

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    # check the custom API key or token
    api_key = request.headers.get("api-key")

    api_key_local = None
    with open("secrets/42_check.txt", "r") as f:
        api_key_local = f.readline()

    if api_key != api_key_local:
        return jsonify({"error": "Invalid API Key"}), 403

    # Path to your service account key
    key_path = "secrets/mpgregression-2358be5d9b3d.json"

    # authenticate and obtain access token
    SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
    creds = service_account.Credentials.from_service_account_file(
        key_path, scopes=SCOPES
    )
    auth_req = grequests.Request()
    creds.refresh(auth_req)
    access_token = creds.token

    # forward the request to the Vertex AI Endpoint
    vertex_ai_url = "https://europe-west3-aiplatform.googleapis.com/v1beta1/projects/611755511509/locations/europe-west3/endpoints/2743932422184763392:predict"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    # one hot encode the origin column
    payload = request.json["instances"][0]

    payload_length = len(payload)

    # check the payload length
    if payload_length not in (7, 9):
        return (
            jsonify(
                {
                    "error": "The provided data point should have 7 or 9 (if one hot encoded) entries."
                }
            ),
            403,
        )

    # if payload length is 7, modify the payload
    if payload_length == 7:
        on_hot = [0.0] * 3
        on_hot[payload[-1] - 1] = 1.0
        payload = payload[:-1] + on_hot
        request.json["instances"][0] = payload

    # no action needed for payload length of 9
    elif payload_length == 9:
        pass

    response = requests.post(vertex_ai_url, headers=headers, json=request.json)

    if "predictions" in response.json():
        return jsonify({"Prediction ": response.json()["predictions"][0][0]})
    return jsonify(response.json())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
