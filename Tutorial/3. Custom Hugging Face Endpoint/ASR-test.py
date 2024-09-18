import base64
import requests

API_URL = "https://dpmbwhnur9m5nn4z.us-east-1.aws.endpoints.huggingface.cloud"
filepath = "./audio.mp3"

with open(filepath, "rb") as f:
    audio_encoded = base64.b64encode(f.read()).decode("utf-8")

data = {
    "inputs": audio_encoded,
    "parameters": {
    }
}

resp = requests.post(API_URL, json=data, headers={"Authorization": "Bearer hf_XX"})
print(resp.json())
