import requests
API_URL = "https://r0rl95p5bkd64d5d.us-east-1.aws.endpoints.huggingface.cloud"
headers = {
	"Accept" : "application/json",
	"Authorization": "Bearer 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",
	"Content-Type": "audio/wav" 
}

def query(filename):
	with open(filename, "rb") as f:
		data = f.read()
	response = requests.post(API_URL, headers=headers, data=data)
	return response.json()

output = query("./test_audio.wav")
print(output)