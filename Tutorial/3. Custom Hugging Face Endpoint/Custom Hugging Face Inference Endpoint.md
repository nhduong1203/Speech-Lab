# Introduction to Hugging Face Inference Endpoint

**Hugging Face** is a company and open-source community that focuses on natural language processing (NLP) technologies. They are renowned for their contributions to NLP with the Transformers library, which provides easy-to-use APIs and pre-trained models for a wide range of NLP tasks.

**Hugging Face Model Hub** provide a various pretrained model for various task, make it become an important platform for every developer. For example, if you want to use Whisper pretrained model from OpenAI for your ASR task, you can using:

```python
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")

```

**Hugging Face Inference Endpoints:** To bring our model to production, **Hugging Face Inference Endpoints** allow developers to deploy machine learning models quickly and efficiently, particularly those hosted on the Hugging Face Model Hub. This service simplifies the process of turning pre-trained models into production-ready APIs with just a few clicks. With minimal setup, you can deploy any model from the Hugging Face Hub to a cloud provider like AWS, GCP, or Azure.
# Hugging Face Inference Endpoints with Whisper model

Let’s take an example of how to deploy a Whisper model to a **Hugging Face Inference Endpoint**. In the **Hugging Face Model Repository**, navigate to your desired model, then go to **Deploy** and select **Inference Endpoint**.
![[Pasted image 20240918030345.png]]

On the Inference Endpoints page, you'll see the `openai/whisper-large-v3` model that we selected from the Hub. You can also choose a different model for your endpoint here. Give your endpoint a name of your choice.
![[Pasted image 20240918030432.png]]

## Instance Configuration

### Cloud vendor

In the Instance Configuration section, you can choose your preferred cloud vendor, including AWS, Microsoft Azure, or GCP. You can also select the instance that best matches your requirements. Be sure to review its details and pricing.
![[Pasted image 20240918030549.png]]
## Automatic Scale-to-Zero 

Stop your endpoint after a period of inactivity, which helps reduce costs. However, it may take some time to scale back up when needed.
![D:\Speech-Lab\Tutorial\3. Custom Hugging Face Endpoint\image-3.png](file:///d%3A/Speech-Lab/Tutorial/3.%20Custom%20Hugging%20Face%20Endpoint/image-3.png)

### The Endpoint security level 

The most important setting for your endpoint. There are three options; choose the one that best suits your needs:
- **Public**: The endpoint is publicly accessible on the internet, and no authentication is required.
- **Protected**: The endpoint is created in a public subnet managed by Hugging Face, but access requires a Hugging Face token.
- **Private**: The endpoint is only accessible through an intra-region secured AWS PrivateLink connection, and it is not available from the internet.

![D:\Speech-Lab\Tutorial\3. Custom Hugging Face Endpoint\image-4.png](file:///d%3A/Speech-Lab/Tutorial/3.%20Custom%20Hugging%20Face%20Endpoint/image-4.png)

### Advanced Configuration
You can also configure parameters such as the number of replicas, container type, environment variables, and revision under **Advanced Configuration**. Be sure to check it out.
## Usage
Here, we will create an example `openai/whisper-large-v3` endpoint using AWS with a T4 GPU. It will take a few minutes to create the endpoint.
![D:\Speech-Lab\Tutorial\3. Custom Hugging Face Endpoint\image-5.png](file:///d%3A/Speech-Lab/Tutorial/3.%20Custom%20Hugging%20Face%20Endpoint/image-5.png)

Once it's ready, you can test it in the playground.
![D:\Speech-Lab\Tutorial\3. Custom Hugging Face Endpoint\image-6.png](file:///d%3A/Speech-Lab/Tutorial/3.%20Custom%20Hugging%20Face%20Endpoint/image-6.png)

An important way to integrate the endpoint into a product is by accessing it via API. To use a protected Inference Endpoint, you'll need to provide your Hugging Face Access Token, as shown below:
```python
import requests
API_URL = "https://r0rl95p5bkd64d5d.us-east-1.aws.endpoints.huggingface.cloud"

headers = {
    "Accept" : "application/json",
    "Authorization": "Bearer hf_XXX",
    "Content-Type": "audio/wav"
}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

output = query("./test_audio.wav")
print(output)
```
```
Output:
{'text': ' Printing, in the only sense with which we are at present concerned, differs from most, if not from all, the arts and crafts represented in the exhibition'}
```

# Custom Hugging Face Inference Endpoint

## Custom ASR + diarization + summarization Pipeline

In the previous section, we introduced how to create and use an endpoint for a model in the Hugging Face Model Hub. However, with a default model, you can only perform one task per API. In some scenarios, you may want to perform multiple tasks or features within a single API, such as ASR and diarization or summarization. In this section, we will explain how to create a custom Inference Endpoint to support the pipeline including ASR + diarization + summarization. You can follow the code provided [here](https://huggingface.co/sieucun/ASR-Diarization).

First, create your own model. Then, go to `File and Versions`. There are two important files you need to manage:
- `handler.py`: This file defines the flow of your endpoint.
- `config.py`: This file retrieves environment variables for your endpoint and sets inference parameters.

Additionally, include any dependency libraries in `requirements.txt` and other files as needed for your project. For example, the diarization processor will be placed in `diarization_utils.py`.

1. `config.py`

In `config.py`, we will retrieve configuration details about the model we are using. The values for these parameters will be provided via environment variables in **Advanced Configuration** when creating the endpoint. If required, you'll also need to supply your Hugging Face access token.

```python
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import Optional, Literal
class ModelSettings(BaseSettings):
    asr_model: str
    diarization_model: Optional[str]
    summarize_model: Optional[str]
    hf_token: Optional[str]

model_settings = ModelSettings()
```

In addition, we will also retrieve the inference configuration details.
```python
class InferenceConfig(BaseModel):
    task: Literal["transcribe", "translate"] = "transcribe"
    batch_size: int = 24
    chunk_length_s: int = 30
    sampling_rate: int = 16000
    language: Optional[str] = None
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
```

2. `handler.py`
In this file, we need to focus on two functions: `__init__` and `__call__` within the `EndpointHandler` class. First, import the required libraries.
```python
import logging
import torch
import os
import base64

from pyannote.audio import Pipeline
from transformers import pipeline, AutoModelForCausalLM
from diarization_utils import diarize
from huggingface_hub import HfApi
from pydantic import ValidationError
from starlette.exceptions import HTTPException

from config import model_settings, InferenceConfig
```

- `__init__`: This function initializes the model pipeline using the `model_setting` configuration.
```python
def __init__(self, path=""):
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	torch_dtype = torch.float32 if device.type == "cpu" else torch.float16


	self.asr_pipeline = pipeline(
		"automatic-speech-recognition",
		model=model_settings.asr_model,
		torch_dtype=torch_dtype,
		device=device
	)

	if model_settings.diarization_model:
		# diarization pipeline doesn't raise if there is no token
		HfApi().whoami(model_settings.hf_token)
		self.diarization_pipeline = Pipeline.from_pretrained(
			checkpoint_path=model_settings.diarization_model,
			use_auth_token=model_settings.hf_token,
		)
		self.diarization_pipeline.to(device)
	else:
		self.diarization_pipeline = None

	# Use a pipeline as a high-level helper
	self.summarize_pipeline = pipeline(
		"summarization", 
		model=model_settings.summarize_model,
		torch_dtype=torch_dtype,
		device=device
	)
```

- `__call__`: This function handles the execution of our endpoint's workflow.

```python
def __call__(self, inputs):
	file = inputs.pop("inputs")
	file = base64.b64decode(file)
	parameters = inputs.pop("parameters", {})
	try:
		parameters = InferenceConfig(**parameters)
	except ValidationError as e:
		logger.error(f"Error validating parameters: {e}")
		raise HTTPException(status_code=400, detail=f"Error validating parameters: {e}")
		
	asr_outputs = self.asr_pipeline(
		file,
		chunk_length_s=parameters.chunk_length_s,
		batch_size=parameters.batch_size,
		generate_kwargs=generate_kwargs,
		return_timestamps=True,
	)

	if self.diarization_pipeline:
		transcript = diarize(self.diarization_pipeline, file, parameters, asr_outputs)
	else:
		transcript = []

	conversation = ""
	for speech in transcript:
	  conversation += speech['speaker'] + ": " + speech['text'] + ' \n'
	
	summarize_outputs = self.summarize_pipeline(
		conversation, 
		max_length=530, 
		min_length=30, 
		do_sample=False
	)


	return {
		"speakers": transcript,
		"summarize": summarize_outputs,
		"chunks": asr_outputs["chunks"],
		"text": asr_outputs["text"],
	}
```

## Deploy Inference Endpoint

After completing the previous steps, you can deploy the Inference Endpoint following the instructions in the previous section. An important step is to provide the information about the model you want to use via **Environment Variables** in **Advanced Configuration**. If the model requires a Hugging Face access token, make sure to include it. However, it's recommended to store the token in **Secrets** so that it remains hidden from users of the endpoint.
![[Pasted image 20240918155227.png]]

## Usage

Once your **Inference Endpoint** is ready, you can use it via API.

```python
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

resp = requests.post(API_URL, json=data, headers={"Authorization": "Bearer hf_XXX"})

print(resp.json())
```

# Conclusion

In this tutorial, we demonstrated how to utilize **Hugging Face Inference Endpoints** to deploy a model and how to customize an **Endpoint** to serve specific features within a single API. We covered the essential steps for creating and configuring endpoints, including selecting cloud providers, setting security levels, and managing instance configurations.

By following these steps, you can effectively deploy and customize your model to meet various needs, whether for single-task or multi-task scenarios. The ability to create a tailored inference endpoint allows you to optimize performance and functionality according to your specific requirements. With these tools and techniques, you can integrate powerful machine learning capabilities into your applications seamlessly.