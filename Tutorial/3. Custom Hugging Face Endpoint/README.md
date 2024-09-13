# Introduction to Huggingface Inference Endpoint
    
**Hugging Face** is a company and open-source community that focuses on natural language processing (NLP) technologies. They are renowned for their contributions to NLP with the Transformers library, which provides easy-to-use APIs and pre-trained models for a wide range of NLP tasks.

**Hugging Face Model Hub** provide a various pretrained model for various task, make it become an important platform for every developer. For example, if you want to use Whisper pretrained model from OpenAI for your ASR task, you can using: 

```python
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small")
```

**Hugging Face Inference Endpoints:** To bring our model to production, Hugging Face Inference Endpoints allow developers to deploy machine learning models quickly and efficiently, particularly those hosted on the Hugging Face Model Hub. This service simplifies the process of turning pre-trained models into production-ready APIs with just a few clicks. With minimal setup, you can deploy any model from the Hugging Face Hub to a cloud provider like AWS, GCP, or Azure.

# Hugging Face Inference Endpoints with Whisper model

Let's take a example of how to deploy a Whisper model to an Hugging Face Inference Endpoints. 

On  the Hugging Face Model Repository, go to your expected model -> Deploy -> Inference Enpoint.

![alt text](image-1.png)

On inference endpoints page, you can see the `openai/whisper-small` model that we have choosen from Hub, you also can choose another model here for your Endpoint. Named your endpoint with your favorite name.

At **Instance Configuration**, you can choose your favorite cloud provider, include: AWS, Microsoft Azure and GCP. Also here, you can choose the instance that match your requirement. Check it information and its price.
![alt text](image.png)

Automatic Scale-to-zero can turn off your Endpoint after a period that have no activity. It's saving cost, but in other side, it take some time to scale back up.
![alt text](image-3.png)

**Endpoint security level** is the most important setting in our Endpoint. There are 3 options, choose it suitale for your purpose.
- Public: The Endpoint is public on internet. Everyone can use it, no authentication is required.
- Protect: The Endpoint is created in a public subnet manage by hugging face, but you need to provide a hugging face token to access the Endpoint.
- Private: A private Endpoint is only available through an intra-region secured AWS PrivateLink connection. Private Endpoints are not available from the Internet.
![alt text](image-4.png)

You can also config some parameter like Number of replicas, Container Type, Environment Variables, Revision at Advanced configuration. Checkout it.



