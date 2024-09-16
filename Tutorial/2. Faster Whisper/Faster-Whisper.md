## Introduction to Faster Whisper

### Whisper: Robust Speech Recognition via Large-Scale Weak Supervision

Whisper, developed by OpenAI, excels in Automatic Speech Recognition (ASR) tasks by demonstrating high performance and strong generalization across datasets and domains without requiring fine-tuning. Its strength lies in training on an extensive dataset of 680,000 hours of multilingual and multitask supervised data sourced from the web.

At the core of Whisper is a Transformer-based sequence-to-sequence model trained across various speech processing tasks: multilingual speech recognition, speech translation, spoken language identification, and voice activity detection. This diversity makes Whisper a robust ASR model.
### CTranslate2 and Faster-Whisper: Optimizing Transformer Model Inference

**CTranslate2** is a C++ and Python library designed for efficient inference with Transformer models. Engineered for high performance, it incorporates optimizations such as weight quantization, layer fusion, and batch reordering. These optimizations enhance speed and minimize memory usage on both CPUs and GPUs.

**CTranslate2** supports a diverse array of model types, catering to various needs:

- **Encoder-decoder** models like Transformer base/big, M2M-100, and Whisper.
- **Decoder-only** models such as GPT-2, GPT-J, and BERT.
- **Encoder-only** models like BERT and XLM-RoBERTa.

**Faster-Whisper** utilizes the Whisper model implemented with **CTranslate2**, offering up to 4 times faster inference speeds with reduced memory usage compared to `openai/whisper`, while maintaining the same accuracy. Further efficiency gains are achievable through 8-bit quantization on both CPU and GPU.
## What's make it fast?

- **Optimized Execution:** The framework achieves fast and efficient execution on both CPU and GPU through a variety of advanced optimizations. These include layer fusion, padding removal, batch reordering, in-place operations, and caching mechanisms.

- **Quantization and Precision Reduction:** The framework supports model serialization and computation with weights of reduced precision, including FP16, BF16, INT16, INT8, INT4. These techniques contribute to improved performance and reduced model size.

- Supports multiple CPU architectures, including x86-64 and AArch64/ARM64, with optimized backends like Intel MKL and oneDNN.

- The framework offers parallel and asynchronous processing, dynamic memory management, reduced disk footprint through quantization, and simple integration with minimal dependencies.

## Benchmark

For reference, here's the time and memory usage that are required to transcribe 13 minutes of audio using different implementations.

- Large-v2 model on GPU

| Implementation  | Precision | Beam size | Time  | Max. GPU memory | Max. CPU memory |
| --------------- | --------- | --------- | ----- | --------------- | --------------- |
| openai/whisper  | fp16      | 5         | 4m30s | 11325MB         | 9439MB          |
| faster-whisper  | fp16      | 5         | 54s   | 4755MB          | 3244MB          |
| faster-whisper  | int8      | 5         | 59s   | 3091MB          | 3117MB          |

- Small model on CPU

| Implementation | Precision | Beam size | Time   | Max. memory |
| -------------- | --------- | --------- | ------ | ----------- |
| openai/whisper | fp32      | 5         | 10m31s | 3101MB      |
| whisper.cpp    | fp32      | 5         | 17m42s | 1581MB      |
| whisper.cpp    | fp16      | 5         | 12m39s | 873MB       |
| faster-whisper | fp32      | 5         | 2m44s  | 1675MB      |
| faster-whisper | int8      | 5         | 2m04s  | 995MB       |

# CTranslate2 Whisper - Fast inference engine for Transformer models

## Preparation
- To install Ctranslate2 Whisper, run:
	
```
!pip install git+https://github.com/openai/whisper.git
!pip install transformers[torch]>=4.23 
!pip install --upgrade ctranslate2 
```

- Import libraries
```python
import whisper
import numpy as np
import librosa
import torchaudio
from transformers import WhisperTokenizer,WhisperProcessor
import torch
import ctranslate2
from torchaudio.utils import download_asset
import IPython.display as ipd
```
- Prepare example audio file
```python
SAMPLE_WAV = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
wave_form, sampling_rate = torchaudio.load(SAMPLE_WAV)
wave_form = wave_form.numpy()
ipd.Audio(wave_form, rate=sampling_rate)
```

## Convert Whisper model to Ctranslate2 Whisper

The Whisper model come in various versions, including whisper-tiny, whisper-small, whisper-base, and whisper-large. You can load the Whisper model from Hugging Face. 

- First, we need to initialize the Whisper processor, which is responsible for encoding the input and decoding the output into a suitable format.
```python
processor = WhisperProcessor.from_pretrained("openai/whisper-base")

tokenizer = processor.tokenizer
```
- CTranslate2 provide command to converts the Whisper model to the CTranslate2 format, saves it in the whisper-base directory, copies the specified configuration files, applies quantization.
```
!ct2-transformers-converter --model openai/whisper-base --output_dir whisper-base \
--copy_files tokenizer.json preprocessor_config.json --quantization float16 --force
```
## CTranslate2 Whisper

- Load the Ctranslate2 Whisper model using the code below and set it up to run on the CPU. Alternatively, it is possible to use a GPU by setting the device variable to `cuda`.
```python
translator = ctranslate2.models.Whisper("./whisper-base", device="cpu")
```
- We use the processor to extract mel-spectrograms from audio.
```python
inputs = processor(wave_form, return_tensors="np", sampling_rate=sampling_rate)
```
- CTranslate2 encodes the mel-spectrograms into a `StorageView` using the `ctranslate2.models.Whisper.encode()` function. CTranslate2 uses `StorageView` as its input feature.
```python
features = ctranslate2.StorageView.from_array(inputs.input_features)
```

## CTranslate2 Whisper Method

1. **Detect language:** This method uses the `StorageView` encoding feature to detect the language from the audio.
```python
results = translator.detect_language(features)
language, probability = results[0][0]
print("Detected language %s with probability %f" % (language, probability))
```
```
Output:
Detected language <|en|> with probability 0.995563
```

2. **Generate:** The generate method creates text from audio features and prompts.
```python
prompt = processor.tokenizer.convert_tokens_to_ids(
	[
		"<|startoftranscript|>",
		language,
		"<|transcribe|>",
		"<|notimestamps|>", # Remove this token to generate timestamps.
	]
)

# Run generation for the 30-second window.
results = translator.generate(features, [prompt])
transcription = processor.decode(results[0].sequences_ids[0])
print(transcription)
```
```
Output: 
I had that curiosity beside me at this moment.
```

3. **Align:** Computes the alignment between the text tokens and the encoding features. We need to provide input parameters for this method, including:
	- `start_sequence` is the initial set of tokens or starting point for the alignment process.
	- `text_tokens` are the tokens of the text that the audio features should be aligned with.
	- `num_frames` is the number of non-padding frames in the audio features.	
```python
# Tokenize the text
text_tokens = tokenizer(transcription, return_tensors="pt").input_ids.tolist()
start_sequence = processor.tokenizer.convert_tokens_to_ids(
	[
		"<|startoftranscript|>",
		language,
		"<|transcribe|>",
	]
)

# Perform alignment
alignment_result = translator.align(
	features=features,
	start_sequence=start_sequence,
	text_tokens=text_tokens,
	num_frames=features.shape[-1]
)
```
```
Example Output:
[WhisperAlignmentResult(
alignments=[(0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),,...],
text_token_probs=[0.0, 0.0, 0.9292299151420593, 0.6885610222816467, 0.9857771992683411, 0.9825729131698608, ...]
)]
```

Here’s a simplified breakdown:

- **Alignments:** A list of tuples where each tuple represents a mapping between the encoded audio frames and text tokens. For instance, the tuple (1, 5) indicates that encoded the audio frame 5 corresponds to text token 1.

- **Text Token Probabilities:** A list of probabilities for each text token, indicating the confidence level for each token being represented in the audio. 
```python
# Extract the probabilities of each text token from the alignment result
probs = alignment_result[0].text_token_probs

# Print the header for the output table
print("Token Text Token Probability\n")

# Iterate through the list of probabilities
for i in range(len(probs)):
	# Skip tokens with a probability of 0.0 (usually special tokens or non-relevant tokens)
	if probs[i] == 0:
		continue

	# Decode the text token corresponding to the current index
	# The `processor.batch_decode` function converts token indices to actual text tokens
	token = processor.batch_decode([[text_tokens[0][i]]])[0]

	# Get the probability of the current token
	prob = probs[i]
	
	# Print the token and its probability, formatted for readability
	print(f'{token:<20} {prob:.4f}')
```
```
Output:
Token   Text-Token-Probability
I               0.9292 
had             0.6886 
that            0.9858 
curiosity       0.9826 
beside          0.9067 
me              0.9950 
at              0.9475 
this            0.9877 
moment          0.9983 
.               0.8566
```

# Faster Whisper Transcription with CTranslate2

In the previous section, we explored **CTranslate2** and the functionality of its various methods. In this section, we will introduce a specific implementation of the Whisper model, leveraging the operational principles of **CTranslate2**, known as **Faster-Whisper**.

Faster-Whisper is a re-implementation of OpenAI's Whisper model using **CTranslate2**, a high-performance inference engine designed for Transformer models. This implementation achieves up to a 4x increase in inference speed compared to `openai/whisper`, while maintaining equivalent accuracy and requiring less memory. Additionally, its efficiency can be further enhanced through 8-bit quantization on both CPU and GPU, optimizing performance even more.

## Preparation
- To install **Faster-Whisper**, run:
```
!pip install --force-reinstall "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/refs/heads/master.tar.gz" 
```

## Usage
1. **Load the model**

Example:
```python
model_size = "base" # Use the "base" model
device, compute_type = "gpu", "float16" # Set computation device and precision

# Initialize the Whisper model
model = WhisperModel(model_size, device=device, compute_type=compute_type)
```

Consider your requirements, if you want to use the `large-v3` Whisper model, run it on a `GPU`, and utilize the `int8` format, you can load the model using:

```python
model_size = "large-v3" # Use the "base" model
device, compute_type = "cuda", "int8" # Set computation device and precision

# Initialize the Whisper model
model = WhisperModel(model_size, device=device, compute_type=compute_type)
```
2.  **Sequential Inference with Faster-Whisper**

First, we use the **Sequential inference** method for the Whisper model, where the input audio is segmented and processed sequentially. Each audio segment is transcribed one by one, allowing the model to handle the input step by step.

Here’s a summary of how the Sequential inference method handles transcription:
- **Sequential Processing:** This method segments the audio and processes each segment in sequence. This allows the model to generate text from each audio chunk while maintaining context from previous segments.
- **Language Detection & VAD:** The method can automatically detect the language if it is not specified and filter out non-speech segments using voice activity detection (VAD).
- **Final Output:** The method returns a generator that yields transcribed segments along with additional transcription details.

```python
import time
from faster_whisper import WhisperModel

# Configuration parameters for transcription
word_timestamps = False # Do not include word-level timestamps
vad_filter = True # Apply Voice Activity Detection to remove non-speech segments
temperature = 0.0 # Use deterministic transcription
language = "en" # Set language to English

# Transcribe the audio file with the specified settings
segments, transcription_info = model.transcribe(
	SAMPLE_WAV,
	word_timestamps=word_timestamps,
	vad_filter=vad_filter,
	temperature=temperature,
	language=language,
)
```

Let's break down the output:
- `segments` is a generator, so the transcription process begins only when you start iterating over it.
- `transcription_info` (of type `TranscriptionInfo`) includes information about the audio and transcription, such as language, language probability, duration, and more.

`segments` represent transcription parts extracted from the audio file, including start time, end time, and transcribed text. Iterating through these segments allows you to extract and process the full transcription.

```python
for segment in segments:
	row = {
		"start": segment.start,
		"end": segment.end,
		"text": segment.text,
	}
	
	if word_timestamps:
		row["words"] = [
			{"start": word.start, "end": word.end, "word": word.word}
			for word in segment.words
		]
	
	print(row)
```
```
Output:
{'start': 0.21, 'end': 3.21, 'text': ' I had that curiosity beside me at this moment.'}
```

Additionally, to extracts word-level timestamps from an audio file, providing the start and end times for each word.
```python
segments, _ = model.transcribe(SAMPLE_WAV, word_timestamps=True)
for segment in segments:
	for word in segment.words:
		print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
```
```
Output:
[0.00s -> 0.68s] I 
[0.68s -> 0.82s] had 
[0.82s -> 1.00s] that 
[1.00s -> 1.62s] curiosity 
[1.62s -> 2.12s] beside 
[2.12s -> 2.42s] me 
[2.42s -> 2.58s] at 
[2.58s -> 2.74s] this 
[2.74s -> 3.08s] moment.
```

3. **Batched inference faster-whisper**

Parallel processing of audio chunks can significantly enhance inference performance compared to sequential processing methods, such as those used in sequential inference with **Faster-Whisper**. This method involves:
- Breaking the audio into semantically meaningful chunks.
- Transcribing these chunks in parallel (as batches), utilizing a faster feature extraction process.

This approach results in considerably faster transcription, especially for long audio files, without sacrificing accuracy.
```python
from faster_whisper import WhisperModel, BatchedInferencePipeline

# Configuration settings
model_size = "large-v3" # Use the "large-v3" Whisper model
device, compute_type = "cuda", "float16" # Set computation to use GPU and float16 precision
  
# Initialize the Whisper model
model = WhisperModel(model_size, device=device, compute_type=compute_type)
  
# Wrap the model in a BatchedInferencePipeline for batch processing
batched_model = BatchedInferencePipeline(model=model)

# Perform transcription on the audio file using batch processing with batch_size of 16
segments, info = batched_model.transcribe(filename, batch_size=16)

for segment in segments:
	print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
```
```
Output:
[0.49s -> 3.34s]  I had that curiosity beside me at this moment.
```


# Benchmark Practice

In this session, we will benchmark three models: the **Original Whisper**, **CTranslate2-Whisper**, and **Faster-Whisper**. The input audio is 10 minutes and 21 seconds long. We will use `large-v3` version and compare the speed of these three models in processing this audio.

We conducted experiments in a Colab environment using a `T4 GPU`.
## Original Whisper

For the **Original Whisper**, we will use the HuggingFace implementation, which is slightly faster than the `openai/whisper` implementation available on GitHub.

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained( model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
	"automatic-speech-recognition",
	model=model,
	tokenizer=processor.tokenizer,
	feature_extractor=processor.feature_extractor,
	torch_dtype=torch_dtype,
	device=device,
)

origin_start = time.time()
wave_form, sampling_rate = librosa.load(audio_file, sr=16000, mono=True)
result = pipe(wave_form)
origin_end = time.time()
```

The **Original Whisper** model take `69s` to completely transcript the audio with a high quality transcription.

## CTranslate2 Whisper

- Prepare the model
```python
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
tokenizer = processor.tokenizer

#The original model was converted with the following command:
!ct2-transformers-converter --model openai/whisper-large-v3 --output_dir whisper-large-v3 --copy_files tokenizer.json preprocessor_config.json --quantization float16 --force

translator = ctranslate2.models.Whisper("/content/whisper-large-v3", device="cuda")
```
- Audio transcription
```python
start_time_ct2 = time.time()
wave_form, sampling_rate = librosa.load(audio_file, sr=16000, mono=True)

inputs = processor(wave_form, return_tensors="np", sampling_rate=sampling_rate)
features = ctranslate2.StorageView.from_array(inputs.input_features)

results = translator.detect_language(features)
language, probability = results[0][0]

prompt = processor.tokenizer.convert_tokens_to_ids(
	[
	"<|startoftranscript|>",
	language,
	"<|transcribe|>",
	"<|notimestamps|>", # Remove this token to generate timestamps.
	]
)
  
# Run generation for the 30-second window.
results = translator.generate(features, [prompt])
transcription = processor.decode(results[0].sequences_ids[0])
end_time_ct2 = time.time()
```

Surprisingly, **CTranslate2-Whisper** took only `3.9 seconds` to transcribe the `10m21s` audio with high-quality output. This speed outperforms **Original Whisper** and uses less memory while maintaining quality transcription.

## Faster-Whisper

For **Faster-Whisper**, we will use  `BatchedInferencePipeline` with `bacth_size=16`

```python
from faster_whisper import WhisperModel, BatchedInferencePipeline

# Configuration settings
word_timestamps = False # Disable word-level timestamps
temperature = 0.0 # Set temperature to 0.0 for deterministic transcription
model_size = "large-v3" # Use the "large-v3" Whisper model
device, compute_type = "cuda", "float16" # Set computation to use GPU and float16 precision

# Initialize the Whisper model
model = WhisperModel(model_size, device=device, compute_type=compute_type)

# Wrap the model in a BatchedInferencePipeline for batch processing
batched_model = BatchedInferencePipeline(model=model)

# Perform transcription on the audio file using batch processing with batch_size of 16
faster_start = time.time()
segments, info = batched_model.transcribe(audio_file, batch_size=16, word_timestamps=word_timestamps, temperature=temperature,)
faster_end = time.time()
```

**Faster-Whisper** with `BatchedInferencePipeline` took `4.3 seconds` to complete the transcription. Note that **Faster-Whisper** not only transcribes the audio to text but also performs segmentation. Despite this additional processing, the result of `4.3 seconds` is competitive with **CTranslate2-Whisper**, which only performs transcription. This is a strong performance and still outperforms the **Original Whisper** model.

# Conclusion

In this tutorial, we introduced **CTranslate2-Whisper** and **Faster-Whisper** as advanced alternatives to the original Whisper model. By conducting some simple experiments, we demonstrated that **CTranslate2-Whisper** offers exceptional performance, completing the transcription of a 10-minute and 21-second audio in just `3.9 seconds` with high-quality output and reduced memory usage. **Faster-Whisper**, even with its additional segmentation capabilities, achieved a commendable `4.3 seconds` for transcription, remaining competitive with **CTranslate2-Whisper** and outperforming the original Whisper model.

Both **CTranslate2-Whisper** and **Faster-Whisper** show significant improvements in speed and efficiency, making them valuable tools for high-performance transcription tasks. Whether you prioritize speed, quality, or additional features like segmentation, these models provide excellent options for various use cases.