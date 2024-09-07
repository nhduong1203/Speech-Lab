# Data Augmentation in Speech Processing

## Introduction to Data Augmentation

Data augmentation is a crucial technique for training deep learning models. In low-resource environments, where data may be limited, data augmentation helps enrich the training set by applying various transformations to the existing data.

Data augmentation is not only beneficial when data is scarce but also when we want to introduce new features or increase the diversity of our training data. 

For example, by adjusting the lighting in an image to simulate different times of day, we can train a vision model to perform well regardless of lighting conditions throughout the day.

<-image for example->

To summarize, data augmentation is typically used in the following scenarios:

- When there is insufficient data, and we want to enrich our dataset through transformations of the existing data.
- To increase the diversity of the dataset or adapt it to specific scenarios, which reduces the cost of data collection and enhances the robustness of the model.

Different types of data require different augmentation techniques and strategies. In this tutorial, we will focus on hands-on augmentation techniques specifically for audio processing

## Preparation
## Import libraries
```python
import torch
import torchaudio
import torchaudio.functional as F

import matplotlib.pyplot as plt

from IPython.display import Audio
```

## Load audio file
```python
sample_audio = "./asset/audio.wav"
waveform1, sample_rate = torchaudio.load(sample_audio, channels_first=False)

```