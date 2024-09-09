# Data Augmentation in Speech Processing

## Introduction to Data Augmentation

Data augmentation is a crucial technique for training deep learning models. In low-resource environments, where data may be limited, data augmentation helps enrich the training set by applying various transformations to the existing data.

Data augmentation is not only beneficial when data is scarce but also when we want to introduce new features or increase the diversity of our training data. 

For example, by adjusting the lighting in an image to simulate different times of day, we can train a vision model to perform well regardless of lighting conditions throughout the day.

<-image for example->

To summarize, data augmentation is typically used in the following scenarios:


- When there is insufficient data, and we want to enrich our dataset through transformations of the existing data.
- To increase the diversity of the dataset or adapt it to specific scenarios, which reduces the cost of data collection and enhances the robustness of the model.

Different types of data require different augmentation techniques and strategies. In this tutorial, we will focus on hands-on augmentation techniques specifically for audio processing. We will use `audiomentations`, a library design for audio augmentation.

## Preparation

### Install libraries
```bash
pip install -r requirements.txt
```
### Import libraries
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

## Audio Augmentation
### Time stretching 
Time stretching in audio processing is a technique used to alter the duration of an audio signal without changing its pitch.

```
from audiomentations import TimeStretch

transform = TimeStretch(
    min_rate=0.8,
    max_rate=1.25,
    leave_length_unchanged=True,
    p=0.5
)

augmented_sound = transform(my_waveform_ndarray, sample_rate=16000)
```

This code will adjust the speed of your audio file by 
 - A rate chosen randomly from a uniform distribution within the range `[min_rate, max_rate]`. 
 - `p` is the probability to apply this transformation.

When adjusting the speed of your audio, the duration of the audio may change. You can control whether the duration remains unchanged by setting the `leave_length_unchanged` parameter:

- Set `leave_length_unchanged=True` (default) to keep the duration of the audio the same.
- Set `leave_length_unchanged=False` if you want the duration to change with the speed adjustment.

For instance, if you want to speed up your audio to a rate of 1.25, you can use the following code:
```
from audiomentations import TimeStretch

transform = TimeStretch(
    min_rate=1.25,
    max_rate=1.25,
    leave_length_unchanged=True,
    p=1.0
)

augmented_sound = transform(data, sample_rate=16000)
ipd.Audio(augmented_sound, rate=sampling_rate)
```


For example:

```
from audiomentations import TimeStretch

transform = TimeStretch(
    min_rate=0.7,
    max_rate=0.7,
    leave_length_unchanged=True,
    p=1.0
)

augmented_sound = transform(data, sample_rate=16000)
ipd.Audio(augmented_sound, rate=sampling_rate)
```

**Note:** Phase vocoding is used here, but it can degrade audio quality by blurring transients, changing timbre, and distorting pitch. This may lead to reduced sharpness, clarity, or naturalness, especially at extreme settings.

### Shift pitch

Pitch shifting is a technique in audio processing that alters the pitch of an audio signal without changing its duration. 


```
from audiomentations import PitchShift

transform = PitchShift(
    min_semitones=-5.0,
    max_semitones=5.0,
    p=1.0
)

augmented_sound = transform(my_waveform_ndarray, sample_rate=44100)
```

This code adjusts the pitch of your audio file by a semitone randomly chosen from a uniform distribution within the range `[min_rate, max_rate]`. The probability of applying this transformation is given by `p`.



**Note:** This process uses time stretching (via phase vocoding) followed by resampling. Phase vocoding can reduce audio quality by blurring transients, altering timbre, and distorting pitch, which may lead to less sharpness, clarity, or naturalness.

### Add Noise

Add Noise is a technique used in audio augmentation to enhance the robustness and versatility of audio processing models. 

By introducing various types of noise—such as white noise, Gaussian noise, or background noise—into the audio data, this method helps simulate different acoustic environments and recording conditions. 


For example, to add **Gaussian Noise**:

```
from audiomentations import AddGaussianNoise

transform = AddGaussianNoise(
    min_amplitude=0.001,
    max_amplitude=0.015,
    p=1.0
)

augmented_sound = transform(my_waveform_ndarray, sample_rate=16000)
```

### Shift

```
from audiomentations import TimeMask

transform = TimeMask(
    min_band_part=0.1,
    max_band_part=0.15,
    fade=True,
    p=1.0,
)

augmented_sound = transform(my_waveform_ndarray, sample_rate=16000)
```

## SpecAugment: A Simple Data Augmentation Method for ASR

In the previous session, we explored various data augmentation techniques for raw audio. Audio augmentation is crucial for training deep learning models, as it enhances model robustness.

Deep learning models for speech processing typically use spectrograms rather than raw audio. Inspired by this, Google Brain introduced SpecAugment in 2019. This method applies three key distortions—frequency masking, time masking, and time warping—directly to spectrograms. These distortions simulate different acoustic conditions, improving model generalization to noisy, distorted, or variable speech.

SpecAugment is simple, computationally inexpensive, and shows promising results for training ASR systems.

The key idea behind SpecAugment is to augment the training data by applying distortions directly to the spectrograms of audio signals. This method involves three main operations:

- **Frequency Masking**: Randomly masking out continuous bands of frequencies in the spectrogram to make the model less sensitive to missing frequency components.
- **Time Masking**: Randomly masking out segments of the time axis to simulate variations in speech timing and make the model more resilient to temporal distortions.
- **Time Warping**: Slightly warping the time axis of the spectrogram to introduce temporal variations.