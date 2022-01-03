---
layout: post
title: "A Timeline of Transformers for Speech"
date: 2021-12-31 18:00:00 +0800
---

Since the arrival of the [Transformer architecture](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) in 2017, transformers have made their way into all domains of machine learning. As [Andrej Karpathy of Tesla noted](https://twitter.com/karpathy/status/1468370605229547522), transformers are quickly becoming the go-to architecture as either a strong baseline or state-of-the-art performance for most problems in language, vision, and now speech.

In speech processing, we recently saw a wave of new transformer-based models leveraging clever tricks to apply the original transformer architecture on continuous acoustic data. And popular libraries like HuggingFace’s Transformers have been implementing more and more of these speech models.

![A Timeline of Transformers for Speech](/assets/images/transformers-speech/timeline-transformers-speech.png)

I'd like to cover here some of the most impactful transformer-based speech models of recent years. This list is not exhaustive as we recently saw an explosion of such models. I also focused mainly here on self-supervised models and ASR-related solutions.

---

## 2018: Starting to replace CNNs with Transformers


The transformer architecture was originally built for automatic language translation, and most of the attention initially focused on natural language processing applications. Only in 2018 did researchers start to think about how to apply the transformer architecture to speech. We saw multiple works trying to replace convolutional neural networks with transformers in speech transducer models, with increased performance over traditional architectures.

As more recent self-supervised approaches have taken the spotlight, I decided not to focus too much on models from that year and instead start focusing on models from 2019 onward.

---

## 2019: VQ-Wav2vec, Mockingjay, DiscreteBERT

### VQ-Wav2vec

The [original wav2vec model]({% post_url 2021-06-29-illustrated-wav2vec %}) did not use transformers and instead relied on two distinct convolutional networks. Its immediate successor, vq-wav2vec, re-used the same architecture but applied a quantization process to transform continuous acoustic features into discrete units (*VQ* stands for *Vector Quantization*). These discrete representations were then used to train a BERT model, a large transformer encoder network.

![Architecture of vq-wav2vec](/assets/images/transformers-speech/vq-wav2vec.png)

In contrast to the original BERT model, only the Masked Language Modelling objective was used, as the next sentence prediction objective could hinder performance for speech.

---

### Mockingjay

This model created by researchers at National Taiwan University experimented with applying transformer encoders directly to the continuous audio features, rather than discretizing first like vq-wav2vec. Inspired by BERT, the pre-training process also masks a random subset of audio frames for prediction. However, since the frames were not discretized, the model tries to reconstruct the original features frame, and an L1 reconstruction loss is used as the objective (hence the model is named after a bird that mimics sound).

![Architecture of Mockingjay](/assets/images/transformers-speech/mockingjay.png){:width="400" style="display: block; margin: auto;"}

The authors report improved performance on a variety of downstream tasks like phoneme classification, speaker recognition, and sentiment classification, after either extracting latent representations from the last layer of the model, a combination of intermediate hidden representations (weighted like ELMo), or directly fine-tuning the model to the downstream tasks.

---

### DiscreteBERT

With the same architecture and training process as vq-wav2vec, DiscreteBERT confirms the benefits of using discrete units as an input for the transformer encoder rather than continuous features. There is some slight difference in the training approach compared to vq-wav2vec, and you can have a look at [the paper](https://arxiv.org/abs/1911.03912) for more details.

![Architecture of DiscreteBERT](/assets/images/transformers-speech/discrete-bert.png)

---

## 2020: Conformer, Wav2vec 2.0, DeCoAR 2.0...

### Conformer

In the first half of 2020, researchers at Google combined convolution neural networks to exploit local features with transformers to model the global context and achieved state-of-the-art performance on LibriSpeech. This out-performed previous Transformer-only transducers by a large margin, demonstrating the advantage of combining CNN with Transformer for end-to-end speech recognition. Their architecture replaces the Transformer blocks with Conformer blocks, which add a convolution operation.

![Conformer Architecture](/assets/images/transformers-speech/conformer.png){:width="400" style="display: block; margin: auto;"}
*The conformer model combines self-attention with convolution wrapped in two feed-forward modules.*

---

### Wav2vec 2.0 / XLSR-53

A continuation of the wav2vec series, wav2vec 2.0 replaces the convolutional context network of the original architecture with a transformer encoder. Despite the use of discrete speech units and a quantization module like the vq-wav2vec model, wav2vec 2.0 goes back to the original contrastive objective used in the first version of wav2vec, rather than the BERT's masked language modeling objective.

![Wav2vec 2.0, based on a transformer encoder](/assets/images/illustrated-wav2vec/wav2vec2_architecture_pretraining.png)
*Image from [An Illustrated Tour of Wav2vec 2.0]({% post_url 2021-09-30-illustrated-wav2vec-2 %}).*

A few days after releasing the wav2vec 2.0 preprint, Facebook AI also released **XLSR-53**, a multi-lingual version of the wav2vec 2.0 model trained in 53 languages.

---

### w2v-Conformer

In a preprint released in October 2020, a team at Google merged the Conformer architecture with the wav2vec 2.0 pre-training objective and [noisy student training](https://paperswithcode.com/method/noisy-student) to reach new state-of-the-art results on LibriSpeech speech recognition. The team also scaled the model to a much larger dimension than the original wav2vec 2.0 paper, with a *Conformer XXL* model using 1 billion parameters.

![w2v-Conformer Architecture](/assets/images/transformers-speech/w2v-conformer.png)

---

### DeCoAR 2.0

This model from Amazon Web Services is the successor of a non-transformer LSTM-based model named DeCoAR (Deep Contextualized Acoustic Representations), which takes its inspiration directly from the popular [ELMo model](https://en.wikipedia.org/wiki/ELMo) in natural language processing. This second version replaces the bi-directional LSTM layers with a transformer encoder and uses a combination of the wav2vec 2.0 loss and another reconstruction loss as its objective.

![DeCoAR 2.0 Architecture](/assets/images/transformers-speech/decoar-2.png)

---

## 2021: UniSpeech, HuBERT, XLS-R, BigSSL...

### UniSpeech

In early 2021, Microsoft released a multi-task model that combines a self-supervised learning objective (same as wav2vec 2.0) with a supervised ASR objective (Connectionist temporal classification). This joint optimization allowed for better alignment of the discrete speech units with the phonetic structure of the audio, improving performance on multi-lingual speech recognition and audio domain transfer.

![UniSpeech Architecture](/assets/images/transformers-speech/unispeech.png)
*Image from the original [UniSpeech paper](https://arxiv.org/abs/2101.07597)*

---

### HuBERT

Coming also from Facebook/Meta AI, HuBERT re-uses the wav2vec 2.0 architecture but replaces the contrastive objective by BERT's original masked language modeling objective. This is made possible by a pre-training process alternating between two steps: a clustering step where pseudo-labels are assigned to short segments of speech, and a prediction step where the model is trained to predict these pseudo-labels at randomly-masked positions in the original audio sequence.

![HuBERT Explained](/assets/images/illustrated-hubert/hubert_explained.png)
*Image from [HuBERT: How to Apply BERT to Speech, Visually Explained]({% post_url 2021-10-30-hubert-visually-explained %}).*

---

### w2v-BERT

w2v-BERT comes from researchers at Google Brain and MIT and combines concepts from wav2vec 2.0, BERT, and Conformer. Similar to w2v-Conformer, it re-uses the wav2vec 2.0 architecture but with transformer layers replaced by conformer layers. It also combines the contrastive loss of wav2vec 2.0 with the masked language modeling objective of BERT, allowing for an end-to-end training process of MLM without the need to alternate between processes like HuBERT. Its largest version, *w2v-BERT XXL*, scales to **1 billion parameters**, similar to the largest w2v-Conformer model.

![w2v-BERT Architecture](/assets/images/transformers-speech/w2v-bert.png){:width="400" style="display: block; margin: auto;"}

---

### XLS-R

A scaled-up version of XLSR-53, based on wav2vec 2.0. This very large model uses **2 billion parameters** and is trained on half a million hours of speech in 128 different languages. This is more than twice the original 53 languages used by XLSR-53. XLS-R attains state-of-the-art performance in speech translation to English and language identification.

![XLS-R Architecture](/assets/images/transformers-speech/xls-r.png)
*Image source from [Meta AI blog](https://ai.facebook.com/blog/xls-r-self-supervised-speech-processing-for-128-languages/)*

The model also seems to perform as well as English-only pre-trained versions of wav2vec 2.0 when translating from English speech, hinting at the potential of multi-lingual models compared to monolingual ones.

---

### BigSSL

[BigSSL](https://arxiv.org/abs/2109.13226) is a continuation of Google's effort with its w2v-Conformer, scaling up both the model size and data (unlabeled and labeled data), and trained on approximately one million hours of audio. It is the largest of such speech Transformer models so far, with a staggering **8 billion parameters**. This work highlights the benefits of scaling up speech models which, just like in natural language processing, benefit downstream tasks when trained on more data and with more parameters.

---

### UniSpeech-SAT / WavLM

Both coming from Microsoft, the UniSpeech-SAT and WavLM models follow the HuBERT framework while focusing on data-augmentation during the pre-training stage to improve speaker representation learning and speaker-related downstream tasks.

![UniSpeech-SAT Architecture](/assets/images/transformers-speech/unispeech-sat.png)

The WavLM model is especially efficient for downstream tasks, it is currently leading the [SUPERB leaderboard](https://superbbenchmark.org/leaderboard), a performance benchmark for re-using speech representations in a variety of tasks such as automatic speech recognition, phoneme recognition, speaker identification, emotion recognition...

---

## What does 2022 hold for speech processing?

In light of all the innovations that happened in 2021, we can be quite optimistic for 2022! We will surely see even larger transformer models and better ways to transfer knowledge from pre-trained models to downstream tasks. In another post I’ve listed what I think could be some of the most interesting [trends in speech processing for 2022]({% post_url 2021-11-30-speech-processing-trends-2022 %}).



