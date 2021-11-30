---
layout: post
title: "2022 Trends for Speech Processing"
date: 2021-11-30 20:00:00 +0800
---

It’s an exciting time to be working in speech processing. The field has seen a lot of transformations and changes just over the past few years. In this post, I’d like to share my perspective on how I see the field moving forward in 2022, based on what we saw recently.

---

## More Transformers

After revolutionizing the field of natural language processing, and then computer vision, the [transformer architecture](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) is now being adopted more and more in the speech processing community. It still feels like early days though, when you compare the diversity of transformer-based speech models compared to the multitude of such models in NLP.

Most visible so far has been the [wav2vec 2.0 model]({% post_url 2021-09-30-illustrated-wav2vec-2 %}) as well as [HuBERT]({% post_url 2021-10-30-hubert-visually-explained %}), both coming from Facebook AI (now called Meta AI). The use of transformers, where each position attends to every other one, enables much richer contextual representations of audio sequences, compared to previous methods which relied on recurrent neural networks like LSTM to build sequence-level embeddings.

![Wav2vec 2.0, based on a transformer encoder](/assets/images/illustrated-wav2vec/wav2vec2_architecture_pretraining.png)
*The wav2vec 2.0 model is based on the transformer encoder. Image from [An Illustrated Tour of Wav2vec 2.0]({% post_url 2021-09-30-illustrated-wav2vec-2 %}).*

---

## Larger Models and Datasets

As the field will likely follow what happened in NLP after the introduction of transformers, we’ll see bigger models emerge in the next year. We are still far from what we see in NLP with GPT-3 and its 175 billion parameters though. The largest transformer-based speech model (**XLS-R**) has slightly more than 2 billion parameters, around the same order of magnitude as the largest version of GPT-2, back in 2019.

![NLP Model Size](/assets/images/speech-processing-trends-2022/nlp_model_size.jpg)

*There seems to be no end in sight to NLP models scaling ([image source](https://huggingface.co/blog/large-language-models)).*

Speech datasets too are being scaled up in size. The [Common Voice initiative](https://commonvoice.mozilla.org/en) by Mozilla has already collected more than 13,000 hours of voice samples from 75,000+ speakers in 76 languages. The [VoxPopuli dataset](https://ai.facebook.com/blog/voxpopuli-the-largest-open-multilingual-speech-corpus-for-ai-translation-and-more/), open-sourced this year, offers 400,000 hours of unlabelled speech in 23 languages, making it the largest open speech dataset so far.

---

## Less Supervision

As model sizes scale, so must training data. While we will see bigger labeled datasets, this will be far from being enough to feed the ever-larger capacity of these models. New approaches will likely be developed to train larger models more effectively in a self-supervised way, without the need to label data. We already saw a lot of different self-supervised training methods based on contrastive losses or unsupervised clustering to create pseudo-labels. More will come.

Fully unsupervised ASR systems are also starting to make their apparition: [wav2vec-U](https://ai.facebook.com/blog/wav2vec-unsupervised-speech-recognition-without-supervision/), released this year, shows comparable performance to the best supervised models from only a few years ago, while using no labeled data at all.

![Wav2vec-U Performance](/assets/images/speech-processing-trends-2022/wav2vec-u-performance.jpg)
*Wav2vec-U shows impressive performance on speech recognition without using any labeled data (Librispeech benchmark) ([image source](https://ai.facebook.com/blog/wav2vec-unsupervised-speech-recognition-without-supervision/)).*

---

## Multilingual Models

The rise of [self-supervised learning]({% post_url 2020-12-31-self-supervised-learning %}) has opened the door to effective speech recognition for low-resource languages for which labeled datasets aren't available, or too small to be able to generalize from. The just-released [XLS-R model](https://ai.facebook.com/blog/xls-r-self-supervised-speech-processing-for-128-languages/), the largest of such models yet with over 2 billion parameters, has been trained on 128 languages, more than twice its predecessor. Interestingly, such large multilingual models improve performance much more when scaling the model size. This will probably lead to a preference for multilingual models over single language ones in the future.

![XLS-R speech translation performance](/assets/images/speech-processing-trends-2022/xls-r_speech_translation.png)
*Speech translation performance of XLS-R from target language to English ([image source](https://ai.facebook.com/blog/xls-r-self-supervised-speech-processing-for-128-languages/)).*

---

## Textless NLP

Generative models like GPT-3 are certainly impressive, but they don't capture all the subtlety of languages. This year researchers at Meta AI released the [Generative Spoken Language Model (GSLM)](https://ai.facebook.com/blog/textless-nlp-generating-expressive-speech-from-raw-audio/), capable of generating speech continuation without any dependency on text. This kind of textless NLP model enables much richer language expression by capturing much more para-linguistic information than just semantics. Tone, emotions, speakers voice characteristics... all these elements can now be encoded both for classification or generation.

![Generative Spoken Language Model](/assets/images/speech-processing-trends-2022/generative-spoken-language-model.jpg)
*Architecture of the Generative Spoken Language Model, a textless generative model ([image source](https://ai.facebook.com/blog/textless-nlp-generating-expressive-speech-from-raw-audio/)).*

Such progress in speech processing could lead to many breakthrough applications, like expressive searchable speech audio, better voice interfaces and assistants, and immersive entertainment.

---

## Better Libraries and Tools

![HuggingFace Library](/assets/images/speech-processing-trends-2022/huggingface-transformers.png)

Speech processing used to be much harder, but things are now changing fast. This year, the popular HuggingFace Transformers library started implementing speech models like wav2vec 2.0 and HuBERT. The [SpeechBrain tookit](https://speechbrain.github.io/index.html) has also seen much progress, and the [TorchAudio library](https://github.com/pytorch/audio) got some serious improvements this year, with new models added like Tacotron2 as well as easy access to some of the most popular speech datasets. We should expect more to come soon. Exciting times ahead!


