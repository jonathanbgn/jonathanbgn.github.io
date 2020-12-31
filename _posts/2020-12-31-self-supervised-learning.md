---
layout: post
title: "The Rise of Self-Supervised Learning"
date: 2020-12-31 12:00:00 +0800
---

![Self-Supervised Learning](/assets/images/puzzle.jpg)

Since the Deep Learning wave started in the early 2010s, there has been much hype but also disappointments as many people feel that AI failed to deliver. I believe a big part of this is due to unrealistic expectations driven by amazing progress in research which doesn't translate so well in real life and the industry. But why? What is it that is so different between the academic world and the real world?

I want to expand here a bit on this particular challenge and explain **why the paradigm that we call self-supervised learning could be about to change that**. Self-supervision is not new, but it has seen a resurgence of interest over the past few years, not the least thanks to the relentless popularization of the concept by [Yann Lecun with its cake analogy](https://syncedreview.com/2019/02/22/yann-lecun-cake-analogy-2-0/). I’ll first share a bit about the motivation and history behind the idea. Then, I'll give some examples of its impact on some of the main machine learning domains.

# It’s all about data (no, seriously)

I’m guessing that by now you must have heard once or twice about the importance of data. According to some, it might even be the “new oil”. Yet not all data is equal. Specifically what we care the most about in deep learning is **labeled data: data manually annotated by humans** (for example, the description of what is in an image). It is now evident that deep learning works incredibly well on tasks for which we have large labeled datasets. However, the kind of data available for use in real-world applications is incredibly messy and often doesn’t comes with any label.

The truth is, most of the time, **real-life labeled data is extremely scarce**. Bear witness the fast-growing labeling industry that is [getting established](https://www.ft.com/content/56dde36c-aa40-11e9-984c-fac8325aaa04). There’s a good chance that you will have to label it yourself as step one when you start a new machine learning project. Even if you are very lucky and the data is already labeled, there might still be issues with the labeling quality or consistency, or there might just be too much noise in your dataset. This problem is even worst if you’re working on an application that is particularly niche or a domain/language that is uniquely different from the rest.

What I’m saying might seem evident, but in this era of new sexy optimization algorithms or model architecture like Transformers, people sometimes forget how essential data is in the picture. For example, when asked about what caused the “deep learning revolution”, a common answer is the “ImageNet Moment” attributed to the 2012 release of the [AlexNet model architecture](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) which attained much better performance on the ImageNet competition.

But what is often omitted in the story is that ImageNet was released only 2 years before. One could probably argue that with the release of such a high-quality large labeled dataset it was only a matter of time before someone comes up with a deep learning architecture that demonstrates the superiority of deep learning over other techniques. In fact, the AlexNet architecture is strikingly similar to the 1998’s LeNet-5 architecture, except of course much larger and deeper (it was also the first to stack convolutional layers directly on top of each other, removing intermediary pooling layers). Considering this, “the ImageNet moment” could be in 2010 upon the first release of the dataset, rather than the creation of a new architecture.

![Computer Vision Datasets](/assets/images/dataset_size_vision.png)

[Revisiting Unreasonable Effectiveness of Data in Deep Learning Era](https://arxiv.org/pdf/1707.02968.pdf)

# Why we need self-supervised learning

The problem is clear: **we don’t have enough labeled data for real-life applications**. Should we spend more effort to label existing data then? Well, the thing is labeling data can be extremely costly and time-consuming. Moreover, deep learning has an insatiable appetite for labeled data. You can easily need in the order of millions of labeled examples for your model to generalize effectively. So for many concrete use cases, this is just not a practical solution. On the other hand, there are tons of data without labels out there. In fact, **unlabeled data is free**.

**Self-supervised learning makes the promise to use all data available, even without labels.** The basic idea is simple: instead of using independent human-made labels, we use one part of the input as the label itself. So self-supervised learning allows to automatically generate labels from the data without any external assistance.

> Self-supervised learning is a game-changing paradigm, it effectively shifts the bottleneck from the data to the compute capacity.

An easy-to-grasp example of this concept is a language model: a model trained to predict the next word in a sentence. One can automatically download a large amount of text from the Internet and automatically split a sentence into its beginning and the next word to predict. It costs nothing to create one labeled example and the amount of textual data on the Internet is now the limit. In a way, for some tasks, the amount of data available is practically “unlimited”, and the compute power now becomes the limit for reaching better performance. The recent [GPT-3 model](https://en.wikipedia.org/wiki/GPT-3) is the perfect example of this.

![Language Model](/assets/images/quotes_language_model.png)

*A language model like the GPT family predicts the next token based on what already exists.*


# Transferring knowledge where it is needed

But the true power of self-supervised learning does not lie in using it just for a few tasks like language modeling for which we can generate an infinite amount of labeled data. **The key is to use it as a transfer learning method**, where we transfer the knowledge contained in a model trained on a task for which plenty of data is available to the objective task for which data is limited. This is usually done in two training steps. In the first step we *pre-train* our model in a self-supervised way on a *pretext task* such as language modeling (also called the *auxiliary task* in the literature). In the second step, we *fine-tune* our model on the downstream task, which is the one that we care about. This second step is done in a standard supervised way, trained on human-labeled data.

![Transfer Learning](/assets/images/transfer_learning.png)

*Transfer learning works by re-using knowledge of a model trained on another task for which data is abundant ([Image credit](https://datascience.aero/transfer-learning-aviation/))*

An important point here is the choice of the pretext task. This task must make sense such that solving it must be in some way useful to the downstream task. In other words, the pretext task should force the model to encode the data to a meaningful representation that can be re-used for fine-tuning later. For example, a language model trained in a self-supervised way needs to learn about meaning and grammar to effectively predict the next word. This linguistic knowledge can be re-used in a downstream task like predicting the sentiment of a text.

# Robust performance on small data

While we still need some domain-specific labeled data for the second fine-tuning step, pre-training first in a self-supervised way has been shown to immensely improve the performance on the downstream task, even when very few labeled data is available for fine-tuning. For example, the [ULMFiT paper](https://arxiv.org/abs/1801.06146) showed that it is possible to reach great performance for text classification with only 100 labeled examples. More recently, a new paper from DeepMind outperformed the original AlexNet performance on ImageNet with only 13 labeled examples per class.

![Computer Vision Datasets](/assets/images/cpc_self_supervised_performance.png)

[Data-Efficient Image Recognition with Contrastive Predictive Coding](https://arxiv.org/pdf/1905.09272.pdf)


There is evidence that self-supervised learning could be closer to the way humans learn than other machine learning paradigms. Surely, children don’t need their parents to tell them 1 million times what a car is. Just pointing to a few different cars over a few years is enough for a child to understand the concept of a car. This tends to support the hypothesis that, even before being told what a car is by a supervisor (parent), children somehow already self-developed prior concepts (size and shape of cars, wheels, motion…) to help them understand the world better. Indeed it has been shown that, at a very young age, babies tend to learn largely by observation solely, with very little interaction.

> Self-supervised learning does not remove the need for labeled data, but it greatly reduces the need for it and makes it practical to deploy deep learning models for use cases in which labeled data is (very) limited.

And this is exactly why it might be the key to unlock the promises of deep learning in real-world applications and for the industry. Think about it, since the digital revolution began, companies have been collecting tons of proprietary data in the hope of profiting from it in the future. However, they don’t know what to do with it so far as most of it is messy, unstructured, and most importantly unlabeled. Bringing self-supervised learning in the equation could help unleash the latent value from these tons of data.

# A very short history of self-supervised learning

While there has been a recent surge of interest in self-supervised learning, this is **not a new idea**. Back in 2006, Geoffrey Hinton et al. published a [seminal paper](https://www.cs.toronto.edu/~hinton/absps/ncfast.pdf) about pre-training Restricted Boltzmann Machines in an unsupervised way followed by a fine-tuning phase. This created a lot of interest in the method, but it quickly faded as people realized that if you have a large labeled dataset, then it is enough to train directly in a supervised way.

![Interest in Self-Supervised Learning](/assets/images/self_supervised_learning_interest.png)

But the idea might be even older. The earliest usage of the term on Google Scholar is in a 1978 paper, and multiple papers were referencing it [back in the 1990s](https://twitter.com/phillip_isola/status/1216902657702617093). Even before the use of the term itself, related ideas like language models were already popularized. The first N-Gram language models are at least as old as Claud Shannon’s 1948 seminal paper [A Mathematical Theory of Communications](http://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf), which set the foundation of Information Theory.

That the concept is now experiencing a resurgence is thus less due to a breakthrough in theory but more to the exponentially available quantity of unlabeled data and the increasing need to be able to extract value from it.

# Natural language processing

NLP is THE field in which self-supervised learning first became hugely popular, and as such has many examples of the technique. The most straight-forward example that I already mentioned in the language model, predicting the next word in a sentence. Despite being one of the earliest examples of the method, it is one of the most popular today due to big models like [GPT-3](https://en.wikipedia.org/wiki/GPT-3).

But language models are far from being the only examples in the field. A variety of word embeddings approaches such as [word2vec](https://en.wikipedia.org/wiki/Word2vec) or [GloVe](https://en.wikipedia.org/wiki/GloVe_(machine_learning)) also revolutionized the field back in the 2010s. The idea was simple, instead of predicting the next word, we could just ask the model to predict a word based on its context or vice versa. We then obtain so-called distributed representations of words (embeddings) which encode meaningful information about words and can be re-used in all sorts of problems.

Today, the most popular self-supervised approach is arguably the masked language modeling one, which the now famous [BERT model](https://en.wikipedia.org/wiki/BERT_(language_model)) exemplifies. By “masking” random words in the text and asking the model to fill the holes, we can learn meaningful representations that can be later used or fine-tuned in a variety of downstream tasks.

The quest for the optimal pretext task is far from complete! Just this month at NeurIPS 2020, Facebook AI shared a promising **new self-supervised learning technique: paraphrasing**. Their new model, [MARGE](https://arxiv.org/abs/2006.15020), finds related documents and then paraphrases them to reconstruct the original. This approach even allows for good enough performance on some tasks without a fine-tuning step on extra labeled data.

![Paraphrasing](/assets/images/nlp_paraphrasing.png)


# Computer vision

Unlike NLP, self-supervision has not yet seen widespread adoption here, mostly because current techniques are still a bit too complex to implement. Yet this might change very soon. Influenced by the success of the concept in NLP, research has been very active in the area. This year alone, there’s been an explosion of papers on self-supervised learning in image recognition, and performance obtained through self-supervision is starting to match end-to-end supervised models even when labeled data is abundant like for ImageNet.

[Many kinds of pretext tasks](https://www.fast.ai/2020/01/13/self_supervised/) are being used today, such as colorization (black & white to colors), patch placement (ordering patches cut from an image like a puzzle), or inpainting (filling an automatically created gap). One of the most interesting idea is the one of contrastive learning, where we train a neural network to generate consistent representations between different views of the same image (transformed such as by cropping or rotation) and distant representations of different images. A good example of the idea is the [SimCLR framework](https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html) from Google.

Yet there are still some important problems that must be solved before we see wider adoption of self-supervision in vision. Often, many "tricks" (data augmentation, negative sampling...) are used during the self-supervised phase, and these can be very domain-specific and hard to implement. Moreover, the losses can be quite complex, and difficult to optimize properly.


# Speech processing

Last but not least, self-supervised learning is also just starting to gain traction in speech processing. I am personally very enthusiastic about the future of this field, as I’m currently working on [Speech Emotion Recognition]({% post_url 2020-10-31-emotion-recognition-transfer-learning-wav2vec %}), a hard task for which labeled data is extremely scarce. Using self-supervised learning, we were able to beat strong baselines while **using only 100 examples per emotion** and [improve upon the state-of-the art](https://arxiv.org/abs/2011.05585) when using all data. I deeply believe that self-supervised learning is the key that will allow the emotion recognition field to make large progress in the coming years, and progress from basic emotion detection to more complex characterization of moods and states-of-mind.

The [wav2vec model](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/), inspired by word2vec, is in my opinion the most successful attempt so far for bringing self-supervised learning into the field. The model saw successive improvement and version 2.0 was just published at NeurIPS this year. While the model was originally only made of convolutional networks, the Facebook AI team then added a Transformer network to the architecture and, inspired by NLP, added an automatic discretization of inputs to be able to use some of the NLP models that work well on discrete data. The method allowed the creation of [robust Automatic Speech Recognition (ASR) models with only 10 minutes of labeled data!](https://arxiv.org/abs/2010.11430)

![Wav2vec Model](/assets/images/wav2vec_objective.jpg)

*Wav2vec uses contextual representations to distinguish between real and fake future samples of audio. Here "CUH" is the correct sample.*

Wav2vec is not the only self-supervised approach in this domain. Google AI has been working with the huge [AudioSet dataset](https://research.google.com/audioset/) which contains more than 2 million audio clips to create unsupervised embeddings for speech. The [resulting embeddings called TRILL](https://ai.googleblog.com/2020/06/improving-speech-representations-and.html) allowed to reach very competitive performance and even state-of-the-art on a variety of tasks such as speaker identification, language identification, emotion detection, or even dementia detection.

# The future will be self-supervised

![Number of Self-Supervised Learning Papers](/assets/images/self_supervised_learning_papers.png)

*Number of papers containing the term "self-supervised learning"*

I believe that self-supervised learning will soon become the gold standard in most fields of machine learning. As the techniques become better and deep learning frameworks remove implementation complexity, there will be wider adoption across all areas. We are also starting to see public repositories of models trained in a self-supervised way that anyone can download for their use case, such as on the [HuggingFace website](https://huggingface.co/models). The mass sharing of already trained self-supervised model will help democratize the use of the technique to every application.

