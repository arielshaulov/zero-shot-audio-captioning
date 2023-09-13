# ZERO-SHOT AUDIO CAPTIONING VIA AUDIBILITY GUIDANCE

<p align="center">
<img src="docs/part1.gif" width="800px"/> 

<a href="https://hila-chefer.github.io/Conceptor/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 
 <a href="https://arxiv.org/abs/2306.00966"><img src="https://img.shields.io/badge/arXiv-2306.00966-b31b1b.svg" height=20.5></a>

> The task of audio captioning is similar in essence to tasks
such as image and video captioning. However, it has received
much less attention. We propose three desiderata for captioning audio – (i) fluency of the generated text, (ii) faithfulness
of the generated text to the input audio, and the somewhat
related (iii) audibility, which is the quality of being able to
be perceived based only on audio. Our method is a zero-shot
method, i.e., we do not learn to perform captioning. Instead,
captioning occurs as an inference process that involves three
networks that correspond to the three desired qualities: (i) A
Large Language Model, in our case, for reasons of convenience, GPT-2, (ii) A model that provides a matching score
between an audio file and a text, for which we use a multimodal matching network called ImageBind, and (iii) A text
classifier, trained using a dataset we collected automatically
by instructing GPT-4 with prompts designed to direct the generation of both audible and inaudible sentences. We present
our results on the AudioCap dataset, demonstrating that audibility guidance significantly enhances performance compared
to the baseline, which lacks this objective.

## Description  
Official implementation of the paper ZERO-SHOT AUDIO CAPTIONING VIA AUDIBILITY GUIDANCE.
 <br>

## Concept Explanation with Conceptor
<p align="center">
<img src="docs/part2.jpg" width="800px"/>  
<br>
Given a concept of interest (e.g., a president) and a text-to-image model, we generate a set of images to visually represent the concept. Conceptor then learns to decompose the concept into a small set of interpretable tokens, with the objective of reconstructing the generated images. The decomposition reveals interesting behaviors such as reliance on exemplars (e.g., "Obama", "Biden").
</p>

## Single-image Decomposition with Conceptor
<p align="center">
<img src="docs/part3.jpg" width="800px"/>  
<br>
 Given a single image from the concept, our method extracts the tokens in the decomposition that caused the generation of the image. For example, a snail is decomposed into a combination of ladybug and winding due to the structure of its body, and the texture of its shell.
</p>

## Concept Manipulation with Conceptor
<p align="center">
<img src="docs/manipulation.jpg" width="600px"/>  
<br>
 Our method enables fine-grained concept manipulation by modifying the coefficient corresponding to a token of interest. For example, by manipulating the coefficient corresponding to the token abstract in the decomposition of the concept sculpture, we can make an input sculpture more or less abstract.
</p>

## Citing our paper
If you make use of our work, please cite our paper:
```
@article{shaharabany2023zero,
        title={Zero-Shot Audio Captioning via Audibility Guidance},
        author={Shaharabany, Tal and Shaulov, Ariel and Wolf, Lior},
        journal={arXiv preprint arXiv:2309.03884},
        year={2023}
      }
```