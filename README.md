FORK

Purpose of this fork is to support training AudioLDM models with Tango code, including audio diffusion conditioned on audio.


# TANGO: Text to Audio using iNstruction-Guided diffusiOn
<!-- ![cover](img/tango-neurips.png) -->

[Paper](https://arxiv.org/pdf/2304.13731.pdf) | [Model](https://huggingface.co/declare-lab/tango) | [Website and Examples](https://tango-web.github.io/) | [More Examples](https://github.com/declare-lab/tango/blob/master/samples/README.md) | [Demo](https://huggingface.co/spaces/declare-lab/tango)

:fire: The demo of **TANGO** is live on [Huggingface Space](https://huggingface.co/spaces/declare-lab/tango)

📣 We are inviting collaborators and sponsors to train **TANGO** on larger datasets, like AudioSET

<br>

**TANGO** is a latent diffusion model (LDM) for text-to-audio (TTA) generation. **TANGO** can generate realistic audios including human sounds, animal sounds, natural and artificial sounds and sound effects from textual prompts. We use the frozen instruction-tuned LLM Flan-T5 as the text encoder and train a UNet based diffusion model for audio generation. We perform comparably to current state-of-the-art models for TTA across both objective and subjective metrics, despite training the LDM on a 63 times smaller dataset. We release our model, training, inference code, and pre-trained checkpoints for the research community.

<!-- ## Abstract
The immense scale of the recent large language models (LLM) allows many interesting properties, such as, instruction- and chain-of-thought-based fine-tuning, that has significantly improved zero- and few-shot performance in many natural language processing (NLP) tasks. Inspired by such successes, we adopt such an instruction-tuned LLM _Flan-T5_ as the text encoder for text-to-audio generation, where the prior works either pre-trained a joint text-audio encoder or used a non-instruction-tuned model, such as, _T5_. Consequently, our latent diffusion model (LDM)-based approach (_**Tango**_) outperforms the state-of-the-art AudioLDM, despite training the LDM on a 63 times smaller dataset and keeping the text encoder frozen. This improvement can also be attributed to the adoption of audio pressure level-based sound mixing for training set augmentation, whereas the prior methods take a random mix. -->

<p align="center">
  <img src=img/tango.png />
</p>

## Quickstart Guide

Download the **TANGO** model and generate audio from a text prompt:

```python
import IPython
import soundfile as sf
from tango import Tango

tango = Tango("declare-lab/tango")

prompt = "An audience cheering and clapping"
audio = tango.generate(prompt)
sf.write(f"{prompt}.wav", audio, samplerate=16000)
IPython.display.Audio(data=audio, rate=16000)
```
[CheerClap.webm](https://user-images.githubusercontent.com/13917097/233851915-e702524d-cd35-43f7-93e0-86ea579231a7.webm)

The model will be automatically downloaded and saved in cache. Subsequent runs will load the model directly from cache.

The `generate` function uses 100 steps by default to sample from the latent diffusion model. We recommend using 200 steps for generating better quality audios. This comes at the cost of increased run-time.

```python
prompt = "Rolling thunder with lightning strikes"
audio = tango.generate(prompt, steps=200)
IPython.display.Audio(data=audio, rate=16000)
```
[Thunder.webm](https://user-images.githubusercontent.com/13917097/233851929-90501e41-911d-453f-a00b-b215743365b4.webm)

<!-- [MachineClicking](https://user-images.githubusercontent.com/25340239/233857834-bfda52b4-4fcc-48de-b47a-6a6ddcb3671b.mp4 "sample 1") -->

Use the `generate_for_batch` function to generate multiple audio samples for a batch of text prompts:

```python
prompts = [
    "A car engine revving",
    "A dog barks and rustles with some clicking",
    "Water flowing and trickling"
]
audios = tango.generate_for_batch(prompts, samples=2)
```
This will generate two samples for each of the three text prompts.

More generated samples are shown [here](https://github.com/declare-lab/tango/blob/master/samples/README.md).

## Prerequisites

Our code is built on pytorch version 1.13.1+cu117. We mention `torch==1.13.1` in the requirements file but you might need to install a specific cuda version of torch depending on your GPU device type.

Install `requirements.txt`. You will also need to install the `diffusers` package from the directory provided in this repo:

```bash
git clone https://github.com/declare-lab/tango/
cd tango
pip install -r requirements.txt
cd diffusers
pip install -e .
```

You might also need to install `libsndfile1` for soundfile to work properly in linux:

```bash
(sudo) apt-get install libsndfile1
```

## Datasets

Follow the instructions given in the [AudioCaps repository](https://github.com/cdjkim/audiocaps) for downloading the data. The audio locations and corresponding captions are provided in our `data` directory. The `*.json` files are used for training and evaluation. Once you have downloaded your version of the data you should be able to map it using the file ids to the file locations provided in our `data/*.json` files.

Note that we cannot distribute the data because of copyright issues.

## How to train?
We use the `accelerate` package from Hugging Face for multi-gpu training. Run `accelerate config` from terminal and set up your run configuration by the answering the questions asked.

You can now train **TANGO** on the AudioCaps dataset using:

```bash
accelerate launch train.py \
--text_encoder_name="google/flan-t5-large" \
--scheduler_name="stabilityai/stable-diffusion-2-1" \
--unet_model_config="configs/diffusion_model_config.json" \
--freeze_text_encoder --augment --snr_gamma 5 \
```

The argument `--augment` uses augmented data for training as reported in our paper. We recommend training for at-least 40 epochs, which is the default in `train.py`.

To start training from our released checkpoint use the `--hf_model` argument.

```bash
accelerate launch train.py \
--hf_model "declare-lab/tango" \
--unet_model_config="configs/diffusion_model_config.json" \
--freeze_text_encoder --augment --snr_gamma 5 \
```

Check `train.py` and `train.sh` for the full list of arguments and how to use them.

## How to make inferences?

Checkpoint from training will be saved in the `saved/*/` directory.

To perform audio generation and objective evaluation in AudioCaps test set from a checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
--original_args="saved/*/summary.jsonl" \
--model="saved/*/best/pytorch_model_2.bin" \
```

Check `inference.py` and `inference.sh` for the full list of arguments and how to use them.

We use functionalities from `audioldm_eval` for objective evalution in `inference.py`. It requires the gold reference audio files and generated audio files to have the same name. You need to create the directory `data/audiocaps_test_references/subset` and keep the reference audio files there. The files should have names as following: `output_0.wav`, `output_1.wav` and so on. The indices should correspond to the corresponding line indices in `data/test_audiocaps_subset.json`.

We use the term _subset_ as some data instances originally released in AudioCaps have since been removed from YouTube and are no longer available. We thus evaluated our models on all the instances which were available as of 8th April, 2023.

We use wandb to log training and infernce results.

## Experimental Results



|           **Model**            |  **Datasets**  | **Text** | **#Params** |         FD ↓          |   KL ↓   |  FAD ↓   |         OVL ↑          |   REL ↑   |
|:------------------------------:|:--------------:|:--------:|:-----------:|:---------------------:|:--------:|:--------:|:----------------------:|:---------:|
|          Ground truth          |       −        |    −     |      −      |           −           |    −     |    −     |         91.61          |   86.78   |
|                                |                |          |             |                       |          |          |                        |           |
|           DiffSound            |     AS+AC      |    ✓     |    400M     |         47.68         |   2.52   |   7.75   |           −            |     −     |
|           AudioGen             | AS+AC+8 others |    ✗     |    285M     |           −           |   2.09   |   3.13   |           −            |     −     |
|           AudioLDM-S           |       AC       |    ✗     |    181M     |         29.48         |   1.97   |   2.43   |           −            |     −     |
|           AudioLDM-L           |       AC       |    ✗     |    739M     |         27.12         |   1.86   |   2.08   |           −            |     −     |
|                                |                |          |             |                       |          |          |                        |           |
| AudioLDM-M-Full-FT<sup>‡</sup> | AS+AC+2 others |    ✗     |    416M     |         26.12         | **1.26** |   2.57   |         79.85          |   76.84   |
|  AudioLDM-L-Full<sup>‡</sup>   | AS+AC+2 others |    ✗     |    739M     |         32.46         |   1.76   |   4.18   |         78.63          |   62.69   |
|       AudioLDM-L-Full-FT       | AS+AC+2 others |    ✗     |    739M     |       **23.31**       |   1.59   |   1.96   |           −            |     −     |
|                                |                |          |             |                       |          |          |                        |           |
|              TANGO             |       AC       |    ✓     |    866M     |         24.52         |   1.37   | **1.59** |       **85.94**        | **80.36** |

## Citation
Please consider citing the following article if you found our work useful:

```bibtex
@article{ghosal2023tango,
  title={Text-to-Audio Generation using Instruction Tuned LLM and Latent Diffusion Model},
  author={Ghosal, Deepanway and Majumder, Navonil and Mehrish, Ambuj and Poria, Soujanya},
  journal={arXiv preprint arXiv:2304.13731},
  year={2023}
}
```
## Limitations
TANGO is trained on the small AudioCaps dataset so it may not generate good audio samples related to concepts that it has not seen in training (e.g. _singing_). For the same reason, TANGO is not always able to finely control its generations over textual control prompts. For example, the generations from TANGO for prompts _Chopping tomatoes on a wooden table_ and _Chopping potatoes on a metal table_ are very similar. _Chopping vegetables on a table_ also produces similar audio samples. Training text-to-audio generation models on larger datasets is thus required for the model to learn the composition of textual concepts and varied text-audio mappings. 

We are training another version of TANGO on larger datasets to enhance its generalization, compositional and controllable generation ability.

## Acknowledgement
We borrow the code in `audioldm` and `audioldm_eval` from the [AudioLDM](https://github.com/haoheliu/AudioLDM) [repositories](https://github.com/haoheliu/audioldm_eval). We thank the AudioLDM team for open-sourcing their code.
