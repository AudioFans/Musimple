# Musimple:Text2Music with DiT Made simple

Due to repository size limitations, the complete dataset and checkpoints are available on Hugging Face: [https://huggingface.co/ZheqiDAI/Musimple](https://huggingface.co/ZheqiDAI/Musimple).

## Introduction

This repository provides a simple and clear implementation of a **Text-to-Music Generation** pipeline using a **DiT (Diffusion Transformer)** model. The codebase includes key components such as **model training**, **inference**, and **evaluation**. We use the **GTZAN dataset** as an example to demonstrate a minimal, working pipeline for text-conditioned music generation.

The repository is designed to be easy to use and customize, making it simple to reproduce our results on a single **NVIDIA RTX 4090 GPU**. Additionally, the code is structured to be flexible, allowing you to modify it for your own tasks and datasets. 

We plan to continue maintaining and improving this repository with new features, model improvements, and extended documentation in the future.

## Features

- **Text-to-Music Generation**: Generate music directly from text descriptions using a DiT model.
- **GTZAN Example**: A simple pipeline using the GTZAN dataset to demonstrate the workflow.
- **End-to-End Pipeline**: Includes model training, inference, and evaluation with support for generating audio files.
- **Customizable**: Easy to modify and extend for different datasets or use cases.
- **Single GPU Training**: Optimized for training on a single RTX 4090 GPU but adaptable to different hardware setups.

## Requirements

Before using the code, ensure that the following dependencies are installed:

- Python >= 3.9
- CUDA (if available)
- Required Python libraries from `requirements.txt`

You can install the dependencies using:

```bash
conda create -n musimple python=3.9
conda activate musimple
pip install -r requirements.txt
```


## Data Preprocessing

To begin with, you will need to download the **GTZAN dataset**. Once downloaded, you can use the `gtzan_split.py` script located in the `tools` directory to split the dataset into training and testing sets. Run the following command:

```bash
python gtzan_split.py --root_dir /path/to/gtzan/genres --output_dir /path/to/output/directory
```

Next, convert the audio files into an HDF5 format using the gtzan2h5.py script:

```bash
python gtzan2h5.py --root_dir /path/to/audio/files --output_h5_file /path/to/output.h5 --config_path bigvgan_v2_22khz_80band_256x/config.json --sr 22050
```

Preprocessed Data
If this process seems cumbersome, don’t worry! **We have already preprocessed the dataset**, and you can find it in the **musimple/dataset** directory. You can download and use this data directly to skip the preprocessing steps.

Data Breakdown
In this preprocessing stage, there are two main parts:

Text to Latent Transformation: We use a Sentence Transformer to convert text labels into latent representations.
Audio to Mel Spectrogram: The original audio files are converted into mel spectrograms.
Both the latent representations and mel spectrograms are stored in an HDF5 file, making them easily accessible during training and inference.

## Training

To begin training, simply navigate to the `Musimple` directory and run the following command:

```bash
cd Musimple
python train.py
```

Configurable Parameters
All training-related parameters can be adjusted in the configuration file located at:
```
./config/train.yaml
```
This allows you to easily modify aspects like the learning rate, batch size, number of epochs, and more to suit your hardware or dataset requirements.

We also provide a **pre-trained checkpoint** trained for two days on a single **NVIDIA RTX 4090**. You can use this checkpoint for inference or fine-tuning. The key training parameters for this checkpoint are as follows:

- `batch_size`: 48  
- `mel_frames`: 800  
- `lr`: 0.0001  
- `num_epochs`: 100000  
- `sample_interval`: 250  
- `h5_file_path`: './dataset/gtzan_train.h5'  
- `device`: 'cuda:4'  
- `input_size`: [80, 800]  
- `patch_size`: 8  
- `in_channels`: 1  
- `hidden_size`: 384  
- `depth`: 12  
- `num_heads`: 6  
- `checkpoint_dir`: 'gtzan-ck'

You can modify the model architecture and parameters in the `train.yaml` configuration file to compare your models against ours. We will continue to release more checkpoints and models in future updates.

## Inference

Once you have trained your own model, you can perform inference using the trained model. To do so, run the following command:

```bash
python sample.py --checkpoint ./gtzan-ck/model_epoch_20000.pt \
                 --h5_file ./dataset/gtzan_test.h5 \
                 --output_gt_dir ./sample/gt \
                 --output_gen_dir ./sample/gn \
                 --segment_length 800 \
                 --sample_rate 22050
```
You can also try running inference using our pre-trained model to familiarize yourself with the inference process. We have saved some inference results in the sample folder as a demo. However, due to the limited size of our model, the generated results are not of the highest quality and are intended as simple examples to guide further evaluation.

## Evaluation

For the evaluation phase, we highly recommend creating a new environment and using the evaluation library available at [Generated Music Evaluation](https://github.com/HarlandZZC/generated_music_evaluation). This repository provides detailed instructions on setting up the environment and how to use the evaluation tools. New features and functionality will be added to this library over time.

Once you have set up the environment following the instructions from the evaluation repository, you can run the following script to evaluate your generated music:

```bash
python eval.py \
    --ref_path  ../sample/gt \
    --gen_path ../sample/gn \
    --id2text_csv_path ../gtzan-test.csv \
    --output_path ./output \
    --device_id 0 \
    --batch_size 32 \
    --original_sample_rate 24000 \
    --fad_sample_rate 16000 \
    --kl_sample_rate 16000 \
    --clap_sample_rate 48000 \
    --run_fad 1 \
    --run_kl 1 \
    --run_clap 1
```

This script evaluates the generated music against reference music, producing evaluation metrics such as CLAP, KL, and FAD scores.

## To-Do

The following features and improvements are planned for future updates:

- **EMA Model**: Implement Exponential Moving Average (EMA) for model weights to stabilize training and improve final generation quality.
- **Long-Term Music Fine-tuning**: Explore fine-tuning the model to generate longer-term music with more coherent structures.
- **VAE Integration**: Integrate a Variational Autoencoder (VAE) to improve latent space representations and potentially enhance generation diversity.
- **T5-based Text Conditioning**: Add T5 to enhance text conditioning, improving the control and accuracy of the text-to-music generation process.