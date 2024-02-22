# Low-Latency Neural Speech Phase Prediction based on Parallel Estimation Architecture and Anti-Wrapping Losses for Speech Generation Tasks
### Yang Ai, Zhen-Hua Ling

we proposed a novel low-latency speech phase prediction model which predicts wrapped phase spectra directly from amplitude spectra by neural networks.<br/>
We provide our implementation as open source in this repository.

**Abstract :**
This paper presents a novel neural speech phase prediction model which predicts wrapped phase spectra directly from amplitude spectra. The proposed model is a cascade of a residual convolutional network and a parallel estimation architecture. The parallel estimation architecture is a core module for direct wrapped phase prediction. This architecture consists of two parallel linear convolutional layers and a phase calculation formula, imitating the process of calculating the phase spectra from the real and imaginary parts of complex spectra and strictly restricting the predicted phase values to the principal value interval. To avoid the error expansion issue caused by phase wrapping, we design anti-wrapping training losses defined between the predicted wrapped phase spectra and natural ones by activating the instantaneous phase error, group delay error and instantaneous angular frequency error using an anti-wrapping function. We mathematically demonstrate that the anti-wrapping function should possess three properties, namely parity, periodicity and monotonicity. We also achieve low-latency streamable phase prediction by combining causal convolutions and knowledge distillation training strategies. For both analysis-synthesis and specific speech generation tasks, experimental results show that our proposed neural speech phase prediction model outperforms the iterative phase estimation algorithms and neural network-based phase prediction methods in terms of phase prediction precision, efficiency and robustness. Compared with HiFi-GAN-based waveform reconstruction method, our proposed model also shows outstanding efficiency advantages while ensuring the quality of synthesized speech. To the best of our knowledge, we are the first to directly predict speech phase spectra from amplitude spectra only via neural networks.

Visit our [demo website](https://yangai520.github.io/LL-NSPP) for audio samples.

## Requirements
```
torch==1.8.1+cu111
numpy==1.21.6
librosa==0.9.1
tensorboard==2.8.0
soundfile==0.10.3
matplotlib==3.1.3
```

## Data Preparation
For training, write the list paths of training set and validation set to `input_training_wav_list` and `input_validation_wav_list` in `config.json`, respectively. And then put the teacher model path to `teacher_checkpoint_file_load` in `config.json`. For teacher model training, you can refer to [NSPP](https://github.com/yangai520/NSPP).

For generation, we provide two ways to read data:

(1) set `test_input_log_amp_dir` to `0` in `config.json` and write the test set waveform path to `test_input_wavs_dir` in `config.json`, the generation process will load the waveform, extract the log amplitude spectra, predict the phase spectra and reconstruct the waveform;

(2) set `test_input_log_amp_dir` to `1` in `config.json` and write the log amplitude spectra (size is `(n_fft/2+1)*frames`) path to `test_input_log_amp_dir` in `config.json`, the generation process will dierctly load the log amplitude spectra, predict the phase spectra and reconstruct the waveform.

## Training
Run using GPU:
```
CUDA_VISIBLE_DEVICES=0 python train.py
```
Using TensorBoard to monitor the training process:
```
tensorboard --logdir=cp_NSPP/logs
```

## Generation:
Write the checkpoint path to `checkpoint_file_load` in `config.json`.

Run using GPU:
```
CUDA_VISIBLE_DEVICES=0 python generation.py
```
Run using CPU:
```
CUDA_VISIBLE_DEVICES=CPU python generation.py
```

## Citation
```
@article{ai2024low,
  title={Low-Latency Neural Speech Phase Prediction based on Parallel Estimation Architecture and Anti-Wrapping Losses for Speech Generation Tasks},
  author={Ai, Yang and Ling, Zhen-Hua},
  year={2024}
}
```
