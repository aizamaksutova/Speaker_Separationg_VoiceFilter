# Speaker Separation
Project on speaker separation within the project DL in Audio. This rep contains my implementation of VoiceFilter model and all the steps to reimplement the pipeline
## Model choice

For my implementation of VoiceFilter I used the model architecture which was presented within the paper [VoiceFilter: Targeted Voice Separation by Speaker-Conditioned Spectrogram Masking](https://arxiv.org/abs/1810.04826).


![](./assets/voicefilter.png)



## Result

- Training took about 25 hours on one NVIDIA P100 GPU, yet could not reach the desired SI-SDR. Though the trend in metrics increase was very promising and the model definitely just needs more training time to hit good quality.

| Metrics             | Ours |
| ---------------------- | ----- |
| Median SI-SDR on LibriSpeech dev-clean     | 1.138 |
| Median PESQ on LibriSpeech dev-clean     |  1.22 |

### Dependencies

```
pip install -r requirements.txt
python3 load_model.py
```

### Dataset Generation

As there are not prepared datasets for this kind of task, I had to create the datasets on my own. This dataset solution includes WHAM noises too, which will afterwards be used to generate mixes of audio with two speakers and additional noise.

Here is the pipeline how one could do it for future research.

```
conda install -c conda-forge sox
git clone https://github.com/JorisCos/LibriMix
cd LibriMix 
./generate_librimix.sh storage_dir
mv storage_dir ./your_main_repo
```
In generate_librimix.sh you should choose only 2 speakers for this exact task. 

After generating the dataset place utils/normalize-resample.sh to the head directory with all of your data to convert from .flac to .wav

```
vim normalize-resample.sh # set "N" as your CPU core number.
chmod a+x normalize-resample.sh
./normalize-resample.sh # this may take long
```

Then run in the Speaker_Separationg_VoiceFilter repo the following code

```
python3 generator.py -c config/data_convertion.yaml -d storage_dir/LibriSpeech -o wav_data -p 40 -n wav_data
```
This will output triplets of target.wav, ref.wav and mixed.wav which you will use for training


### Train VoiceFilter


1. Get pretrained model for speaker recognition system

    VoiceFilter utilizes speaker recognition system ([d-vector embeddings](https://google.github.io/speaker-id/publications/GE2E/)).

    This model was trained with [VoxCeleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) dataset,
    where utterances are randomly fit to time length [70, 90] frames.
    Tests are done with window 80 / hop 40 and have shown equal error rate about 1%.
    Data used for test were selected from first 8 speakers of [VoxCeleb1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) test dataset, where 10 utterances per each speakers are randomly selected.

    The model can be downloaded at [this GDrive link](https://drive.google.com/file/d/1YFmhmUok-W76JkrfA0fzQt3c-ZsfiwfL/view?usp=sharing).


2. Training process

     Specify your `train_dir`, `test_dir` at `config.yaml` and then run
    ```
    python trainer.py -c [config yaml] -e [path of embedder pt file] -m [name]
    ```

    In my case it was
    ```
    python3 trainer.py -c config/convert.yaml --checkpoint_path chkpt/vf_exp2/chkpt_1000.pt -e embedder.pt -m vf_exp3
    ```
    This will create `chkpt/name` and `logs/name` at base directory

3. View tensorboardX

    ```
    tensorboard --logdir ./logs
    ```
    
    My training loss for the final experiment:


![](./assets/sisya-loss.png)


4. Resuming from checkpoint

    ```
    python trainer.py -c [config yaml] --checkpoint_path [chkpt/name/chkpt_{step}.pt] -e [path of embedder pt file] -m name
    ```



## To evaluate my results first download checkpoint

## Very important, please do

```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hBYrwcw96KIjjkBcH2GZawOWXAZ5lEXO' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hBYrwcw96KIjjkBcH2GZawOWXAZ5lEXO" -O best_model.pt && rm -rf /tmp/cookies.txt

python3 inference.py -c config/convert.yaml -e embedder.pt --checkpoint_path best_model.pt -o results
```



## Possible improvments

- Try power-law compressed reconstruction error as loss function, instead of MSE. (See [#14](https://github.com/mindslab-ai/voicefilter/issues/14))

## Author

[Seungwon Park](http://swpark.me) at MINDsLab (yyyyy@snu.ac.kr, swpark@mindslab.ai)

## License

Apache License 2.0

This repository contains codes adapted/copied from the followings:
- [utils/adabound.py](./utils/adabound.py) from https://github.com/Luolc/AdaBound (Apache License 2.0)
- [utils/audio.py](./utils/audio.py) from https://github.com/keithito/tacotron (MIT License)
- [utils/hparams.py](./utils/hparams.py) from https://github.com/HarryVolek/PyTorch_Speaker_Verification (No License specified)
- [utils/normalize-resample.sh](./utils/normalize-resample.sh.) from https://unix.stackexchange.com/a/216475
