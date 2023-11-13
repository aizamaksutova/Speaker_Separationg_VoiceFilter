import os
import glob
import tqdm
import torch
import random
import librosa
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count
import soundfile as sf
from utils.audio import Audio
from utils.hparams import HParam


def formatter(dir_, form, num):
    return os.path.join(dir_, form.replace('*', '%06d' % num))

def formatter_wav(form, num):
    return os.path.join('', form.replace('*', '%06d' % num))

def vad_merge(w):
    intervals = librosa.effects.split(w, top_db=20)
    temp = list()
    for s, e in intervals:
        temp.append(w[s:e])
    return np.concatenate(temp, axis=None)


def mix(hp, args, audio, num, ref_wav, target_wav, mixed_wav, train):
    srate = hp.audio.sample_rate
    dir_ = os.path.join(args.out_dir, 'train' if train else 'test')



    # save magnitude spectrograms
    target_mag, _ = audio.wav2spec(target_wav)
    mixed_mag, _ = audio.wav2spec(mixed_wav)


    target_mag_path = formatter(dir_, hp.form.target.mag, num)
    mixed_mag_path = formatter(dir_, hp.form.mixed.mag, num)

    torch.save(torch.from_numpy(target_mag), target_mag_path)
    torch.save(torch.from_numpy(mixed_mag), mixed_mag_path)

    # save selected sample as text file. d-vec will be calculated soon
    # dvec_text_path = formatter(dir_, hp.form.dvec, num)
    # with open(dvec_text_path, 'w') as f:
    #     f.write(s1_dvec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-d', '--input_dir', type=str, default=None,
                        help="Directory of input dataset, containing wavs for target speaker, reference wav and mixed wav of two speakers")
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help="Directory of output training triplet")
    parser.add_argument('-p', '--process_num', type=int, default=None,
                        help='number of processes to run. default: cpu_count')
    parser.add_argument('--vad', type=int, default=0,
                        help='apply vad to wav file. yes(1) or no(0, default)')

    parser.add_argument('-t', '--train', type=str, default=None,
                        help='If you need to change only the train or test')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'test'), exist_ok=True)

    hp = HParam(args.config)

    cpu_num = cpu_count() if args.process_num is None else args.process_num

    if args.libri_dir is None and args.voxceleb_dir is None:
        raise Exception("Please provide directory of data")

    if args.libri_dir is not None:
        if args.train == 'train' or args.train is None:
            train_folders = [x for x in glob.glob(os.path.join(args.libri_dir, 'train', '*'))
                                if os.path.isdir(x)]
        if args.train == 'test' or args.train is None:
            test_folders = [x for x in glob.glob(os.path.join(args.libri_dir, 'test', '*'))]

    elif args.voxceleb_dir is not None:
        all_folders = [x for x in glob.glob(os.path.join(args.voxceleb_dir, '*'))
                            if os.path.isdir(x)]
        train_folders = all_folders[:-20]
        test_folders = all_folders[-20:]

    train_spk = [glob.glob(os.path.join(spk, '**', hp.form.input), recursive=True)
                    for spk in train_folders]
    train_spk = [x for x in train_spk if len(x) >= 2]

    test_spk = [glob.glob(os.path.join(spk, '**', hp.form.input), recursive=True)
                    for spk in test_folders]
    test_spk = [x for x in test_spk if len(x) >= 2]

    audio = Audio(hp)

    def train_wrapper(num):
        target_wav = random.sample(train_spk, 1)
        target_wav_path = formatter_wav(target_wav, num)
        id_speaker = target_wav_path.split('-')[0]
        mixed_wav = 
        s2 = random.choice(spk2)
        mix(hp, args, audio, num, s1_dvec, s1_target, s2, train=True)

    def test_wrapper(num):
        spk1, spk2 = random.sample(test_spk, 2)
        s1_dvec, s1_target = random.sample(spk1, 2)
        s2 = random.choice(spk2)
        mix(hp, args, audio, num, s1_dvec, s1_target, s2, train=False)

    arr = list(range(10**5))
    with Pool(cpu_num) as p:
        r = list(tqdm.tqdm(p.imap(train_wrapper, arr), total=len(arr)))

    arr = list(range(10**2))
    with Pool(cpu_num) as p:
        r = list(tqdm.tqdm(p.imap(test_wrapper, arr), total=len(arr)))
