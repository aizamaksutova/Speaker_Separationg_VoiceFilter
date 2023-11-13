import os
import glob
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
import numpy as np

from utils.audio import Audio


def create_dataloader(hp, args, train):
    def train_collate_fn(batch):
        dvec_list = list()
        target_mag_list = list()
        mixed_mag_list = list()
        target_wav_list = list()
        mixed_wav_list = list()
        mixed_phase_list = list()

        for dvec_mel, target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase in batch:
            dvec_list.append(dvec_mel)
            target_mag_list.append(target_mag)
            mixed_mag_list.append(mixed_mag)
            target_wav_list.append(torch.tensor(target_wav))
            mixed_wav_list.append(torch.tensor(mixed_wav))
            mixed_phase_list.append(mixed_phase)

        target_mag_list = torch.stack(target_mag_list, dim=0)
        mixed_mag_list = torch.stack(mixed_mag_list, dim=0)
        target_wav_list = torch.stack(target_wav_list, dim=0)
        mixed_wav_list = torch.stack(mixed_wav_list, dim=0)

        return dvec_list, target_wav_list, mixed_wav_list, target_mag_list, mixed_mag_list, mixed_phase_list

    def test_collate_fn(batch):
        return batch

    if train:
        return DataLoader(dataset=VFDataset(hp, args, True),
                          batch_size=hp.train.batch_size,
                          shuffle=True,
                          num_workers=hp.train.num_workers,
                          collate_fn=train_collate_fn,
                          pin_memory=True,
                          drop_last=True,
                          sampler=None)
    else:
        return DataLoader(dataset=VFDataset(hp, args, False),
                          collate_fn=test_collate_fn,
                          batch_size=1, shuffle=False, num_workers=0)


class VFDataset(Dataset):
    def __init__(self, hp, args, train):
        def find_all(file_format):
            return sorted(glob.glob(os.path.join(self.data_dir, file_format)))
        self.hp = hp
        self.args = args
        self.train = train
        self.data_dir = hp.data.train_dir if train else hp.data.test_dir

        self.dvec_wav_list = find_all(hp.form.reference.wav)
        self.target_wav_list = find_all(hp.form.target.wav)
        self.mixed_wav_list = find_all(hp.form.mixed.wav)

        assert len(self.dvec_wav_list) == len(self.target_wav_list) == len(self.mixed_wav_list), "number of training files must match"
        assert len(self.dvec_wav_list) != 0, \
            "no training file found"

        self.audio = Audio(hp)

    def __len__(self):
        return len(self.dvec_wav_list)

    def __getitem__(self, idx):


        dvec_wav, _ = librosa.load(self.dvec_wav_list[idx], sr=self.hp.audio.sample_rate)
        if not isinstance(dvec_wav, np.ndarray):
            raise ValueError("Audio data is not a numpy array")

        dvec_mel = self.audio.get_mel(dvec_wav)
        dvec_mel = torch.from_numpy(dvec_mel).float()


        if self.train: # need to be fast
            target_wav, _ = librosa.load(self.target_wav_list[idx], sr=self.hp.audio.sample_rate)
            mixed_wav, _ = librosa.load(self.mixed_wav_list[idx], sr=self.hp.audio.sample_rate)
            target_mag, _ = self.wav2magphase(target_wav)
            mixed_mag, mixed_phase = self.wav2magphase(mixed_wav)
            target_mag = torch.from_numpy(target_mag)
            mixed_mag = torch.from_numpy(mixed_mag)
            return dvec_mel, target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase
        else:
            target_wav, _ = librosa.load(self.target_wav_list[idx], sr=self.hp.audio.sample_rate)
            mixed_wav, _ = librosa.load(self.mixed_wav_list[idx], sr=self.hp.audio.sample_rate)
            target_mag, _ = self.wav2magphase(target_wav)
            mixed_mag, mixed_phase = self.wav2magphase(mixed_wav)
            target_mag = torch.from_numpy(target_mag)
            mixed_mag = torch.from_numpy(mixed_mag)
            # mixed_phase = torch.from_numpy(mixed_phase)
            return dvec_mel, target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase

    def wav2magphase(self, wav):
        mag, phase = self.audio.wav2spec(wav)
        return mag, phase
