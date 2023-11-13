import os
import glob
import torch
import librosa
import argparse
import numpy as np

from utils.audio import Audio
from utils.hparams import HParam
from model.model import VoiceFilter
from model.embedder import SpeechEmbedder

import soundfile as sf

from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio import PerceptualEvaluationSpeechQuality

from datasets.dataloader import create_dataloader


def main(args, hp, test_loader):

    si_sdr_score = []
    pesq_score = []
    si_sdr = ScaleInvariantSignalDistortionRatio()
    wb_pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')
    with torch.no_grad():
        model = VoiceFilter(hp).cuda()
        chkpt_model = torch.load(args.checkpoint_path)['model']
        model.load_state_dict(chkpt_model)
        model.eval()

        embedder = SpeechEmbedder(hp).cuda()
        chkpt_embed = torch.load(args.embedder_path)
        embedder.load_state_dict(chkpt_embed)
        embedder.eval()
        num = 0
        for batch in test_loader:
            for i in range(len(batch)):
                dvec_mel, target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase = batch[i]

                dvec_mel = dvec_mel.cuda()
                target_mag = target_mag.unsqueeze(0).cuda()
                mixed_mag = mixed_mag.unsqueeze(0).cuda()

                dvec = embedder(dvec_mel)
                dvec = dvec.unsqueeze(0)
                est_mask = model(mixed_mag, dvec)
                est_mag = est_mask * mixed_mag

                mixed_mag = mixed_mag[0].cpu().detach().numpy()
                target_mag = target_mag[0].cpu().detach().numpy()
                est_mag = est_mag[0].cpu().detach().numpy()

                audio = Audio(hp)
                est_wav = audio.spec2wav(est_mag, mixed_phase)
                est_mask = est_mask[0].cpu().detach().numpy()

                si_sdr_score_i = si_sdr(torch.tensor(est_wav), torch.tensor(target_wav))
                si_sdr_score.append(si_sdr_score_i.item())

                pesq_score_i = wb_pesq(torch.tensor(est_wav), torch.tensor(target_wav))
                pesq_score.append(pesq_score_i.item())

                os.makedirs(args.out_dir, exist_ok=True)
                out_path = os.path.join(args.out_dir, f'result_{num}.wav')
                sf.write(out_path, est_wav, 16000)
                num += 1
    
    
    print('Average si_sdr score', np.mean(si_sdr_score))
    print('Average pesq score', np.mean(pesq_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-e', '--embedder_path', type=str, required=True,
                        help="path of embedder model pt file")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file")
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help='directory of output')

    args = parser.parse_args()
    hp = HParam(args.config)
    testloader = create_dataloader(hp, args, train=False)


    main(args, hp, testloader)

