import os
import math
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import traceback

from .adabound import AdaBound
from .audio import Audio
from .evaluation import validate
from model.model import VoiceFilter
from model.embedder import SpeechEmbedder
from utils.sisdr import si_sdr


def train(args, pt_dir, chkpt_path, trainloader, testloader, writer, logger, hp, hp_str):
    # load embedder
    embedder_pt = torch.load(args.embedder_path)
    embedder = SpeechEmbedder(hp).cuda()
    embedder.load_state_dict(embedder_pt)
    embedder.eval()

    audio = Audio(hp)
    num_epochs = hp.train.num_epochs
    model = VoiceFilter(hp).cuda()
    if hp.train.optimizer == 'adabound':
        optimizer = AdaBound(model.parameters(),
                             lr=hp.train.adabound.initial,
                             final_lr=hp.train.adabound.final)
    elif hp.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hp.train.adam)
    else:
        raise Exception("%s optimizer not supported" % hp.train.optimizer)

    if hp.train.lr_scheduler == 'OneCycleLR':
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=hp.train.steps_per_epoch, epochs=100000)


    step = 0

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']

        # will use new given hparams.
        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams is different from checkpoint.")
    else:
        logger.info("Starting new training run")

    try:
        criterion = nn.MSELoss()
        for _ in range(num_epochs):
            model.train()
            for dvec_mels, target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase in trainloader:
                target_mag = target_mag.cuda()
                mixed_mag = mixed_mag.cuda()

                dvec_list = list()
                for mel in dvec_mels:
                    mel = mel.cuda()
                    dvec = embedder(mel)
                    dvec_list.append(dvec)
                dvec = torch.stack(dvec_list, dim=0)
                dvec = dvec.detach()

                mask = model(mixed_mag, dvec)
                output = mixed_mag * mask

                loss = criterion(output, target_mag)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                step += 1

                loss = loss.item()
                if loss > 1e8 or math.isnan(loss):
                    logger.error("Loss exploded to %.02f at step %d!" % (loss, step))
                    raise Exception("Loss exploded")
                # write loss to tensorboard
                if step % hp.train.summary_interval == 0:
                    
                    target_mag = target_mag[0].unsqueeze(0)
                    mixed_mag = mixed_mag[0].unsqueeze(0)

                    dvec = dvec[0].unsqueeze(0)
                    est_mask = model(mixed_mag, dvec)
                    est_mag = est_mask * mixed_mag

                    mixed_mag = mixed_mag[0].cpu().detach().numpy()
                    target_mag = target_mag[0].cpu().detach().numpy()
                    est_mag = est_mag[0].cpu().detach().numpy()
                    est_wav = audio.spec2wav(est_mag, mixed_phase[0])
                    est_mask = est_mask[0].cpu().detach().numpy()

                    si_sdr_score = si_sdr(est_wav, target_wav)
                    writer.log_training(loss, si_sdr_score,
                                mixed_wav[0], target_wav[0], est_wav,
                                mixed_mag.T, target_mag.T, est_mag.T, est_mask.T,
                                scheduler.get_last_lr()[0],
                                step)
                    logger.info("Wrote summary at step %d" % step)

                # 1. save checkpoint file to resume training
                # 2. evaluate and save sample to tensorboard
                if step % hp.train.checkpoint_interval == 0:
                    save_path = os.path.join(pt_dir, 'chkpt_%d.pt' % step)
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'hp_str': hp_str,
                    }, save_path)
                    logger.info("Saved checkpoint to: %s" % save_path)
                    validate(audio, model, embedder, testloader, writer, step)
    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
