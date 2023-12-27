import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from tqdm import tqdm
import logging
from dataset import ReactionDataset
from model import ReactFace
from utils import AverageMeter
from model.losses import VAELoss
from metric import *
from dataset import get_dataloader

def parse_arg():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Param
    parser.add_argument('--dataset-path', default="./data", type=str, help="dataset path")
    parser.add_argument('--split', type=str, help="split of dataset", choices=["val", "test"], required=True)
    parser.add_argument('--resume', default="", type=str, help="checkpoint path")
    parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size (default: 4)')
    parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--img-size', default=256, type=int, help="size of train/test image data")
    parser.add_argument('--crop-size', default=224, type=int, help="crop size of train/test image data")
    parser.add_argument('-seq-len', default=751, type=int, help="length of clip")
    parser.add_argument('-max-seq-len', default=751, type=int, help="max length of clip")
    parser.add_argument('--clip-length', default=751, type=int, help="len of video clip")
    parser.add_argument('--window-size', default=8, type=int, help="prediction window-size for online mode")
    parser.add_argument('--feature-dim', default=128, type=int, help="feature dim of model")
    parser.add_argument('--audio-dim', default=39, type=int, help="feature dim of audio")
    parser.add_argument('--_3dmm-dim', default=58, type=int, help="feature dim of 3dmm")
    parser.add_argument('--emotion-dim', default=25, type=int, help="feature dim of emotion")
    parser.add_argument('--online', action='store_true', help='online / offline method')
    parser.add_argument('--outdir', default="./results", type=str, help="result dir")
    parser.add_argument('--device', default='cuda', type=str, help="device: cuda / cpu")
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--kl-p', default=0.0002, type=float, help="hyperparameter for kl-loss")

    args = parser.parse_args()
    return args


# Train
def val(args, model, val_loader, criterion, render):
    losses = AverageMeter()
    rec_losses = AverageMeter()
    kld_losses = AverageMeter()
    model.eval()

    out_dir = os.path.join(args.outdir, args.split)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    listener_emotion_list = []
    speaker_emotion_list = []
    all_listener_emotion_list = []

    for batch_idx, (speaker_video_clip, speaker_audio_clip, speaker_emotion, _, _, _, listener_emotion, listener_3dmm, listener_references, _, _, _) in enumerate(tqdm(val_loader)):
        if torch.cuda.is_available():
            speaker_video_clip, speaker_audio_clip, listener_emotion, listener_3dmm, listener_references = \
                speaker_video_clip.cuda(), speaker_audio_clip.cuda(), listener_emotion.cuda(), listener_3dmm.cuda(), listener_references.cuda()
                #speaker_video_clip[:,:750].cuda(), speaker_audio_clip[:,:750].cuda(), listener_emotion[:,:750].cuda(), listener_3dmm[:,:750].cuda(), listener_references[:,:750].cuda()

        with torch.no_grad():
            listener_3dmm_out, listener_emotion_out, distribution = model(speaker_video_clip, speaker_audio_clip)

            loss, rec_loss, kld_loss = criterion[0](listener_emotion, listener_3dmm, listener_emotion_out, listener_3dmm_out, distribution)


            losses.update(loss.data.item(), speaker_video_clip.size(0))
            rec_losses.update(rec_loss.data.item(), speaker_video_clip.size(0))
            kld_losses.update(kld_loss.data.item(), speaker_video_clip.size(0))
            B = speaker_video_clip.shape[0]
            listener_emotion_list.append(listener_emotion_out.cpu())
            speaker_emotion_list.append(speaker_emotion)

    listener_emotion = torch.cat(listener_emotion_list, dim = 0)
    speaker_emotion = torch.cat(speaker_emotion_list, dim = 0)
    all_listener_emotion_list.append(listener_emotion.unsqueeze(1))

    print("-----------------Repeat 9 times-----------------")
    for i in range(9):
        listener_emotion_list = []
        for batch_idx, (speaker_video_clip, speaker_audio_clip, _, _, _, _, _, _, _, _, _, _) in enumerate(tqdm(val_loader)):
            if torch.cuda.is_available():
                speaker_video_clip, speaker_audio_clip = \
                    speaker_video_clip.cuda(), speaker_audio_clip.cuda()
                    #peaker_video_clip[:,:750].cuda(), speaker_audio_clip[:,:750].cuda()
            with torch.no_grad():
                _, listener_emotion_outs, _ = model(speaker_video_clip, speaker_audio_clip)
                listener_emotion_list.append(listener_emotion_outs[:,:750].cpu())
        listener_emotion = torch.cat(listener_emotion_list, dim=0)
        all_listener_emotion_list.append(listener_emotion.unsqueeze(1))
    all_listener_emotion = torch.cat(all_listener_emotion_list, dim=1)

    print("-----------------Evaluating Metric-----------------")
    FRC = compute_FRC_mp(args, all_listener_emotion, listener_emotion)
    FRD = compute_FRD_mp(args, all_listener_emotion, listener_emotion)
    FRDvs = compute_FRDvs(all_listener_emotion)
    FRVar  = compute_FRVar(all_listener_emotion)
    smse  = compute_s_mse(all_listener_emotion)
    #TLCC = compute_TLCC(all_listener_emotion, speaker_emotion)

    return losses.avg, rec_losses.avg, kld_losses.avg, FRC, FRD, FRDvs, FRVar, smse, 0


def main(args):
    val_loader = get_dataloader(args, "val", load_audio=True, load_video_s=True,  load_emotion_l=True, load_3dmm_l=True, load_ref=True)
    model = ReactFace(img_size = args.img_size, output_3dmm_dim = args._3dmm_dim, output_emotion_dim = args.emotion_dim, feature_dim = args.feature_dim, max_seq_len = args.max_seq_len, window_size = args.window_size, device = args.device)
    criterion = [VAELoss(args.kl_p).cuda()]

    if args.resume != '':
        checkpoint_path = args.resume
        print("Resume from {}".format(checkpoint_path))
        checkpoints = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        state_dict = checkpoints['state_dict']
        model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        model = model.cuda()
        
    render = None

    val_loss, rec_loss, kld_loss, FRC, FRD, FRDvs, FRVar, smse, TLCC = val(args, model, val_loader, criterion, render)
    print("{}_loss: {:.5f}   {}_rec_loss: {:.5f}  {}_kld_loss: {:.5f} ".format(args.split, val_loss, args.split, rec_loss, args.split, kld_loss))
    print("Metric: | FRC: {:.5f}| FRD: {:.5f}| FRDvs: {:.5f}| FRVar: {:.5f}| S-MSE: {:.5f}| TLCC: {:.5f}".format(FRC, FRD, FRDvs, FRVar, smse, TLCC))


# ---------------------------------------------------------------------------------


if __name__=="__main__":
    args = parse_arg()
    os.environ["NUMEXPR_MAX_THREADS"] = '16'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    main(args)

