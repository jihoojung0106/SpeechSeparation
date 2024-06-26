import glob
import torchvision
from talk_model.networks import Resnet18_
from argparse import ArgumentParser
from os.path import join
from talk_model.talkNetModel import talkNetModel
from torch.autograd import Variable
import torch
import librosa
import matplotlib.pyplot as plt

from vv_utils.utils import object_collate
from torch.utils.data import DataLoader
from soundfile import write
from torchaudio import load
from tqdm import tqdm
from data.audioVisual_dataset import *
from sgmse.model import ScoreModel,MyScoreModel
from sgmse.util.other import ensure_dir, pad_spec
from sgmse.data_module import MySpecsDataModule
import numpy as np
def test_normalize(samples, desired_rms=0.1, eps=1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    samples_normalized = samples * (desired_rms / rms)
    return samples_normalized, rms
def test_denormalize(normalized_samples, original_rms, desired_rms=0.1):
    original_rms_tensor = torch.tensor(original_rms)
    samples_denormalized = normalized_samples * (original_rms_tensor.cuda() / desired_rms)
    return samples_denormalized

def spec_fwd(spec):
    e = 0.5 #0.5
    spec = spec.abs()**e * torch.exp(1j * spec.angle())
    return spec * 0.15 #0.15

def find_max_numeric_folder(directory):
    max_num = None
    
    for folder_name in os.listdir(directory):
        if folder_name.isdigit():  # Check if the folder name is composed of digits
            folder_num = int(folder_name)
            if max_num is None or folder_num > max_num:
                max_num = folder_num
                
    return max_num
def create_next_numeric_folder(directory):
    max_folder_number = find_max_numeric_folder(directory)
    
    if max_folder_number is None:
        next_folder_number = 0  # If no numeric folders exist, start with 0
    else:
        next_folder_number = max_folder_number + 1
    
    new_folder_path = os.path.join(directory, str(next_folder_number))
    os.makedirs(new_folder_path)
    return new_folder_path
def build_facial(self, pool_type='maxpool', fc_out=128, with_fc=True, weights=None):
        pretrained = False
        original_resnet = torchvision.models.resnet18(pretrained)
        net = Resnet18_(original_resnet, pool_type=pool_type, with_fc=with_fc, fc_in=512, fc_out=fc_out)

        if len(weights) > 0:
            print('Loading weights for facial attributes analysis stream')
            pretrained_state = torch.load(weights)
            model_state = net.state_dict()
            pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
            model_state.update(pretrained_state)
            net.load_state_dict(model_state)
        return net


def load_and_preprocess_data(common_path1, common_path2, data_path, audio_sampling_rate, audio_window, num_frames, window_size, hop_size, n_fft, vision_transform, lipreading_preprocessing_func):
    # 데이터 경로 설정
    video_path_A1 = os.path.join(data_path, "mp4", common_path1 + ".mp4")
    mouthroi_path_A1 = os.path.join(data_path, "mouth_roi_hdf5", common_path1 + ".h5")
    audio_path_A1 = os.path.join(data_path, "aac", common_path1 + ".m4a")

    video_path_B = os.path.join(data_path, "mp4", common_path2 + ".mp4")
    mouthroi_path_B = os.path.join(data_path, "mouth_roi_hdf5", common_path2 + ".h5")
    audio_path_B = os.path.join(data_path, "aac", common_path2 + ".m4a")

    # 데이터 로드
    mouthroi_A1 = load_mouthroi(mouthroi_path_A1)
    mouthroi_B = load_mouthroi(mouthroi_path_B)
    frame_rate, audio_A1 = read_m4a(audio_path_A1)
    _, audio_B = read_m4a(audio_path_B)

    audio_A1 = audio_A1 / 32768
    audio_B = audio_B / 32768

    mouthroi_A1, audio_A1 = get_mouthroi_audio_pair(mouthroi_A1, audio_A1, audio_window, num_frames, audio_sampling_rate)
    mouthroi_B, audio_B = get_mouthroi_audio_pair(mouthroi_B, audio_B, audio_window, num_frames, audio_sampling_rate)

    frame_A_list = []
    frame_B_list = []
    for _ in range(1):  # number_of_identity_frames = 1
        frame_A = load_frame(video_path_A1)
        frame_B = load_frame(video_path_B)
        frame_A = vision_transform(frame_A)
        frame_B = vision_transform(frame_B)
        frame_A_list.append(frame_A)
        frame_B_list.append(frame_B)

    frames_A = torch.stack(frame_A_list).squeeze()
    frames_B = torch.stack(frame_B_list).squeeze()

    mouthroi_A1 = lipreading_preprocessing_func(mouthroi_A1)
    mouthroi_B = lipreading_preprocessing_func(mouthroi_B)

    audio_A1, rms_A1 = test_normalize(audio_A1)
    audio_B, rms_B = test_normalize(audio_B)
    audio_mix1 = (audio_A1 + audio_B) / 2

    audio_spec_A1 = generate_spectrogram_complex(audio_A1, window_size, hop_size, n_fft)
    audio_spec_B = generate_spectrogram_complex(audio_B, window_size, hop_size, n_fft)
    audio_spec_mix1 = generate_spectrogram_complex(audio_mix1, window_size, hop_size, n_fft)

    audio_spec_A1, audio_spec_B, audio_spec_mix1 = spec_fwd(audio_spec_A1), spec_fwd(audio_spec_B), spec_fwd(audio_spec_mix1)

    return {
        "mouthroi_A1": mouthroi_A1,
        "frame_A": frames_A,
        "frame_B": frames_B,
        "audio_spec_A1": audio_spec_A1,
        "audio_spec_mix1": audio_spec_mix1,
        "rms_A1": rms_A1,
        "audio_length": audio_A1.size,
        "frame_rate": frame_rate,
    }

def run_inference(args):
    # 하이퍼파라미터 설정
    data_path = "/dataset/VoxCeleb2"
    audio_length = 2.55
    audio_sampling_rate = 16000
    num_frames = 64
    window_size = 400
    hop_size = 160
    n_fft = 512

    # 데이터 전처리 함수 설정
    lipreading_preprocessing_func = get_preprocessing_pipelines()["val"]
    vision_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 경로 설정
    common_path1 = "train/id00562/_HozbUvc988/00133"
    common_path2 = "train/id00154/2pSNL5YdcoQ/00003"

    # 데이터 로드 및 전처리
    data = load_and_preprocess_data(
        common_path1, common_path2, data_path, audio_sampling_rate, audio_length * audio_sampling_rate,
        num_frames, window_size, hop_size, n_fft, vision_transform, lipreading_preprocessing_func
    )

    mouthroi_A1 = data["mouthroi_A1"]
    frames_A = data["frame_A"]
    frames_B = data["frame_B"]
    audio_spec_A1 = data["audio_spec_A1"]
    audio_spec_mix1 = data["audio_spec_mix1"]
    rms_A1 = data["rms_A1"]
    audio_length = data["audio_length"]
    frame_rate = data["frame_rate"]

    new_folder_path = create_next_numeric_folder("output")
    print(f"New directory created: {new_folder_path}")

    save_as_wav(frame_rate, audio_A1 * 32768, os.path.join(new_folder_path, "A1.wav"))
    save_as_wav(frame_rate, audio_B * 32768, os.path.join(new_folder_path, "B.wav"))
    save_as_wav(frame_rate, (audio_A1 + audio_B) / 2 * 32768, os.path.join(new_folder_path, "mix.wav"))

    # 모델 로드 및 설정
    model = MyScoreModel.load_from_checkpoint(args.ckpt, base_dir='', batch_size=1, num_workers=1, kwargs=dict(gpu=False))
    model.eval(no_ema=False)
    model.cuda()

    talknetmodel = model.talknetmodel
    facialnetmodel = model.facialnetmodel

    mouthroi_A1_embed = talknetmodel(mouthroi_A1.unsqueeze(0).unsqueeze(0).squeeze(1).cuda())
    batch_size = mouthroi_A1_embed.shape[0]
    time_ = mouthroi_A1_embed.shape[1]
    identity_feature_A = facialnetmodel(Variable(frames_A.unsqueeze(0).cuda(), requires_grad=False)).squeeze(2).squeeze(2)
    embedding_dim = identity_feature_A.shape[1]
    identity_feature_A = identity_feature_A.repeat(1, time_).view(batch_size, time_, embedding_dim)
    visual_embedding = torch.cat((mouthroi_A1_embed, identity_feature_A), dim=2)

    y = audio_spec_mix1[:, :-1, :].unsqueeze(0).cuda()
    if not args.withVisual:
        visual_embedding = torch.zeros_like(visual_embedding)

    # Reverse sampling
    sampler = model.get_pc_sampler(
        'reverse_diffusion', args.corrector, y, visual_embedding.cuda(), N=args.N, corrector_steps=args.corrector_steps, snr=args.snr
    )
    sample, _ = sampler()

    # Backward transform in time domain
    spec, x_hat = model.to_audio(sample.squeeze(), T_orig=audio_length)
    visualize_and_save_spectrogram(spec, os.path.join(new_folder_path, "generated.png"), hop_size)

    # Renormalize
    x_hat = test_denormalize(x_hat, rms_A1)

    # Write enhanced wav file
    out = os.path.join(new_folder_path, "generated.wav")
    write(out, x_hat.cpu().numpy(), 16000)
    print(f"Generated audio saved to {out}")
