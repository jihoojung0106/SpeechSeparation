import glob
import torchvision
from talk_model.networks import Resnet18_
from argparse import ArgumentParser
from os.path import join
from talk_model.talkNetModel import talkNetModel
from torch.autograd import Variable
import torch
import librosa
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sgmse.util.other import pad_spec,load_json
from vv_utils.utils import object_collate
from torch.utils.data import DataLoader
from soundfile import write
from torchaudio import load
from tqdm import tqdm
from data.audioVisual_dataset import *
from sgmse.model import MyScoreModel

import numpy as np
from models.lipreading_model import Lipreading

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
def build_facial(pool_type='maxpool', fc_out=128, with_fc=True, weights="checkpoints/facial_best.pth"):
        pretrained = False
        original_resnet = torchvision.models.resnet18(pretrained)
        net = Resnet18_(original_resnet, pool_type=pool_type, with_fc=with_fc, fc_in=512, fc_out=fc_out)

        if weights is not None:
            print('Loading weights for facial attributes analysis stream')
            pretrained_state = torch.load(weights)
            model_state = net.state_dict()
            pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
            model_state.update(pretrained_state)
            net.load_state_dict(model_state)
        return net
    
def build_lipreadingnet(config_path="configs/lrw_snv1x_tcn2x.json", weights='checkpoints/lipreading_best.pth', extract_feats=True):
        if os.path.exists(config_path):
            args_loaded = load_json(config_path)
            print('Lipreading configuration file loaded.')
            tcn_options = { 'num_layers': args_loaded['tcn_num_layers'],
                            'kernel_size': args_loaded['tcn_kernel_size'],
                            'dropout': args_loaded['tcn_dropout'],
                            'dwpw': args_loaded['tcn_dwpw'],
                            'width_mult': args_loaded['tcn_width_mult']}                 
        net = Lipreading(tcn_options=tcn_options,
                        backbone_type=args_loaded['backbone_type'],
                        relu_type=args_loaded['relu_type'],
                        width_mult=args_loaded['width_mult'],
                        extract_feats=extract_feats)

        if len(weights) > 0:
            print('Loading weights for lipreading stream')
            net.load_state_dict(torch.load(weights))
        return net


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--ckpt", type=str,  default="logs/12/last.ckpt",help='Path to model checkpoint.')
    parser.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), default="ald", help="Corrector class for the PC sampler.")
    parser.add_argument("--corrector_steps", type=int, default=1, help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.5, help="SNR value for (annealed) Langevin dynmaics.")
    parser.add_argument("--N", type=int, default=300, help="Number of reverse steps")
    parser.add_argument("--withVisual", type=bool, default=True)
    args = parser.parse_args()

    checkpoint_file = args.ckpt 
    corrector_cls = args.corrector

    # Settings
    sr = 16000
    snr = args.snr
    N = args.N
    withVisual=args.withVisual
    corrector_steps = args.corrector_steps
    
    audio_length = 2.55
    audio_sampling_rate = 16000
    seed = 42
    num_frames = 64
    number_of_identity_frames = 1
    data_path = "/dataset/VoxCeleb2"
    normalization = True
    audio_augmentation = False
    audio_normalization = True
    window_size = 400
    hop_size = 160
    n_fft = 512
    audio_window = int(audio_length * audio_sampling_rate)
    lipreading_preprocessing_func = get_preprocessing_pipelines()["val"]
    h5f_path = os.path.join(data_path, "train.h5") 
    h5f = h5py.File(h5f_path, 'r')
    videos_path = list(h5f['videos_path'][:])
    videos_path = [x.decode("utf-8") for x in videos_path]
    videos_path = [s.replace('val', 'train') for s in videos_path]

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    vision_transform_list = [transforms.Resize(224), transforms.ToTensor()]
    vision_transform_list.append(normalize)
    vision_transform = transforms.Compose(vision_transform_list)
    spec_abs_exponent=0.5
    spec_factor = 0.15
    # /Users/jihoojung/Desktop/snu/6-1/marg/mp4/id00061/0G9G9oyFHI8/00001.mp4
    common_path2="train/id00419/1zffAxBod_c/00006"
    #common_path2="train/id00154/2pSNL5YdcoQ/00003"
    common_path1="train/id00061/0G9G9oyFHI8/00001"
    # /Users/jihoojung/Desktop/snu/6-1/marg/mp4/id00419/1zffAxBod_c
    new_folder_path = create_next_numeric_folder("output")
    print(f"New directory created: {new_folder_path}")
    
    
    video_path_A1 = os.path.join("/dataset/VoxCeleb2/mp4",common_path1+".mp4")
    mouthroi_path_A1 = os.path.join("/dataset/VoxCeleb2/mouth_roi_hdf5",common_path1+".h5")
    audio_path_A1 = os.path.join("/dataset/VoxCeleb2/aac",common_path1+".m4a")
        
    video_path_B = os.path.join("/dataset/VoxCeleb2/mp4",common_path2+".mp4")
    mouthroi_path_B = os.path.join("/dataset/VoxCeleb2/mouth_roi_hdf5",common_path2+".h5")
    audio_path_B = os.path.join("/dataset/VoxCeleb2/aac",common_path2+".m4a")

    mouthroi_A1 = load_mouthroi(mouthroi_path_A1)
    mouthroi_B = load_mouthroi(mouthroi_path_B)
    frame_rate, audio_A1 = read_m4a(audio_path_A1)
    _, audio_B = read_m4a(audio_path_B)
    
    audio_A1 = audio_A1 / 32768
    audio_B = audio_B / 32768
    
    mouthroi_A1, audio_A1 = get_mouthroi_audio_pair(mouthroi_A1, audio_A1, audio_window, num_frames, audio_sampling_rate)
    mouthroi_B, audio_B = get_mouthroi_audio_pair(mouthroi_B, audio_B, audio_window, num_frames, audio_sampling_rate)
    save_as_wav(frame_rate, audio_A1*32768, os.path.join(new_folder_path,"A1.wav"))
    save_as_wav(frame_rate, audio_B*32768, os.path.join(new_folder_path,"B.wav"))
    save_as_wav(frame_rate, (audio_A1 + audio_B) / 2*32768, os.path.join(new_folder_path,"mix.wav"))
    
    frame_A_list = []
    frame_B_list = []
    for i in range(number_of_identity_frames):
        frame_A = load_frame(video_path_A1)
        frame_B = load_frame(video_path_B)
        frame_A = vision_transform(frame_A)
        frame_B = vision_transform(frame_B)
        frame_A_list.append(frame_A)
        frame_B_list.append(frame_B)
    frames_A = torch.stack(frame_A_list).squeeze()
    frames_B = torch.stack(frame_B_list).squeeze()

    mouthroi_A1 = lipreading_preprocessing_func(mouthroi_A1) #(64,88,88)
    mouthroi_B = lipreading_preprocessing_func(mouthroi_B)#(64,88,88)
    
    
    mouthroi_A1=torch.FloatTensor(mouthroi_A1).unsqueeze(0).unsqueeze(0)
    frame_A=frame_A.unsqueeze(0)
    audio_A1,rms_A1 = test_normalize(audio_A1)
    audio_B,rms_B = test_normalize(audio_B)
    audio_mix1 = (audio_A1 + audio_B) / 2 #float64,(40800,)
    audio_spec_A1 = generate_spectrogram_complex(audio_A1, window_size, hop_size, n_fft) #(2,257,256)->(257,256)
    audio_spec_B = generate_spectrogram_complex(audio_B, window_size, hop_size, n_fft) #(2,257,256)
    audio_spec_mix1 = generate_spectrogram_complex(audio_mix1, window_size, hop_size, n_fft) #(2,257,256)
    visualize_and_save_spectrogram(audio_spec_A1, os.path.join(new_folder_path,"A1.png"),hop_size)
    visualize_and_save_spectrogram(audio_spec_mix1, os.path.join(new_folder_path,"mix.png") ,hop_size)
    visualize_and_save_spectrogram(audio_spec_B, os.path.join(new_folder_path,"B.png"),hop_size)
    audio_spec_A1, audio_spec_B,audio_spec_mix1 = spec_fwd(audio_spec_A1), spec_fwd(audio_spec_B), spec_fwd(audio_spec_mix1)
    
    
    # mouthroi_B = torch.FloatTensor(mouthroi_B).unsqueeze(0).squeeze(1)
    audio_spec_A1=audio_spec_A1[:, :-1, :].unsqueeze(0).cuda()
    audio_spec_B = audio_spec_B[:, :-1, :].unsqueeze(0).cuda()
    audio_spec_mix1= audio_spec_mix1[:, :-1, :].unsqueeze(0).cuda()
    

    # Load score model 
    model = MyScoreModel.load_from_checkpoint(checkpoint_file, base_dir='', batch_size=1, num_workers=1, kwargs=dict(gpu=False))
    model.eval(no_ema=False)
    model.cuda()
    
    facialnetmodel=build_facial().cuda()
    net_lipreading=build_lipreadingnet().cuda()
    
    
    mouthroi_A1_embed = net_lipreading(Variable(mouthroi_A1.cuda(), requires_grad=False), 64) #(2,512,1,64)
    identity_feature_A = facialnetmodel(Variable(frame_A.cuda(), requires_grad=False))
    identity_feature_A = F.normalize(identity_feature_A, p=2, dim=1) #(2,128,1,1)
    identity_feature_A = identity_feature_A.repeat(1, 1, 1, mouthroi_A1_embed.shape[-1]) #(2,128,1,64)
    visual_feature_A1 = torch.cat((identity_feature_A, mouthroi_A1_embed), dim=1) #(2,640,1,64)
    visual_embedding=visual_feature_A1.permute(0, 3, 1, 2).squeeze(3)
    y=audio_spec_mix1
    if not withVisual:
        visual_embedding = torch.zeros_like(visual_embedding)
    # Reverse sampling
    # sampler=model.get_ode_sampler(y.cuda(),visual_embedding.cuda(), N=N)
    sampler = model.get_pc_sampler( #y는 
        'reverse_diffusion', corrector_cls, y.cuda(),visual_embedding.cuda(), N=N,  #corrector_cls = ald, N = 30
        corrector_steps=corrector_steps, snr=snr) #corrector_steps=1, snr=0.5, N=30
    sample, _ = sampler() #pc_sampler() 이게 진짜 샘플링하는 과정 (1,1,256,256)
    
    # Backward transform in time domain
    T_orig=40720
    spec,x_hat = model.to_audio(sample.squeeze(), T_orig)
    visualize_and_save_spectrogram(spec, os.path.join(new_folder_path,"generated.png"),160)
    # Renormalize
    x_hat=test_denormalize(x_hat,rms_A1)
    
    # Write enhanced wav file
    out=os.path.join(new_folder_path,"generated.wav")
    
    # out_path=output_path1+"_generated.wav"
    write(out, x_hat.cpu().numpy(), 16000)
    print(f"{out} 에 저장함.")