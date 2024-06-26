import numpy as np
import glob
from torch.autograd import Variable
from tensorboard import summary
from tqdm import tqdm
from torchaudio import load, save
import torch
import os
import torch.nn.functional as F
from argparse import ArgumentParser
import time
from soundfile import write
from models.networks import Resnet18
from data.audioVisual_dataset import *
from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module import SpecsDataModule
from sgmse.sdes import SDERegistry
from sgmse.model import StochasticRegenerationModel, MyScoreModel, DiscriminativeModel
from models.lipreading_model import Lipreading

from sgmse.util.other import *

import matplotlib.pyplot as plt
import torchvision
EPS_LOG = 1e-10

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
        net = Resnet18(original_resnet, pool_type=pool_type, with_fc=with_fc, fc_in=512, fc_out=fc_out)

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
	# Tags
	debug=False
	base_parser = ArgumentParser(add_help=False)
	parser = ArgumentParser()

	for parser_ in (base_parser, parser):
		parser_.add_argument("--ckpt", type=str, required=False)
		parser_.add_argument("--mode", default="storm", choices=["score-only", "denoiser-only", "storm"])

		parser_.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), default="ald", help="Corrector class for the PC sampler.")
		parser_.add_argument("--corrector-steps", type=int, default=1, help="Number of corrector steps")
		parser_.add_argument("--snr", type=float, default=0.5, help="SNR value for (annealed) Langevin dynamics.")
		parser_.add_argument("--N", type=int, default=50, help="Number of reverse steps")

	args = parser.parse_args()
	args.ckpt="/logs/storm/mode=regen-joint-training_sde=OUVESDE_score=ncsnpp_denoiser=ncsnpp_condition=both_data=wsj0_ch=1/version_16/checkpoints/last.ckpt"
	
	#Checkpoint
	checkpoint_file = args.ckpt

	# Settings
	model_sr = 16000
	withVisual=True
	# Load score model 
	if args.mode == "storm":
		model_cls = StochasticRegenerationModel
	elif args.mode == "score-only":
		model_cls = MyScoreModel
	elif args.mode == "denoiser-only":
		model_cls = DiscriminativeModel
	if debug:
		model = model_cls.load_from_checkpoint(
		checkpoint_file, base_dir="",
		batch_size=1, num_workers=0, debug=True,kwargs=dict(gpu=False)
	)
	else:
		model = model_cls.load_from_checkpoint(
			checkpoint_file, base_dir="",
			batch_size=1, num_workers=0, kwargs=dict(gpu=False)
		)
	model.eval(no_ema=False)
	model.cuda()

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
	common_path1="train/id00562/_HozbUvc988/00133"
    #common_path2="train/id00154/2pSNL5YdcoQ/00003"
	common_path2="train/id00061/0G9G9oyFHI8/00001"
	new_folder_path = create_next_numeric_folder("output")
	print(f"New directory created: {new_folder_path}")
    
	if debug:
		real_part = torch.rand(1, 1, 256, 256)
		imaginary_part = torch.rand(1, 1, 256, 256)
		y = torch.complex(real_part, imaginary_part).cuda()
		visual_embedding=torch.rand(1, 64, 640).cuda()
	else:
		print("wo debug")
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
	Y_denoised,sample=model.separate(y,visual_embedding) #(1,1,256,256)
	
	T_orig=40720
	spec,x_hat = model.to_audio(sample.squeeze(), T_orig)
	spec_y,x_hat_y=model.to_audio(Y_denoised.squeeze(),T_orig)
	visualize_and_save_spectrogram(spec, os.path.join(new_folder_path,"generated.png"),160)
	visualize_and_save_spectrogram(spec, os.path.join(new_folder_path,"denoised.png"),160)
		
 # Renormalize
	# x_hat=test_denormalize(x_hat,rms_A1)
	# x_hat_y=test_denormalize(x_hat_y,rms_A1)

	# Write enhanced wav file
	out=os.path.join(new_folder_path,"generated.wav")
	out_=os.path.join(new_folder_path,"denoised.wav")
	# out_path=output_path1+"_generated.wav"
	write(out, x_hat.cpu().numpy(), 16000)
	write(out_, x_hat_y.cpu().numpy(), 16000)
	print(f"{out} 에 저장함.")
