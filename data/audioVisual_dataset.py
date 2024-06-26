#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from torch.utils.data import Dataset
import sys
import matplotlib.pyplot as plt

import wave
from torchaudio import load
import cv2
import os.path
import librosa
from scipy.io import wavfile
import h5py
import random
from random import randrange
import glob
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import torchvision.transforms as transforms
import torch
from vv_utils.lipreading_preprocess import *
import time
# from torchvision.transforms import Resize
from vv_utils.video_reader import VideoReader
import ffmpeg
from pydub import AudioSegment
import io
def read_m4a(file_path):
    audio = AudioSegment.from_file(file_path, format='m4a')
    audio = audio.set_channels(1)  # 스테레오를 모노로 변환 (필요한 경우)
    samples = np.array(audio.get_array_of_samples())
    return audio.frame_rate, samples
def read_m4a_with_wav(filename):
    out, _ = (
        ffmpeg
        .input(filename)
        .output('pipe:', format='wav')
        .run(capture_stdout=True, capture_stderr=True)
    )
    wav, sample_rate = load(io.BytesIO(out))
    return wav, sample_rate
def save_as_wav(frame_rate, samples, output_file_path):
    # Convert samples to bytes
    samples = samples.astype(np.int16).tobytes()
    
    # Write the WAV file
    with wave.open(output_file_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # Sample width in bytes
        wav_file.setframerate(frame_rate)
        wav_file.writeframes(samples)

def normalize(samples, desired_rms = 0.1, eps = 1e-4):
  rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
  samples = samples * (desired_rms / rms)
  return samples
  
def get_preprocessing_pipelines():
    # -- preprocess for the video stream
    preprocessing = {}
    # -- LRW config
    crop_size = (88, 88)
    resize_size = (112, 112)
    (mean, std) = (0.421, 0.165)
    preprocessing['train'] = Compose([
                                Normalize( 0.0,255.0 ),
                                RandomCrop(crop_size),
                                Resize(resize_size),
                                HorizontalFlip(0.5),
                                Normalize(mean, std) ])
    preprocessing['val'] = Compose([
                                Normalize( 0.0,255.0 ),
                                CenterCrop(crop_size),
                                Resize(resize_size),
                                Normalize(mean, std) ])
    preprocessing['test'] = preprocessing['val']
    return preprocessing
#비디오 파일에서 임의의 프레임을 추출하여 이미지로 변환
def load_frame(clip_path):
    video_reader = VideoReader(clip_path, 1)
    start_pts, time_base, total_num_frames = video_reader._compute_video_stats()
    end_frame_index = total_num_frames - 1
    if end_frame_index < 0:
        clip, _ = video_reader.read(start_pts, 1)
    else:
        clip, _ = video_reader.read(random.randint(0, end_frame_index) * time_base, 1)
    frame = Image.fromarray(np.uint8(clip[0].to_rgb().to_ndarray())).convert('RGB')
    return frame

def save_mouthroi(mouthroi):
    save_path = "tmp/mouthroi_image_{}.png"
    for i in range(mouthroi.shape[0]):
        img = mouthroi[i]
        cv2.imwrite(save_path.format(i), img)
    print("이미지가 성공적으로 저장되었습니다.")
    
    
def get_mouthroi_audio_pair(mouthroi, audio, window, num_of_mouthroi_frames, audio_sampling_rate):
    audio_start = randrange(0, audio.shape[0] - window + 1)
    audio_sample = audio[audio_start:(audio_start+window)] #(40800), window=40800
    frame_index_start = int(round(audio_start / audio_sampling_rate * 25)) #num_of_mouthroi_frames=64
    mouthroi = mouthroi[frame_index_start:(frame_index_start + num_of_mouthroi_frames), :, :]
    # save_mouthroi(mouthroi)
    return mouthroi, audio_sample #(128,96,96)->(64,96,96), (40800,

def load_mouthroi(filename):
    try:
        if filename.endswith('npz'):
            return np.load(filename)['data']
        elif filename.endswith('h5'):
            with h5py.File(filename, 'r') as hf:
                return hf["data"][:]
        else:
            return np.load(filename)
    except IOError:
        print( "Error when reading file: {}".format(filename) )
        sys.exit()

def generate_spectrogram_magphase(audio, stft_frame, stft_hop, n_fft):
    spectro = librosa.core.stft(audio, hop_length=stft_hop, n_fft=n_fft, win_length=stft_frame, center=True)
    spectro_mag, spectro_phase = librosa.core.magphase(spectro)
    spectro_mag = np.expand_dims(spectro_mag, axis=0)
    with_phase=False
    if with_phase:
        spectro_phase = np.expand_dims(np.angle(spectro_phase), axis=0)
        return spectro_mag, spectro_phase
    else:
        return spectro_mag

def generate_spectrogram_complex(audio, stft_frame, stft_hop, n_fft):
    spectro = librosa.core.stft(audio, hop_length=stft_hop, n_fft=n_fft, win_length=stft_frame, center=True)
    # audio_tensor=torch.from_numpy(audio).unsqueeze(0)
    # spectro_tensor = torch.stft(audio_tensor, hop_length=stft_hop, n_fft=n_fft, win_length=stft_frame, center=True)
    
    # real = np.expand_dims(np.real(spectro), axis=0)
    # imag = np.expand_dims(np.imag(spectro), axis=0)
    # spectro_two_channel = np.concatenate((real, imag), axis=0)
    spectro=spectro.astype(np.complex64)
    spectro=torch.from_numpy(spectro).unsqueeze(0)
    return spectro

def augment_image(image):
    if(random.random() < 0.5):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.random()*0.6 + 0.7)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.random()*0.6 + 0.7)
    return image

def augment_audio(audio):
    audio = audio * (random.random() * 0.2 + 0.9) # 0.9 - 1.1
    audio[audio > 1.] = 1.
    audio[audio < -1.] = -1.
    return audio
man_ids = [
    "id00017", "id00061", "id00081", "id00154", "id00562", "id00817", "id00866", 
    "id00926", "id01041", "id01066", "id01106", "id01298", "id01437", "id01509", 
    "id01541", "id01593", "id01822", "id01892", "id01989", "id02019", "id02057", 
    "id02317", "id02542", "id02576", "id02577", "id02685", "id02745", "id03030", 
    "id03041", "id03178", "id03347", "id03524", "id03677", "id03789", "id03839", 
    "id03862", "id04006", "id04094", "id04119", "id04232", "id04253", "id04276", 
    "id04295", "id04366", "id04478", "id04536"
]
woman_ids = [
    "id00419", "id00812", "id01000", "id01224", "id01228", "id01333", "id01460",
    "id01567", "id01618", "id02086", "id02181", "id02286", "id02445", "id02465",
    "id02548", "id02725", "id03127", "id03382", "id03969", "id03978", "id03980",
    "id03981", "id04030", "id04570"
]


class AudioVisualDataset(Dataset):
    def __init__(self,spec_transform):
        super(AudioVisualDataset, self).__init__()
        # self.opt = opt
        self.audio_length = 2.55
        self.audio_sampling_rate = 16000
        self.seed = 42
        self.num_frames = 64
        self.mode = "train"
        self.number_of_identity_frames = 1
        self.data_path = "/dataset/VoxCeleb2"
        self.normalization = True
        self.audio_augmentation = False
        self.audio_normalization = True
        self.window_size = 400
        self.hop_size = 160
        self.n_fft = 512
        self.batchSize = 4
        self.num_batch = 50000
        self.validation_batches = 30
        self.audio_window = int(self.audio_length * self.audio_sampling_rate)
        self.spec_transform=spec_transform
        random.seed(self.seed)
        self.lipreading_preprocessing_func = get_preprocessing_pipelines()[self.mode]
        
        # Load videos path from hdf5 file
        h5f_path = os.path.join(self.data_path, self.mode + '.h5') 
        h5f = h5py.File(h5f_path, 'r')
        self.videos_path = list(h5f['videos_path'][:])
        self.videos_path = [x.decode("utf-8") for x in self.videos_path]
        self.videos_path = [s.replace('val', 'train') for s in self.videos_path]

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        vision_transform_list = [transforms.Resize(224), transforms.ToTensor()]
        if self.normalization:
            vision_transform_list.append(normalize)
        self.vision_transform = transforms.Compose(vision_transform_list)
        self.spec_abs_exponent=0.5
        self.spec_factor = 0.15
        self.man_videos = [path for path in self.videos_path if any(vid in path for vid in man_ids)]
        self.woman_videos = [path for path in self.videos_path if any(vid in path for vid in woman_ids)]
        
    def __getitem__(self, index):
        if random.choice([True, False]):
            videos1 = random.sample(self.man_videos, 1)
            videos2 = random.sample(self.woman_videos, 1)
        else:
            videos1 = random.sample(self.woman_videos, 1)
            videos2 = random.sample(self.man_videos, 1)
        videos2Mix=[videos1[0],videos2[0]]
        # videos2Mix = random.sample(self.videos_path, 2) #get two videos
        #sample two clips for speaker A
        videoA_clips = os.listdir(videos2Mix[0])
        clipPair_A = random.choices(videoA_clips, k=2) #randomly sample two clips
        #clip A1
        video_path_A1 = os.path.join(videos2Mix[0], clipPair_A[0])
        mouthroi_path_A1 = os.path.join(videos2Mix[0].replace('/mp4/', '/mouth_roi_hdf5/'), clipPair_A[0].replace('.mp4', '.h5'))
        audio_path_A1 = os.path.join(videos2Mix[0].replace('/mp4/', '/aac/'), clipPair_A[0].replace('.mp4', '.m4a'))
        #clip A2
        video_path_A2 = os.path.join(videos2Mix[0], clipPair_A[1])
        mouthroi_path_A2 = os.path.join(videos2Mix[0].replace('/mp4/', '/mouth_roi_hdf5/'), clipPair_A[1].replace('.mp4', '.h5'))
        audio_path_A2 = os.path.join(videos2Mix[0].replace('/mp4/', '/aac/'), clipPair_A[1].replace('.mp4', '.m4a'))
        #sample one clip for person B
        videoB_clips = os.listdir(videos2Mix[1])
        clipB = random.choice(videoB_clips) #randomly sample one clip
        video_path_B = os.path.join(videos2Mix[1], clipB)
        mouthroi_path_B = os.path.join(videos2Mix[1].replace('/mp4/', '/mouth_roi_hdf5/'), clipB.replace('.mp4', '.h5'))
        audio_path_B = os.path.join(videos2Mix[1].replace('/mp4/', '/aac/'), clipB.replace('.mp4', '.m4a'))

        #start_time = time.time()
        mouthroi_A1 = load_mouthroi(mouthroi_path_A1)
        mouthroi_A2 = load_mouthroi(mouthroi_path_A2)
        mouthroi_B = load_mouthroi(mouthroi_path_B)
        _, audio_A1 = read_m4a(audio_path_A1)
        # audio_A1_wav,_=read_m4a_with_wav(audio_path_A1)
        _, audio_A2 = read_m4a(audio_path_A2)
        _, audio_B = read_m4a(audio_path_B)
        audio_A1 = audio_A1 / 32768
        audio_A2 = audio_A2 / 32768
        audio_B = audio_B / 32768

        if not (len(audio_A1) > self.audio_window and len(audio_A2) > self.audio_window and len(audio_B) > self.audio_window):
            return self.__getitem__(index)
        
        mouthroi_A1, audio_A1 = get_mouthroi_audio_pair(mouthroi_A1, audio_A1, self.audio_window, self.num_frames, self.audio_sampling_rate)
        mouthroi_A2, audio_A2 = get_mouthroi_audio_pair(mouthroi_A2, audio_A2, self.audio_window, self.num_frames, self.audio_sampling_rate)
        mouthroi_B, audio_B = get_mouthroi_audio_pair(mouthroi_B, audio_B, self.audio_window, self.num_frames, self.audio_sampling_rate)
        
        frame_A_list = []
        frame_B_list = []
        for i in range(self.number_of_identity_frames):
            frame_A = load_frame(video_path_A1)
            frame_B = load_frame(video_path_B)
            if self.mode == 'train':
                frame_A = augment_image(frame_A)
                frame_B = augment_image(frame_B)
            frame_A = self.vision_transform(frame_A)
            frame_B = self.vision_transform(frame_B)
            frame_A_list.append(frame_A)
            frame_B_list.append(frame_B)
        frames_A = torch.stack(frame_A_list).squeeze()
        frames_B = torch.stack(frame_B_list).squeeze() 

        if not (mouthroi_A1.shape[0] == self.num_frames and mouthroi_A2.shape[0] == self.num_frames and mouthroi_B.shape[0] == self.num_frames):
            return self.__getitem__(index)

        #transform mouthrois and audios
        mouthroi_A1 = self.lipreading_preprocessing_func(mouthroi_A1) #(64,88,88)
        mouthroi_A2 = self.lipreading_preprocessing_func(mouthroi_A2)#(64,88,88)
        mouthroi_B = self.lipreading_preprocessing_func(mouthroi_B)#(64,88,88)
        
        #transform audio
        if(self.audio_augmentation and self.mode == 'train'):
            audio_A1 = augment_audio(audio_A1)
            audio_A2 = augment_audio(audio_A2)
            audio_B = augment_audio(audio_B)
        if self.audio_normalization:
            audio_A1 = normalize(audio_A1)
            audio_A2 = normalize(audio_A2)
            audio_B = normalize(audio_B)
                
        #get audio spectrogram
        audio_mix1 = (audio_A1 + audio_B) / 2 #float64,(40800,)
        audio_mix2 = (audio_A2 + audio_B) / 2
        # print(audio_mix1.shape)
        
        audio_spec_A1 = generate_spectrogram_complex(audio_A1, self.window_size, self.hop_size, self.n_fft) #(2,257,256)->(257,256)
        audio_spec_A2 = generate_spectrogram_complex(audio_A2, self.window_size, self.hop_size, self.n_fft) #(2,257,256)
        audio_spec_B = generate_spectrogram_complex(audio_B, self.window_size, self.hop_size, self.n_fft) #(2,257,256)
        audio_spec_mix1 = generate_spectrogram_complex(audio_mix1, self.window_size, self.hop_size, self.n_fft) #(2,257,256)
        audio_spec_mix2 = generate_spectrogram_complex(audio_mix2, self.window_size, self.hop_size, self.n_fft) #(2,257,256)
        
       

        audio_spec_A1, audio_spec_A2,audio_spec_B,audio_spec_mix1,audio_spec_mix2 = self.spec_transform(audio_spec_A1), self.spec_transform(audio_spec_A2),self.spec_transform(audio_spec_B), self.spec_transform(audio_spec_mix1),self.spec_transform(audio_spec_mix2)
        
        data = {}
        data['mouthroi_A1'] = torch.FloatTensor(mouthroi_A1).unsqueeze(0)
        data['mouthroi_A2'] = torch.FloatTensor(mouthroi_A2).unsqueeze(0)
        data['mouthroi_B'] = torch.FloatTensor(mouthroi_B).unsqueeze(0)
        data['frame_A'] = frames_A #(3,224,224)
        data['frame_B'] = frames_B #(3,224,224)

        data['audio_spec_A1'] = audio_spec_A1[:, :-1, :]
        data['audio_spec_A2'] = audio_spec_A2[:, :-1, :]
        data['audio_spec_B'] = audio_spec_B[:, :-1, :]
        data['audio_spec_mix1'] = audio_spec_mix1[:, :-1, :]
        data['audio_spec_mix2'] = audio_spec_mix2[:, :-1, :]
        return data

    def __len__(self):
        if self.mode == 'train':
            return self.num_batch
        elif self.mode == 'val':
            return self.validation_batches

    def name(self):
        return 'AudioVisualDataset'


def test_normalize(samples, desired_rms=0.1, eps=1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    samples_normalized = samples * (desired_rms / rms)
    return samples_normalized, rms
def visualize_and_save_spectrogram(spectro, output_image_path,stft_hop):
    spectro = spectro.cpu().numpy().squeeze()
    magnitude = np.abs(spectro)
    log_magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_magnitude, sr=16000, hop_length=stft_hop, x_axis='time', y_axis='log', cmap=None)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.close()
    
if __name__=="__main__":
    dataset = AudioVisualDataset(spec_transform=None)
    print(dataset[0])
    print()