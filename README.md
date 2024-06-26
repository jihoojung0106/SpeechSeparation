# Audio-Visual Speech Separation with Diffusion-based Generative Models

This project performs Audio-Visual Speech Separation using a diffusion model.
THERE IS TWO BRANCH(STORM, SGMSE). NEITHER OF THEM IS WORKING AS I WISH. 

## Installation

```bash
pip install -r requirements.txt
```


## Dataset 
I used [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) dataset. You can also find it on [huggingface](https://huggingface.co/datasets/ProgramComputer/voxceleb/tree/main/vox2). 
The dataset preprocessing steps follow those outlined in [VisualVoice](https://github.com/facebookresearch/VisualVoice). 
Since the pre-processed mouth ROIs are too large to download in full, I only downloaded the VoxCeleb2 validation set and renamed it as "train" to use it for training. For the full dataset, please refer to the [original VisualVoice code](https://github.com/facebookresearch/VisualVoice).
The pre-processed mouth ROIs can be downloaded as follows:
```
# mounth ROIs for VoxCeleb2 (train: 1T; val: 20G; seen_heard_test: 88G; unseen_unheard_test: 20G)
wget http://dl.fbaipublicfiles.com/VisualVoice/mouth_roi_val.tar.gz

# Directory structure of the dataset:
#    ├── VoxCeleb2                          
#    │       └── [mp4]               (contain the face tracks in .mp4)
#    │                └── [train]
#    │                
#    │       └── [audio]             (contain the audio files in .m4a)
#    │                └── [train]
#    │                
#    │       └── [mouth_roi]         (contain the mouth ROIs in .h5)
#    │                └── [train]
#    │                
```

2. Download the hdf5 files that contain the data paths, and then modify the hdf5 file accordingly by changing the paths to have the correct root prefix of your own.
```
wget http://dl.fbaipublicfiles.com/VisualVoice/hdf5/VoxCeleb2/val.h5
```

3. Please download the pretrained weights for facial model in [here](https://drive.google.com/file/d/1R-6QJ8fWDHqb3jrjWaYacZtfgCrROWhC/view?usp=sharing) and put it under the checkpoints/.

## Inference

ex) 
```bash
python separation.py --ckpt [path-to-ckpt]
```
You can download the pretrained weight for the model in [here](https://drive.google.com/file/d/1R-6QJ8fWDHqb3jrjWaYacZtfgCrROWhC/view?usp=sharing). 

## Training
```bash
python train.py --batch_size 1  --accumulate_grad_batches 1024

```
If you want to resume training from a checkpoint, add --ckpt to the arguments. Don't forget to change the GPU IDs on line 115 in train.py.


## References

- [Speech Enhancement and Dereverberation with Diffusion-based Generative Models](https://github.com/sp-uhh/sgmse)
- [VisualVoice: Audio-Visual Speech Separation with Cross-Modal Consistency](https://github.com/facebookresearch/VisualVoice)
- [Seeing Through the Conversation: Audio-Visual Speech Separation based on Diffusion Model](https://arxiv.org/abs/2310.19581)
# speechldm
# speechldm
# speechldm
# SpeechSeparation
# SpeechSeparation
