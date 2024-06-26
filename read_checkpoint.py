import torch

# 체크포인트 파일 경로
checkpoint_path = 'logs/41/last.ckpt'

# # 체크포인트 파일 로드
# checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# # 체크포인트에서 에포크 정보 가져오기
# epoch = checkpoint['epoch']
# global_step = checkpoint['global_step']

# print(f"Checkpoint was saved at epoch {epoch}, global step {global_step}")
# best_model_score = None
# if 'callbacks' in checkpoint:

#     for key, value in checkpoint['callbacks'].items():
#         # print(key, value)
#         if 'best_model_score' in value:
#             best_model_score = value['best_model_score']
#             break

# print(f"Checkpoint was saved at epoch {epoch}, global step {global_step}, best model score (loss): {best_model_score}")

import torch


# .ckpt 파일 로드
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# 체크포인트의 키 확인
print(checkpoint.keys())

# 모델 가중치 확인
model_state_dict = checkpoint['state_dict']
print(model_state_dict.keys())
