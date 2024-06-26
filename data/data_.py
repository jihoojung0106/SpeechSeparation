import pandas as pd

# 입력 파일 경로와 출력 파일 경로를 지정합니다.
input_file_path = 'data/vox2_vox2_meta.csv'  # 여기에 실제 파일 경로를 입력하세요.
output_file_path = 'data/filtered_vox2_meta.csv'  # 여기에 실제 파일 경로를 입력하세요.

# 필터링할 VoxCeleb2 ID 목록을 정의합니다.
desired_ids = [
    "id00017", "id00812", "id01066", "id01437", "id01618", "id02086", "id02542", "id02745", "id03382", "id03969", 
    "id04094", "id04366", "id00061", "id00817", "id01106", "id01460", "id01822", "id02181", "id02548", "id03030", 
    "id03524", "id03978", "id04119", "id04478", "id00081", "id00866", "id01224", "id01509", "id01892", "id02286", 
    "id02576", "id03041", "id03677", "id03980", "id04232", "id04536", "id00154", "id00926", "id01228", "id01541", 
    "id01989", "id02317", "id02577", "id03127", "id03789", "id03981", "id04253", "id04570", "id00419", "id01000", 
    "id01298", "id01567", "id02019", "id02445", "id02685", "id03178", "id03839", "id04006", "id04276", "id00562", 
    "id01041", "id01333", "id01593", "id02057", "id02465", "id02725", "id03347", "id03862", "id04030", "id04295"
]

# CSV 파일을 읽어옵니다.
data = pd.read_csv(input_file_path)

# 필요한 열의 공백을 제거합니다.
data.columns = data.columns.str.strip()
# print(data['VoxCeleb2 ID'])
data.columns = data.columns.str.strip()

# 각 열의 데이터 값의 공백도 제거합니다.
data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
# 필터링된 데이터를 생성합니다.
filtered_data = data[data['VoxCeleb2 ID'].isin(desired_ids)]

# 필터링된 데이터를 새로운 CSV 파일로 저장합니다.
filtered_data.to_csv(output_file_path, index=False)

print(f"Filtered data saved to {output_file_path}")
