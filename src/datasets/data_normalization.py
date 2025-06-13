import json
import os
import numpy as np


def find_global_min_max(data_dir):
    """遍历所有数据文件，计算每个packet vector分量的全局最小值和最大值"""
    min_vals = None
    max_vals = None

    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), 'r') as f:
                for line in f:
                    sample = json.loads(line.strip())
                    for flow in sample[0]:
                        for pv in flow:
                            vector = np.array(pv, dtype=np.float32)
                            if min_vals is None:
                                min_vals = vector.copy()
                                max_vals = vector.copy()
                            else:
                                min_vals = np.minimum(min_vals, vector)
                                max_vals = np.maximum(max_vals, vector)

    return min_vals, max_vals


def normalize_dataset(data_dir, output_dir, min_vals, max_vals):
    """使用找到的最小最大值，对整个数据集进行归一化，并保存"""
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(data_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with open(input_path, 'r') as infile, open(output_path, 'w', newline='\n') as outfile:
                for line in infile:
                    sample = json.loads(line.strip())
                    normalized_trace = []

                    for flow in sample[0]:
                        normalized_flow = []
                        for packet_vector in flow:
                            vector = np.array(packet_vector, dtype=np.float32)
                            if np.all(vector == 0):
                                norm_vector = vector.tolist()
                            else:
                                norm_vector = (vector - min_vals) / (max_vals - min_vals + 1e-8)  # 避免除零错误
                                norm_vector = [round(val, 4) for val in norm_vector.tolist()]  # 限制小数位数为6
                            normalized_flow.append(norm_vector)
                        normalized_trace.append(normalized_flow)

                    normalized_sample = (normalized_trace, sample[1])
                    outfile.write(json.dumps(normalized_sample) + "\n")


if __name__ == "__main__":
    data_directory = "D:/traffic_dataset/interval"
    output_directory = "D:/traffic_dataset/interval/normalized"

    print("Finding global min and max values...")
    min_values, max_values = find_global_min_max(data_directory)
    print("Min values:", min_values)
    print("Max values:", max_values)

    print("Normalizing dataset...")
    normalize_dataset(data_directory, output_directory, min_values, max_values)
    print("Normalization complete. Output saved in", output_directory)
