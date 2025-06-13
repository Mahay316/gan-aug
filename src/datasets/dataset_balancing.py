import math
import os

def get_average_sample_count(data_dir):
    sample_cnt = 0
    file_cnt = 0
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_cnt += 1
            with open(os.path.join(data_dir, filename), 'r') as f:
                for line in f:
                    sample_cnt += 1

    return math.floor(sample_cnt / file_cnt)


def balancing_dataset(data_dir, output_dir, average_cnt):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(data_dir, filename)
            output_path = os.path.join(output_dir, filename)

            written_cnt = 0
            with open(input_path, 'r') as infile, open(output_path, 'w', newline='\n') as outfile:
                for line in infile:
                    if written_cnt >= average_cnt:
                        break

                    outfile.write(line + '\n')
                    written_cnt += 1


if __name__ == '__main__':
    data_directory = "D:/traffic_dataset/span/normalized"
    output_directory = "D:/traffic_dataset/span/balanced"

    average_cnt = get_average_sample_count(data_directory)
    print(average_cnt)
    balancing_dataset(data_directory, output_directory, average_cnt)
