import os.path
import pickle

type2label = {
    'nontor': 0,
    'tor': 1,
    'obfs4': 2,
    'webtunnel': 3,
    'snowflake': 4,
    'dnstt': 5,
    'shadowsocks': 6
    # 'conjure': 7,
    # 'cloak': 8,
}


def write_dataset_stat(data_dir: str, output: str = 'dataset.idx'):
    trace_offset = []
    index_to_filename = []
    index_to_label = []

    for file in os.listdir(data_dir):
        if not file.startswith('pt_'):
            continue

        offset = 0
        trace_offset_per_type = [0]
        with open(os.path.join(data_dir, file), 'r') as f:
            for line in f:
                offset += len(line)
                trace_offset_per_type.append(offset)

        trace_offset_per_type.pop()
        trace_offset.append(trace_offset_per_type)
        index_to_filename.append(file)

        traffic_type = file.split('_')[1]
        label = type2label[traffic_type]
        index_to_label.append(label)

        print(f'Finished processing traffic: {traffic_type} -> {label}')

    with open(os.path.join(data_dir, output), 'wb') as out:
        pickle.dump(trace_offset, out)
        pickle.dump(index_to_filename, out)
        pickle.dump(index_to_label, out)


def read_sample(idx, root_dir, index_file: str = 'dataset.idx'):
    with open(os.path.join(root_dir, index_file), 'rb') as f:
        trace_offset = pickle.load(f)
        index_to_filename = pickle.load(f)

    print(index_to_filename)
    with open(os.path.join(root_dir, index_to_filename[0]), 'r') as f:
        f.seek(trace_offset[0][idx])
        print(f.readline())


if __name__ == '__main__':
    write_dataset_stat('D:/traffic_dataset/span/balanced')
    # read_sample(3, 'D:/traffic_dataset/interval/normalized')
