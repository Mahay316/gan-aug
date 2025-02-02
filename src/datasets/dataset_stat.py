import os.path
import pickle

# traffic filename -> traffic type
dataset_list = {
    'pt_tor.txt': 'Tor',
    'pt_normal.txt': 'Non-Tor',
    'pt_obfs4.txt': 'Obfs4',
    'pt_meek.txt': 'Meek',
    'pt_fte.txt': 'FTE',
    'pt_webtunnel.txt': 'WebTunnel'
}


def write_dataset_stat(root_dir: str, output: str = 'dataset.idx'):
    trace_offset = []
    index_to_filename = []

    for file, label in dataset_list.items():
        offset = 0
        trace_offset_per_type = [0]
        with open(os.path.join(root_dir, file), 'r') as f:
            for line in f:
                offset += len(line)
                trace_offset_per_type.append(offset)

        trace_offset_per_type.pop()
        trace_offset.append(trace_offset_per_type)
        index_to_filename.append(file)

        print(f'Finished processing traffic: {label}')

    with open(os.path.join(root_dir, output), 'wb') as out:
        pickle.dump(trace_offset, out)
        pickle.dump(index_to_filename, out)


def read_sample(idx, root_dir, index_file: str = 'dataset.idx'):
    with open(os.path.join(root_dir, index_file), 'rb') as f:
        trace_offset = pickle.load(f)

    with open(os.path.join(root_dir, 'pt_tor.txt'), 'r') as f:
        f.seek(trace_offset[0][idx])
        print(f.readline())


if __name__ == '__main__':
    write_dataset_stat('D:/BaiduNetdiskDownload/Tor Traffic/')
    # read_sample(2, 'D:/BaiduNetdiskDownload/Tor Traffic/')
