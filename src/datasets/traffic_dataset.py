import json
import os.path

import torch
import pickle

from torch.utils.data import Dataset, DataLoader


class TrafficDataset(Dataset):
    def __init__(self, data_dir, index_file='dataset.idx', transform=None):
        self.data_dir = data_dir
        self.transform = transform

        with open(os.path.join(data_dir, index_file), 'rb') as f:
            self.dataset_index = pickle.load(f)
            self.index_to_filename = pickle.load(f)

        self.file_pool = []
        for filename in self.index_to_filename:
            self.file_pool.append(
                open(os.path.join(data_dir, filename), 'r')
            )

    def calculate_file_and_offset(self, idx):
        file_idx = 0

        for file in self.dataset_index:
            if idx < len(file):
                break  # idx falls into this file
            else:
                idx -= len(file)
                file_idx += 1

        return file_idx, self.dataset_index[file_idx][idx]

    def __len__(self):
        return sum([len(x) for x in self.dataset_index])

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        file_idx, offset = self.calculate_file_and_offset(idx)
        target_file = self.file_pool[file_idx]
        target_file.seek(offset)

        trace = json.loads(target_file.readline())
        sample = (trace[0], trace[1])

        return sample


def my_collate_fn(data):
    traces = []
    labels = []

    for sample in data:
        traces.append(torch.tensor(sample[0]))
        labels.append(sample[1])
    return traces, labels


if __name__ == '__main__':
    root_dir = 'D:/traffic_dataset'
    cd = TrafficDataset(root_dir)

    train, val, test = torch.utils.data.random_split(cd, [0.8, 0.1, 0.1])
    dl = DataLoader(train, batch_size=32, shuffle=True, collate_fn=my_collate_fn)

    print(len(dl))

    for batch_idx, samples in enumerate(dl):
        traces, labels = samples
        for trace in traces:
            print(trace)
        break
