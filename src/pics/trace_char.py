import matplotlib.pyplot as plt
import numpy as np

from traffic_stat import get_packet_count, get_byte_count

# traffic filename -> traffic type
dataset_list = {
    'pt_tor.txt': 'Tor',
    'pt_normal.txt': 'Non-Tor',
    'pt_obfs4.txt': 'Obfs4',
    'pt_meek.txt': 'Meek',
    'pt_fte.txt': 'FTE',
    'pt_webtunnel.txt': 'WebTunnel'
}


def draw_cdf(samples, label):
    data_sorted = sorted(samples)
    cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    plt.plot(data_sorted, cdf, label=label, marker='none', linestyle='-')


def draw_line(sample, label):
    values, counts = np.unique(sample, return_counts=True)
    total = float(np.sum(counts).item())
    counts = counts / total
    plt.plot(values, counts, label=label, marker='none', linestyle='-')


root_dir = 'D:/BaiduNetdiskDownload/Tor Traffic/'

# draw packet count line
for file, label in dataset_list.items():
    ret = get_packet_count(root_dir + file)
    samples = list(map(lambda x: sum(x), ret))
    draw_line(samples, label)

plt.xscale('log')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.xlabel('The number of packets in trace')
plt.ylabel('Probability Distribution')
plt.legend()
plt.show()

plt.clf()

# draw packet length cdf
for file, label in dataset_list.items():
    ret = get_packet_count(root_dir + file)
    samples = list(map(lambda x: sum(x), ret))
    draw_cdf(samples, label)

plt.xscale('log')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.xlabel('The number of packets in trace')
plt.ylabel('Probability')
plt.legend()
plt.show()

plt.clf()

# draw byte count cdf
for file, label in dataset_list.items():
    ret = get_byte_count(root_dir + file)
    samples = list(map(lambda x: sum(x), ret))
    draw_line(samples, label)

plt.xscale('log')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.xlabel('The total bytes in trace')
plt.ylabel('Probability Distribution')
plt.legend()
plt.show()

plt.clf()
# draw byte count line
for file, label in dataset_list.items():
    ret = get_byte_count(root_dir + file)
    samples = list(map(lambda x: sum(x), ret))
    draw_cdf(samples, label)

plt.xscale('log')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.xlabel('The total bytes in trace')
plt.ylabel('Probability')
plt.legend()
plt.show()
