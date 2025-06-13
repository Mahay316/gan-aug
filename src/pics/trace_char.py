import matplotlib.pyplot as plt
import numpy as np

from traffic_stat import get_packet_count, get_byte_count

# traffic filename -> traffic type
dataset_list = ['nontor', 'tor', 'obfs4', 'webtunnel', 'snowflake', 'dnstt', 'shadowsocks']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

def draw_cdf(samples, label):
    data_sorted = sorted(samples)
    cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    plt.plot(data_sorted, cdf, label=label, marker='none', linestyle='-')


def draw_line(sample, label):
    values, counts = np.unique(sample, return_counts=True)
    total = float(np.sum(counts).item())
    counts = counts / total
    plt.plot(values, counts, label=label, marker='none', linestyle='-')


seg_method = 'interval'
root_dir = f'D:/traffic_dataset/{seg_method}/'


def get_filename(label):
    return f'pt_{label}_{seg_method}.txt'


# draw packet count line
for label in dataset_list:
    ret = get_packet_count(root_dir + get_filename(label))
    samples = list(map(lambda x: sum(x), ret))
    draw_line(samples, label)

plt.xscale('log')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.xlabel('The number of packets in trace')
plt.ylabel('Probability Distribution')
plt.legend()
plt.tight_layout()
plt.show()

plt.clf()

# draw packet length cdf
for label in dataset_list:
    ret = get_packet_count(root_dir + get_filename(label))
    samples = list(map(lambda x: sum(x), ret))
    draw_cdf(samples, label)

plt.xscale('log')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.xlabel('The number of packets in trace')
plt.ylabel('Probability')
plt.legend()
plt.show()
plt.tight_layout()
plt.clf()

# draw byte count line
for label in dataset_list:
    ret = get_byte_count(root_dir + get_filename(label))
    samples = list(map(lambda x: sum(x), ret))
    draw_line(samples, label)

plt.xscale('log')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.xlabel('The total bytes in trace')
plt.ylabel('Probability Distribution')
plt.legend()
plt.tight_layout()
plt.show()

plt.clf()
# draw byte count cdf
for label in dataset_list:
    ret = get_byte_count(root_dir + get_filename(label))
    samples = list(map(lambda x: sum(x), ret))
    draw_cdf(samples, label)

plt.xscale('log')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.xlabel('The total bytes in trace')
plt.ylabel('Probability')
plt.legend()
plt.tight_layout()
plt.show()
