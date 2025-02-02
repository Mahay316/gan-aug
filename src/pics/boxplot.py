import matplotlib.pyplot as plt
import numpy as np
from traffic_stat import get_byte_count, get_packet_count

# traffic filename -> traffic type
dataset_list = {
    'pt_tor.txt': 'Tor',
    'pt_normal.txt': 'Non-Tor',
    # 'pt_obfs4.txt': 'Obfs4',
    # 'pt_meek.txt': 'Meek',
    # 'pt_fte.txt': 'FTE',
    # 'pt_webtunnel.txt': 'WebTunnel'
}

# root_dir = 'D:/BaiduNetdiskDownload/Tor Traffic/'
#
# sample_list = []
# for file, label in dataset_list.items():
#     ret = get_packet_count(root_dir + file)
#     samples = list(map(lambda x: sum(x), ret))
#     sample_list.append(samples)
# plt.boxplot(sample_list)

plt.boxplot([[1, 2, 3, 5,7, 333, 333, 333], [2, 2, 2, 2, 2, 5]])

# plt.xticks(list(range(len(dataset_list))), list(dataset_list.values()))
plt.xlabel('Obfuscator')
plt.ylabel('Flow Length (Num of Packets)')

# Display the plot
plt.show()
