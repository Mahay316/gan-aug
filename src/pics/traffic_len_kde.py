# use KDE(Kernel Density Estimation) to estimate the distribution of trace length and flow length
import os.path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

from traffic_stat import get_packet_count


def estimate_distro(data, xlabel: str, ylabel: str, bin_width: int = 2):
    max_val = np.max(data)
    min_val = np.min(data)

    # 将数据转换为二维数组 (scikit-learn 需要二维输入)
    data = data[:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth=1)
    kde.fit(data)

    # 计算密度估计值 (KDE 计算的是对数密度，需要取指数)
    x_vals = np.linspace(min_val, max_val, 1000)[:, np.newaxis]
    log_dens = kde.score_samples(x_vals)
    pdf_vals = np.exp(log_dens)

    # 绘制直方图和 KDE 曲线
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=50, density=True, alpha=0.5, label="Histogram")  # 归一化直方图
    plt.plot(x_vals, pdf_vals, label="KDE Estimate (Gaussian)", linewidth=2)  # KDE 曲线

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend()
    plt.grid(linestyle=':')
    plt.show()


if __name__ == '__main__':
    data_dir = r'D:\traffic_dataset\span'
    filename = r'pt_nontor_span.txt'

    packet_cnt = get_packet_count(os.path.join(data_dir, filename))
    flow_len = np.array([item for row in packet_cnt for item in row])
    flow_len = flow_len[flow_len < 1000]

    # target_value = 2
    # flow_len[flow_len == target_value] = np.clip(np.floor(np.abs(np.random.normal(loc=0, scale=150, size=np.sum(flow_len == target_value))) + 1), 1, 400)
    trace_len = np.array([len(flow) for flow in packet_cnt])

    print(len(flow_len))
    # estimate_distro(flow_len, 'Flow Length', 'Probability Density')
    estimate_distro(trace_len, 'Trace Length', 'Probability Density')
