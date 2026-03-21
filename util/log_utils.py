import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def log_ecc(file_path, length, total_size, parity_len, backup_count, parity_acc, indexs):
    with open(file_path, 'a') as f:
        if isinstance(indexs, torch.Tensor):
            if indexs.is_cuda:
                indexs = indexs.cpu()
            # 将张量转换为NumPy数组
            indexs = indexs.numpy()
        f.write(f'\ntotal_size: {total_size}\n')
        f.write(f'水印信息长度 : {length}\n')
        f.write(f'冗余位数 : {parity_len}\n')
        f.write(f'冗余备份数: {backup_count}\n')
        f.write(f'冗余最终还原率 : {parity_acc*100:.2f}%\n')
        f.write(f'冗余错误index: {indexs}\n')




def plot_error_density(file_path, error_proportion, secret_length):
    # 合并所有比例并转成百分比
    error_percentages = np.concatenate(error_proportion) * 100

    # 定义区间：0~10,10~20,...,90~100
    bins = np.arange(0, 110, 10)  # 边界 [0,10,20,...,100]
    
    # 计算每个区间的样本数
    hist, bin_edges = np.histogram(error_percentages, bins=bins)
    hist_prop = hist / hist.sum()  # 频率

    # 计算区间中心，用于绘图
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(9, 6))
    bar_width = 6  # 柱宽为 8，区间宽 10，留出间隔

    # —— 在这里修改柱子的颜色 ——  
    # 例如 color='skyblue'，或 color='#FF6666'，甚至传入一个列表指定每个柱子的颜色
    plt.bar(
        bin_centers,
        hist_prop,
        width=bar_width,
        align='center',
        color="#E2DAAA",      # ← 这里设置柱状图颜色
        edgecolor='white'     # ← 可选：设置柱子边框颜色以加强间隔感
    )

    # 标题与坐标轴
    plt.title("Bit-flipping Distribution")
    plt.xlabel("Error Position Distribution in Latent (%)")
    plt.ylabel("Frequency (Proportion)")

    # 设置刻度标签为“0–10%”形式
    labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}%" for i in range(len(bin_edges)-1)]
    plt.xticks(bin_centers, labels)

    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.show()