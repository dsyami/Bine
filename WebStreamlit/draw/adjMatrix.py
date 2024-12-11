import matplotlib.pyplot as plt

def plot_heatmaps(adj_list, label, index, save_path=None):
    # 设置每个画布上的图片数量（行数和列数）
    rows_per_canvas = 4
    cols_per_canvas = 4

    num_images = adj_list.shape[0]
    num_canvases = (num_images // (rows_per_canvas * cols_per_canvas)) + (1 if num_images % (rows_per_canvas * cols_per_canvas) != 0 else 0)

    figs = []

    for canvas_idx in range(num_canvases):
        fig, axes = plt.subplots(rows_per_canvas if num_images - canvas_idx * (rows_per_canvas * cols_per_canvas) >= rows_per_canvas else (num_images - canvas_idx * (rows_per_canvas * cols_per_canvas)) // cols_per_canvas + (1 if (num_images - canvas_idx * (rows_per_canvas * cols_per_canvas)) % cols_per_canvas != 0 else 0),
                                 cols_per_canvas if num_images - canvas_idx * (rows_per_canvas * cols_per_canvas) >= cols_per_canvas else (num_images - canvas_idx * (rows_per_canvas * cols_per_canvas)) % rows_per_canvas + (1 if (num_images - canvas_idx * (rows_per_canvas * cols_per_canvas)) // rows_per_canvas != 0 else 0),
                                 figsize=(10, 10), squeeze=False)  # 使用squeeze=False确保axes是二维数组
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        start_idx = canvas_idx * (rows_per_canvas * cols_per_canvas)
        end_idx = min((canvas_idx + 1) * (rows_per_canvas * cols_per_canvas), num_images)
        current_idx = 0  # 用于跟踪当前填充的子图索引

        for ax_row in range(axes.shape[0]):
            for ax_col in range(axes.shape[1]):
                ax = axes[ax_row, ax_col]
                if current_idx < end_idx - start_idx:
                    img_idx = start_idx + current_idx
                    cax = ax.imshow(adj_list[img_idx], cmap='viridis', aspect='equal', origin='lower')
                    ax.axis('off')
                    label_dir = {
                        0 : "L",
                        1 : "R",
                        2 : "B",
                        3 : "F"
                    }
                    import numpy as np
                    ax.set_title(f'Label: {label_dir[np.argmax(label[img_idx])]}')
                    # 添加颜色条到右侧
                    fig.colorbar(cax, ax=ax, location='right', pad=0.05)  # pad参数调整颜色条与图的距离
                    # 如果需要，可以在这里为每个子图添加标题
                    current_idx += 1
                else:
                    # 如果当前子图不需要填充，则隐藏它（或者可以选择删除它，但隐藏更简单）
                    fig.delaxes(ax)  # 删除当前轴（子图），防止它显示为空白

        # 更新布局以适应可能已删除的子图
        fig.tight_layout()

        plt.suptitle(f'Adj Matrix for Subject {index + 1}')  # 为当前画布添加标题

        # 调整图形边缘的空白区域，以增加标题与图片之间的空间
        plt.subplots_adjust(top=0.95)  # 通过减小 top 参数的值来增加空间

        if save_path:
            save_file = f"{save_path}_canvas_{canvas_idx+1}.png"
            plt.savefig(save_file)
            print(f"Saved: {save_file}")
        # else:
        #     plt.show()

        figs.append(fig)
    return figs