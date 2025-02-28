import numpy as np

def generate_normal_matrix(rows, cols, edge_value):
    """
    生成一个正态分布矩阵，中心值最大，边缘值接近于 edge_value。

    Args:
        rows: 矩阵的行数。
        cols: 矩阵的列数。
        edge_value: 边缘值。

    Returns:
        一个 NumPy 矩阵。
    """

    # 创建一个所有元素都初始化为 edge_value 的矩阵
    matrix = np.full((rows, cols), edge_value)

    # 计算中心点坐标
    center_row = rows // 2
    center_col = cols // 2

    # 计算标准差，使边缘值接近于 edge_value
    std_dev = min(rows, cols) / 6

    # 生成正态分布值（峰值）
    peak_value = 1.0  # 峰值可以根据需要调整

    for i in range(rows):
        for j in range(cols):
            distance_to_center = np.sqrt((i - center_row)**2 + (j - center_col)**2)
            # 将正态分布的值加到 edge_value 上
            matrix[i, j] += (peak_value - edge_value) * np.exp(-(distance_to_center**2) / (2 * std_dev**2))

    return matrix

# 示例：生成一个 5x5 的矩阵，边缘值接近于 0.1
matrix = generate_normal_matrix(5, 5, 0.5)
print(matrix)