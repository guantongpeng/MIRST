import os
import random
import re

import cv2
import numpy as np




# 灰体辐射强度计算函数
def radiation_wavelength(epsilon, T, wavelength, r):
    """
    计算灰体辐射强度在特定波长下，并考虑距离的衰减。
    :param epsilon: 辐射率（无量纲）
    :param T: 温度（K）
    :param wavelength: 波长（m）
    :param r: 距离（m）
    :return: 距离 r 处的辐射强度（W/m²）
    """
    # 常数定义
    h = 6.626e-34  # 普朗克常数 (J·s)
    c = 3e8        # 光速 (m/s)
    k = 1.381e-23  # 玻尔兹曼常数 (J/K)
        
    # 使用普朗克定律计算在特定波长下的辐射强度
    I_lambda = (2 * h * c**2) / (wavelength**5) * (1 / (np.exp((h * c) / (wavelength * k * T)) - 1))
    
    # 应用逆平方定律，计算距离 r 处的辐射强度
    I_r = epsilon * I_lambda / r**2
    
    return I_r


def generate_random_numbers_sum_to_one(n):
    # 生成n个随机正数，这些数的总和为1
    random_numbers = np.random.rand(n)
    return random_numbers / np.sum(random_numbers)


def generate_multiple_sets_of_random_numbers(N, n):
    # 生成N组随机正数，每组包含n个数，这些数的总和为1
    sets = []
    for _ in range(N):
        sets.append(generate_random_numbers_sum_to_one(n))
    return np.array(sets)


def generate_arrays(M, N):
    arrays = []
    corresponding_sums = []

    # 生成M个数组
    for _ in range(M):
        length = random.randint(1, N)  # 数组的长度在1到N之间
        array = [
            random.choice([0, 1, 2, 3]) for _ in range(length)
        ]  # 随机生成0,1,2,3中的整数
        arrays.append(array)

        # 生成与该数组长度完全对应的和为1的数组
        sum_array = generate_random_numbers_sum_to_one(length)
        corresponding_sums.append(sum_array.tolist())

    return arrays, corresponding_sums


def load_images_from_folder(folder):
    """
    Load images from a specified folder.
    """
    images = []

    for filename in sorted(
        os.listdir(folder), key=lambda x: int(re.search(r"\d+", x).group())
    ):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return np.array(images)


def save_images_and_annotations(output_folder, images, annotations):
    """
    Save generated images and their annotations to the specified folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, (image, annot) in enumerate(zip(images, annotations)):
        image_filename = os.path.join(output_folder, f"{i:04d}.png")
        annot_filename = os.path.join(output_folder, f"{i:04d}.txt")

        # Save image
        cv2.imwrite(image_filename, image)

        # Save annotations
        # with open(annot_filename, 'w') as f:
        #     for bbox in annot:
        #         f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

    # elif shape_type == 'rectangle':
    #     # Rectangle shape
    #     start_row = random.randint(0, h // 4)
    #     start_col = random.randint(0, w // 4)
    #     end_row = random.randint(h // 2, h - 1)
    #     end_col = random.randint(w // 2, w - 1)
    #     _temp = np.random.uniform(0.8, 1.2) * scale
    #     target[start_row:end_row, start_col:end_col] = 255 - np.clip(_temp * eta + theta, 0, 255)
    #     print(np.clip(_temp * eta + theta, 0, 255))

    # elif shape_type == 'template' and template_path:
    #     # TODO
    #     # Load predefined template
    #     template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    #     # Resize template to fit within the image size
    #     scale_factor = random.uniform(0.5, 1.5)
    #     new_size = (int(template.shape[1] * scale_factor), int(template.shape[0] * scale_factor))
    #     resized_template = cv2.resize(template, new_size, interpolation=cv2.INTER_LINEAR)

    #     # Rotate template randomly
    #     M = cv2.getRotationMatrix2D((resized_template.shape[1] // 2, resized_template.shape[0] // 2), angle, 1)
    #     rotated_template = cv2.warpAffine(resized_template, M, (resized_template.shape[1], resized_template.shape[0]))

    #     # Place the rotated template randomly within the target image
    #     x_offset = random.randint(0, w - rotated_template.shape[1])
    #     y_offset = random.randint(0, h - rotated_template.shape[0])
    #     for y in range(rotated_template.shape[0]):
    #         for x in range(rotated_template.shape[1]):
    #             if rotated_template[y, x] > 128:  # Threshold to keep only the shape
    #                 target[y + y_offset, x + x_offset] = random.uniform(100, 255)


# def DISTG_with_background_alignment(I, max_num_targets, target_shape_range, bbox_size, size_variation_factor=0.05, rotation_factor=5):
#     """
#     Modified DISTG with consistent target shapes, dynamic size and rotation adjustments, boundary bounce logic,
#     and background-based brightness control (η and θ).
#     """
#     # N = random.randint(1, max_num_targets)  # Number of targets
#     N = max_num_targets
#     L, H, W = I.shape
#     annotations = []

#     # Initialize target positions and movements
#     target_positions = np.random.randint(0, [W, H], size=(N, 2))
#     target_motions = np.stack([np.cos(np.random.uniform(-np.pi, np.pi, N)), np.sin(np.random.uniform(0, 2 * np.pi, N))], axis=-1)

#     # Track previous features for optical flow calculation
#     lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#     feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
#     prev_gray = None
#     prev_features = None

#     # For consistent target shapes

#     for i in range(L):

#         target_shapes = ['circle', 'ellipse', 'polygon']
#         target_shape_ids = np.random.randint(0, len(target_shapes), size=N)
#         print(target_shape_ids)

#         frame_annotations = []
#         current_frame = I[i]
#         current_gray = current_frame.astype(np.uint8)

#         # Calculate background brightness statistics
#         theta = np.mean(current_gray)
#         eta = np.std(current_gray)
#         bg_diff = (np.max(current_gray) - np.min(current_gray)) / eta

#         # Optical flow background alignment
#         if i == 0:
#             background_displacement = (0, 0)
#             prev_gray = current_gray
#             prev_features = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
#         else:
#             if prev_features is not None and len(prev_features) > 0:
#                 next_features, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, prev_features, None, **lk_params)
#                 valid_prev = prev_features[status == 1] if status is not None else []
#                 valid_next = next_features[status == 1] if status is not None else []
#                 if len(valid_prev) > 0 and len(valid_next) > 0:
#                     displacement = np.mean(valid_next - valid_prev, axis=0)
#                 else:
#                     displacement = (0, 0)
#                 background_displacement = displacement
#                 prev_gray = current_gray
#                 prev_features = valid_next.reshape(-1, 1, 2) if len(valid_next) > 0 else None
#             else:
#                 background_displacement = (0, 0)
#                 prev_gray = current_gray
#                 prev_features = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

#         # Generate targets and add them to the frame
#         for j in range(N):
#             # Update target position based on background displacement and motion
#             x, y = target_positions[j]
#             motion_x, motion_y = target_motions[j]
#             x += background_displacement[0] + motion_x + np.random.uniform(-0.2, 0.2)
#             y += background_displacement[1] + motion_y + np.random.uniform(-0.2, 0.2)

#             # Boundary bounce logic
#             if x < 0 or x >= W:
#                 motion_x *= -1  # Reverse horizontal direction
#                 x = np.clip(x, 0, W - 1)
#             if y < 0 or y >= H:
#                 motion_y *= -1  # Reverse vertical direction
#                 y = np.clip(y, 0, H - 1)
#             target_positions[j] = [x, y]
#             target_motions[j] = [motion_x, motion_y]

#             # Generate target shape and size
#             shape_type = target_shapes[target_shape_ids[j]]
#             size_factor = 1 + random.uniform(-size_variation_factor, size_variation_factor)
#             angle = random.uniform(-rotation_factor, rotation_factor)
#             h = random.randint(*target_shape_range[0])
#             w = random.randint(*target_shape_range[1])

#             Tj = generate_random_target(h, w, shape_type, size_factor, angle, eta, theta, bg_diff)

#             # Calculate target bounding box
#             x_start, y_start = int(x - h // 2), int(y - w // 2)
#             x_end, y_end = x_start + h, y_start + w
#             target_height = min(H, y_end) - max(0, y_start)
#             target_width = min(W, x_end) - max(0, x_start)

#             if Tj.shape[0] != target_height or Tj.shape[1] != target_width:
#                 Tj_resized = cv2.resize(Tj, (target_width, target_height))
#             else:
#                 Tj_resized = Tj

#             I[i, max(0, y_start):min(H, y_end), max(0, x_start):min(W, x_end)] += Tj_resized.astype(I.dtype)

#             # Record the bounding box
#             bbox = [x_start, y_start, x_end, y_end]
#             frame_annotations.append(bbox)

#         annotations.append(frame_annotations)

#     return I, annotations


def _target_motion_transform(
    target,
    shape_type,
    scale,
    h,
    w,
    position,
    motion,
    angle,
    zoom_factor,
    margin,
    W,
    H,
    zoom_accum=0.0,
):
    """
    Modify the target based on motion, rotation, scaling with accumulated zoom factor.
    """
    x, y = position
    motion_x, motion_y = motion

    # 处理边界情况
    if x < margin or x >= (W - margin):
        motion_x *= -1  # 如果接近边界，改变方向
        x = np.clip(x, margin, W - margin - 1)
    if y < margin or y >= (H - margin):
        motion_y *= -1  # 如果接近边界，改变方向
        y = np.clip(y, margin, H - margin - 1)

    # 旋转
    if angle != 0:
        M_rotate = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        target = cv2.warpAffine(target, M_rotate, (w, h))

    # 累积缩放因子
    zoom_accum += zoom_factor  # 累加缩放因子

    # 当累积的缩放因子达到1时，进行实际的尺寸变化
    # zoom_factor_actual = 1 + zoom_accum
    # _resize_w, _resize_h = int(w * zoom_factor_actual), int(h * zoom_factor_actual)
    # if (_resize_w != w) or (_resize_h != h):
    #     target = cv2.resize(target, (int(w * zoom_factor_actual), int(h * zoom_factor_actual)))

    # 更新目标的位置
    new_position = (x + motion_x, y + motion_y)

    return target, new_position, (motion_x, motion_y), zoom_accum  # {目标信息}
