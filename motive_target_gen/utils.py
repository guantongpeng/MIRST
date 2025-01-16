import os
import random
import re
import shutil

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
    c = 3e8  # 光速 (m/s)
    k = 1.381e-23  # 玻尔兹曼常数 (J/K)

    # 使用普朗克定律计算在特定波长下的辐射强度
    I_lambda = (
        (2 * h * c**2)
        / (wavelength**5)
        * (1 / (np.exp((h * c) / (wavelength * k * T)) - 1))
    )

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
    Load images from a specified folder in alphabetical order of filenames.
    """
    images = []

    # Sort filenames alphabetically
    for filename in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return np.array(images)


def clear_folder(folder_path):
    # 检查文件夹是否为空
    if not os.listdir(folder_path):
        pass
    else:
        # 清空文件夹
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"delete {file_path} error: {e}")


def save_images_and_annotations(output_folder, images, annotations):
    """
    Save generated images and their annotations to the specified folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    clear_folder(output_folder)

    for i, (image, annot) in enumerate(zip(images, annotations)):
        image_filename = os.path.join(output_folder, f"{i:04d}.png")
        annot_filename = os.path.join(output_folder, f"{i:04d}.txt")

        # Save image
        cv2.imwrite(image_filename, image)
        # # Save annotation
        # with open(annot_filename, 'w') as annot_file:
        #     annot_file.write(annot)
        #     print(f"Annotation saved: {annot_filename}")

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
