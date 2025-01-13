import argparse
import random

import cv2
import numpy as np
from utils import *


# 初始化参数：中心点、形状（圆形、椭圆、多边形）、大小，像素值（依据背景与距离自适应）# TODO 
def target_generator(h, w, shape_type, background_std, background_mean, background_diff, target_distance=3000, template_path=None):
    target = np.full((h, w), -1, dtype=np.uint8)
    
    epsilon = 0.8     # 灰体的辐射率
    T = 600           # 灰体的温度 (K)
    wavelength = 10e-6 # 波长 (10 micrometers，即红外波段)
    
    I_r = radiation_wavelength(epsilon, T, wavelength, target_distance)
    
    eta = target_distance + 0.5 # 计算辐射强度折算像素值
    # scale = 0.5
    scale = background_diff * eta

    # 目标初始化，根据形状类型生成目标
    if shape_type == 'circle':
        center = (h // 2, w // 2)
        radius = max(min(h, w) // 4, 1)
        for i in range(h):
            for j in range(w):
                if (i - center[0]) ** 2 + (j - center[1]) ** 2 <= radius ** 2:
                    _temp = np.exp(-((i - center[0]) ** 2 + (j - center[1]) ** 2) / (2 * radius ** 2)) * scale
                    target[i, j] = np.clip(_temp * background_std + background_mean, 0, 255)
        target_info = {'shape_type': 'circle', 'target': target, 'params':target.astype(np.uint8)}

    elif shape_type == 'ellipse':
        center = (h // 2, w // 2)
        axis1 = max(random.randint(h // 4, h // 2), 1)
        axis2 = max(random.randint(w // 4, w // 2), 1)
        for i in range(h):
            for j in range(w):
                if ((i - center[0]) / axis1) ** 2 + ((j - center[1]) / axis2) ** 2 <= 1:
                    _temp = np.exp(-(((i - center[0]) / axis1) ** 2 + ((j - center[1]) / axis2) ** 2)) * scale
                    target[i, j] = np.clip(_temp * background_std + background_mean, 0, 255)
        target_info = {'shape_type': 'ellipse', 'target': target, 'params':target.astype(np.uint8)}
        
    elif shape_type == 'polygon':
        num_points = random.randint(3, 10)
        points = np.array([[random.randint(0, h), random.randint(0, w)] for _ in range(num_points)], np.int32)
        points = points.reshape((-1, 1, 2))
        _temp = np.random.uniform(0.8, 1.2) * scale
        color = np.clip(_temp * background_std + background_mean, 0, 255)
        cv2.fillPoly(target, [points], color)
        target_info = {'shape_type': 'polygon', 'target': target, 'params':[[points], color]}
        
    return target_info  # 返回目标的形状和相关参数

def motion_params_init(init_target_info, max_velocity, min_init_velocity, max_init_velocity, mode_slice, min_acceleration_factor, max_acceleration_factor):

    # 获取每个目标的初始化参数
    target_positions = init_target_info['target_positions']               # N,2 (x,y)
    init_sizes = init_target_info['init_sizes']                           # N,2 (h,w)
    init_shapes = init_target_info['init_shapes']                         # N ('circle', 'ellipse', 'polygon')
    target_distances = init_target_info['target_distances']               # N (z)    
    init_pixel_values = init_target_info['init_pixel_values']             # N个(h,w) 辐射强度（反应为像素值）与距离也有关系
    target_nums = init_target_info['target_nums']
 
    # 设计每个目标的N段运动模式, modes relay_time
    motion_modes = generate_arrays(target_nums, mode_slice)
       
    # 设计每个目标的运动方向参数
    motion_direction = np.random.choice([-1, 1], size=(target_nums, 3))   # N, (x,y,z)
    motion_direction_coef = generate_multiple_sets_of_random_numbers(target_nums, 3)
    direction_coef = motion_direction * motion_direction_coef

    # 运动速度、加速度参数设计
    acceleration_factor = np.random.uniform(min_acceleration_factor, max_acceleration_factor)
    init_velocity = np.random.randint(min_init_velocity, max_init_velocity, size=target_nums)     # N
    
    # TODO 设计合理的加速度和变化率
    a0 = np.random.rand()  # 初始加速度
    k = np.random.rand()   # 加速度变化率
    acceleration_params = (init_velocity, acceleration_factor, a0, k)
    # TODO 自旋运动模式设计
    spin_params = None
    
    # TODO 仿射变换形状变换（目标旋转）、目标辐射强度(像素值)（基于红外辐射模型）、目标
    
    # TODO 微弱随机扰动
    
    return motion_modes, direction_coef, acceleration_params, spin_params

def get_displacement(motion_mode, init_velocity, max_velocity, direction_coef, acceleration_factor, a0, k, t):
    
    x_coef, y_coef, z_coef = direction_coef
    
    if motion_mode == 'uniform':  # 匀速
        velocity = init_velocity  # 速度是恒定的
        displacement = velocity * t  # 位移随时间线性增加
        
    elif motion_mode == 'uniform_acceleration':  # 匀加速
        velocity = acceleration_factor * t  # 速度随时间增加
        displacement = 0.5 * acceleration_factor * t**2  # 位移为加速运动

    elif motion_mode == 'variable_acceleration':  # 变加速
        acceleration = a0 + k * t  # 加速度随时间变化
        velocity = a0 * t + 0.5 * k * t**2  # 速度随时间变化（从初始加速度开始增加）
        displacement = a0 * t**2 + (1/6) * k * t**3  # 位移为变加速运动的积分形式
        
    elif motion_mode == 'random_motion':                # 随机运动
        random_acceleration = np.random.uniform(-1, 1)  # 每步的随机加速度变化
        velocity += random_acceleration                 # 根据加速度和时间步长更新速度
        displacement += velocity                        # 根据速度和时间步长更新位移
        
    else:
        raise
    
    displacement_x = displacement * x_coef
    displacement_y = displacement * y_coef
    displacement_z = displacement * z_coef
    
    return displacement_x, displacement_y, displacement_z

def get_spin_info(angle):
    w, h = target.shape[0], target.shape[1]
    # 2D仿射变换
    M_rotate = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    target = cv2.warpAffine(target, M_rotate, (w, h))
    # TODO 3D建模+旋转+投影
    return target

def get_targets_pixel_values():
    
    # 目标占据像素四舍五入
    
    # 像素值变化依据距离计算红外辐射强度进行换算
    # 旋转变换（2D仿射变换，3D旋转计算+投影计算）

    # 累积缩放因子
    zoom_accum += zoom_factor  # 累加缩放因子

    # 当累积的缩放因子达到1时，进行实际的尺寸变化
    # zoom_factor_actual = 1 + zoom_accum
    # _resize_w, _resize_h = int(w * zoom_factor_actual), int(h * zoom_factor_actual)
    # if (_resize_w != w) or (_resize_h != h):
    #     target = cv2.resize(target, (int(w * zoom_factor_actual), int(h * zoom_factor_actual)))
    
    return
  
def new_target_gen(): # 生成新目标
    return

def handle_critical_target(H, W):
    # 处理目标消失情况
    # 1.删除目标
    # 2.更改目标运动方向
    # 3.重新初始化目标位置
    if x < 0 or x >= (W):
        motion_x *= -1  # 如果接近边界，改变方向
        x = np.clip(x, 0, W - 1)
    if y < 0 or y >= (H):
        motion_y *= -1  # 如果接近边界，改变方向
        y = np.clip(y, 0, H - 1)
    return

def motion_target_update(init_target_info, target_nums, img_nums, t):
    
    for target_num in range(target_nums):
    # 模拟每一个目标的运动更新   
        for t in range(img_nums):
            # 模拟每一帧的运动更新
            # 运动模式和方向决定了每一帧的计算

            # 位移计算
            displacement = get_displacement()
            # 更新位置
            position += np.array([displacement[0], displacement[1], displacement[2]])
            # 自旋计算
            spin_info = get_spin_info()
            
            handle_critical_target()
            new_target_gen()

            # 更新像素值
            new_pixel_value = get_targets_pixel_values(spin_info)
            
            # 保存每一帧的运动更新参数
            move_values = {'position': position.copy(),
                           'size': new_size,
                           'rotation_angle': rotation_angle,
                           'pixel_value': new_pixel_value
                           }
            targets_move_values.append(move_values)
            
    return move_values

# 逐个目标基于前一帧逐帧计算，传递目标前一帧信息和运动模式
def MISTG(input_images, max_num_targets):
    """
    Moving Infrared Small Target Generate and Background Alignment.
    """
    target_nums = max_num_targets
    img_nums, img_h, img_w = input_images.shape
    
    annotations = []            # 标签
    output_images = input_images.copy()
    
    ##################################  params init  ######################################
    focal_length = np.array([25, 50, 100, 200, 300]) * 1e-3              # 红外相机焦距 m
    F = focal_length[1]                                                  # 选择使用红外相机焦距 m
    pixel_size = 15 * 1e-6                                               # 像元尺寸 m
    
    min_target_size = 1                                                  # 初始化目标尺寸最小值 m
    max_target_size = 5                                                  # 初始化目标尺寸最大值 m
    min_init_distance = 3000                                             # 最小目标初始化距离 m
    max_init_distance = 5000                                             # 最大目标初始化距离 m   
    margin = 10                                                          # 初始化目标距离图像边框最小距离限制     
    max_velocity = 700                                                   # 目标最大速度 m/s
    min_init_velocity = 200                                              # 初始化目标最小速度 m/s
    max_init_velocity = 500                                              # 初始化目标最大速度 m/s

    time2maxspeed = 2                                                    # 加速到最大速度所需时间 s
    min_acceleration_factor = -100                                       # 最小加速度，考虑匀减速运动 m/s^2
    max_acceleration_factor = max_velocity // time2maxspeed              # 最大加速度 m/s^2    
    rotation_factor = 5                                                  # 初始化最大目标自旋转角度 
    
    fps = 100                                                            # 视频帧率
    mode_slice = 4                                                       # 运动模式允许的最大段数

    # Initialize target positions and attributes
    target_positions = np.random.randint(margin, [img_w-margin, img_h-margin], size=(target_nums, 2))
    target_distances = np.random.randint(min_init_distance, max_init_distance, size=target_nums)
    target_real_size = np.random.randint(min_target_size, max_target_size, size=target_nums)          
    # target_init_velocity = np.random.randint(min_init_velocity, max_init_velocity, size=target_nums) # m/s
    pixel_scale = (target_distances * pixel_size) / F       # 1px distance, 50mm: 5000->1.5m 3000->0.9m
    target_pixel = np.round(target_real_size / pixel_scale)
    print(target_pixel)

    # pixel_displacement = (target_init_velocity / fps) / pixel_scale
    
    ##################################  target init  ######################################
    target_shapes = ['circle', 'ellipse', 'polygon']                                        
    target_shape_ids = np.random.randint(0, len(target_shapes), size=target_nums)        # 初始化目标形状
    targets_info = [] #  帧/目标数/target id: {位置、大小、形状、像素}
    # First frame: Initialize targets
    init_targets_info = []   
    for j in range(target_nums):
        shape_type = target_shapes[target_shape_ids[j]]
        h = w = target_pixel[j] # TODO 修改
        target_info = target_generator(h, w, shape_type, background_std, background_mean, background_diff, target_distances[j])
        target = target_info['target']
        init_targets_info.append(target_info)
        x, y = target_positions[j]
            
        # Generate targets and add them to the frame
        output_images[i, max(0, y):min(img_h, y+h), max(0, x):min(img_w, x+w)] = target
        
    targets_info.append({f'{i}':init_targets_info})    
    # 计算出与初始化目标的位移、旋转角度，再根据目标初始化方式修改输入参数生成目标即可
    targets_motion_params = motion_params_init(init_targets_info, max_velocity, min_init_velocity, max_init_velocity, mode_slice, min_acceleration_factor, max_acceleration_factor)

    # Track previous features for optical flow calculation
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    prev_gray = None
    prev_features = None
    
    
    
    for i in range(img_nums):
        current_frame = input_images[i]
        current_gray = current_frame.astype(np.uint8)

        # Calculate background brightness statistics
        background_mean = np.mean(current_gray)
        background_std = np.std(current_gray)
        background_diff = (np.max(current_gray) - np.min(current_gray)) / background_std  
                
        if i == 0:  
            # Optical flow background alignment
            prev_gray = current_gray
            prev_features = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

        else:
            # Optical flow background alignment
            if prev_features is not None and len(prev_features) > 0:
                next_features, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, prev_features, None, **lk_params)
                valid_prev = prev_features[status == 1] if status is not None else []
                valid_next = next_features[status == 1] if status is not None else []
                if len(valid_prev) > 0 and len(valid_next) > 0:
                    displacement = np.mean(valid_next - valid_prev, axis=0)
                else:
                    displacement = (0, 0)
                background_displacement = displacement
                prev_gray = current_gray
                prev_features = valid_next.reshape(-1, 1, 2) if len(valid_next) > 0 else None
            else:
                background_displacement = (0, 0)
                prev_gray = current_gray
                prev_features = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
                
            # Subsequent frames: Update targets based on motion
            prev_targets_info = targets_info[f'{i-1}']
            init_targets_info = targets_info['0']
            iframe_targets_info = []
            for j in range(target_nums):
                # 传入参数：初始帧参数，前一帧参数，运动模式
                target_info = motion_target_update(background_displacement, init_targets_info[j], prev_targets_info[j], t=1/fps)
                iframe_targets_info.append(target_info)
                target = target_info['target']
                target_displacement = target_info['displacement']
                x, y = int(target_displacement[0] + background_displacement[0]), int(target_displacement[1] + background_displacement[1])

                mask = target != 0
                output_images[i, max(0, y):min(img_h, y+h), max(0, x):min(img_w, x+w)] = np.where(mask, target, input_images[i, max(0, y):min(img_h, y+h), max(0, x):min(img_w, x+w)])
            
            targets_info.append({f'{i}':iframe_targets_info})
            
    return output_images, annotations


if __name__ == '__main__':
    # 参数配置
    folder_path = "/home/guantp/Infrared/datasets/DUAB/data4/"   # Replace with your folder containing images
    output_folder = "/home/guantp/Infrared/datasets/temp_dataset/"
    max_num_targets = 50
    
    input_images = load_images_from_folder(folder_path)
    output_images, annotations = MISTG(input_images, max_num_targets)
    save_images_and_annotations(output_folder, output_images, annotations)
