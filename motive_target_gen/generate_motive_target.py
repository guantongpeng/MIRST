import argparse
import random

import cv2
import numpy as np
from utils import *


def circle_gen(h, w, target, scale, background_std, background_mean):
    
    center = (h // 2, w // 2)
    radius = max(min(h, w) // 4, 1)
    
    for i in range(h):
        for j in range(w):
            if (i - center[0]) ** 2 + (j - center[1]) ** 2 <= radius ** 2:
                _temp = np.exp(-((i - center[0]) ** 2 + (j - center[1]) ** 2) / (2 * radius ** 2)) * scale
                target[i, j] = np.clip(_temp * background_std + background_mean, 0, 255)
                
    return 

def ellipse_gen(h, w, target, scale, background_std, background_mean, axis=None):
    
    center = (h // 2, w // 2)
    
    if axis:
        axis1, axis2 = axis
    else:
        axis1 = max(random.randint(h // 4, h // 2), 1)
        axis2 = max(random.randint(w // 4, w // 2), 1)
        
    for i in range(h):
        for j in range(w):
            if ((i - center[0]) / axis1) ** 2 + ((j - center[1]) / axis2) ** 2 <= 1:
                _temp = np.exp(-(((i - center[0]) / axis1) ** 2 + ((j - center[1]) / axis2) ** 2)) * scale
                target[i, j] = np.clip(_temp * background_std + background_mean, 0, 255)  
                     
    return axis1, axis2

def polygon_gen(h, w, target, scale, background_std, background_mean, points=None):
    
    if points:
        points = points
    else:
        num_points = random.randint(3, 10)
        points = np.array([[random.randint(0, h), random.randint(0, w)] for _ in range(num_points)], np.int32)
        points = points.reshape((-1, 1, 2))
        
    _temp = np.random.uniform(0.8, 1) * scale
    color = np.clip(_temp * background_std + background_mean, 0, 255)
    cv2.fillPoly(target, points, color) 
    
    return points
       
# 初始化参数：中心点、形状（圆形、椭圆、多边形）、大小，像素值
def target_generator(target, shape_type, background_std, background_mean, background_diff, target_distance=3000, base_distance = 3000):
   
    h, w = target.shape
    eta = np.clip(np.log(np.log(base_distance**2 / target_distance**2 + 1) + 1), 0, 1) # TODO 计算辐射强度折算像素值，待优化
    scale = background_diff * eta # 基准值原来设计的是scale = 0.5

    # 目标初始化，根据形状类型生成目标
    if shape_type == 'circle':
        circle_gen(h, w, target, scale, background_std, background_mean)
        target_info = {'shape_type': 'circle', 'target': target, 'params':[h, w]}

    elif shape_type == 'ellipse':
        axis = ellipse_gen(h, w, target, scale, background_std, background_mean)
        target_info = {'shape_type': 'ellipse', 'target': target, 'params':[h, w, axis]}
        
    elif shape_type == 'polygon':
        points = polygon_gen(h, w, target, scale, background_std, background_mean)
        target_info = {'shape_type': 'polygon', 'target': target, 'params':points}

    return target_info  # 返回目标的形状和相关参数

def motion_params_init(panoramic_target_nums, max_velocity, min_init_velocity, max_init_velocity, mode_slice, min_acceleration_factor, max_acceleration_factor):
     
    # 设计每个目标的N段运动模式, 返回：modes relay_time
    motion_modes = generate_arrays(panoramic_target_nums, mode_slice)
       
    # 设计每个目标的运动方向参数
    motion_direction = np.random.choice([-1, 1], size=(panoramic_target_nums, 3))   # N, (x,y,z)
    motion_direction_coef = generate_multiple_sets_of_random_numbers(panoramic_target_nums, 3)
    direction_coef = motion_direction * motion_direction_coef

    # 运动速度、加速度参数设计
    target_init_velocity = np.random.randint(min_init_velocity, max_init_velocity, size=panoramic_target_nums) # m/s
    acceleration_factor = np.random.uniform(min_acceleration_factor, max_acceleration_factor, size=panoramic_target_nums)
    
    # TODO 设计合理的加速度和变化率
    a0 = np.random.rand()                                                            # 初始加速度
    k = np.random.rand()                                                             # 加速度变化率
    acceleration_params = (target_init_velocity, acceleration_factor, a0, k)
    # TODO 自旋运动模式设计
    spin_params = None
    
    # TODO 仿射变换形状变换（目标自旋转）
    
    # TODO 微弱随机扰动
        
    return motion_modes, direction_coef, acceleration_params, spin_params

# 根据不同运动模式计算目标位移
def get_displacement(motion_mode, init_velocity, max_velocity, direction_coef, acceleration_factor, a0, k, t, fps):
    x_coef, y_coef, z_coef = direction_coef

    if motion_mode == 0:  # 'uniform' 匀速
        velocity = min(init_velocity, max_velocity)  # 确保速度不超过最大速度
        displacement = velocity * t  # 位移随时间线性增加

    elif motion_mode == 1:  # 'uniform_acceleration' 匀加速
        velocity = init_velocity + acceleration_factor * t  # 速度随时间增加
        velocity = min(velocity, max_velocity)  # 确保速度不超过最大速度
        displacement = init_velocity * t + 0.5 * acceleration_factor * t**2  # 位移考虑初始速度

    elif motion_mode == 2:  # 'variable_acceleration' 变加速
        acceleration = a0 + k * t  # 加速度随时间变化
        velocity = init_velocity + a0 * t + 0.5 * k * t**2  # 速度随时间变化，考虑初始速度
        velocity = min(velocity, max_velocity)  # 确保速度不超过最大速度
        displacement = init_velocity * t + 0.5 * a0 * t**2 + (1/6) * k * t**3  # 位移考虑初始速度和变加速

    elif motion_mode == 3:  # 'random_motion' 随机运动
        # 初始化速度和位移
        velocity = init_velocity
        displacement = 0

        # 动态时间步长
        time_step = 1 / fps  # 使用帧率作为时间步长
        num_steps = int(t * fps)  # 计算时间步长内的步数

        # 随机生成加速度变化
        random_accelerations = np.random.uniform(-1, 1, num_steps)
        velocities = np.cumsum(random_accelerations) + velocity
        velocities = np.clip(velocities, 0, max_velocity)  # 确保速度不超过最大速度
        displacements = np.cumsum(velocities) * time_step
        displacement = displacements[-1]

    else:
        raise ValueError("Invalid motion mode")

    displacement_x = displacement * x_coef
    displacement_y = displacement * y_coef
    displacement_z = displacement * z_coef

    return displacement_x, displacement_y, displacement_z

# 计算所有目标的累计位移
def get_cumulative_displacements(motion_modes, time_ratios, total_frames, fps, target_init_velocity, max_velocity, target_direction_coef, target_acceleration_factor, init_acceleration, acceleration_change_rate, mode_slice, target_nums, x_range, y_range, z_range): 
    frame_time = 1 / fps
    all_cumulative_displacements = []

    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    for target_index in range(target_nums):
        target_cumulative_displacements = []
        cumulative_displacement_x = 0
        cumulative_displacement_y = 0
        cumulative_displacement_z = 0

        init_velocity = target_init_velocity[target_index]
        direction_coef = target_direction_coef[target_index]
        acceleration_factor = target_acceleration_factor[target_index]
        a0 = init_acceleration[target_index]
        k = acceleration_change_rate[target_index]

        target_motion_modes, target_time_ratios = motion_modes[target_index], time_ratios[target_index]

        for mode, ratio in zip(target_motion_modes, target_time_ratios):
            num_frames = int(ratio * total_frames)
            for _ in range(num_frames):
                displacement_x, displacement_y, displacement_z = get_displacement(
                    mode, init_velocity, max_velocity, direction_coef, acceleration_factor, a0, k, frame_time, fps)

                cumulative_displacement_x += displacement_x
                cumulative_displacement_y += displacement_y
                cumulative_displacement_z += displacement_z

                # 目标移动边界检查是否超出范围，如果超出则调整方向
                if cumulative_displacement_x < x_min or cumulative_displacement_x > x_max:
                    direction_coef = (-direction_coef[0], direction_coef[1], direction_coef[2])
                    cumulative_displacement_x = max(min(cumulative_displacement_x, x_max), x_min)
                if cumulative_displacement_y < y_min or cumulative_displacement_y > y_max:
                    direction_coef = (direction_coef[0], -direction_coef[1], direction_coef[2])
                    cumulative_displacement_y = max(min(cumulative_displacement_y, y_max), y_min)
                if cumulative_displacement_z < z_min or cumulative_displacement_z > z_max:
                    direction_coef = (direction_coef[0], direction_coef[1], -direction_coef[2])
                    cumulative_displacement_z = max(min(cumulative_displacement_z, z_max), z_min)
                target_cumulative_displacements.append((cumulative_displacement_x, cumulative_displacement_y, cumulative_displacement_z))

            if len(target_cumulative_displacements) >= total_frames:
                break

        # 如果生成的帧数少于总帧数，填充剩余帧数
        while len(target_cumulative_displacements) < total_frames:
            target_cumulative_displacements.append(target_cumulative_displacements[-1])

        all_cumulative_displacements.append(target_cumulative_displacements[:total_frames])

    return np.array(all_cumulative_displacements)

def get_cumulative_spin():
    return 

def get_spin_info(angle):
    w, h = target.shape[0], target.shape[1]
    # 2D仿射变换
    M_rotate = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    target = cv2.warpAffine(target, M_rotate, (w, h))
    # TODO 3D建模+旋转+投影
    return target

  
def new_target_gen():
    # 新动态目标生成
    return 

def handle_critical_target():
    # 处理目标消失情况
    # 1.删除目标
    # 2.更改目标运动方向
    # 3.重新初始化目标位置
    return 


# 逐个目标基于前一帧逐帧计算，传递目标前一帧信息和运动模式
def MISTG(input_images, max_num_targets):
    
    """
    Moving Infrared Small Target Generate and Background Alignment.
    """
    
    init_target_nums = max_num_targets
    img_nums, img_h, img_w = input_images.shape
    print("Image Size: ", img_h, img_w)                    # 512 640
    
    annotations = []            # 标签
    output_images = input_images.copy()


    ###########################################  background displacement  ##########################################
    # Track previous features for optical flow calculation
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    prev_gray = None
    prev_features = None
    
    # background_displacement_list = []
    relative_displacement_list = []
    total_displacement = np.array([0.0, 0.0])
    
    for i in range(img_nums):
        current_frame = input_images[i]
        current_gray = current_frame.astype(np.uint8)
                
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
                    displacement = np.mean(valid_prev - valid_next, axis=0)
                else:
                    displacement = (0, 0)
                background_displacement = displacement
                prev_gray = current_gray
                prev_features = valid_next.reshape(-1, 1, 2) if len(valid_next) > 0 else None
            else:
                background_displacement = (0, 0)
                prev_gray = current_gray
                prev_features = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
            
            # background_displacement_list.append(background_displacement)
            # 累计背景位移
            total_displacement = background_displacement + total_displacement
            relative_displacement_list.append(total_displacement)   
            
    # background_displacement_array = np.array(background_displacement_list)
    relative_displacement_array = np.array(relative_displacement_list)

    y_displacement_range_min = np.min(relative_displacement_array[:,0])
    y_displacement_range_max = np.max(relative_displacement_array[:,0])
    x_displacement_range_min = np.min(relative_displacement_array[:,1])
    x_displacement_range_max = np.max(relative_displacement_array[:,1])
    print("Background Displacement Range, ", "X Range: ", [x_displacement_range_min, x_displacement_range_max], ", Y Range: ", [y_displacement_range_min, y_displacement_range_max])
    print("Final Background Displacement: ", relative_displacement_array[-1])
    ##############################################################################################################
    
                
    ###########################################  params init  ##########################################
    focal_length = np.array([25, 50, 100, 200, 300]) * 1e-3              # 红外相机焦距 m
    F = focal_length[1]                                                  # 选择使用红外相机焦距 m
    pixel_size = 15 * 1e-6                                               # 像元尺寸 m
    
    min_target_size = 1                                                  # 初始化目标尺寸最小值 m
    max_target_size = 5                                                  # 初始化目标尺寸最大值 m
    min_init_distance = 3000                                             # 最小目标初始化距离 m
    max_init_distance = 5000                                             # 最大目标初始化距离 m   
    base_distance = 3000                                                 # 目标基准距离，根据此距离修正目标辐射强度以修正像素值
    margin = 10                                                          # 初始化目标距离图像边框最小距离限制     
    max_velocity = 700                                                   # 目标最大速度 m/s
    min_init_velocity = 200                                              # 初始化目标最小速度 m/s
    max_init_velocity = 500                                              # 初始化目标最大速度 m/s
    
    # TODO 修改参数
    acceleration_factor_range = 400  # 加速度因子 (m/s^2)
    init_acceleration = 100  # 初始加速度 (m/s^2)
    acceleration_change_rate_range = 50  # 加速度变化率 (m/s^3)

    time2maxspeed = 2                                                    # 加速到最大速度所需时间 s
    min_acceleration_factor = -100                                       # 最小加速度，考虑匀减速运动 m/s^2
    max_acceleration_factor = max_velocity // time2maxspeed              # 最大加速度 m/s^2    
    rotation_factor = 5                                                  # 初始化最大目标自旋转角度 
    
    fps = 100                                                            # 视频帧率
    mode_slice = 4                                                       # 运动模式允许的最大段数
    target_density = init_target_nums / (img_h * img_w)                  # 场景中目标密度
    z_displacement_range_min = 500
    z_displacement_range_max = 10000
    
    x_range = (x_displacement_range_max - x_displacement_range_min) + img_w
    y_range = (y_displacement_range_max - y_displacement_range_min) + img_h
    init_target_nums = int(target_density * (x_range * y_range))
    print("Panoramic Target Number: ", init_target_nums)
    
    # 全场景图像坐标范围，以第一帧图像左下角为坐标原点
    x_min = np.ceil(x_displacement_range_min + margin)
    y_min = np.ceil(y_displacement_range_min  + margin)
    x_max = np.floor(x_displacement_range_max + img_w - margin)
    y_max = np.floor(y_displacement_range_max + img_h - margin)
    
    target_positions = np.random.randint([x_min, y_min, min_init_distance], [x_max, y_max, max_init_distance], size=(init_target_nums, 3))         # N*3 (x,y,z)
    target_distances = target_positions[::,-1]
    target_real_size = np.random.randint(min_target_size, max_target_size, size=init_target_nums)          
    # target_init_velocity = np.random.randint(min_init_velocity, max_init_velocity, size=init_target_nums) # m/s
    pixel_scale = (target_distances * pixel_size) / F          # 1px distance, 50mm: 5000->1.5m 3000->0.9m
    target_pixel = np.round(target_real_size / pixel_scale)    # 目标占据像素
    # print(target_pixel)
    ####################################################################################################
    
    
    
    ##################################  first frame target init  ######################################
    # pixel_displacement = (target_init_velocity / fps) / pixel_scale
    
    target_shapes = ['circle', 'ellipse', 'polygon']                                        
    target_shape_ids = np.random.randint(0, len(target_shapes), size=init_target_nums)        # 初始化目标形状
    targets_info = [] #  帧/目标数/target id: {位置、大小、形状、像素}
    # First frame: Initialize targets
    init_targets_info = []   
    # Calculate background brightness statistics
    first_frame_img = input_images[0].astype(np.uint8)
    background_mean = np.mean(first_frame_img)
    background_std = np.std(first_frame_img)
    background_diff = (np.max(first_frame_img) - np.min(first_frame_img)) / background_std  
    
    for j in range(init_target_nums):
        shape_type = target_shapes[target_shape_ids[j]]
        h = w = int(target_pixel[j]) # TODO 待优化
        x, y, z = target_positions[j]
        
        if 0 <= x < img_w and 0 <= y < img_h:
            img_target_background = input_images[0, y:y+h, x:x+w]
            target_info = target_generator(img_target_background, shape_type, background_std, background_mean, background_diff, z, base_distance=base_distance)
            target = target_info['target']
            init_targets_info.append(target_info)
           
            # Generate targets and add them to the frame
            output_images[0, max(0, y):min(img_h, y+h), max(0, x):min(img_w, x+w)] = target
        
    targets_info.append({'0':init_targets_info})
    #####################################################################################################
    

    ##########################################  target update  ##########################################  

    # 随机初始化运动速度参数
    targets_init_velocity = np.random.randint(min_init_velocity, max_init_velocity, size=init_target_nums)
    targets_acceleration_factor = np.random.uniform(min_acceleration_factor, max_acceleration_factor, size=init_target_nums)
    targets_init_acceleration = np.random.rand(init_target_nums) * 20
    targets_acceleration_change_rate = np.random.rand(init_target_nums) * 5
    motion_modes, time_ratios = generate_arrays(init_target_nums, mode_slice) # 每个目标的运动模式数量在1到4之间
    
    # 初始化目标运动方向参数
    motion_direction = np.random.choice([-1, 1], size=(init_target_nums, 3))   # N, (x,y,z)
    motion_direction_coef = generate_multiple_sets_of_random_numbers(init_target_nums, 3)
    targets_direction_coef = motion_direction * motion_direction_coef

    # 计算出后续全部帧所有目标与第一帧的位移、旋转变化
    targets_cumulative_displacements = get_cumulative_displacements(motion_modes, time_ratios, img_nums-1, fps, targets_init_velocity, max_velocity, targets_direction_coef, targets_acceleration_factor, targets_init_acceleration, targets_acceleration_change_rate, mode_slice, init_target_nums, [x_displacement_range_min, x_displacement_range_max], [y_displacement_range_min, y_displacement_range_max], [z_displacement_range_min, z_displacement_range_max]) # return shape:[targets_num, img_num, 3]

    targets_cumulative_spin = get_cumulative_spin() # TODO
 
    
       
    # 以第一帧的左下角作为整个场景的坐标原点（0,0），目标动态更新与计算，目标更新依据背景位移多少进行更新    
    for i in range(1, img_nums):
        current_frame = input_images[i]
        current_gray = current_frame.astype(np.uint8)
                
        background_mean = np.mean(current_gray)
        background_std = np.std(current_gray)
        background_diff = (np.max(current_gray) - np.min(current_gray)) / background_std  

        # 基于位移与旋转变化和目标初始化信息修改输入参数生成更新后目标
        update_target_positions = target_positions + targets_cumulative_displacements[:, i-1, :] 
        update_target_distances = update_target_positions[::,-1]
        update_pixel_scale = (update_target_distances * pixel_size) / F                     

        # 更新目标大小
        update_target_pixel = np.round(target_real_size / update_pixel_scale)   
                      
        # 更新像素值
        # 输入参数：初始帧参数，位移变化、旋转变化
        for j in range(init_target_nums):
            shape_type = target_shapes[target_shape_ids[j]]
            h = w = int(update_target_pixel[j]) # TODO 待优化
            x, y, z = target_positions[j]

            x = int(x - relative_displacement_array[i-1, 0])
            y = int(y - relative_displacement_array[i-1, 1])

            # 判断目标是否显示在图像中，只需要计算在现在这一帧图像内的目标，依据目标中心点和大小判定，简化：忽略目标大小，仅依据中心点判定
            if 0 <= x < img_w and 0 <= y < img_h:
                img_target_background = input_images[0, y:y+h, x:x+w]
                target_info = target_generator(img_target_background, shape_type, background_std, background_mean, background_diff, z, base_distance=base_distance)
                target = target_info['target']
                # Generate targets and add them to the frame
                output_images[i, max(0, y):min(img_h, y+h), max(0, x):min(img_w, x+w)] = target 
        
        # mask = target != 0
        # output_images[i, max(0, y):min(img_h, y+h), max(0, x):min(img_w, x+w)] = np.where(mask, target, input_images[i, max(0, y):min(img_h, y+h), max(0, x):min(img_w, x+w)])
            
    return output_images, annotations
    ######################################################################################################


if __name__ == '__main__':
    # 参数配置
    # folder_path = "/home/guantp/Infrared/datasets/mydata/250110/M615/300_2min_1min20s_4"   # Replace with your folder containing images
    folder_path = "/home/guantp/Infrared/MIRST/motive_target_gen/data4"
    output_folder = "/home/guantp/Infrared/MIRST/motive_target_gen/imgs/"
    max_num_targets = 10
    
    input_images = load_images_from_folder(folder_path)
    output_images, annotations = MISTG(input_images, max_num_targets)
    annotations = ["Annotation " + str(i) for i in range(399)]
    save_images_and_annotations(output_folder, output_images, annotations)
    print("Moving Infrared Small Target Generate Finish.")
    
# TODO 未完成部分以及优化
# 目标mask生成
# 使用三维模型 + 投影优化目标形状
# 添加目标自旋转模型
# 像素值根据距离和大气传输模型仿真优化