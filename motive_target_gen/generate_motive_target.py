import os
import shutil
import random
import trimesh
import cv2
import numpy as np

from motive_mode import (
    get_cumulative_displacements,
    update_rotation,
    generate_3d_targets_rotation_sequences
)

from projection_3d import (
    normalize_mesh, 
    get_projection, 
    calculate_target_angles, 
    generate_ir_target_intensity
)

from utils import (
    generate_multiple_sets_of_random_numbers,
    generate_arrays,
    load_images_from_folder
)

def circle_gen(h, w, target, scale, background_std, background_mean): #TODO 待优化
    """
    生成一个连续的不规则类圆形红外目标
    参数:
    h, w: 目标区域高度和宽度
    target: 背景图像
    scale: 目标强度缩放因子
    background_std: 背景标准差
    background_mean: 背景均值
    """
    center = (w // 2, h // 2)
    radius = max(min(h, w) // 4, 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 使用矩阵运算
    y, x = np.ogrid[:h, :w]
    
    # 添加形状扭曲，但减小扭曲程度
    angle = np.random.uniform(0, 2*np.pi)
    distortion = np.random.uniform(0.85, 1.15, (2,))  # 减小扭曲范围
    
    # 坐标变换
    x_rot = (x - center[0]) * np.cos(angle) + (y - center[1]) * np.sin(angle)
    y_rot = -(x - center[0]) * np.sin(angle) + (y - center[1]) * np.cos(angle)
    dist_from_center = np.sqrt((x_rot * distortion[0])**2 + (y_rot * distortion[1])**2)
    
    # 使用更平滑的边缘扰动
    theta = np.arctan2(y - center[1], x - center[0])
    # 减少高频扰动，使用更低频率的变化
    edge_noise = 1 + 0.15 * np.cos(2 * theta) + 0.1 * np.sin(2 * theta)
    # 添加平滑的随机扰动
    smooth_noise = cv2.GaussianBlur(np.random.normal(0, 0.1, (h, w)), (5, 5), 2)
    edge_noise += smooth_noise
    
    # 创建基础目标区域
    target_region = (dist_from_center * edge_noise) <= radius
    
    # 使用形态学操作确保连续性
    kernel = np.ones((3, 3), np.uint8)
    target_region = cv2.morphologyEx(target_region.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    target_region = cv2.morphologyEx(target_region, cv2.MORPH_OPEN, kernel)
    
    # 使用改进的强度计算方式
    intensity = np.zeros((h, w))
    valid_region = target_region > 0
    intensity[valid_region] = np.exp(-0.7 * (dist_from_center[valid_region] / radius)**1.5)
    
    # 添加平滑的纹理变化
    texture_scale = cv2.GaussianBlur(np.random.uniform(0.9, 1.1, (h, w)), (5, 5), 2)
    noise = cv2.GaussianBlur(np.random.normal(0, 0.1, (h, w)), (3, 3), 1)
    intensity = intensity * texture_scale + noise * intensity
    
    # 添加平滑的热点
    num_hotspots = np.random.randint(1, 3)  # 减少热点数量
    for _ in range(num_hotspots):
        offset = np.random.uniform(-radius*0.2, radius*0.2, 2)  # 减小偏移范围
        hotspot_dist = np.sqrt((x - (center[0] + offset[0]))**2 + 
                              (y - (center[1] + offset[1]))**2)
        hotspot = np.exp(-0.5 * (hotspot_dist / (radius * 0.3))**2) * \
                 np.random.uniform(0.2, 0.3)
        intensity += hotspot * (target_region > 0)
    
    # 应用最终的平滑
    intensity = cv2.GaussianBlur(intensity, (3, 3), 0.5)
    
    # 裁剪并应用强度
    intensity = np.clip(intensity, 0, 1)
    target_values = intensity * scale * background_std + background_mean
    target_values = np.clip(target_values, 0, 255)
    
    # 更新目标区域
    target[valid_region] = target_values[valid_region]
    mask[valid_region] = 255
    
    return target, mask

def ellipse_gen(h, w, target, scale, background_std, background_mean, axis=None):
    """
    生成一个连续的不规则类椭圆形红外目标
    """
    center = (w // 2, h // 2)
    if axis is None:
        axis1 = max(random.randint(h // 4, h // 2), 1)
        axis2 = max(random.randint(w // 4, w // 2), 1)
    else:
        axis1, axis2 = axis
    
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 使用矩阵运算
    y, x = np.ogrid[:h, :w]
    
    # 添加平滑的旋转和扭曲
    angle = np.random.uniform(0, 2*np.pi)
    x_rot = (x - center[0]) * np.cos(angle) + (y - center[1]) * np.sin(angle)
    y_rot = -(x - center[0]) * np.sin(angle) + (y - center[1]) * np.cos(angle)
    
    # 减小扭曲程度
    distortion = np.random.uniform(0.85, 1.15, (2,))
    dist = np.sqrt((x_rot * distortion[0] / axis2)**2 + 
                   (y_rot * distortion[1] / axis1)**2)
    
    # 使用更平滑的边缘扰动
    theta = np.arctan2(y - center[1], x - center[0])
    edge_noise = 1 + 0.1 * np.cos(2 * theta) + 0.08 * np.sin(2 * theta)
    smooth_noise = cv2.GaussianBlur(np.random.normal(0, 0.08, (h, w)), (5, 5), 2)
    edge_noise += smooth_noise
    
    # 创建目标区域并确保连续性
    target_region = (dist * edge_noise) <= 1
    kernel = np.ones((3, 3), np.uint8)
    target_region = cv2.morphologyEx(target_region.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    target_region = cv2.morphologyEx(target_region, cv2.MORPH_OPEN, kernel)
    
    # 计算强度
    intensity = np.zeros((h, w))
    valid_region = target_region > 0
    intensity[valid_region] = np.exp(-0.7 * (dist[valid_region] * 1.5)**1.5)
    
    # 添加平滑的纹理和噪声
    texture = cv2.GaussianBlur(np.random.uniform(0.9, 1.1, (h, w)), (5, 5), 2)
    noise = cv2.GaussianBlur(np.random.normal(0, 0.08, (h, w)), (3, 3), 1)
    intensity = intensity * texture + noise * intensity
    
    # 添加平滑的热点
    num_hotspots = np.random.randint(1, 3)
    for _ in range(num_hotspots):
        offset = np.random.uniform(-0.2, 0.2, 2)
        hotspot_dist = np.sqrt(((x - (center[0] + offset[0]*axis2))/axis2)**2 + 
                              ((y - (center[1] + offset[1]*axis1))/axis1)**2)
        hotspot = np.exp(-0.5 * (hotspot_dist * 1.5)**2) * \
                 np.random.uniform(0.2, 0.3)
        intensity += hotspot * (target_region > 0)
    
    # 最终平滑处理
    intensity = cv2.GaussianBlur(intensity, (3, 3), 0.5)
    
    # 裁剪并应用强度
    intensity = np.clip(intensity, 0, 1)
    target_values = intensity * scale * background_std + background_mean
    target_values = np.clip(target_values, 0, 255)
    
    # 更新目标区域
    target[valid_region] = target_values[valid_region]
    mask[valid_region] = 255
    
    return target, mask, axis1, axis2

 
def generate_irregular_polygon_points(num_points):
    """
    生成不规则多边形的顶点
    
    参数:
    num_points: 多边形顶点数量
    返回: 极坐标角度数组
    """
    # 初始化第一个顶点的极坐标角度
    initial_angle = np.random.uniform(0, 2 * np.pi)
    
    # 生成其他顶点的不规则极坐标角度
    angles = np.sort(np.random.uniform(0, 2 * np.pi, num_points - 1))
    angles = np.concatenate([[initial_angle], angles + initial_angle]) % (2 * np.pi)
    return angles

def polygon_gen(h, w, target, pixel_scale, background_std, background_mean, angles=None):

    if angles is None:
        num_points = random.randint(3, 10)
        angles = generate_irregular_polygon_points(num_points)
    
    # 计算多边形的半径
    radius = 1  # 使用标准化的半径

    # 计算顶点的标准化坐标
    points = np.array([[radius * np.cos(angle), radius * np.sin(angle)] for angle in angles])
    
    # 计算目标的缩放因子
    max_dim = max(h, w)
    scale = max_dim / 2

    # 缩放顶点并移动到目标中心
    points = points * scale
    points[:, 0] += w / 2  # 移动到目标中心
    points[:, 1] += h / 2  # 移动到目标中心
    points = points.astype(np.int32).reshape((-1, 1, 2))

    # 生成多边形颜色
    pixel_color = np.random.uniform(0.8, 1) * pixel_scale
    color = np.clip(pixel_color * background_std + background_mean, 0, 255)
    
    # 绘制多边形
    cv2.fillPoly(target, [points], color)
    
    return angles

def load_3d_model_library(models_dir):
    """
    加载3D模型库
    
    参数:
    models_dir: 模型库目录路径
    返回: (加载的模型列表, 模型文件路径列表)
    """
    models = []
    model_paths = []
    
    # 确保目录存在
    if not os.path.exists(models_dir):
        print(f"warning: models library directory not found: {models_dir}")
        return models, model_paths
        
    # 获取所有.obj文件
    for file in os.listdir(models_dir):
        if file.endswith('.obj'):
            model_path = os.path.join(models_dir, file)
            try:
                mesh = trimesh.load(model_path, force='mesh')
                mesh.fix_normals()
                mesh = normalize_mesh(mesh)
                models.append(mesh)
                model_paths.append(model_path)
                print(f"load 3d model success: {file}")
            except Exception as e:
                print(f"load 3d model failed: {file}: {e}")
    
    if not models:
        print("warning: no valid 3d model")
    else:
        print(f"load {len(models)} 3d models")
    
    return models, model_paths

def add_target_to_background(background, target_mask, target_size_ratio=0.1, peak_temp=250, 
                           falloff_sigma=2.0, min_temp=220):
    """
    将红外目标添加到背景图像块中
    
    参数:
    background: 背景图像块
    target_mask: 目标掩码
    target_size_ratio: 目标相对于背景的大小比例
    peak_temp: 目标峰值温度
    falloff_sigma: 温度衰减系数
    min_temp: 最小温度
    """
    # 获取背景图像块尺寸
    bg_h, bg_w = background.shape
    
    # 获取目标掩码的轮廓
    contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("target mask not found contours")
    
    # 获取目标的边界框
    x, y, w, h = cv2.boundingRect(np.concatenate(contours))
    
    # 计算目标的期望大小
    target_size = int(min(bg_w, bg_h) * target_size_ratio)
    scale = target_size / max(w, h)
    new_w = max(int(w * scale), 1)
    new_h = max(int(h * scale), 1)
    
    # 确保目标尺寸不超过背景块
    if new_w > bg_w or new_h > bg_h:
        scale = min(bg_w/w, bg_h/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
    
    # 缩放掩码，保持原始形状
    target_mask_resized = cv2.resize(target_mask[y:y+h, x:x+w], (new_w, new_h))
    
    # 生成红外特征的目标强度
    target_intensity = generate_ir_target_intensity(
        target_mask_resized, 
        peak_temp=peak_temp,
        falloff_sigma=falloff_sigma,
        min_temp=min_temp
    )
    
    # 将目标放在背景块中心
    start_x = (bg_w - new_w) // 2
    start_y = (bg_h - new_h) // 2
    
    # 创建结果图像
    result = background.copy()
    
    # 在目标位置应用强度值
    mask_region = target_mask_resized > 0
    result[start_y:start_y+new_h, start_x:start_x+new_w] = \
        np.where(mask_region, 
                target_intensity, 
                result[start_y:start_y+new_h, start_x:start_x+new_w])
    
    return result, (start_x, start_y, new_w, new_h), mask_region

def target_generator(target_background, shape_type, background_std, background_mean, background_diff, target_distance, base_distance=3000, params=None, models_library=None):
    """
    生成目标和对应的掩码
    参数:
    target_background: 背景图像
    shape_type: 目标形状类型 ('circle', 'ellipse', '3d_projection')
    background_std: 背景标准差
    background_mean: 背景均值
    background_diff: 背景最大最小值差异
    target_distance: 目标距离
    base_distance: 基准距离，用于计算辐射强度
    params: 目标形状参数
    models_library: 3D模型库
    """
    h, w = target_background.shape
    mask = np.zeros((h, w), dtype=np.uint8)  # 初始化掩码图像

    # 计算辐射强度折算像素值
    eta = np.clip(np.log(np.log(base_distance**2 / target_distance**2 + 1) + 1), 0, 1) 
    scale = background_diff * eta # 基准值原来设计的是scale = 0.5

    if shape_type == '3d_projection':
        if not models_library or not models_library[0]:
            print("warning: 3d model library is empty, switch to circle target")
            shape_type = 'circle'
            params = None
    
    # 目标初始化，根据形状类型生成目标
    if shape_type == 'circle':
        target, mask = circle_gen(h, w, target_background, scale, background_std, background_mean)
        target_info = {'shape_type': 'circle', 'target': target, 'mask': mask, 'params': None}

    elif shape_type == 'ellipse':
        target, mask, axis1, axis2 = ellipse_gen(h, w, target_background, scale, background_std, background_mean, params)
        target_info = {'shape_type': 'ellipse', 'target': target, 'mask': mask, 'params': (axis1, axis2)}

    # elif shape_type == 'polygon':
    #     angles = polygon_gen(h, w, target_background, scale, background_std, background_mean, params)
    #     target_info = {'shape_type': 'polygon', 'target': target, 'params':angles}
    
    elif shape_type == '3d_projection':
        if params is None or 'model_idx' not in params:
            model_idx = np.random.randint(0, len(models_library[0]))
            rotation = (0, 0, 0)
        else:
            model_idx = params['model_idx']
            rotation = params['rotation']
        
        mesh = models_library[0][model_idx]
        target_pos = np.array([0, 0, target_distance]) # TODO: 需要修改目标位置
        azimuth, elevation, _ = calculate_target_angles(target_pos)
        
        # 获取投影掩码
        print("----------in 3d projection------------")
        target_mask = get_projection(
            mesh.copy(),
            azimuth,
            elevation,
            distance=target_distance/200,
            target_rotation=rotation
        )
        print("----------out 3d projection------------")
        # 使用add_target_to_background函数处理目标
        target, bbox, mask_region = add_target_to_background(
            target_background,
            target_mask,
            target_size_ratio=1,
            peak_temp=250,
            falloff_sigma=2.0,
            min_temp=220
        )
        start_x, start_y, new_w, new_h = bbox
        # 确保掩码为二值图像
        mask[start_y:start_y+new_h, start_x:start_x+new_w] = mask_region * 255
        
        target_info = {
            'shape_type': '3d_projection', 
            'target': target, 
            'mask': mask, 
            'params': {
                'model_idx': model_idx,
                'rotation': rotation,
                'distance': target_distance,
                'bbox': bbox
            }
        }

    return target_info


def MISTG(input_images, max_num_targets, output_folder):
    """
    Moving Infrared Small Target Generate and Background Alignment
    运动红外弱小目标生成与背景对齐
    
    主要步骤:
    1. 背景位移计算 - 使用光流法计算背景运动
    2. 参数初始化 - 设置相机参数、目标参数等
    3. 第一帧目标初始化 - 生成初始目标
    4. 目标更新 - 根据运动模型更新后续帧的目标
    
    参数:
    input_images: 输入背景图像序列
    max_num_targets: 最大目标数量
    output_folder: 输出文件夹路径
    """
    
    # 初始化参数
    init_target_nums = max_num_targets
    img_nums, img_h, img_w = input_images.shape
    print("Image Number: ", img_nums)                    # 512 640
    print("Image Size: ", [img_h, img_w])                    # 512 640
    
    output_images = input_images.copy()
    output_images_mask = np.zeros_like(input_images)


    ###########################################  background displacement  ##########################################
    """
    背景位移计算部分:
    1. 使用Lucas-Kanade光流法计算连续帧之间的背景运动
    2. 累积计算总位移,用于后续目标位置更新
    """
    
    # Track previous features for optical flow calculation
    # Lucas-Kanade光流法参数
    lk_params = dict(
        winSize=(21, 21),  # 增大窗口大小
        maxLevel=3,  # 增加金字塔层数
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)  # 调整终止条件
    )

    # 角点检测参数
    feature_params = dict(
        maxCorners=200,  # 增加检测到的角点数
        qualityLevel=0.4,  # 提高角点质量等级
        minDistance=5,  # 减小角点之间的最小距离
        blockSize=9  # 增加块大小
    )

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

    x_displacement_range_min = np.min(relative_displacement_array[:,0])
    x_displacement_range_max = np.max(relative_displacement_array[:,0])
    y_displacement_range_min = np.min(relative_displacement_array[:,1])
    y_displacement_range_max = np.max(relative_displacement_array[:,1])
      
    print("Background Displacement Range, ", "X Range: ", 
          [x_displacement_range_min, x_displacement_range_max], 
          ", Y Range: ", [y_displacement_range_min, y_displacement_range_max])
    
    print("Final Background Displacement: ", relative_displacement_array[-1])
    ##############################################################################################################
    
                
    ###########################################  params init  ##########################################
    # TODO 参数控制优化
    F = 50 * 1e-3                                                        # 选择使用红外相机焦距 m
    pixel_size = 15 * 1e-6                                               # 像元尺寸 m
    
    min_target_size = 1                                                  # 初始化目标尺寸最小值 m
    max_target_size = 5                                                  # 初始化目标尺寸最大值 m
    min_init_distance = 500                                             # 最小目标初始化距离 m
    max_init_distance = 1000                                             # 最大目标初始化距离 m   
    z_range_min = 500                                                     # 目标最近距离，超出距离调转方向
    z_range_max = 6500                                                   # 目标最远距离，超出距离调转方向                      
    
    base_distance = 3000                                                 # 目标基准距离，根据此距离修正目标辐射强度以修正像素值
    margin = 10                                                          # 初始化目标距离图像边框最小距离限制     
    max_velocity = 1400                                                   # 目标最大速度 m/s
    min_init_velocity = 200                                              # 初始化目标最小速度 m/s
    max_init_velocity = 500                                              # 初始化目标最大速度 m/s
    
    init_acceleration_range = [50, 500]                                  # 初始加速度 (m/s^2)
    acceleration_change_rate_range = [-50, 500]                           # 加速度变化率 (m/s^3)

    time2maxspeed = 2                                                    # 加速到最大速度所需时间 s
    min_acceleration_factor = -100                                       # 最小加速度，考虑匀减速运动 m/s^2
    max_acceleration_factor = max_velocity // time2maxspeed              # 最大加速度 m/s^2    
    rotation_factor = 5                                                  # 初始化最大目标自旋转角度 
    
    fps = 100                                                            # 视频帧率
    mode_slice = 4                                                       # 运动模式允许的最大段数
    target_density = init_target_nums / (img_h * img_w)                  # 场景中目标密度
    
    x_range = (x_displacement_range_max - x_displacement_range_min) + img_w
    y_range = (y_displacement_range_max - y_displacement_range_min) + img_h
    init_target_nums = int(target_density * (x_range * y_range))
    print("Panoramic Target Number: ", init_target_nums)
    
    models_dir = "/home/guantp/Infrared/MIRST/motive_target_gen/3d_models/"
    
    # 全场景图像坐标范围，以第一帧图像左下角为坐标原点
    x_min = np.ceil(x_displacement_range_min + margin)
    y_min = np.ceil(y_displacement_range_min  + margin)
    x_max = np.floor(x_displacement_range_max + img_w - margin)
    y_max = np.floor(y_displacement_range_max + img_h - margin)
    
    # 目标位置初始化
    target_positions = np.random.randint([x_min, y_min, min_init_distance], 
                                         [x_max, y_max, max_init_distance], 
                                         size=(init_target_nums, 3))         # N*3 (x,y,z)
    target_distances = target_positions[::,-1]
    target_real_size = np.random.randint(min_target_size, 
                                         max_target_size, 
                                         size=init_target_nums)          

    pixel_scale = (target_distances * pixel_size) / F          # 1px distance, 50mm: 5000->1.5m 3000->0.9m
    target_pixel = np.round(target_real_size / pixel_scale)    # 目标占据像素
    print(target_pixel)
    ####################################################################################################
    
    
    
    ##################################  first frame target init  ###################################### 
    # 加载3D模型库
    models_library = load_3d_model_library(models_dir)
    
    target_shapes = ['circle', 'ellipse']
    # if models_library[0]:  # 如果成功加载了模型
    #     target_shapes.append('3d_projection')   
    # 记录每个目标的类型
    target_shape_ids = {}
    target_params = {}
    
    for j in range(init_target_nums):
        # target_shape_ids[j] = np.random.choice(target_shapes, p=[0.2, 0.1, 0.7])
        target_shape_ids[j] = np.random.choice(target_shapes, p=[0.5, 0.5])
        if target_shape_ids[j] == '3d_projection':
            target_params[j] = {
                'model_idx': np.random.randint(0, len(models_library[0]))
            }
        else:
            target_params[j] = None  
              
    # 只为3D投影目标生成旋转序列
    all_rotations = generate_3d_targets_rotation_sequences(img_nums, fps, target_shape_ids)
                    
    # target_shape_ids = np.random.randint(0, len(target_shapes), size=init_target_nums)        # 初始化目标形状
    init_targets_info = []
    # First frame: Initialize targets
    # Calculate background brightness statistics
    first_frame_img = input_images[0].astype(np.uint8)
    background_mean = np.mean(first_frame_img)
    background_std = np.std(first_frame_img)
    background_diff = (np.max(first_frame_img) - np.min(first_frame_img)) / background_std  
    print("######## The 1 frame ########")
    for j in range(init_target_nums):

        shape_type = target_shape_ids[j]
        h = w = int(target_pixel[j]) # TODO 待优化
        x, y, z = target_positions[j]

        if (0+w) <= x < (img_w-w) and (0+h) <= y < (img_h-h):
            img_target_background = input_images[0][y:y+h, x:x+w]
            
            print(f"target {j}: y, x: {y}, {x}, h, w: {h}, {w}", shape_type)
            # print(img_target_background.shape)
            
            init_target_info = target_generator(
                img_target_background, 
                shape_type, 
                background_std, 
                background_mean, 
                background_diff, 
                z,
                base_distance=base_distance,
                params=None,
                models_library=models_library
            )
            
            target = init_target_info['target']       
            mask = init_target_info['mask']
            init_targets_info.append(init_target_info)
            
            # Generate targets and add them to the frame
            output_images[0, y:y+h, x:x+w] = target
            output_images_mask[0, y:y+h, x:x+w] = mask
            
        else: # TODO 待优化
            init_target_info = target_info = {
            'shape_type': shape_type, 
            'target': np.array([]), 
            'mask': np.array([]),
            'params': None
        }
            init_targets_info.append(init_target_info)        
    

    # 保存当前帧图像
    output_image_path = os.path.join(output_folder, f"output_image_0.png")
    cv2.imwrite(output_image_path, output_images[0])
    # 保存掩码图像
    mask_output_path = os.path.join(output_folder, f"output_image_0_mask.png")
    cv2.imwrite(mask_output_path, output_images_mask[0])        
    #####################################################################################################
    

    ##########################################  target update  ##########################################  
    """
    目标更新部分:
    1. 初始化目标运动参数(速度、加速度、运动方向等)
    2. 计算目标在整个序列中的累积位移
    3. 根据背景位移和目标运动更新每一帧的目标位置和外观
    """
    
    # 随机初始化运动参数
    targets_init_velocity = np.random.randint(min_init_velocity, 
                                              max_init_velocity, 
                                              size=init_target_nums)
    
    targets_acceleration_factor = np.random.uniform(min_acceleration_factor, 
                                                    max_acceleration_factor, 
                                                    size=init_target_nums)
    
    targets_init_acceleration = np.random.randint(init_acceleration_range[0],
                                                  init_acceleration_range[1], 
                                                  size=init_target_nums)
    
    targets_acceleration_change_rate = np.random.randint(acceleration_change_rate_range[0], 
                                                         acceleration_change_rate_range[1], 
                                                         size=init_target_nums)
    
    motion_modes, time_ratios = generate_arrays(init_target_nums, mode_slice) # 每个目标的运动模式数量在1到4之间
    
    # 初始化目标运动方向参数
    motion_direction = np.random.choice([-1, 1], size=(init_target_nums, 3))   # N, (x,y,z)
    motion_direction_coef = generate_multiple_sets_of_random_numbers(init_target_nums, 3)
    targets_direction_coef = motion_direction * motion_direction_coef

    # 计算目标累积位移
    targets_cumulative_displacements = get_cumulative_displacements(
        motion_modes, 
        time_ratios,
        img_nums-1,
        fps,
        targets_init_velocity,
        max_velocity,
        targets_direction_coef,
        targets_acceleration_factor,
        targets_init_acceleration,
        targets_acceleration_change_rate,
        init_target_nums,
        [x_displacement_range_min, x_displacement_range_max],
        [y_displacement_range_min, y_displacement_range_max],
        [z_range_min, z_range_max],
        target_positions,
        img_w,
        img_h
    )  # 返回形状: [targets_num, img_num, 3]
        
    if target_shape_ids[j] == '3d_projection':
        params = target_params[j].copy()  # 复制基础参数
        params['rotation'] = tuple(all_rotations[j][i])  # 添加当前帧的旋转参数
    else:
        params = None
    
    target_info = target_generator(
        img_target_background, 
        target_shape_ids[j],  # 使用记录的目标类型
        background_std, 
        background_mean, 
        background_diff, 
        z,
        base_distance=base_distance,
        params=params,
        models_library=models_library
    )
 
      
    # 以第一帧的左下角作为整个场景的坐标原点（0,0），目标动态更新与计算，目标更新依据背景位移多少进行更新    
    for i in range(1, img_nums):
        print("######## The ", i+1, " frame ########")
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
        # print(update_target_pixel)
        # 更新像素值
        # 输入参数：初始帧参数，位移变化、旋转变化
        for j in range(init_target_nums):
            shape_type = target_shape_ids[j]
            h = w = int(update_target_pixel[j]) # TODO 待优化
            x, y, z = update_target_positions[j]

            x = int(x - relative_displacement_array[i-1, 0])
            y = int(y - relative_displacement_array[i-1, 1]) # 减去背景位移,计算背景坐标变换后物体与新的坐标系的相对位置
            
            # 判断目标是否显示在图像中，只需要计算在现在这一帧图像内的目标，依据目标中心点和目标大小判定
            # 简化：忽略目标大小，仅依据中心点判定
            if (0+w) <= x < (img_w-w) and (0+h) <= y < (img_h-h):
                img_target_background = current_gray[y:y+h, x:x+w]
                print(f"target {j}: y, x: {y}, {x}, h, w: {h}, {w}", shape_type)
                
                target_info = target_generator(img_target_background, 
                                               shape_type, 
                                               background_std, 
                                               background_mean, 
                                               background_diff, 
                                               z, 
                                               base_distance=base_distance, 
                                               params=init_targets_info[j]['params'],
                                               models_library=models_library)
                
                target = target_info['target']
                mask = target_info['mask']
                # Generate targets and add them to the frame
                output_images[i, y:y+h, x:x+w] = target 
                output_images_mask[i, y:y+h, x:x+w] = mask
                if shape_type == '3d_projection':
                    params = target_info['params']
                    current_rotation = params['rotation']
                    new_rotation = update_rotation(current_rotation, i, fps, params)
                    params['rotation'] = new_rotation
                    # 更新目标信息
                    target_info['params'] = params
                    
        
        # 保存当前帧图像
        output_image_path = os.path.join(output_folder, f"output_image_{i}.png")
        cv2.imwrite(output_image_path, output_images[i])
        # 保存掩码图像
        mask_output_path = os.path.join(output_folder, f"output_image_{i}_mask.png")
        cv2.imwrite(mask_output_path, output_images_mask[i])   
    return output_images
    ######################################################################################################


if __name__ == '__main__':
    # 参数配置
    folder_path = "/home/guantp/Infrared/datasets/mydata/250110/M615/100_1min_4"   # Replace with your folder containing images
    # folder_path = "/home/guantp/Infrared/MIRST/motive_target_gen/bg_imgs"
    # folder_path = "/home/guantp/Infrared/datasets/复杂背景下红外弱小运动目标检测数据集/train/100/"
    output_folder = "/home/guantp/Infrared/MIRST/motive_target_gen/motive_target_imgs/"

    # 如果输出文件夹存在，则删除整个文件夹
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    # 重新创建文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    max_num_targets = 20

    input_images = load_images_from_folder(folder_path)
    output_images = MISTG(input_images, max_num_targets, output_folder)
    print("Moving Infrared Small Target Generate Finish.")
    
# TODO 未完成部分以及优化

# 使用三维模型 + 投影优化目标形状
# 添加目标自旋转模型
# 背景抖动模糊图像处理
# 像素值根据距离和大气传输模型仿真优化
# 设计合理的运动速度、加速度和变化率
# 运动参数添加微弱随机扰动
# 目标mask生成
# 光流法优化