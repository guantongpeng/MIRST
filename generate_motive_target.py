import cv2
import numpy as np
import os
import random
import re

def load_images_from_folder(folder):
    """
    Load images from a specified folder.
    """
    images = []

    for filename in sorted(os.listdir(folder), key=lambda x: int(re.search(r'\d+', x).group())):
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


def target_init(h, w, shape_type, eta=1.0, theta=0.0, bk_diff=None, template_path=None):
    target = np.full((h, w), -1, dtype=np.uint8)
    scale = bk_diff * 0.5  # 调整目标大小的因子

    # 目标初始化，根据形状类型生成目标
    if shape_type == 'circle':
        center = (h // 2, w // 2)
        radius = max(min(h, w) // 4, 1)
        for i in range(h):
            for j in range(w):
                if (i - center[0]) ** 2 + (j - center[1]) ** 2 <= radius ** 2:
                    _temp = np.exp(-((i - center[0]) ** 2 + (j - center[1]) ** 2) / (2 * radius ** 2)) * scale
                    target[i, j] = np.clip(_temp * eta + theta, 0, 255)
        target_info = {'shape_type': 'circle', 'params':target.astype(np.uint8)}

    elif shape_type == 'ellipse':
        center = (h // 2, w // 2)
        axis1 = max(random.randint(h // 4, h // 2), 1)
        axis2 = max(random.randint(w // 4, w // 2), 1)
        for i in range(h):
            for j in range(w):
                if ((i - center[0]) / axis1) ** 2 + ((j - center[1]) / axis2) ** 2 <= 1:
                    _temp = np.exp(-(((i - center[0]) / axis1) ** 2 + ((j - center[1]) / axis2) ** 2)) * scale
                    target[i, j] = np.clip(_temp * eta + theta, 0, 255)
        target_info = {'shape_type': 'ellipse', 'params':target.astype(np.uint8)}
        
    elif shape_type == 'polygon':
        num_points = random.randint(3, 10)
        points = np.array([[random.randint(0, h), random.randint(0, w)] for _ in range(num_points)], np.int32)
        points = points.reshape((-1, 1, 2))
        _temp = np.random.uniform(0.8, 1.2) * scale
        color = np.clip(_temp * eta + theta, 0, 255)
        cv2.fillPoly(target, [points], color)
        import pdb
        pdb.set_trace()
        target_info = {'shape_type': 'polygon', 'params':[[points], color]}
        
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
    return target_info  # 返回目标的形状和相关参数


def target_gen(h, w, shape_type, eta=1.0, theta=0.0, bk_diff=None, template_path=None): # 用于生成M-1帧的像素值
    target = np.full((h, w), -1, dtype=np.uint8)
    scale = bk_diff * 0.5  # 调整目标大小的因子

    # 目标初始化，根据形状类型生成目标
    if shape_type == 'circle':
        center = (h // 2, w // 2)
        radius = max(min(h, w) // 4, 1)
        for i in range(h):
            for j in range(w):
                if (i - center[0]) ** 2 + (j - center[1]) ** 2 <= radius ** 2:
                    _temp = np.exp(-((i - center[0]) ** 2 + (j - center[1]) ** 2) / (2 * radius ** 2)) * scale
                    target[i, j] = np.clip(_temp * eta + theta, 0, 255)
        target_info = {'shape_type': 'circle', 'params':target.astype(np.uint8)}

    elif shape_type == 'ellipse':
        center = (h // 2, w // 2)
        axis1 = max(random.randint(h // 4, h // 2), 1)
        axis2 = max(random.randint(w // 4, w // 2), 1)
        for i in range(h):
            for j in range(w):
                if ((i - center[0]) / axis1) ** 2 + ((j - center[1]) / axis2) ** 2 <= 1:
                    _temp = np.exp(-(((i - center[0]) / axis1) ** 2 + ((j - center[1]) / axis2) ** 2)) * scale
                    target[i, j] = np.clip(_temp * eta + theta, 0, 255)
        target_info = {'shape_type': 'ellipse', 'params':target.astype(np.uint8)}
        
    elif shape_type == 'polygon':
        num_points = random.randint(3, 10)
        points = np.array([[random.randint(0, h), random.randint(0, w)] for _ in range(num_points)], np.int32)
        points = points.reshape((-1, 1, 2))
        _temp = np.random.uniform(0.8, 1.2) * scale
        color = np.clip(_temp * eta + theta, 0, 255)
        cv2.fillPoly(target, [points], color)
        target_info = {'shape_type': 'polygon', 'params':[[points], color]}
    return target_info 


def update_motion_params(motion_mode, motion_direction, init_velocity, max_velocity, acceleration_factor, t):
    if motion_mode == 'uniform':  # 匀速
        velocity = max_velocity  # 速度是恒定的
        displacement = velocity * t  # 位移随时间线性增加
        x_coef = np.random.random()
        y_coef = np.random.random()
        z_coef = np.random.random()
    elif motion_mode == 'uniform_acceleration':  # 匀加速
        velocity = acceleration_factor * t  # 速度随时间增加
        displacement = 0.5 * acceleration_factor * t**2  # 位移为加速运动
        x_coef =
        y_coef = 
    elif motion_mode == 'variable_acceleration':  # 变加速
        velocity = acceleration_factor * np.sin(t * np.pi / M)  # 变加速度，示例使用正弦函数
        displacement = velocity * t
    else:  # 非标准运动模式
        velocity = max_velocity * np.cos(t * np.pi / M)  # 自定义复杂的非标准运动模式
        displacement = velocity * t
    
    # 根据运动方向调整位置
    if motion_direction == 'circular':
        angle = np.pi * t / M
        displacement_x = np.cos(angle) * displacement * x_coef
        displacement_y = np.sin(angle) * displacement * y_coef
    elif motion_direction == 'approaching':
        displacement_x = -displacement * x_coef
        displacement_y = -displacement * y_coef
    elif motion_direction == 'receding':
        displacement_x = displacement * x_coef
        displacement_y = displacement * y_coef
    elif motion_direction == 'combined':
        angle = np.pi * t / M
        displacement_x = np.cos(angle) * displacement
        displacement_y = np.sin(angle) * displacement
    else:
        displacement_x = displacement
        displacement_y = displacement
    return 
    
def generate_random_numbers_sum_to_one(n):
    """
    生成n个随机正数，这些数的总和为1
    """
    random_numbers = np.random.rand(n)
    return random_numbers / np.sum(random_numbers)

def generate_multiple_sets_of_random_numbers(N, n):
    """
    生成N组随机正数，每组包含n个数，这些数的总和为1
    """
    sets = []
    for _ in range(N):
        sets.append(generate_random_numbers_sum_to_one(n))
    return np.array(sets)
   
def target_motion_transform(target_nums, target_info, img_nums, HZ=50):
    
    # 初始参数
    # TODO
    target_positions = target_info['']               # N,2 (x,y)
    initial_sizes = target_info['']                  # N,2 (h,w)
    initial_shapes = target_info['']                 # N ('circle', 'ellipse', 'polygon')
    target_distances = target_info['']               # N (z)    
    initial_pixel_values = target_info['']           # N个(h,w) 辐射强度（反应为像素值）与距离也有关系
    
    
    _motion_direction = np.random.choice([-1, 1], size=(target_nums, 3))   # N, (x,y,z)
    motion_direction_coef = generate_multiple_sets_of_random_numbers(target_nums, 3)
    motion_direction = _motion_direction * motion_direction_coef
    
    max_velocity = 700
    acceleration_factor = np.random.randint(0, max_velocity//2)
    init_velocity = np.random.randint(0, max_velocity, size=target_nums)     # N
    motion_modes = np.random.randint(0, 3, size=target_nums) # N 每一个目标的运动模式
    targets_move_values = []
    
    '''
    x,y移动像素计算
    
    目标占据像素h,w 目标距离d
    
    h * d
    '''
    
    for target_num in range(target_nums):
    # 模拟每一个目标的运动更新
        
        for t in range(img_nums):
            # 模拟每一帧的运动更新
            # 运动模式和方向决定了每一帧的计算

            displacement = update_motion_params(motion_modes[target_num], motion_direction[target_num], init_velocity[target_num], max_velocity, acceleration_factor, t)
            # 更新位置
            position += np.array([displacement[0], displacement[1]])

            # 计算缩放因子，目标离相机的距离影响大小
            scale_factor = np.interp(position[0], [-500, 500], scaling_factor_range)  # 用x位置模拟距离影响
            
            # 更新目标大小
            new_size = size * scale_factor
            
            # 计算旋转角度
            rotation_angle = np.interp(t, [0, M], rotation_range)
            
            # 更新像素值，假设大小越大像素值越高，模拟目标亮度变化
            new_pixel_value = np.clip(initial_pixel_value * (1 + scale_factor), 0, 255)
            
            # 保存每一帧的运动更新参数
            move_values = {'position': position.copy(),
                           'size': new_size,
                           'rotation_angle': rotation_angle,
                           'pixel_value': new_pixel_value
                           }
            targets_move_values.append(move_values)
            
    return move_values


def DISTG_with_background_alignment(I, max_num_targets, target_shape_range, bbox_size, size_variation_factor=0.05, rotation_factor=5, margin=10):
    """
    Modify DISTG to use zoom accumulation and background alignment.
    """
    target_nums = max_num_targets
    img_nums, img_H, img_W = I.shape
    HZ = 50
    proportion = 10 # 多少m代表1px
    annotations = []

    # Initialize target positions, movements, and attributes
    target_positions = np.random.randint(margin, [img_W-margin, img_H-margin], size=(target_nums, 2))
    target_distances = np.random.randint(margin, [3000, 5000], size=target_nums)
    
    target_motions = np.stack([np.cos(np.random.uniform(-np.pi, np.pi, target_nums)), np.sin(np.random.uniform(-np.pi, np.pi, target_nums))], axis=-1)

    # Randomly assign fixed target attributes (shape, size, brightness)
    target_shapes = ['circle', 'ellipse', 'polygon']
    target_shape_ids = np.random.randint(0, len(target_shapes), size=target_nums)
    
    real_target_sizes = [(
                            random.randint(*target_shape_range[0]),  # Height 假设2m到10m
                            random.randint(*target_shape_range[1])   # Width 
                        ) for _ in range(target_nums)]
    
    target_sizes = (real_target_sizes / target_distances) * 2500
    
    
    '''
    distance=[1000, 5000]
    init_target_size=[2,10]
    real_size=[2,10]
    
    distance = 5000
    1px -> 5m
    distance = 3000
    1px -> 
    '''
    
    '''
    红外辐射强度与距离平方成正比，即像素值变化规律
    生成分割label与检测框
    '''
    zoom_accum_values = np.zeros(target_nums)  # 用来跟踪每个目标的累积缩放因子

    # Track previous features for optical flow calculation
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    prev_gray = None
    prev_features = None

    for i in range(img_nums):
        current_frame = I[i]
        current_gray = current_frame.astype(np.uint8)

        # Calculate background brightness statistics
        theta = np.mean(current_gray)
        eta = np.std(current_gray)
        bg_diff = (np.max(current_gray) - np.min(current_gray)) / eta
        
        targets_info = [] #  帧/目标数/target id: {位置、大小、形状、像素}
                
        if i == 0:  
            # Optical flow background alignment
            prev_gray = current_gray
            prev_features = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
            
            # First frame: Initialize targets
            init_targets_info = []
            for j in range(target_nums):
                shape_type = target_shapes[target_shape_ids[j]]
                
                h, w = target_sizes[j]
                target_info = target_init(h, w, shape_type, eta, theta, bg_diff)
                target = target_info['params']
                init_targets_info.append(target_info)
                x, y = target_positions[j]
                    
                # Generate targets and add them to the frame
                I[i, max(0, y):min(img_H, y+h), max(0, x):min(img_W, x+w)] = target
            targets_info.append({f'{i}':init_targets_info})
            import pdb
            pdb.set_trace()
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
            _targets_info = []
            for j in range(target_nums):
                shape_type = target_shapes[target_shape_ids[j]]
                h, w = target_sizes[j]
                target, new_position, motion, zoom_accum_values[j], target_info = target_motion_transform(
                    target, shape_type, bg_diff, h, w, target_positions[j], target_motions[j], 
                    rotation_factor, size_variation_factor, margin, img_W, img_H, zoom_accum_values[j])
                _targets_info.append(target_info)
                # Update positions and target information
                target_positions[j] = new_position
                target_motions[j] = motion

                # Generate and update targets on frame
                x, y = int(new_position[0] + background_displacement[0]), int(new_position[1] + background_displacement[1])

                mask = target != 0
                I[i, max(0, y):min(img_H, y+h), max(0, x):min(img_W, x+w)] = np.where(mask, target, I[i, max(0, y):min(img_H, y+h), max(0, x):min(img_W, x+w)])
            
            targets_info.append({'i':_targets_info})
            
    return I, annotations



if __name__ == '__main__':
    # Example usage
    folder_path = "/home/guantp/Infrared/datasets/DUAB/data4/"   # Replace with your folder containing images
    input_images = load_images_from_folder(folder_path)  # Load real image frames

    max_num_targets = 2
    target_shape_range = ((30, 50), (30, 50))  # Target shape range ((min_h, max_h), (min_w, max_w))
    bbox_size = 10  # Bounding box size

    output_folder = "/home/guantp/Infrared/datasets/temp_dataset/"   # Replace with your desired output folder
    output_images, annotations = DISTG_with_background_alignment(input_images, max_num_targets, target_shape_range, bbox_size)
    save_images_and_annotations(output_folder, output_images, annotations)


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


def _target_motion_transform(target, shape_type, scale, h, w, position, motion, angle, zoom_factor, margin, W, H, zoom_accum=0.0):
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

    return target, new_position, (motion_x, motion_y), zoom_accum # {目标信息}