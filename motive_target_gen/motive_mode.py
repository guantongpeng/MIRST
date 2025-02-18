import numpy as np

# 根据不同运动模式计算目标位移
def get_displacement(motion_mode, 
                     init_velocity, 
                     max_velocity, 
                     direction_coef, 
                     acceleration_factor, 
                     a0, 
                     k, 
                     frame, 
                     fps, 
                     random_accelerations):
    
    x_coef, y_coef, z_coef = direction_coef
    
    t = 1 / fps
    
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

    # elif motion_mode == 3:  # 'random_motion' 随机运动
    #     time_step = 1 / fps  # 使用帧率作为时间步长

    #     # 初始化速度和位移
    #     velocity = init_velocity
    #     displacement = 0

    #     # 获取前frame个加速度变化
    #     accelerations = random_accelerations[:frame]
        
    #     # 计算速度
    #     velocity = np.cumsum(accelerations) * acceleration_factor + velocity
    #     velocity = np.clip(velocity, 0, max_velocity)  # 确保速度不超过最大速度

    #     # 计算位移
    #     displacements = np.cumsum(velocity) * time_step * direction_coef
        
    #     # 获取当前帧的位移
    #     displacement = displacements[-1] if len(displacements) > 0 else 0

    else:
        raise ValueError("Invalid motion mode")

    displacement_x = displacement * x_coef
    displacement_y = displacement * y_coef
    displacement_z = displacement * z_coef
    return displacement_x, displacement_y, displacement_z, velocity

# 计算所有目标的累计位移
def get_cumulative_displacements(motion_modes, 
                                 time_ratios, 
                                 total_frames, 
                                 fps, 
                                 target_init_velocity, 
                                 max_velocity, 
                                 target_direction_coef, 
                                 target_acceleration_factor, 
                                 init_acceleration, 
                                 acceleration_change_rate, 
                                 target_nums, 
                                 x_range, 
                                 y_range, 
                                 z_range, 
                                 target_positions, 
                                 img_w, 
                                 img_h): 
    
    all_cumulative_displacements = []

    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    
    for target_index in range(target_nums):
        target_cumulative_displacements = []
        cumulative_displacement_x = 0
        cumulative_displacement_y = 0
        cumulative_displacement_z = 0

        x0, y0, z0 = target_positions[target_index]
        init_velocity = target_init_velocity[target_index]
        direction_coef = target_direction_coef[target_index]
        acceleration_factor = target_acceleration_factor[target_index]
        a0 = init_acceleration[target_index]
        k = acceleration_change_rate[target_index]

        target_motion_modes, target_time_ratios = motion_modes[target_index], time_ratios[target_index]
        
        cur_velocity = init_velocity
        
        for mode, ratio in zip(target_motion_modes, target_time_ratios):
            num_frames = int(ratio * total_frames)
            if mode == 3:
                random_accelerations = np.random.uniform(-1, 1, num_frames)
            else:
                random_accelerations = None
                
            for frame in range(num_frames):       
                displacement_x, displacement_y, displacement_z, cur_velocity = get_displacement(mode, 
                                                                                                cur_velocity, 
                                                                                                max_velocity, 
                                                                                                direction_coef, 
                                                                                                acceleration_factor, 
                                                                                                a0, 
                                                                                                k, 
                                                                                                frame+1, 
                                                                                                fps, 
                                                                                                random_accelerations)
                
                cumulative_displacement_x = displacement_x + cumulative_displacement_x
                cumulative_displacement_y = displacement_y + cumulative_displacement_y
                cumulative_displacement_z = displacement_z + cumulative_displacement_z

                # 目标移动边界检查是否超出范围，如果超出则调整方向
                if cumulative_displacement_x < (x_min-x0) or cumulative_displacement_x > (x_max+img_w-x0):
                    direction_coef = (-direction_coef[0], direction_coef[1], direction_coef[2])
                    cumulative_displacement_x = max(min(cumulative_displacement_x, x_max+img_w-x0), x_min-x0)
                if cumulative_displacement_y < (y_min-y0) or cumulative_displacement_y > (y_max+img_h-y0):
                    direction_coef = (direction_coef[0], -direction_coef[1], direction_coef[2])
                    cumulative_displacement_y = max(min(cumulative_displacement_y, y_max+img_h-y0), y_min-y0)
                if cumulative_displacement_z < (z_min-z0) or cumulative_displacement_z > (z_max-z0):
                    direction_coef = (direction_coef[0], direction_coef[1], -direction_coef[2])
                    cumulative_displacement_z = max(min(cumulative_displacement_z, z_max-z0), z_min-z0)

                target_cumulative_displacements.append((cumulative_displacement_x, cumulative_displacement_y, cumulative_displacement_z))

            if len(target_cumulative_displacements) >= total_frames:
                break

        # 如果生成的帧数少于总帧数，填充剩余帧数
        while len(target_cumulative_displacements) < total_frames:
            target_cumulative_displacements.append(target_cumulative_displacements[-1])

        all_cumulative_displacements.append(target_cumulative_displacements[:total_frames])

    return np.array(all_cumulative_displacements)


def update_rotation(rotation, frame, fps, maneuver_params=None):
    """
    更新飞机旋转角度，模拟更真实的飞行姿态变化。
    
    参数:
    - rotation: 当前旋转角度 (pitch, yaw, roll)
    - frame: 当前帧数
    - fps: 帧率
    - maneuver_params: 机动参数字典，包含机动类型和相关参数
    
    返回:
    - new_rotation: 更新后的旋转角度 (pitch, yaw, roll)
    """
    t = frame / fps
    
    # 基本姿态保持（小幅振荡）
    base_pitch = rotation[0] + np.sin(t * 0.5) * 0.2  # 俯仰角小幅上下振荡
    base_yaw = rotation[1] + np.sin(t * 0.3) * 0.1    # 偏航角小幅左右振荡
    base_roll = rotation[2] + np.sin(t * 0.4) * 0.15  # 滚转角小幅摆动
    
    # 大气湍流引起的随机扰动（使用低通滤波的随机噪声）
    turbulence_scale = 0.1
    noise_pitch = np.random.normal(0, 0.1) * turbulence_scale
    noise_yaw = np.random.normal(0, 0.1) * turbulence_scale
    noise_roll = np.random.normal(0, 0.1) * turbulence_scale
    
    # 机动动作（如果有）
    maneuver_pitch = 0
    maneuver_yaw = 0
    maneuver_roll = 0
    
    if maneuver_params:
        maneuver_type = maneuver_params.get('type', None)
        if maneuver_type == 'turn':
            # 转向机动
            turn_progress = (t - maneuver_params['start_time']) / maneuver_params['duration']
            if 0 <= turn_progress <= 1:
                turn_angle = maneuver_params['angle']
                maneuver_roll = np.sin(turn_progress * np.pi) * 20  # 最大20度侧倾
                maneuver_yaw = turn_angle * (1 - np.cos(turn_progress * np.pi)) / 2
                maneuver_pitch = -np.sin(turn_progress * np.pi) * 5  # 轻微俯冲
        elif maneuver_type == 'climb':
            # 爬升机动
            climb_progress = (t - maneuver_params['start_time']) / maneuver_params['duration']
            if 0 <= climb_progress <= 1:
                climb_angle = maneuver_params['angle']
                maneuver_pitch = climb_angle * np.sin(climb_progress * np.pi)
    
    # 合成最终旋转角度
    new_rotation = (
        base_pitch + noise_pitch + maneuver_pitch,
        base_yaw + noise_yaw + maneuver_yaw,
        base_roll + noise_roll + maneuver_roll
    )
    
    # 限制角度范围
    new_rotation = (
        new_rotation[0] % 360,   # 限制俯仰角范围
        new_rotation[1] % 360,    # 偏航角循环
        new_rotation[2] % 360,    # 限制滚转角范围
    )
    
    return new_rotation

def generate_rotation_pattern(t, pattern_type):
    """
    生成单个旋转模式的角度变化。
    
    参数:
    - t: 时间序列
    - pattern_type: 旋转模式类型
    
    返回:
    - rotations: shape为(len(t), 3)的数组，包含(pitch, yaw, roll)角度
    """
    rotations = np.zeros((len(t), 3))
    
    if pattern_type == 'stable':
        # 稳定模式：小幅振荡
        rotations[:, 0] = np.sin(t * 0.5) * 1.0  # 俯仰角
        rotations[:, 1] = np.sin(t * 0.3) * 0.8  # 偏航角
        rotations[:, 2] = np.sin(t * 0.4) * 0.5  # 滚转角
        
    elif pattern_type == 'fast_yaw':
        # 快速偏航模式
        rotations[:, 0] = np.sin(t * 0.8) * 2.0   # 小幅俯仰
        rotations[:, 1] = t * 30                   # 持续偏航
        rotations[:, 2] = np.sin(t * 1.2) * 5.0   # 中等滚转
        
    elif pattern_type == 'roll_dominant':
        # 滚转主导模式
        rotations[:, 0] = np.sin(t * 0.5) * 3.0   # 中等俯仰
        rotations[:, 1] = np.sin(t * 0.3) * 5.0   # 中等偏航
        rotations[:, 2] = np.sin(t * 2.0) * 20.0  # 大幅滚转
        
    elif pattern_type == 'pitch_oscillation':
        # 俯仰振荡模式
        rotations[:, 0] = np.sin(t * 1.5) * 15.0  # 大幅俯仰振荡
        rotations[:, 1] = np.sin(t * 0.2) * 3.0   # 小幅偏航
        rotations[:, 2] = np.sin(t * 0.8) * 8.0   # 中等滚转
        
    elif pattern_type == 'complex':
        # 复杂运动模式：多频率叠加
        rotations[:, 0] = (np.sin(t * 1.0) * 8.0 + 
                          np.sin(t * 2.3) * 3.0)  # 俯仰
        rotations[:, 1] = (np.sin(t * 0.7) * 10.0 + 
                          np.sin(t * 1.5) * 5.0)  # 偏航
        rotations[:, 2] = (np.sin(t * 1.2) * 12.0 + 
                          np.sin(t * 2.5) * 4.0)  # 滚转
    
    return rotations

def generate_rotation_sequence(num_frames, fps):
    """
    生成完整的旋转序列，包含多个阶段的不同旋转模式。
    
    参数:
    - num_frames: 总帧数
    - fps: 帧率
    
    返回:
    - rotations: shape为(num_frames, 3)的数组，包含每一帧的(pitch, yaw, roll)角度
    """
    t = np.arange(num_frames) / fps
    rotations = np.zeros((num_frames, 3))
    
    # 可用的旋转模式
    patterns = ['stable', 'fast_yaw', 'roll_dominant', 'pitch_oscillation', 'complex']
    
    # 随机生成2-4个阶段
    num_phases = np.random.randint(2, 5)
    phase_points = np.sort(np.random.choice(num_frames - 1, num_phases - 1, replace=False))
    phase_points = np.append(phase_points, num_frames)
    start_idx = 0
    
    # 为每个阶段随机选择一个旋转模式
    prev_pattern = None
    for end_idx in phase_points:
        # 选择一个与前一个不同的模式
        available_patterns = [p for p in patterns if p != prev_pattern]
        pattern = np.random.choice(available_patterns)
        prev_pattern = pattern
        
        phase_t = t[start_idx:end_idx] - t[start_idx]
        phase_rotations = generate_rotation_pattern(phase_t, pattern)
        
        # 如果不是第一个阶段，需要平滑过渡
        if start_idx > 0:
            # 创建过渡区间
            transition_frames = min(int(fps * 0.5), (end_idx - start_idx) // 4)  # 0.5秒或四分之一阶段长度
            weights = np.linspace(0, 1, transition_frames)
            
            # 平滑过渡
            for i in range(transition_frames):
                blend_idx = start_idx + i
                rotations[blend_idx] = (rotations[blend_idx] * (1 - weights[i]) + 
                                      phase_rotations[i] * weights[i])
            
            # 复制剩余帧
            rotations[start_idx + transition_frames:end_idx] = phase_rotations[transition_frames:]
        else:
            rotations[start_idx:end_idx] = phase_rotations
        
        start_idx = end_idx
    
    # 添加小幅随机扰动
    turbulence = np.random.normal(0, 0.1, rotations.shape)
    rotations += turbulence
    
    # 限制角度范围
    rotations[:, 0] = np.clip(rotations[:, 0], -30, 30)    # 俯仰角限制
    rotations[:, 1] = rotations[:, 1] % 360                # 偏航角循环
    rotations[:, 2] = np.clip(rotations[:, 2], -45, 45)    # 滚转角限制
    
    return rotations

def generate_3d_targets_rotation_sequences(num_frames, fps, target_types):
    """
    仅为3D投影目标生成旋转序列
    
    参数:
    - num_frames: 总帧数
    - fps: 帧率
    - target_types: 字典，键为目标ID，值为目标类型
    
    返回:
    - rotations_dict: 字典，键为目标ID，值为该目标的旋转序列数组（仅3D目标）
    """
    rotations_dict = {}
    
    for target_id, target_type in target_types.items():
        if target_type == '3d_projection':
            rotations_dict[target_id] = generate_rotation_sequence(num_frames, fps)
    
    return rotations_dict