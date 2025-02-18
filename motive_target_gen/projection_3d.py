import numpy as np
import trimesh
import cv2
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor  # 用于并行处理

def normalize_mesh(mesh):
    """归一化模型尺寸和位置"""
    scale = np.max(mesh.extents)
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= (1.0 / scale)
    center = mesh.centroid
    translation = np.eye(4)
    translation[:3, 3] = -center
    # 合并变换矩阵，减少多次 apply_transform 调用
    transform = scale_matrix @ translation
    mesh.apply_transform(transform)
    return mesh

def project_points(points_3d, camera_matrix, dist_coeffs=None):
    """使用OpenCV投影3D点到2D平面"""
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)
    points_2d, _ = cv2.projectPoints(points_3d, 
                                     rvec=np.zeros(3), 
                                     tvec=np.zeros(3), 
                                     cameraMatrix=camera_matrix, 
                                     distCoeffs=dist_coeffs)
    return points_2d.reshape(-1, 2)

def rotate_mesh(mesh, pitch=0, yaw=0, roll=0):
    """
    按照欧拉角旋转模型
    pitch: 俯仰角（绕X轴），单位：度
    yaw: 偏航角（绕Y轴），单位：度
    roll: 滚转角（绕Z轴），单位：度
    """
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    roll_rad = np.radians(roll)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    Ry = np.array([
        [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])
    Rz = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad), np.cos(roll_rad), 0],
        [0, 0, 1]
    ])
    R = Ry @ Rx @ Rz  # 顺序：roll -> pitch -> yaw
    transform = np.eye(4)
    transform[:3, :3] = R
    mesh.apply_transform(transform)
    return mesh

def get_projection(mesh, azimuth, elevation, distance=3.0, target_rotation=(0, 0, 0), fov=60, fast_mode=True):
    """
    优化后的3D模型投影函数
    参数:
      mesh: trimesh模型
      azimuth: 水平角
      elevation: 垂直角
      distance: 目标距离（会通过系数调整）
      target_rotation: 目标自旋转角（俯仰、偏航、滚转，单位度）
      fov: 视场角（默认60度）
      fast_mode: 是否跳过轮廓平滑步骤以加速投影生成（True时仅返回基础投影）
    """
    # 限幅
    azimuth = np.clip(azimuth, -fov/2, fov/2)
    elevation = np.clip(elevation, -fov/2, fov/2)
    
    mesh = mesh.copy()  # 避免修改原始模型
    
    # 如果面片数量太多，则进行简化（当面数大于300时）
    if len(mesh.faces) > 300:
        try:
            mesh = mesh.simplify_quadratic_decimation(300)
        except Exception as e:
            pass
    
    # 进行自旋转
    mesh = rotate_mesh(mesh, *target_rotation)
    
    # 定义输出图像大小及相机内参
    w, h = 512, 512
    f = w / (2 * np.tan(np.radians(fov/2)))
    camera_matrix = np.array([
        [f, 0, w/2],
        [0, f, h/2],
        [0, 0, 1]
    ])
    
    # 根据 azimuth、elevation 计算摄像机位置（增加一个缩放系数以保证模型充分呈现）
    azimuth_rad = np.radians(azimuth)
    elevation_rad = np.radians(elevation)
    base_distance = distance * 5.0
    x = base_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
    y = base_distance * np.sin(elevation_rad)
    z = base_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
    camera_pos = np.array([x, y, z])
    
    # 设定观察目标为原点
    target = np.array([0, 0, 0])
    forward = target - camera_pos
    norm = np.linalg.norm(forward)
    if norm == 0:
        norm = 1
    forward = forward / norm
    right = np.cross(np.array([0, 1, 0]), forward)
    right_norm = np.linalg.norm(right)
    if right_norm == 0:
        right_norm = 1
    right = right / right_norm
    up = np.cross(forward, right)
    
    # 构造视图矩阵
    view_matrix = np.eye(4)
    view_matrix[:3, :3] = np.stack([right, up, -forward])
    view_matrix[:3, 3] = -camera_pos
    mesh.apply_transform(view_matrix)
    
    vertices = mesh.vertices
    points_2d = project_points(vertices, camera_matrix)
    
    # 初始化空白图像
    image = np.zeros((h, w), dtype=np.uint8)
    
    # 按面片中心深度排序，实现简单遮挡
    face_centers = vertices[mesh.faces].mean(axis=1)
    depths = face_centers[:, 2]
    sorted_face_indices = np.argsort(depths)[::-1]  # 从远到近绘制
    
    for idx in sorted_face_indices:
        face = mesh.faces[idx]
        pts = points_2d[face]
        if (np.any(pts[:, 0] >= 0) and np.any(pts[:, 0] < w) and 
            np.any(pts[:, 1] >= 0) and np.any(pts[:, 1] < h) and 
            not np.any(np.isnan(pts)) and not np.any(np.isinf(pts))):
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            pts_int = pts.astype(np.int32)
            if cv2.contourArea(pts_int) > 0:
                # 三角形必定为凸，使用 fillConvexPoly 加速绘制
                cv2.fillConvexPoly(image, pts_int, 255)
    
    if not fast_mode:
        # 平滑轮廓：保留最大轮廓区域，可提高投影效果，但开销较大
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(image)
            cv2.fillPoly(mask, [max_contour], 255)
            image = mask
    return image

def generate_ir_target_intensity(mask, peak_temp=600, falloff_sigma=10000, min_temp=599):
    """
    生成符合红外目标特征的强度值
    mask: 二值掩码
    peak_temp: 目标中心最高温度对应的像素值
    falloff_sigma: 温度衰减系数
    min_temp: 目标边缘最低温度
    """
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return np.zeros_like(mask)
    center_y = np.mean(y_indices)
    center_x = np.mean(x_indices)
    y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
    distances = np.hypot(x - center_x, y - center_y)
    temp_range = peak_temp - min_temp
    intensity = min_temp + temp_range * np.exp(-distances**2 / (2 * falloff_sigma**2))
    intensity = np.clip(intensity, 0, 255).astype(np.uint8)
    return np.where(mask > 0, intensity, 0)

def add_target_to_image(background, target_mask, target_size_ratio=0.1, peak_temp=250, 
                          falloff_sigma=2.0, min_temp=220, fixed_position=False):
    """
    将红外目标添加到背景图像中
    fixed_position: 是否将目标固定在图像中心
    """
    bg_h, bg_w = background.shape
    contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("no contours found in target mask")
    x, y, w, h = cv2.boundingRect(np.concatenate(contours))
    target_size = int(min(bg_w, bg_h) * target_size_ratio)
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    target_mask_resized = cv2.resize(target_mask[y:y+h, x:x+w], (new_w, new_h))
    target_intensity = generate_ir_target_intensity(
        target_mask_resized, 
        peak_temp=peak_temp,
        falloff_sigma=falloff_sigma,
        min_temp=min_temp
    )
    max_x = bg_w - new_w
    max_y = bg_h - new_h
    rand_x = np.random.randint(0, max_x + 1)
    rand_y = np.random.randint(0, max_y + 1)
    result = background.copy()
    mask_region = target_mask_resized > 0
    result[rand_y:rand_y+new_h, rand_x:rand_x+new_w] = np.where(
        mask_region, 
        target_intensity, 
        result[rand_y:rand_y+new_h, rand_x:rand_x+new_w]
    )
    return result, (rand_x, rand_y, new_w, new_h)

def get_target_positions(min_distance=100, max_distance=500, num_positions=10, fov=60):
    """
    生成目标的随机位置（相对于相机原点）
    """
    target_positions = []
    effective_fov = fov * 0.8  # 使用80%的视场范围
    for _ in range(num_positions):
        distance = np.random.uniform(min_distance, max_distance)
        azimuth = np.random.uniform(-effective_fov/2, effective_fov/2)
        elevation = np.random.uniform(-effective_fov/2, effective_fov/2)
        azimuth_rad = np.radians(azimuth)
        elevation_rad = np.radians(elevation)
        x = distance * np.sin(azimuth_rad)
        y = distance * np.sin(elevation_rad)
        z = distance * np.cos(azimuth_rad) * np.cos(elevation_rad)
        if z > 0:
            target_positions.append(np.array([x, y, z]))
    return target_positions

def calculate_target_angles(target_pos, camera_pos=np.array([0, 0, 0])):
    """
    根据目标和相机的相对位置计算方位角和仰角
    """
    relative_pos = target_pos - camera_pos
    distance = np.linalg.norm(relative_pos)
    azimuth = np.degrees(np.arctan2(relative_pos[0], relative_pos[2]))
    elevation = np.degrees(np.arcsin(relative_pos[1] / distance))
    return azimuth, elevation, distance

def is_in_fov(target_pos, fov=60):
    """
    检查目标是否在相机视场内
    """
    azimuth, elevation, _ = calculate_target_angles(target_pos)
    if target_pos[2] <= 0:
        return False
    return (abs(azimuth) <= fov/2) and (abs(elevation) <= fov/2)

def process_and_save_projection(args):
    """
    辅助函数，用于并行处理目标投影生成任务
    参数：
      args: (mesh, background, target_pos, rotation, i, j, save_dir)
    """
    mesh, background, target_pos, rotation, i, j, save_dir = args
    azimuth, elevation, distance = calculate_target_angles(target_pos)
    try:
        mask = get_projection(
            mesh.copy(),
            azimuth,
            elevation,
            distance=distance / 200,  # 调整距离比例
            target_rotation=rotation
        )
        result, (x, y, w, h) = add_target_to_image(
            background,
            mask,
            target_size_ratio=0.1,
            peak_temp=250,
            falloff_sigma=1.2,
            min_temp=200
        )
        filename = f'result_pos{i}_rot{j}_x{target_pos[0]:.0f}_y{target_pos[1]:.0f}_z{target_pos[2]:.0f}.png'
        output_path = os.path.join(save_dir, filename)
        cv2.imwrite(output_path, result)
        return f"save image: {filename}, target position: x={x}, y={y}, w={w}, h={h}"
    except Exception as e:
        return f"error: {e}"

if __name__ == '__main__':
    mesh_path = "/home/guantp/Infrared/MIRST/motive_target_gen/3d_models/air.obj"
    background_path = "/home/guantp/Infrared/MIRST/motive_target_gen/bg_imgs/GERGB-20250110-153334611_50_30s_4.jpg"
    
    # 预先读取背景图像
    background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
    if background is None:
        raise ValueError(f"cannot read background image: {background_path}")
    
    mesh = trimesh.load(mesh_path, force='mesh')
    print(f"original model info:")
    print(f"vertex number: {len(mesh.vertices)}")
    print(f"face number: {len(mesh.faces)}")
    print(f"model size: {mesh.extents}")
    
    mesh.fix_normals()
    mesh = normalize_mesh(mesh)
    print(f"\nnormalized model info:")
    print(f"model size: {mesh.extents}")
    print(f"model center: {mesh.centroid}")
    
    save_dir = '/home/guantp/Infrared/MIRST/motive_target_gen/3d_projection_images'
    if os.path.exists(save_dir):
        for file in os.listdir(save_dir):
            file_path = os.path.join(save_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"delete {file_path} error: {e}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"\noutput folder is cleaned: {save_dir}")
    
    target_positions = get_target_positions(
        min_distance=200,
        max_distance=800,
        num_positions=10  # 生成更多目标时可加速利用并行
    )
    
    target_rotations = [
        (0, 0, 0),       # 无旋转
        (30, 0, 0),      # 俯仰30度
        (0, 45, 0),      # 偏航45度
        (0, 0, 90),      # 滚转90度
    ]
    
    # 组合所有投影任务参数
    tasks = []
    for i, target_pos in enumerate(target_positions):
        for j, rotation in enumerate(target_rotations):
            tasks.append((mesh, background, target_pos, rotation, i, j, save_dir))
    
    # 利用并行处理加速任务
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_and_save_projection, tasks))
    for res in results:
        print(res)