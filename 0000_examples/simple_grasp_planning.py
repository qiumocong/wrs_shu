from __future__ import annotations

from typing import Any

import open3d as o3d
import numpy as np
import trimesh

import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import robot_sim.end_effectors.gripper.dh76.dh76 as dh76



def generate_pcd_from_mesh(obj_path: str|o3d.Path, voxel_size=0.002) -> o3d.geometry.PointCloud:
    """
    通过读取网格文件生成点云，并进行下采样。
    返回 open3d 点云对象。
    """
    mesh = o3d.io.read_triangle_mesh(obj_path) # 读取网格文件
    mesh.compute_vertex_normals() # 计算顶点法线
    pcd = mesh.sample_points_uniformly(number_of_points=100000) # 从网格均匀采样10万个点
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size) # 体素下采样
    return pcd_down


def get_detector(base, pcd_np, pcd_normal, pcd_scale, pnt_id=None, diameter=0.01, show=False) -> gm.GeometricModel:
    """
    在点云的指定表面点处生成一个圆柱形检测器。
    """
    surface_pnt = pcd_np[pnt_id]
    surface_normal = -pcd_normal[pnt_id]
    detector = gm.gen_cylinder(radius=diameter, height=pcd_scale,
                               homomat=rm.homomat_from_posrot(surface_pnt, rm.rotmat_from_normal(surface_normal)))
    detector.objtrm.export("tst.stl")
    d = gm.GeometricModel("tst.stl")
    if show:
        d.set_rgba([1, 0, 0, 0.5])
        d.attach_to(base)
    return d

def find_antipodal(detector, anchor, anchor_id, anchor_normal, target_pnts, target_pnts_id, target_normal,
                   threshold=1.0):
    """
    在目标点集中寻找与锚点形成对抗点对的点。
    返回对抗点列表、对抗点ID列表、对抗点法线列表和对抗点接触向量列表。
    通过检测器判断点是否在物体内部，并根据法线方向判断是否为对抗点对。
    其中，threshold参数用于控制法线夹角的阈值。
    夹角余弦值越接近-1，表示夹角越接近180度。
    """
    checker = trimesh.base.ray.ray_pyembree.RayMeshIntersector(detector.objtrm)
    inside_points = []
    inside_points_id = []
    inside_normal = []
    anti_points = []
    anti_points_id = []
    anti_normal = []
    anti_contact_vectors = []
    try:
        inner_tf_list = checker.contains_points(target_pnts)
    except Exception as e:
        return [], [], [], []
    for i, item in enumerate(inner_tf_list):
        if item:
            inside_points.append(target_pnts[i])
            inside_points_id.append(target_pnts_id[i])
            inside_normal.append(target_normal[i])
            contact_vector = rm.unit_vector(target_pnts[i] - anchor)
            judge = rm.unit_vector(target_normal[i]).dot(rm.unit_vector(anchor_normal)) <= -threshold
            if judge:
                anti_points.append([anchor, target_pnts[i]])
                anti_points_id.append([anchor_id, target_pnts_id[i]])
                anti_normal.append([anchor_normal, target_normal[i]])
                anti_contact_vectors.append(contact_vector)
    return anti_points, anti_points_id, anti_normal, anti_contact_vectors

def get_grasp_info_dict(antipodal_points: list,
                        anti_contact_vectors: list,
                        gripper: dh76.Dh76, obj: cm.CollisionModel,
                        base: wd.World, show: bool) -> list[Any]:
    """
    判断一个点和对应点组成的抓取对是否合法。
    生成抓取信息列表
    """
    grasp_pairs = []
    for i, ap in enumerate(antipodal_points):
        contact_vector = anti_contact_vectors[i]
        jaw_center_rotmat = rm.rotmat_between_vectors(np.array([1, 0, 0]), contact_vector)
        jaw_width = np.linalg.norm(ap[0] - ap[1]) * 1.2  # 夹爪张开宽度略大于对抗点距离
        grasp_center_pos = (ap[0]+ap[1]) / 2
        if jaw_width > 0.076:  # 假设夹爪最大张开宽度为0.076米
            continue
        gripper.grip_at_with_jcpose(gl_jaw_center_pos=grasp_center_pos, gl_jaw_center_rotmat=jaw_center_rotmat, jaw_width=jaw_width)
        # if gripper.is_mesh_collided([obj]):
        #     continue
        if show:
            gripper.gen_meshmodel(rgba=[0, 1, 0, 0.3]).attach_to(base)
        grasp_pairs.append(ap)
    return grasp_pairs

def get_pcd_max_scale(pcd_np):
    x_scale = np.min(pcd_np[:, 0]) - np.max(pcd_np[:, 0])
    y_scale = np.min(pcd_np[:, 1]) - np.max(pcd_np[:, 1])
    z_scale = np.min(pcd_np[:, 2]) - np.max(pcd_np[:, 2])
    max_scale = np.max(np.abs([x_scale, y_scale, z_scale]))
    return max_scale

def main():
    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)

    obj_path = "./objects/box.stl"  # 替换为你的物体模型路径
    pcd = generate_pcd_from_mesh(obj_path, voxel_size=0.01)
    pcd_np = np.asarray(pcd.points)
    pcd_normal = np.asarray(pcd.normals)
    pcd_scale = get_pcd_max_scale(pcd_np)
    obj_mesh = cm.CollisionModel(obj_path)
    obj_mesh.attach_to(base)

    gripper = dh76.Dh76(fingertip_type='r_76')

    for i, pnt in enumerate(pcd_np):
        detector = get_detector(base, pcd_np, pcd_normal, pcd_scale, i, diameter=0.01, show=True)
        antipodal_points, antipodal_points_id, antipodal_normals, antipodal_contact_vectors = find_antipodal(
            detector, pnt, i, pcd_normal[i], pcd_np, list(range(len(pcd_np))), pcd_normal, threshold=0.95)
        grasp_info_dict = get_grasp_info_dict(antipodal_points, antipodal_contact_vectors, gripper, obj_mesh, base, show=True)

        base.run()


if __name__ == "__main__":
    main()