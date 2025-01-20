'''
This demo demonstrates how to perform closest point queries using FCPW.
The full FCPW API can be viewed using the following commands in the Python console:
>>> import fcpw
>>> help(fcpw)
'''

import numpy as np
import polyscope as ps
import fcpw
import argparse
from pathlib import Path
from typing import Callable, Optional

def load_obj(obj_file_path):
    positions = []
    indices = []

    with open(obj_file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                position = list(map(float, line.strip().split()[1:]))
                positions.append(np.array(position, dtype=np.float32, order='C'))

            elif line.startswith('f '):
                index = [int(idx.split('/')[0]) - 1 for idx in line.strip().split()[1:]]
                indices.append(np.array(index, dtype=np.int32, order='C'))

    return np.array(positions), np.array(indices)

def load_fcpw_scene(positions, indices, build_vectorized_cpu_bvh):
    # load positions and indices
    scene = fcpw.scene_3D()
    scene.set_object_count(1)
    scene.set_object_vertices(positions, 0)
    scene.set_object_triangles(indices, 0)

    # build scene on CPU
    aggregate_type = fcpw.aggregate_type.bvh_surface_area
    print_stats = False
    reduce_memory_footprint = False
    scene.build(aggregate_type, build_vectorized_cpu_bvh,
                print_stats, reduce_memory_footprint)

    return scene

# def perform_closest_point_queries(scene, query_points):
#     # perform cpqs
#     squared_max_radii = np.inf * np.ones(len(query_points), dtype=np.float32)
#     interactions = fcpw.interaction_3D_list()
#     scene.find_closest_points(query_points, squared_max_radii, interactions)

#     # extract closest points
#     closest_points = np.array([i.p for i in interactions])

#     return closest_points

# def perform_gpu_closest_point_queries(gpu_scene, query_points):
#     # perform cpqs on GPU
#     squared_max_radii = np.inf * np.ones(len(query_points), dtype=np.float32)
#     interactions = fcpw.gpu_interaction_list()
#     gpu_scene.find_min_cones(query_points, squared_max_radii, interactions)

#     # extract closest points
#     closest_points = np.array([np.array([i.p.x, i.p.y, i.p.z], dtype=np.float32, order='C') for i in interactions])

#     return closest_points

def perform_gpu_min_cone_queries(gpu_scene, query_data, brute_force=False):
    # perform cpqs on GPU
    
    interactions = fcpw.gpu_interaction_list()
    gpu_scene.find_min_cones(query_data["origin"], query_data["dirs"], query_data["max_cos_half_angle"], query_data["plane_near"], query_data["plane_far"], interactions, brute_force)

    # extract closest points
    closest_points = np.array([np.array([i.p.x, i.p.y, i.p.z], dtype=np.float32, order='C') for i in interactions])
    # print(closest_points)

    return closest_points

def gui_callback(scene, query_data, use_gpu):
    # animate query points
    # for q in query_points:
        # q[0] += 0.001 * np.sin(10.0 * q[1])
        # q[1] += 0.001 * np.cos(10.0 * q[0])

    query_points = query_data["origin"]
    p_hit = query_data["p_hit"]

    # perform closest point queries
    closest_points = None
    if use_gpu:
        closest_points = perform_gpu_min_cone_queries(scene, query_data, False)
        closest_points_ref = perform_gpu_min_cone_queries(scene, query_data, True)
    else:
        pass
        # closest_points = perform_closest_point_queries(scene, query_points)

    diff = closest_points - closest_points_ref
    diff = np.linalg.norm(diff, axis=1)
    acc = np.sum(diff < 1e-6) / len(diff)
    print(acc)

    # plot results
    # query_dir_end_points = query_points + query_dirs * 1.0
    ps.register_point_cloud("cone origin", query_points)
    # ps.register_point_cloud("cone hit", p_hit)
    ps.register_point_cloud("closest points", closest_points)
    ps.register_point_cloud("closest points ref", closest_points_ref)
    edge_positions = np.concatenate([query_points, p_hit, closest_points], axis=0)
    edge_indices_1 = np.array([[i, i + len(query_points)] for i in range(len(query_points))])
    edge_indices_2 = np.array([[i, i + 2 * len(query_points)] for i in range(len(query_points))])
    network1 = ps.register_curve_network("dirs", edge_positions, edge_indices_1)
    network2 = ps.register_curve_network("closest silhouettes", edge_positions, edge_indices_2)
    network1.set_radius(0.003, relative=False)
    network2.set_radius(0.005, relative=False)

    ref_edge_positions = np.concatenate([query_points, closest_points_ref], axis=0)
    edge_indices_3 = np.array([[i, i + len(query_points)] for i in range(len(query_points))])
    network3 = ps.register_curve_network("closest silhouettes ref", ref_edge_positions, edge_indices_3)
    network3.set_radius(0.005, relative=False)

def visualize(scene, positions, indices, query_data, use_gpu):
    # initialize polyscope
    ps.init()
    ps.set_ground_plane_mode("none")

    # register mesh and callback
    ps.register_surface_mesh("mesh", positions, indices)
    gui_callback_no_args = lambda: gui_callback(scene, query_data, use_gpu)
    ps.set_user_callback(gui_callback_no_args)

    # give control to polyscope gui
    ps.show()

def gen_cone_queries(gpu_scene, positions, num_queries):
    np.random.seed(120)
    box_min = np.min(positions, axis=0)
    box_max = np.max(positions, axis=0)
    box_center = (box_min + box_max) * 0.5
    box_size = box_max - box_min
    dist_far = np.linalg.norm(box_max - box_min) * 2.0

    query_dirs = np.random.uniform(-1.0, 1.0, (num_queries, 3)).astype(np.float32)
    query_dirs /= np.linalg.norm(query_dirs, axis=1)[:, None]    
    
    query_points = box_center + query_dirs * dist_far * 0.2

    target = np.random.uniform(box_center - box_size * 0.3, box_center + box_size * 0.3, (num_queries, 3)).astype(np.float32)
    query_dirs = target - query_points
    query_dirs /= np.linalg.norm(query_dirs, axis=1)[:, None] 

    # Try ray intersection
    interactions = fcpw.gpu_interaction_list()
    ray_distance_bounds = np.ones(num_queries, dtype=np.float32) * dist_far
    gpu_scene.intersect(query_points, query_dirs, ray_distance_bounds, interactions)
    
    max_cos_half_angle = []
    p_hit = []
    plane_near = []
    plane_far = []
    for i in range(num_queries):
        plane_near.append([query_dirs[i][0], query_dirs[i][1], query_dirs[i][2], -np.dot(query_dirs[i], query_points[i])])
        if interactions[i].index < 100000000:
            p_hit.append([interactions[i].p.x, interactions[i].p.y, interactions[i].p.z])

            orientation = interactions[i].n.x * query_dirs[i][0] + interactions[i].n.y * query_dirs[i][1] + interactions[i].n.z * query_dirs[i][2]
            if orientation > 0.0:
                interactions[i].n.x *= -1.0
                interactions[i].n.y *= -1.0
                interactions[i].n.z *= -1.0

            plane_d = -(interactions[i].n.x * interactions[i].p.x + interactions[i].n.y * interactions[i].p.y + interactions[i].n.z * interactions[i].p.z)
            plane_far.append([interactions[i].n.x, interactions[i].n.y, interactions[i].n.z, plane_d])
            # print("hit")
        else:
            p_hit.append(query_points[i] + query_dirs[i] * 1.0)
            plane_far.append([-query_dirs[i][0], -query_dirs[i][1], -query_dirs[i][2], -np.dot(-query_dirs[i], query_points[i] + query_dirs[i] * dist_far)])
            # print("miss")

        max_cos_half_angle.append(0.0)

        # test = np.dot(plane_far[i][:3], p_hit[i]) + plane_far[i][3]
        # print(test)

    p_hit = np.array(p_hit, dtype=np.float32)
    plane_near = np.array(plane_near, dtype=np.float32)
    plane_far = np.array(plane_far, dtype=np.float32)
    max_cos_half_angle = np.array(max_cos_half_angle, dtype=np.float32)

    # print(p_hit)
    # print(plane_near)
    # print(plane_far)
    # exit(0)

    query_data = {
        "origin": query_points,
        "dirs": query_dirs,
        "p_hit": p_hit,
        "max_cos_half_angle": max_cos_half_angle,
        "plane_near": plane_near,
        "plane_far": plane_far,
    }
    return query_data

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="fcpw demo")
    parser.add_argument("--use_gpu", action="store_true", help="use GPU")
    args = parser.parse_args()

    # load obj file
    positions, indices = load_obj("bunny_simple_normalized.obj")

    box_min = np.min(positions, axis=0)
    box_max = np.max(positions, axis=0)
    #query_points = np.random.uniform(box_min, box_max, (num_query_points, 3)).astype(np.float32)
    
    # query_points = np.reshape(box_max, (1, 3)) + 0.01
    # query_dirs = np.array([[-1.0, -0.01, -0.01]], dtype=np.float32)

    if args.use_gpu:
        # load fcpw scene
        scene = load_fcpw_scene(positions, indices, False) # NOTE: must build non-vectorized CPU BVH

        # transfer scene to GPU
        fcpw_directory_path = str(Path.cwd().parent)
        print_stats = True
        gpu_scene = fcpw.gpu_scene_3D(fcpw_directory_path, print_stats)
        gpu_scene.transfer_to_gpu(scene)

        # generate random query points for closest point queries
        num_queries = 20
        query_data = gen_cone_queries(gpu_scene, positions, num_queries)

        # visualize scene
        visualize(gpu_scene, positions, indices, query_data, True)

    else:
        pass
        # load fcpw scene
        # scene = load_fcpw_scene(positions, indices, True)

        # visualize scene
        # visualize(scene, positions, indices, query_points, False)

if __name__ == "__main__":
    main()