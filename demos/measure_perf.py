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


def perform_gpu_ray_intersect(gpu_scene, query_data):
    interactions = fcpw.gpu_interaction_list()
    num_queries = query_data["origin"].shape[0]
    ray_distance_bounds = np.ones(num_queries, dtype=np.float32) * 1e8
    gpu_scene.intersect(query_data["origin"], query_data["dirs"], ray_distance_bounds, interactions, check_for_occlusion=True)


def gen_cone_queries(gpu_scene, positions, num_queries):
    # np.random.seed(120)
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
            max_cos_half_angle.append(interactions[i].maxCos)
            # max_cos_half_angle.append(0.0)
            # print("hit")
        else:
            p_hit.append(query_points[i] + query_dirs[i] * 1.0)
            plane_far.append([-query_dirs[i][0], -query_dirs[i][1], -query_dirs[i][2], -np.dot(-query_dirs[i], query_points[i] + query_dirs[i] * dist_far)])
            max_cos_half_angle.append(0.0)
            # print("miss")            

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
    # parser.add_argument("--use_gpu", action="store_true", help="use GPU")
    args = parser.parse_args()

    # load obj file
    positions, indices = load_obj("dragon.obj")
    # positions, indices = load_obj("bunny_simple_normalized.obj")

    # load fcpw scene
    scene = load_fcpw_scene(positions, indices, False) # NOTE: must build non-vectorized CPU BVH

    # transfer scene to GPU
    fcpw_directory_path = str(Path.cwd().parent)
    print_stats = True
    gpu_scene = fcpw.gpu_scene_3D(fcpw_directory_path, print_stats)
    gpu_scene.transfer_to_gpu(scene)

    # generate random query points for closest point queries
    num_queries = 1000000
    query_data = gen_cone_queries(gpu_scene, positions, num_queries)

    print("======== Ray Intersect ========")
    perform_gpu_ray_intersect(gpu_scene, query_data)

    print("======== Cone ========")
    perform_gpu_min_cone_queries(gpu_scene, query_data, False)


if __name__ == "__main__":
    main()