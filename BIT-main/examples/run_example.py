"""
Demo deform.
Deform template mesh based on input silhouettes and camera pose
"""
import time
import json
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np
import imageio
import argparse

import subprocess
import threading

import sys
current_dir = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_dir)
pp_dir = os.path.dirname(parent_dir)
sys.path.append(pp_dir)

import soft_renderer.functional as srf
import soft_renderer as sr
import utils as ut
import communication
import sam_utils as su
import cv2
from validation_selection import ValidationBuffer, pose_signature



from segment_anything import sam_model_registry, SamPredictor
from utils import load_config, pjoin


current_dir = os.path.dirname(os.path.realpath(__file__))

data_dir = os.path.join(current_dir, '../data')

import random  
def set_seed(seed):  
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    if torch.cuda.is_available():  
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    print("shpe anns", type(anns))

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def show_sam_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3)*255, [1]])
        img[m] = color_mask
    img = img[:,:,:3]  
    img = img.astype(np.uint8)  
    # cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
class Model(nn.Module):
    def __init__(self, template_path):
        super(Model, self).__init__()

        # set template mesh
        self.template_mesh = sr.Mesh.from_obj(template_path, normalization=False)
        self.register_buffer('vertices', self.template_mesh.vertices)
        self.register_buffer('faces', self.template_mesh.faces)
        self.register_buffer('textures', self.template_mesh.textures)

        # optimize for displacement map and center
        self.register_parameter('displace', nn.Parameter(torch.zeros_like(self.template_mesh.vertices)))
        self.register_parameter('center', nn.Parameter(torch.zeros(1, 1, 3)))

        # define Laplacian and flatten geometry constraints
        self.laplacian_loss = sr.LaplacianLoss(self.vertices[0].cpu(), self.faces[0].cpu())
        self.flatten_loss = sr.FlattenLoss(self.faces[0].cpu())
        
        print('vertices shape', self.template_mesh.vertices.shape)	# [1 1032 3]
        print('faces shape', self.template_mesh.faces.shape)	# [1 2600 3]
       
    
    def resetMeshFromObj(self, template_path):
        # set template mesh
        self.template_mesh = sr.Mesh.from_obj(template_path)
        self.register_buffer('vertices', self.template_mesh.vertices * 1.0)
        self.register_buffer('faces', self.template_mesh.faces)
        self.register_buffer('textures', self.template_mesh.textures)

        # optimize for displacement map and center
        self.register_parameter('displace', nn.Parameter(torch.zeros_like(self.template_mesh.vertices)))
        self.register_parameter('center', nn.Parameter(torch.zeros(1, 1, 3)))

        # define Laplacian and flatten geometry constraints
        self.laplacian_loss = sr.LaplacianLoss(self.vertices[0].cpu(), self.faces[0].cpu())
        self.flatten_loss = sr.FlattenLoss(self.faces[0].cpu())
    
    def forward(self, batch_size):
        base = torch.log(self.vertices.abs() / (1 - self.vertices.abs()))
        centroid = torch.tanh(self.center)
        vertices = torch.sigmoid(base + self.displace) * torch.sign(self.vertices)
        vertices = F.relu(vertices) * (1 - centroid) - F.relu(-vertices) * (centroid + 1)
        vertices = vertices + centroid
        
        # apply Laplacian and flatten geometry constraints
        laplacian_loss = self.laplacian_loss(vertices).mean()
        flatten_loss = self.flatten_loss(vertices).mean()

        return sr.Mesh(vertices.repeat(batch_size, 1, 1),
                       self.faces.repeat(batch_size, 1, 1)), laplacian_loss, flatten_loss
     
    def get_face_normals(self, vertices, faces):
        face_normals = []
        for batch_idx in range(faces.shape[0]):
            batch_faces = faces[batch_idx]
            batch_vertices = vertices[batch_idx]
    
            v0 = batch_vertices[batch_faces[:, 0]]
            v1 = batch_vertices[batch_faces[:, 1]]
            v2 = batch_vertices[batch_faces[:, 2]]
    
            normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
            normals = normals / torch.norm(normals, dim=-1, keepdim=True)
            
            face_normals.append(normals)
        
        face_normals = torch.stack(face_normals)
        return face_normals
    
def neg_iou_loss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])  # ndimension = 3
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6

    with open('./loss_iou2.txt', 'a') as f:
         f.write(f"{predict.sum()} '\t' {target.sum()} '\t'  {intersect.sum()} {union.sum()} '\t'  {1. - (intersect / union).sum() / intersect.nelement()}\n")

    return 1. - (intersect / union).sum() / intersect.nelement()

def soft_iou_loss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6
    return 1. - (intersect / union).mean()


@torch.no_grad()
def evaluate_model_gate_metrics(
    model, masks, poses, intrinsics, transform, lighting, rasterizer,
    rotation_delta_deg, translation_delta,
):
    if len(masks) == 0:
        return None

    device = model.vertices.device
    poses = poses.to(device=device, dtype=torch.float32)
    intrinsics = intrinsics.to(device=device, dtype=torch.float32)
    targets = torch.from_numpy(masks).to(device=device, dtype=torch.float32)[:, 3]

    def render_losses(eval_poses, eval_intrinsics, eval_targets):
        transform.set_K_list(eval_intrinsics)
        transform.set_T(eval_poses)
        mesh, _, _ = model(len(eval_poses))
        predictions = rasterizer(transform(lighting(mesh)))[:, 3]
        dims = tuple(range(1, predictions.ndimension()))
        intersection = (predictions * eval_targets).sum(dims)
        union = (
            predictions + eval_targets - predictions * eval_targets
        ).sum(dims) + 1e-6
        return 1.0 - intersection / union

    nominal_losses = render_losses(poses, intrinsics, targets)
    angle = np.deg2rad(rotation_delta_deg)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    deltas = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            delta = torch.eye(4, device=device, dtype=torch.float32)
            i, j = (1, 2) if axis == 0 else ((0, 2) if axis == 1 else (0, 1))
            signed_s = sign * s
            delta[i, i], delta[j, j] = c, c
            delta[i, j], delta[j, i] = -signed_s, signed_s
            deltas.append(delta)
    for axis in range(3):
        for sign in (-1.0, 1.0):
            delta = torch.eye(4, device=device, dtype=torch.float32)
            delta[axis, 3] = sign * translation_delta
            deltas.append(delta)

    perturbations = torch.stack(deltas)
    perturbed_loss_columns = []
    for delta in perturbations:
        perturbed_poses = torch.matmul(delta[None, :, :], poses)
        perturbed_loss_columns.append(
            render_losses(perturbed_poses, intrinsics, targets)
        )
    perturbed_losses = torch.stack(perturbed_loss_columns, dim=1)

    best_perturbed_losses = perturbed_losses.min(dim=1).values
    pose_correction_gain = torch.clamp(
        nominal_losses - best_perturbed_losses, min=0.0
    )
    return {
        'mean_iou_loss': float(nominal_losses.mean().item()),
        'temporal_std': float(nominal_losses.std(unbiased=False).item()),
        'pose_inconsistency': float(pose_correction_gain.mean().item()),
        'uncertainty': float(perturbed_losses.std(dim=1, unbiased=False).mean().item()),
        'per_frame_iou_loss': nominal_losses.cpu().tolist(),
    }


def publish_mesh_atomic(mesh, final_path):
    staging_path = os.path.join(
        os.path.dirname(os.path.dirname(final_path)),
        '.publish_{}_{}.tmp.obj'.format(os.getpid(), os.path.basename(final_path)),
    )
    try:
        mesh.save_obj(staging_path, save_texture=False)
        os.replace(staging_path, final_path)
    finally:
        if os.path.exists(staging_path):
            os.remove(staging_path)


def publish_file_atomic(source_path, final_path):
    staging_path = os.path.join(
        os.path.dirname(os.path.dirname(final_path)),
        '.publish_{}_{}.tmp.obj'.format(os.getpid(), os.path.basename(final_path)),
    )
    try:
        shutil.copyfile(source_path, staging_path)
        os.replace(staging_path, final_path)
    finally:
        if os.path.exists(staging_path):
            os.remove(staging_path)


def append_jsonl(path, record):
    with open(path, 'a') as stream:
        stream.write(json.dumps(record, sort_keys=True) + '\n')


def merge_dict(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def save_obj_file(vertices, faces, file_path):
    with open(file_path, 'w') as f:
        f.write(f"#coarse_50.obj\n# \n \n")
        for vertex in vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device) 
    return image.permute(2, 0, 1).contiguous()


    
def main():

    set_seed(int(os.environ.get('BIT_RANDOM_SEED', '42')))

	# RBOT dataset
    # summer_config_file = "./config/rbot/a_regular_ape.yml"
	
    # MOPED dataset
    summer_config_file = pjoin("./config/", sys.argv[1], sys.argv[2]) # 0-4
    # summer_config_file = "./config/moped/black_drill/evaluation/00.yml" # 0-4
    # summer_config_file = "./config/moped/duplo_dude/evaluation/00.yml" # 0-6
    # summer_config_file = "./config/moped/rinse_aid/evaluation/00.yml" # 0-5
    # summer_config_file = "./config/moped/toy_plane/evaluation/00.yml" # 0-4
    # summer_config_file = "./config/moped/vim_mug/evaluation/00.yml" # 0-4

    configs = su.LoadConfigSafety(config_file = summer_config_file)
    print("output_dir: ", configs['output_dir'])
    os.makedirs(configs['output_dir'], exist_ok=True)
    validation_config = configs.get('model_validation', {})
    validation_override = os.environ.get('BIT_MODEL_VALIDATION_JSON')
    if validation_override:
        validation_config = merge_dict(
            validation_config, json.loads(validation_override)
        )
    validation_enabled = bool(validation_config.get('enabled', True))
    consume_once = os.environ.get('BIT_CONSUME_ONCE', '1').lower() not in ('0', 'false', 'no')
    validation_frames = int(validation_config.get('validation_frames', 1))
    validation_min_improvement = float(validation_config.get('min_improvement', 0.01))
    validation_weights = validation_config.get('weights', {})
    validation_iou_weight = float(validation_weights.get('iou', 1.0))
    validation_pose_weight = float(validation_weights.get('pose_consistency', 0.25))
    validation_temporal_weight = float(validation_weights.get('temporal_stability', 0.25))
    validation_uncertainty_weight = float(validation_weights.get('uncertainty', 0.25))
    validation_rotation_delta_deg = float(validation_config.get('rotation_delta_deg', 2.0))
    validation_translation_delta = float(validation_config.get('translation_delta', 0.005))
    validation_max_iou_regression = float(validation_config.get('max_iou_regression', 0.005))
    validation_buffer_size = int(validation_config.get('buffer_size', 32))
    selection_config = validation_config.get('selection', {})
    selection_rotation_weight = float(selection_config.get('rotation_weight', 1.0))
    selection_translation_weight = float(selection_config.get('translation_weight', 100.0))
    selection_quality_weight = float(selection_config.get('quality_weight', 0.25))
    registry_path = os.path.join(
        os.path.dirname(configs['output_dir']), 'model_registry.jsonl'
    )
    candidate_dir = os.path.join(
        os.path.dirname(configs['output_dir']), 'model_candidates'
    )
    os.makedirs(candidate_dir, exist_ok=True)
    with open(registry_path, 'w'):
        pass
    print("model_registry: ", registry_path)
    

    prepare_start_time = time.time()  
    # differentiable renderer
    from_sphere = False	 # Set this to False and consider the original sphere is not a sphere, so that simplify the code logic
    render_size = configs['renderer_img_size']
    img_nums = su.GenerateArray(first_num=configs['reference_imgs_nums'], second_num=configs['estimated_imgs_nums'], len =100) 
    cur_model_deformed_nums = 0
    stable_geometry_version = 0
    max_model_deformed_nums = configs['max_model_deformed_nums'] 
    iteration_nums = configs['iteration_nums_in_each_deform']
    
    batch_size     = configs['batch_size']
    transform      = sr.LookAt(viewing_angle=configs['viewing_angle'])
    lighting       = sr.Lighting()
    rasterizer     = sr.SoftRasterizer(image_size=render_size, sigma_val=1e-4, aggr_func_rgb='hard',near=0.1,far=1000)
    model          = Model(configs['template_mesh']).cuda()
    img_h = configs['height']
    img_w = configs['width']

    # SAM
    sam_checkpoint = pjoin(pp_dir, '..', 'checkpoints', configs['sam']['checkpoint'])
    model_type     = configs['sam']['model']
    device         = configs['sam']['device']
    sam            = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor      = SamPredictor(sam)

    
    save_obj_path = configs['save_obj_path']
    save_index    = configs['save_index']

	# cpp
    lsc = communication.LocalStorageCommunication(
        root=configs['communication']['root'], max_count=img_nums[0],
        from_sphere=from_sphere, consume_once=consume_once,
    )
    # tracker and part of scale
    # here call the eg_BIT
    process = subprocess.Popen([configs['tracker'], summer_config_file])

    loop_nums = 0
    validation_buffer = ValidationBuffer(capacity=validation_buffer_size)
        
    while(True):
        return_code = process.poll()
        if return_code is not None:
            if return_code == 0:
                return
            raise RuntimeError(
                f"Tracker process exited before completion with code {return_code}"
            )

        status = lsc.Scan(sleep_time = 2) # We sleep here to give the scanner time to scan the local folder, so that the C++ and Python can communicate with each other.
        lsc.set_status(status)
        
        dataGroups = lsc.parse_data()
        
        dataGroups.print_info("datgroups scaning")
        lsc.set_base_datagroups(dataGroups=dataGroups)
        if len(dataGroups.rgbs_) == lsc.max_count():
            lsc.add_new_datagroups_2_base(dataGroups=dataGroups)
            dataGroups = lsc.get_dataGroups()
            dataGroups.print_info()

            segmented_masks = su.segment(predictor=predictor, img_width=img_w, img_height=img_h, dataGroups=dataGroups, configs=configs, save_index=save_index)
            lsc.set_max_count(max_count = img_nums[cur_model_deformed_nums+1])
            segmented_masks = ut.FlipMatImgs(segmented_masks, axis = 0)	
            probs = ut.FlipMatImgs(dataGroups.probs_, axis = 0)	
            cnfds = ut.FlipMatImgs(dataGroups.masks_, axis = 0)	
            
            h, w = segmented_masks[0].shape[-2:]
            if w > h:
                segmented_masks = ut.ExtendMatImgs(segmented_masks, top = (w - h))	
                probs = ut.ExtendMatImgs(probs, top = (w - h))	
                cnfds = ut.ExtendMatImgs(cnfds, top = (w - h))	

            probs = ut.ResizeImgs(probs, width = render_size, height = render_size)
            cnfds = ut.ResizeImgs(cnfds, width = render_size, height = render_size)
            segmented_masks = ut.ResizeImgs(segmented_masks, width = render_size, height = render_size)
            for i in range(len(segmented_masks)):
                cv2.imshow("segmented_masks", segmented_masks[i])
                cv2.waitKey(100)

            probs = su.BinaringImgs(probs, 100, 255)
			# Note in this the probs actully is the mask
            probs_npy = ut.Mats2Npy(segmented_masks.copy()).astype('float32')/255
            cnfds_npy = ut.Mats2Npy(cnfds.copy()).astype('float32')/255

            Ts_all = torch.from_numpy(dataGroups.Ts_)
            Ks_all = torch.from_numpy(dataGroups.Ks_).to(dtype=torch.float32)
            candidate_version = cur_model_deformed_nums + 1
            stable_model_path = save_obj_path if cur_model_deformed_nums > 0 else None

            all_indices = list(range(len(probs_npy)))
            validation_buffer.add(probs_npy, dataGroups.Ts_, dataGroups.Ks_)
            validation_records = []
            if validation_enabled and stable_model_path and len(all_indices) >= 3:
                validation_records = validation_buffer.select(
                    validation_frames,
                    rotation_weight=selection_rotation_weight,
                    translation_weight=selection_translation_weight,
                    quality_weight=selection_quality_weight,
                )
            selected_signatures = {
                record['signature'] for record in validation_records
            }
            training_indices = [
                index for index in all_indices
                if pose_signature(dataGroups.Ts_[index]) not in selected_signatures
            ]
            while len(training_indices) < 2 and validation_records:
                validation_records.pop()
                selected_signatures = {
                    record['signature'] for record in validation_records
                }
                training_indices = [
                    index for index in all_indices
                    if pose_signature(dataGroups.Ts_[index]) not in selected_signatures
                ]
            validation_pose_signatures = [
                list(record['signature']) for record in validation_records
            ]

            if stable_model_path:
                model = Model(stable_model_path).cuda()

            optimizer = torch.optim.Adam(model.parameters(), 0.01, betas=(0.5, 0.99))
            transform.set_img_size(max(img_h, img_w))
            training_poses = Ts_all[training_indices]
            training_intrinsics = Ks_all[training_indices]
            training_masks = probs_npy[training_indices]
            transform.set_K_list(training_intrinsics)
            transform.set_T(training_poses)

            loop = tqdm.tqdm(list(range(0, iteration_nums)))
            batch_size = len(training_masks)
            cur_model_deformed_nums = candidate_version
            images_gt = torch.from_numpy(training_masks).cuda()

            print("begin to deform model")
            for i in loop:
                mesh, laplacian_loss, flatten_loss = model(batch_size)

                # render
                mesh = lighting(mesh)
                mesh = transform(mesh)
                images_pred = rasterizer(mesh)
                loss = neg_iou_loss(images_pred[:, 3], images_gt[:, 3]) + 0.1 * laplacian_loss

                loop.set_description('Loss: %.4f' % (loss.item()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                intermediate_result = images_pred.detach().cpu().numpy()[0].transpose((1, 2, 0))
                intermediate_result_flipped = cv2.flip(intermediate_result, 0)
                cv2.imshow("intermediate_result", intermediate_result_flipped)
                cv2.waitKey(1)
                loop_nums = loop_nums + 1

            candidate_mesh = model(1)[0]
            candidate_path = os.path.join(
                candidate_dir, 'candidate_v{:03d}.obj'.format(candidate_version)
            )
            candidate_mesh.save_obj(candidate_path, save_texture=False)

            accepted = True
            stable_validation_loss = None
            candidate_validation_loss = None
            improvement = None
            gate_score = None
            stable_gate_metrics = None
            candidate_gate_metrics = None
            reason = 'initial_model'
            stable_model = None
            base_stable_geometry_version = stable_geometry_version
            if validation_records:
                validation_poses = torch.from_numpy(np.stack([
                    record['pose'] for record in validation_records
                ]))
                validation_intrinsics = torch.from_numpy(np.stack([
                    record['intrinsic'] for record in validation_records
                ])).to(dtype=torch.float32)
                validation_masks = np.stack([
                    record['mask'] for record in validation_records
                ])
                candidate_gate_metrics = evaluate_model_gate_metrics(
                    model, validation_masks, validation_poses,
                    validation_intrinsics, transform, lighting, rasterizer,
                    validation_rotation_delta_deg, validation_translation_delta
                )
                stable_model = Model(stable_model_path).cuda()
                stable_gate_metrics = evaluate_model_gate_metrics(
                    stable_model, validation_masks, validation_poses,
                    validation_intrinsics, transform, lighting, rasterizer,
                    validation_rotation_delta_deg, validation_translation_delta
                )
                candidate_validation_loss = candidate_gate_metrics['mean_iou_loss']
                stable_validation_loss = stable_gate_metrics['mean_iou_loss']
                improvement = stable_validation_loss - candidate_validation_loss
                pose_gain = stable_gate_metrics['pose_inconsistency'] - candidate_gate_metrics['pose_inconsistency']
                temporal_gain = stable_gate_metrics['temporal_std'] - candidate_gate_metrics['temporal_std']
                uncertainty_gain = stable_gate_metrics['uncertainty'] - candidate_gate_metrics['uncertainty']
                gate_score = (
                    validation_iou_weight * improvement
                    + validation_pose_weight * pose_gain
                    + validation_temporal_weight * temporal_gain
                    + validation_uncertainty_weight * uncertainty_gain
                )
                iou_guard_passed = candidate_validation_loss <= (
                    stable_validation_loss + validation_max_iou_regression
                )
                accepted = gate_score >= validation_min_improvement and iou_guard_passed
                if accepted:
                    reason = 'multi_metric_gate_passed'
                elif not iou_guard_passed:
                    reason = 'iou_regression_guard'
                else:
                    reason = 'multi_metric_score_below_threshold'

            published_path = os.path.join(
                configs['output_dir'],
                '_realtime_' + str(candidate_version) + '.obj'
            )
            if accepted:
                publish_mesh_atomic(candidate_mesh, published_path)
            else:
                publish_file_atomic(stable_model_path, published_path)
                model = stable_model
            if accepted:
                stable_geometry_version = candidate_version

            save_obj_path = published_path
            decision = {
                'version': candidate_version,
                'consume_once': consume_once,
                'accepted': accepted,
                'reason': reason,
                'base_stable_geometry_version': base_stable_geometry_version,
                'stable_geometry_version': stable_geometry_version,
                'stable_validation_iou_loss': stable_validation_loss,
                'candidate_validation_iou_loss': candidate_validation_loss,
                'improvement': improvement,
                'gate_score': gate_score,
                'minimum_improvement': validation_min_improvement,
                'stable_gate_metrics': stable_gate_metrics,
                'candidate_gate_metrics': candidate_gate_metrics,
                'gate_weights': {
                    'iou': validation_iou_weight,
                    'pose_consistency': validation_pose_weight,
                    'temporal_stability': validation_temporal_weight,
                    'uncertainty': validation_uncertainty_weight,
                },
                'rotation_delta_deg': validation_rotation_delta_deg,
                'translation_delta': validation_translation_delta,
                'max_iou_regression': validation_max_iou_regression,
                'training_indices': training_indices,
                'validation_pose_signatures': validation_pose_signatures,
                'validation_buffer_size': len(validation_buffer),
                'candidate_path': candidate_path,
                'published_path': published_path,
            }
            append_jsonl(registry_path, decision)
            print("model_validation: ", json.dumps(decision, sort_keys=True))
            
            if cur_model_deformed_nums == max_model_deformed_nums:
                return_code = process.wait()
                if return_code != 0:
                    raise RuntimeError(f"Tracker process exited with code {return_code}")
                return

    
if __name__ == '__main__':
    main()
