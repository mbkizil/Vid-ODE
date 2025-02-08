import matplotlib
matplotlib.use('Agg')

import torch
from torchvision.utils import save_image
import os
import wandb
import utils
import imageio


def save_test_images(opt, preds, batch_dict, path, index):
    preds = preds.cpu().detach()
    if opt.dataset == 'hurricane':
        gt = batch_dict['orignal_data_to_predict'].cpu().detach()
    else:
        gt = batch_dict['data_to_predict'].cpu().detach()

    b, t, c, h, w = gt.shape
    
    if opt.input_norm:
        preds = utils.denorm(preds)
        gt = utils.denorm(gt)
    
    os.makedirs(os.path.join(path, 'pred'), exist_ok=True)
    os.makedirs(os.path.join(path, 'gt'), exist_ok=True)
    
    for i in range(b):
        for j in range(t):
            save_image(preds[i, j, ...], os.path.join(path, 'pred', f"pred_{index + i:03d}_{j:03d}.png"))
            save_image(gt[i, j, ...], os.path.join(path, 'gt', f"gt_{index + i:03d}_{j:03d}.png"))


def make_save_sequence(opt, batch_dict, res):
    """ 4 cases: (interp, extrap) | (regular, irregular) """
    
    b, t, c, h, w = batch_dict['observed_data'].size()

    # Filter out / Select by mask
    if opt.irregular:
        observed_mask = batch_dict["observed_mask"]
        mask_predicted_data = batch_dict["mask_predicted_data"]
        selected_timesteps = int(observed_mask[0].sum())
        
        
        if opt.dataset in ['hurricane']:
            batch_dict['observed_data'] = batch_dict['observed_data'][observed_mask.squeeze(-1).byte(), ...].view(b, selected_timesteps, c, h, w)
            batch_dict['data_to_predict'] = batch_dict['data_to_predict'][mask_predicted_data.squeeze(-1).byte(), ...].view(b, selected_timesteps, c, h, w)
        else:
            batch_dict['observed_data'] = batch_dict['observed_data'] * observed_mask.unsqueeze(-1).unsqueeze(-1)
            batch_dict['data_to_predict'] = batch_dict['data_to_predict'] * mask_predicted_data.unsqueeze(-1).unsqueeze(-1)
        
    # Make sequence to save
    pred = res['pred_y'].cpu().detach()
    
    if opt.extrap:
        inputs = batch_dict['observed_data'].cpu().detach()
        gt_to_predict = batch_dict['data_to_predict'].cpu().detach()
        gt = torch.cat([inputs, gt_to_predict], dim=1)
    else:
        gt = batch_dict['data_to_predict'].cpu().detach()

    time_steps = None

    if opt.input_norm:
        gt = utils.denorm(gt)
        pred = utils.denorm(pred)
    
    return gt, pred, time_steps


import os
import torch
import wandb
from torchvision.utils import save_image
import imageio

def save_extrap_images(opt, gt, pred, path, total_step):
    """
    Creates merged videos for ground truth, prediction, and difference.
    
    Assumptions:
      - gt: tensor of shape (B, 10, C, H, W)
      - pred: tensor of shape (B, 5, C, H, W)
    
    For each sample, the new prediction becomes:
        new_pred = gt[:5] + pred
    (i.e. first 5 frames are taken from gt, then the next 5 from pred, for a total of 10 frames.)
    
    Then, for each frame index 0..9, the corresponding frames from up to 4 samples are
    merged horizontally into one frame. Finally, the videos are written to disk, logged to wandb,
    and then removed from the local disk.
    """
    # Ensure tensors are on CPU and detached.
    gt = gt.cpu().detach()    # shape: (B, 10, C, H, W)
    pred = pred.cpu().detach()  # shape: (B, 5, C, H, W)
    
    B, T_gt, C, H, W = gt.shape  # T_gt should be 10
    _, T_pred, _, _, _ = pred.shape  # T_pred should be 5
    num_samples = min(B, 4)  # Process up to 4 samples

    # Create a new prediction sequence for each sample by concatenating gt[:5] and pred.
    new_preds = []  # Will have shape (num_samples, 10, C, H, W)
    for i in range(num_samples):
        new_pred = torch.cat([gt[i, :5], pred[i]], dim=0)  # (5 + 5 = 10 frames)
        new_preds.append(new_pred.unsqueeze(0))
    new_preds = torch.cat(new_preds, dim=0)  # shape: (num_samples, 10, C, H, W)
    
    # For each of the 10 frames, horizontally merge the frames from each sample.
    merged_gt_frames = []    # For ground truth frames.
    merged_pred_frames = []  # For new prediction frames.
    merged_diff_frames = []  # For difference frames (abs(gt - new_pred)).
    
    for frame_idx in range(T_gt):  # 0 .. 9
        gt_frames = []    # List of ground truth frames from each sample.
        pred_frames = []  # List of new prediction frames from each sample.
        diff_frames = []  # List of difference frames from each sample.
        for i in range(num_samples):
            gt_frame = gt[i, frame_idx]            # shape: (C, H, W)
            pred_frame = new_preds[i, frame_idx]     # shape: (C, H, W)
            diff_frame = torch.abs(gt_frame - pred_frame)
            gt_frames.append(gt_frame)
            pred_frames.append(pred_frame)
            diff_frames.append(diff_frame)
        # Merge frames horizontally (concatenate along the width dimension, dim=2)
        merged_gt = torch.cat(gt_frames, dim=2)    
        merged_pred = torch.cat(pred_frames, dim=2)
        merged_diff = torch.cat(diff_frames, dim=2)
        # Add a time dimension
        merged_gt_frames.append(merged_gt.unsqueeze(0))
        merged_pred_frames.append(merged_pred.unsqueeze(0))
        merged_diff_frames.append(merged_diff.unsqueeze(0))
    
    # Stack all frames to form video tensors of shape (10, C, H, merged_width)
    merged_gt_video = torch.cat(merged_gt_frames, dim=0)
    merged_pred_video = torch.cat(merged_pred_frames, dim=0)
    merged_diff_video = torch.cat(merged_diff_frames, dim=0)
    
    # Helper function: Convert video tensor (T, C, H, W) to numpy array (T, H, W, C)
    def tensor_to_np(video_tensor):
        # If opt.input_norm is True, you might want to denormalize here:
        # video_tensor = utils.denorm(video_tensor)
        # For this example, we assume the values are already in [0, 1].
        video_np = (video_tensor * 255).clamp(0, 255).byte()
        video_np = video_np.permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, C)
        return video_np
    
    gt_video_np = tensor_to_np(merged_gt_video)
    pred_video_np = tensor_to_np(merged_pred_video)
    diff_video_np = tensor_to_np(merged_diff_video)
    
    # Helper: Write numpy video array to an MP4 file using imageio.
    def write_video(video_np, filename, fps=4):
        writer = imageio.get_writer(filename, fps=fps)
        for frame in video_np:
            writer.append_data(frame)
        writer.close()
    
    # Define filenames.
    gt_video_filename = os.path.join(path, f"gt_video_{total_step+1}.mp4")
    pred_video_filename = os.path.join(path, f"pred_video_{total_step+1}.mp4")
    diff_video_filename = os.path.join(path, f"diff_video_{total_step+1}.mp4")
    
    # Write the videos.
    write_video(gt_video_np, gt_video_filename)
    write_video(pred_video_np, pred_video_filename)
    write_video(diff_video_np, diff_video_filename)
    
    # Log the videos to wandb under fixed names.
    wandb.log({
        "extrap_video_gt": wandb.Video(gt_video_filename),
        "extrap_video_pred": wandb.Video(pred_video_filename),
        "extrap_video_diff": wandb.Video(diff_video_filename)
    }, step=total_step+1)
    
    # Delete the video files locally.
    os.remove(gt_video_filename)
    os.remove(pred_video_filename)
    os.remove(diff_video_filename)


def save_interp_images(opt, gt, pred, path, total_step):
    
    pred = pred.cpu().detach()
    data = gt.cpu().detach()
    b, t, c, h, w = data.shape
    
    save_me = []
    for i in range(min([b, 4])):  # save only 4 items
        row = torch.cat([data[i], pred[i]], dim=0)
        if opt.input_norm:
            row = utils.denorm(row)
        if row.size(1) == 1:
            row = row.repeat(1, 3, 1, 1)
        save_me += [row]
    save_me = torch.cat(save_me, dim=0)
    save_image(save_me, os.path.join(path, f"image_{(total_step + 1):08d}.png"), nrow=t)


if __name__ == '__main__':
    pass
