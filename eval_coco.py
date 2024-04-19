import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torchvision import utils as vutils
from torchvision.utils import save_image

from dataset import COCO2017
from object_discovery.slot_attention_model import SlotAttentionModel
from object_discovery.params import (
    merge_namespaces,
    training_coco_params,
    slot_attention_coco_params,

)
from object_discovery.utils import (
    to_rgb_from_tensor,
    warm_and_decay_lr_scheduler,
    cosine_anneal,
    linear_warmup,
    visualize,
    compute_ari,
    sa_segment,
    rescale,
    get_largest_objects,
    cmap_tensor,
    inv_normalize,
)
from ocl_metrics import UnsupervisedMaskIoUMetric, ARIMetric


import time
import torch.optim as optim
import wandb
from tqdm import tqdm
import os
import datetime
import sys
import pandas as pd

def adjust_checkpoint_keys(loaded_checkpoint):
    # Remove 'model.' prefix from each key
    new_state_dict = {key.replace('model.', ''): value for key, value in loaded_checkpoint['state_dict'].items()}
    loaded_checkpoint['state_dict'] = new_state_dict
    return loaded_checkpoint

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    # checkpoint = adjust_checkpoint_keys(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['train_step']
    epoch = checkpoint['epoch']
    return step, epoch

def main(params=None):

    if params is None:
        params = training_coco_params
        params = merge_namespaces(params, slot_attention_coco_params)

    train_dataset = COCO2017(root=params.data_root, split='train', image_size=params.resolution[0], mask_size = params.resolution[0], return_gt_in_train=True)
    val_dataset = COCO2017(root=params.data_root, split='val', image_size=params.resolution[0], mask_size = params.resolution[0], return_gt_in_train=True)

    train_dataloader = DataLoader(
            train_dataset,
            batch_size=params.batch_size,
            shuffle=True,
            num_workers=params.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    val_dataloader = DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    
    model = SlotAttentionModel(
                resolution=params.resolution,
                num_slots=params.num_slots,
                # num_slots = 4, 
                num_iterations=params.num_iterations,
                # num_iterations= 4, 
                slot_size=params.slot_size,
                mlp_hidden_size=params.mlp_hidden_size,
                hidden_dims=params.hidden_dims,
                decoder_resolution=params.decoder_resolution,
                use_separation_loss=params.use_separation_loss,
                use_area_loss=params.use_area_loss,
            )
    
    model = model.to(params.device)

    MBO_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).to(params.device)
    miou_metric = UnsupervisedMaskIoUMetric(matching="hungarian", ignore_background = True, ignore_overlaps = True).to(params.device)
    ari_metric = ARIMetric(foreground = True, ignore_overlaps = True).to(params.device)

    params_model = [{'params': model.parameters()}]

    # define optimizer
    optimizer = optim.Adam(params_model, lr=params.lr_main)
    
    # TODO: load checkpoint
    params.is_load_checkpoint = True
    if(params.is_load_checkpoint):
       step_c, epoch_c = load_checkpoint(model, optimizer, "checkpoints/model_coco_stage_2.ckpt")
       print("The checkpoint has been loaded!")
    
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader):

            recon_combined, recons, masks, _ = model.forward(batch[0].to(params.device))

            masks_gt = batch[2].to(params.device) # N, 128, 128
            mask_ignore = batch[3].to(params.device) # N, 1, H, W
            masks_argmax = masks.argmax(dim=1).squeeze(1) # N, H, W

            true_mask_reshaped = torch.nn.functional.one_hot(masks_gt).to(torch.float32).permute(0,3,1,2).to(params.device) # torch.Size([32, 27, 320, 320])
            pred_mask_reshaped = torch.nn.functional.one_hot(masks_argmax).to(torch.float32).permute(0,3,1,2).to(params.device) # torch.Size([32, 7, 320, 320])

            MBO_metric.update(pred_mask_reshaped, true_mask_reshaped, mask_ignore)
            miou_metric.update(pred_mask_reshaped, true_mask_reshaped, mask_ignore)
            ari_metric.update(pred_mask_reshaped, true_mask_reshaped, mask_ignore)
            
            loss = F.mse_loss(recon_combined, batch[0].to(params.device))
            total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)

        ari = 100 * ari_metric.compute()
        mbo = 100 * MBO_metric.compute()
        miou = 100 * miou_metric.compute()

        df_results = pd.DataFrame([[avg_val_loss, miou.item(), mbo.item(), ari.item()]], 
                 columns=['MSE', 'miou', 'mBO', 'FG-ARI'])

        print(f"Epoch: {epoch_c}, Step: {step_c}:")
        print(df_results)

        recon_combined, recons, masks, slots = model.forward(batch[0][:params.n_samples].to(params.device)) # recon_combined: torch.Size([64, 3, 128, 128]), recons: torch.Size([64, 7, 3, 128, 128]), masks: torch.Size([64, 7, 1, 128, 128]), slots: torch.Size([448, 64, 1, 1])
        masks = masks.cpu()
        images = batch[0][:params.n_samples] # N, 3, 128, 128

        true_mask_c = batch[2][:params.n_samples].cpu() # N, 128, 128

        slots_recon_vis = recons.cpu() * masks + (1 - masks) # torch.Size([64, 7, 3, 128, 128])

        slots_image_vis = images.unsqueeze(dim=1) * masks + 1.0 - masks

        masks_argmax = masks.squeeze(dim=2).argmax(dim=1) # N, H, W

        vis_recon = visualize(inv_normalize(images), true_mask_c, inv_normalize(recon_combined.cpu()), inv_normalize(slots_recon_vis), masks_argmax, inv_normalize(slots_image_vis), N=params.n_samples)

        grid = vutils.make_grid(vis_recon, nrow=2*params.num_slots + 4, pad_value=0.2)[:, 2:-2, 2:-2]
        grid = F.interpolate(grid.unsqueeze(1), scale_factor=1.0, mode='bilinear').squeeze()
        
        # save_image(grid, os.path.join("/home/jiantong/project/python/SAPI/JiantongZhao/object-discovery-pytorch/vis/coco",f'viz_{epoch_c}_step_{step_c}.png'))

        # grid_numpy = grid.permute(1, 2, 0).contiguous().numpy()



if __name__ == "__main__":
    
    main()





