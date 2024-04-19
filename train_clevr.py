import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torchvision import utils as vutils
from torchvision.utils import save_image

from dataset import CLEVRWithMasksDataset, image_transforms, mask_transforms
from object_discovery.slot_attention_model import SlotAttentionModel
from object_discovery.params import (
    merge_namespaces,
    training_params,
    slot_attention_params,
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
)

import time
import torch.optim as optim
import wandb
from tqdm import tqdm
import os
import datetime
import sys

def adjust_checkpoint_keys(loaded_checkpoint):
    # Remove 'model.' prefix from each key
    new_state_dict = {key.replace('model.', ''): value for key, value in loaded_checkpoint['state_dict'].items()}
    loaded_checkpoint['state_dict'] = new_state_dict
    return loaded_checkpoint

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    checkpoint = adjust_checkpoint_keys(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_states'])
    step = checkpoint['global_step']
    epoch = checkpoint['epoch']
    return step, epoch


def main(params=None):

    if params is None:
        params = training_params
        params = merge_namespaces(params, slot_attention_params)



    train_dataset = CLEVRWithMasksDataset(
        data_root=params.data_root,
        clevr_transforms=image_transforms,
        mask_transforms=mask_transforms,
        split="train",
        max_n_objects=params.num_slots - 1,
    )

    val_dataset = CLEVRWithMasksDataset(
            data_root=params.data_root,
            clevr_transforms=image_transforms,
            mask_transforms=mask_transforms,
            split="val",
            max_n_objects=params.num_slots - 1,
        )
        
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
            batch_size=params.val_batch_size,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    

    model = SlotAttentionModel(
                resolution=params.resolution,
                num_slots=params.num_slots,
                num_iterations=params.num_iterations,
                slot_size=params.slot_size,
                use_separation_loss=params.use_separation_loss,
                use_area_loss=params.use_area_loss,
            )
    
    model = model.to(params.device)

    params_model = [{'params': model.parameters()}]

    # define optimizer
    optimizer = optim.Adam(params_model, lr=params.lr_main)
    
    # load checkpoint
    if(params.is_load_checkpoint):
       step_c, epoch_c = load_checkpoint(model, optimizer, params.ckpt_path)

    start = time.time()
    i = 0
    best_val_loss = float('inf')

    # wandb initialization
    os.environ["WANDB_MODE"] = params.wandb_mode
    run = wandb.init(project="Train slot attention on clevr6_mask dataset")

    # create the checkpoint dir
    current_time = datetime.datetime.now()
    folder_name = current_time.strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_path = os.path.join(params.model_dir, folder_name)
    os.makedirs(checkpoint_path, exist_ok=True)

    print(f"Save the checkpoints in: {checkpoint_path}")

    for epoch in range(params.max_epochs):
        # When reaching the max train_step, just save and quit.
        if(i >= params.max_steps):
            state_dict = model.state_dict()
            torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_val_loss, 'train_step': i}, os.path.join(checkpoint_path, params.model_name) + f'_epoch_{epoch}_step_{i}.ckpt')
            break
        
        model.train()
        total_loss = 0
        batch_counter = 0

        for batch in tqdm(train_dataloader):
            i += 1

            if i < params.warmup_steps:
                learning_rate = params.lr_main * (i / params.warmup_steps)
            else:
                learning_rate = params.lr_main
            learning_rate *= (params.decay_rate ** (i / params.decay_steps))

            optimizer.param_groups[0]['lr'] = learning_rate

            recon_combined, recons, masks, _ = model.forward(batch[0].to(params.device))

            loss = F.mse_loss(recon_combined, batch[0].to(params.device))
            
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_counter += 1
            if batch_counter % 100 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_counter}, Batch Loss: {loss.item()}")

            run.log({"learning_rate": optimizer.param_groups[0]['lr'], "train/loss":loss.item()}, step=i)

        total_loss /= len(train_dataloader)
        print("Epoch: {}, Loss: {}, Time: {}".format(epoch, total_loss, datetime.timedelta(seconds=time.time() - start)))
        run.log({"train/loss_average":total_loss}, step=i)
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader):

                recon_combined, recons, masks, _ = model.forward(batch[0].to(params.device))
                
                loss = F.mse_loss(recon_combined, batch[0].to(params.device))
                total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_dataloader)
            print(f"Epoch: {epoch}, Validation Loss: {avg_val_loss}")

            recon_combined, recons, masks, slots = model.forward(batch[0][:params.n_samples].to(params.device)) # recon_combined: torch.Size([64, 3, 128, 128]), recons: torch.Size([64, 7, 3, 128, 128]), masks: torch.Size([64, 7, 1, 128, 128]), slots: torch.Size([448, 64, 1, 1])
            masks = masks.cpu()
            images = batch[0][:params.n_samples] # N, 3, 128, 128
            masks_gt_one_hot = (batch[1][:params.n_samples] / 255.0).permute(0, 1, 4, 2, 3).contiguous() # torch.Size([64, 11, 1, 128, 128])

            N, max_object_num, channel, h, w = masks_gt_one_hot.shape

            masks_gt = masks_gt_one_hot.argmax(dim=1).squeeze(dim=1) # N, H, W

            slots_recon_vis = recons.cpu() * masks + (1 - masks) # torch.Size([64, 7, 3, 128, 128])

            slots_image_vis = images.unsqueeze(dim=1) * masks + 1.0 - masks

            masks_argmax = masks.squeeze(dim=2).argmax(dim=1) # N, H, W

            vis_recon = visualize(to_rgb_from_tensor(images), masks_gt, to_rgb_from_tensor(recon_combined.cpu()), to_rgb_from_tensor(slots_recon_vis), masks_argmax, to_rgb_from_tensor(slots_image_vis), N=params.n_samples)

            grid = vutils.make_grid(vis_recon, nrow=2*params.num_slots + 4, pad_value=0.2)[:, 2:-2, 2:-2]
            grid = F.interpolate(grid.unsqueeze(1), scale_factor=1.0, mode='bilinear').squeeze()
            
            # save_image(grid, os.path.join(params.viz_dir,f'viz_{epoch}.png'))

            grid_numpy = grid.permute(1, 2, 0).contiguous().numpy()
            run.log({"Visualization": wandb.Image(grid_numpy, caption=f"val visualization of epoch {epoch}"), "val/loss_average": avg_val_loss}, step=i)

            # Save the model if it has the best validation loss so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = os.path.join(checkpoint_path, params.model_name + f'_best_epoch_{epoch}_step_{i}.ckpt')
                state_dict = model.state_dict()
                torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'optimizer_state_dict': optimizer.state_dict(),
                            'loss': avg_val_loss, 'train_step': i}, model_path)
                print(f"Saved new best model to {model_path}")

            if not epoch % 5:
                # When using DataParallel, save only the model module's state_dict
                state_dict = model.state_dict()
                torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'optimizer_state_dict': optimizer.state_dict(),
                            'loss': avg_val_loss, 'train_step': i}, os.path.join(checkpoint_path, params.model_name) + f'_epoch_{epoch}_step_{i}.ckpt')

if __name__ == "__main__":
    
    main()