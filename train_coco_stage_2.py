import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torchvision import utils as vutils
from torchvision.utils import save_image
import torch.nn.functional as F

from dataset import COCO2017
from object_discovery.slot_attention_model import SlotAttentionModel
from object_discovery.params import (
    merge_namespaces,
    training_coco_params,
    slot_attention_coco_params,

)
from object_discovery.utils import (
    visualize,
    inv_normalize,
)

from MAE_model import MAE

import time
import torch.optim as optim
import wandb
from tqdm import tqdm
import os
import datetime
import random

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_checkpoint(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
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
            batch_size=16,
            shuffle=True,
            num_workers=params.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    val_dataloader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    
    mae = MAE(4, 3)
    mae = mae.to(params.device)

    if(params.is_load_checkpoint):
       step_c, epoch_c = load_checkpoint(mae, "checkpoints/MAE_best_epoch_45_step_340032.ckpt")
       print("The checkpoint has been loaded!")


    mae.eval()
    
    model = SlotAttentionModel(
                resolution=params.resolution,
                num_slots=params.num_slots,
                num_iterations=params.num_iterations,
                slot_size=params.slot_size,
                mlp_hidden_size=params.mlp_hidden_size,
                hidden_dims=params.hidden_dims,
                decoder_resolution=params.decoder_resolution,
                use_separation_loss=params.use_separation_loss,
                use_area_loss=params.use_area_loss,
            )
    
    model = model.to(params.device)

    params_model = [{'params': model.parameters()}]

    # define optimizer
    optimizer = optim.Adam(params_model, lr=params.lr_main)
    
    # TODO: load checkpoint
    params.is_load_checkpoint = True
    if(params.is_load_checkpoint):
       step_c, epoch_c = load_checkpoint(model, "checkpoints/model_clevr6_mask_stage_2.ckpt")
       print("The checkpoint has been loaded!")
    
    start = time.time()
    i = 0
    best_val_loss = float('inf')

    # wandb initialization
    os.environ["WANDB_MODE"] = 'offline'
    run = wandb.init(project="Train MAE on coco dataset")

    # create the checkpoint dir
    current_time = datetime.datetime.now()
    folder_name = current_time.strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_path = os.path.join("checkpoints/coco_2", folder_name)
    os.makedirs(checkpoint_path, exist_ok=True)

    print(f"Save the checkpoints in: {checkpoint_path}")

    for epoch in range(params.max_epochs):
        # When reaching the max train_step, just save and quit.
        if(i >= params.max_steps):
            state_dict = model.state_dict()
            torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_val_loss, 'train_step': i}, os.path.join(checkpoint_path, "model") + f'_epoch_{epoch}_step_{i}.ckpt')
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

            masks_argmax = masks.argmax(dim=1).squeeze(1) # N, H, W
            pred_mask_reshaped = torch.nn.functional.one_hot(masks_argmax).to(torch.float32).permute(0,3,1,2).to(params.device) # torch.Size([N, 7, H, W])

            # choose masks randomly
            random_ids = [random.randint(0, 6) for _ in range(pred_mask_reshaped.shape[0])]

            ids = [i for i in range(pred_mask_reshaped.shape[0])]

            masks_selected = pred_mask_reshaped[ids, random_ids].unsqueeze(1) # torch.Size([N, 1, H, W])
            
            images = batch[0].to(params.device) # N, 3, H, W
            images_masked = images * (1-masks_selected) # N, 3, H, W

            input_data = torch.cat([images_masked, masks_selected], dim=1)

            outputs = mae(input_data)

            loss_mae = F.mse_loss(outputs*masks_selected, images*masks_selected, reduction='sum') / masks_selected.sum()

            loss_model = F.mse_loss(recon_combined, batch[0].to(params.device))

            loss = loss_model - 0.01 * loss_mae
            
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_counter += 1
            if batch_counter % 100 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_counter}, Batch Loss: {loss.item()}")

            run.log({"learning_rate": optimizer.param_groups[0]['lr'], "train/loss":loss.item()}, step=i)

            # break

        total_loss /= len(train_dataloader)
        print("Epoch: {}, Loss: {}, Time: {}".format(epoch, total_loss, datetime.timedelta(seconds=time.time() - start)))
        run.log({"train/loss_average":total_loss}, step=i)
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader):

                recon_combined, recons, masks, _ = model.forward(batch[0].to(params.device))

                masks_argmax = masks.argmax(dim=1).squeeze(1) # N, H, W
                pred_mask_reshaped = torch.nn.functional.one_hot(masks_argmax).to(torch.float32).permute(0,3,1,2).to(params.device) # torch.Size([N, 7, H, W])

                # choose masks randomly
                random_ids = [random.randint(0, 6) for _ in range(pred_mask_reshaped.shape[0])]

                ids = [i for i in range(pred_mask_reshaped.shape[0])]

                masks_selected = pred_mask_reshaped[ids, random_ids].unsqueeze(1) # torch.Size([N, 1, H, W])
                
                images = batch[0].to(params.device) # N, 3, H, W
                images_masked = images * (1-masks_selected) # N, 3, H, W

                input_data = torch.cat([images_masked, masks_selected], dim=1)

                outputs = mae(input_data)

                loss_mae = F.mse_loss(outputs*masks_selected, images*masks_selected, reduction='sum') / masks_selected.sum()

                loss_model = F.mse_loss(recon_combined, batch[0].to(params.device))
                
                loss = loss_model - 0.01 * loss_mae

                total_val_loss += loss.item()

                # break

            avg_val_loss = total_val_loss / len(val_dataloader)
            print(f"Epoch: {epoch}, Validation Loss: {avg_val_loss}")

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

            grid_numpy = grid.permute(1, 2, 0).contiguous().numpy()
            run.log({"Visualization": wandb.Image(grid_numpy, caption=f"val visualization of epoch {epoch}"), "val/loss_average": avg_val_loss}, step=i)

            # Save the model if it has the best validation loss so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = os.path.join(checkpoint_path, "model" + f'_best_epoch_{epoch}_step_{i}.ckpt')
                state_dict = model.state_dict()
                torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'optimizer_state_dict': optimizer.state_dict(),
                            'loss': avg_val_loss, 'train_step': i}, model_path)
                print(f"Saved new best model to {model_path}")

            if not epoch % 5:
                # When using DataParallel, save only the model module's state_dict
                state_dict = model.state_dict()
                torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'optimizer_state_dict': optimizer.state_dict(),
                            'loss': avg_val_loss, 'train_step': i}, os.path.join(checkpoint_path, "model") + f'_epoch_{epoch}_step_{i}.ckpt')

if __name__ == "__main__":
    set_global_seed(42)
    main()


