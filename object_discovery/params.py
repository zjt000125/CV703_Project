from argparse import Namespace


training_params = Namespace(
    model_type="sa",
    dataset="clevr",
    num_slots=7, # 10
    num_iterations=3,
    accumulate_grad_batches=1,
    data_root="data/clevr_with_masks.h5",
    accelerator="gpu",
    device='cuda:0',
    max_steps=-1,
    num_sanity_val_steps=1,
    num_workers=4,
    is_logger_enabled=True,
    gradient_clip_val=0.0,
    n_samples=8, 
    clevrtex_dataset_variant="full",
    alternative_crop=True,  
    is_load_checkpoint = False, 
    ckpt_path = "checkpoints/model_clevr6_mask_stage_1.ckpt", 
    model_dir="checkpoints/clevr6_mask", 
    model_name = "model_clevr6_mask", 
    warmup_steps = 10000, 
    decay_rate = 0.5, 
    decay_steps = 100000,
    wandb_mode = "offline", 
)


slot_attention_params = Namespace(
    lr_main=4e-4,
    batch_size=64,
    val_batch_size=64,
    resolution=(128, 128),
    slot_size=64,
    max_epochs=1000,
    max_steps=500000,
    weight_decay=0.0,
    mlp_hidden_size=128,
    scheduler="warmup_and_decay",
    scheduler_gamma=0.5,
    warmup_steps_pct=0.02,
    decay_steps_pct=0.2,
    use_separation_loss=None, # "entropy"
    separation_tau_start=60_000,
    separation_tau_end=65_000,
    separation_tau_max_val=0.003,
    separation_tau=None,
    boxworld_group_objects=True,
    use_area_loss=False, # True
    area_tau_start=60_000,
    area_tau_end=65_000,
    area_tau_max_val=0.006,
    area_tau=None,
)



training_coco_params = Namespace(
    model_type="sa",
    dataset="coco",
    num_slots=7, # 10
    num_iterations=6,
    accumulate_grad_batches=1,
    data_root="data/COCO2017",
    accelerator="gpu",
    device='cuda:0',
    max_steps=-1,
    num_sanity_val_steps=1,
    num_workers=4,
    is_logger_enabled=True,
    gradient_clip_val=0.0,
    n_samples=8, 
    clevrtex_dataset_variant="full",
    alternative_crop=True,  # Alternative crop for RAVENS dataset
    is_load_checkpoint = False, 
    ckpt_path = "checkpoints/model_coco_stage_1.ckpt", 
    model_dir="checkpoints/coco/", 
    model_name = "model_coco", 
    warmup_steps = 10000, 
    decay_rate = 0.5, 
    decay_steps = 100000,
    wandb_mode = "offline", 
)


slot_attention_coco_params = Namespace(
    lr_main=4e-4,
    batch_size=32,
    val_batch_size=64,
    resolution=(128, 128),
    slot_size=128,
    max_epochs=1000,
    max_steps=500000,
    weight_decay=0.0,
    mlp_hidden_size=256,
    hidden_dims=(128, 128, 128, 128), 
    decoder_resolution=(8, 8), 
    scheduler="warmup_and_decay",
    scheduler_gamma=0.5,
    warmup_steps_pct=0.02,
    decay_steps_pct=0.2,
    use_separation_loss=None, # "entropy"
    separation_tau_start=60_000,
    separation_tau_end=65_000,
    separation_tau_max_val=0.003,
    separation_tau=None,
    boxworld_group_objects=True,
    use_area_loss=False, # True
    area_tau_start=60_000,
    area_tau_end=65_000,
    area_tau_max_val=0.006,
    area_tau=None,
)


def merge_namespaces(one: Namespace, two: Namespace):
    return Namespace(**{**vars(one), **vars(two)})
