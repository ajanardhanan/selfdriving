import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import datasets and models based on the homework structure
from datasets.road_dataset import load_data
from models import MODEL_FACTORY, save_model

def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    dataset_path: str = "drive_data",
    batch_size: int = 64,
    num_epochs: int = 50,
    lr: float = 1e-3,
    num_workers: int = 4,
    seed: int = 2024,
):
    # 1. Setup Device and Logging
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {model_name} on {device}")
    
    log_dir = Path(exp_dir) / model_name
    logger = SummaryWriter(str(log_dir))
    
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 2. Determine Transform Pipeline
    # Optimization: Only load images ('default') if using CNN. 
    # Use 'state_only' for MLP/Transformer to speed up data loading.
    if model_name == "cnn_planner":
        pipeline = "default"
    else:
        pipeline = "state_only"

    # 3. Load Data
    # Assumes dataset_path contains 'train' and 'val' subdirectories
    train_path = Path(dataset_path) / "train"
    val_path = Path(dataset_path) / "val"

    print(f"Loading training data from {train_path} using '{pipeline}' pipeline...")
    train_loader = load_data(
        dataset_path=str(train_path),
        transform_pipeline=pipeline,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True
    )

    print(f"Loading validation data from {val_path} using '{pipeline}' pipeline...")
    val_loader = load_data(
        dataset_path=str(val_path),
        transform_pipeline=pipeline,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False
    )

    # 4. Initialize Model
    model = MODEL_FACTORY[model_name]().to(device)
    
    # 5. Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_func = nn.L1Loss(reduction='none') # We handle reduction manually for masking

    # 6. Training Loop
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss_accum = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Move data to device
            track_left = batch['track_left'].to(device)
            track_right = batch['track_right'].to(device)
            waypoints = batch['waypoints'].to(device)
            mask = batch['waypoints_mask'].to(device) # Shape (B, N)

            # Determine input based on model type
            if model_name == "cnn_planner":
                image = batch['image'].to(device)
                preds = model(image)
            else:
                preds = model(track_left, track_right)

            # Compute Masked Loss
            # preds: (B, N, 2), waypoints: (B, N, 2)
            raw_loss = loss_func(preds, waypoints) # (B, N, 2)
            
            # Sum over coordinates (x, y) -> (B, N)
            # Note: L1 loss elementwise is |x-x_hat|, sum(-1) adds x_err + y_err
            sample_loss = raw_loss.sum(dim=-1)
            
            # Apply mask: zero out invalid waypoints
            masked_loss = sample_loss * mask
            
            # Normalize by valid points
            loss = masked_loss.sum() / (mask.sum() + 1e-6)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            train_loss_accum += loss.item()
            global_step += 1
            if global_step % 10 == 0:
                logger.add_scalar("train/loss", loss.item(), global_step)
            
            progress_bar.set_postfix(loss=loss.item())

        # 7. Validation Loop
        model.eval()
        val_loss_accum = 0.0
        val_lat_error = 0.0
        val_long_error = 0.0
        total_valid_points = 0

        with torch.no_grad():
            for batch in val_loader:
                track_left = batch['track_left'].to(device)
                track_right = batch['track_right'].to(device)
                waypoints = batch['waypoints'].to(device)
                mask = batch['waypoints_mask'].to(device)

                if model_name == "cnn_planner":
                    image = batch['image'].to(device)
                    preds = model(image)
                else:
                    preds = model(track_left, track_right)

                # Validation Loss calculation (same as train)
                raw_loss = loss_func(preds, waypoints).sum(dim=-1)
                val_loss = (raw_loss * mask).sum()
                
                # Metrics Calculation (L1 Error breakdown)
                # EgoTrackProcessor projects points to [x, z]. 
                # index 0 = X (Lateral), index 1 = Z (Longitudinal/Depth)
                abs_diff = torch.abs(preds - waypoints) # (B, N, 2)
                
                lat_diff = abs_diff[:, :, 0] * mask # Lateral
                long_diff = abs_diff[:, :, 1] * mask # Longitudinal

                val_loss_accum += val_loss.item()
                val_lat_error += lat_diff.sum().item()
                val_long_error += long_diff.sum().item()
                total_valid_points += mask.sum().item()

        # Average metrics
        avg_val_loss = val_loss_accum / (total_valid_points + 1e-6)
        avg_lat_error = val_lat_error / (total_valid_points + 1e-6)
        avg_long_error = val_long_error / (total_valid_points + 1e-6)

        logger.add_scalar("val/loss", avg_val_loss, global_step)
        logger.add_scalar("val/lateral_error", avg_lat_error, global_step)
        logger.add_scalar("val/longitudinal_error", avg_long_error, global_step)

        print(f"Val Loss: {avg_val_loss:.4f} | "
              f"Lat Err: {avg_lat_error:.4f} (Goal < 0.45/0.6) | "
              f"Long Err: {avg_long_error:.4f} (Goal < 0.2/0.3)")

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = save_model(model)
            print(f"Saved best model to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["mlp_planner", "transformer_planner", "cnn_planner"])
    parser.add_argument("--dataset_path", type=str, default="drive_data", help="Path to data containing train/val subfolders")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    
    train(
        model_name=args.model,
        dataset_path=args.dataset_path,
        lr=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )