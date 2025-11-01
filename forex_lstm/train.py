# Ensure we're in the correct working directory for Kaggle
import os
if os.path.exists('/kaggle/working'):
    # If running on Kaggle, change to the working directory
    os.chdir('/kaggle/working')

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

# Optional MT5 import - only needed for live data download
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None
    print("MetaTrader5 not available - live data download disabled")

from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from .data import (download_ticker, load_csv, prepare_ohlc_series, scale_series, save_scaler,
                   generate_smc_labels, generate_enhanced_smc_labels, initialize_mt5,
                   evaluate_walk_forward, detect_trend_direction, calculate_adx)
from .model import LSTMClassifier, FocalLoss
from .utils import create_sequences, SequenceDataset


# Timeframe mapping - fallback values if MT5 not available
if MT5_AVAILABLE:
    TIMEFRAME_MAP = {
        'M15': mt5.TIMEFRAME_M15,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
    }
else:
    # Fallback numeric values (these won't be used for actual MT5 calls)
    TIMEFRAME_MAP = {
        'M15': 1,  # Placeholder
        'H1': 2,   # Placeholder
        'H4': 3,   # Placeholder
        'D1': 4,   # Placeholder
    }


def train_epoch(model, loader, opt, loss_fn, device, grad_clip=0.0):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()

        # Apply gradient clipping if specified
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        opt.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


def eval_model(model, loader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            out = model(X).cpu().numpy()
            preds.append(out)
            trues.append(y.numpy())
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    pred_classes = np.argmax(preds, axis=1)
    accuracy = np.mean(pred_classes == trues)
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="EURUSD",
                        help="MT5 symbol (e.g., EURUSD) or path to CSV. Auto-detects prepared datasets if not specified.")
    parser.add_argument("--timeframe", type=str, default="H1", choices=TIMEFRAME_MAP.keys(),
                        help="Timeframe: M15, H1, H4, D1")
    parser.add_argument("--bars", type=int, default=10000,
                        help="Number of bars to download")
    parser.add_argument("--seq-len", type=int, default=60,
                        help="Sequence length for LSTM")
    parser.add_argument("--batch", type=int, default=32,
                        help="Batch size (smaller for better convergence with small datasets)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Maximum number of epochs (early stopping may stop earlier)")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Initial learning rate (lower for smoother convergence)")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="L2 regularization weight decay (lighter for small datasets)")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd", "rmsprop"],
                        help="Optimizer to use (adamw recommended for better generalization)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum for SGD optimizer (0.9 recommended)")
    parser.add_argument("--nesterov", action="store_true", default=True,
                        help="Use Nesterov momentum for SGD (improves convergence)")
    parser.add_argument("--hidden", type=int, default=256,
                        help="LSTM hidden size (increased for richer features)")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate (increased for better regularization)")
    parser.add_argument("--bidirectional", action="store_true", default=True,
                        help="Use bidirectional LSTM")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience (epochs, increased for more training)")
    parser.add_argument("--min-delta", type=float, default=1e-4,
                        help="Minimum change to qualify as an improvement")
    parser.add_argument("--lr-schedule", action="store_true", default=True,
                        help="Use learning rate scheduler")
    parser.add_argument("--out-dir", type=str, default="/kaggle/working/LSTM_Model_Output",
                        help="Output directory for checkpoints")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--mixed-precision", action="store_true", default=True,
                        help="Use mixed precision training (FP16)")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.8,
                        help="GPU memory fraction to use (0.0-1.0)")
    parser.add_argument("--walk-forward", action="store_true", default=False,
                        help="Use Walk-Forward Validation instead of traditional train/val split")
    parser.add_argument("--trend-filter", action="store_true", default=True,
                        help="Apply trend alignment filtering to signals")
    parser.add_argument("--triple-barrier", action="store_true", default=True,
                        help="Use Triple Barrier Method for signal validation")
    parser.add_argument("--min-adx", type=float, default=20.0,
                        help="Minimum ADX threshold for trend strength filtering")
    parser.add_argument("--quality-threshold", type=float, default=0.0,
                        help="Minimum quality score threshold for signals (0.0=disable, 1.0=max)")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping threshold (0 to disable)")
    parser.add_argument("--no-class-weights", action="store_true", default=False,
                        help="Disable class weighting (use uniform weights)")
    parser.add_argument("--max-class-weight", type=float, default=10.0,
                        help="Maximum allowed class weight to prevent extreme imbalance")
    parser.add_argument("--binary-classification", action="store_true", default=False,
                        help="Convert to binary: class 2 (win) vs classes 0,1 (not-win)")
    parser.add_argument("--focal-loss", action="store_true", default=False,
                        help="Use Focal Loss instead of CrossEntropy (better for extreme imbalance)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma parameter (higher = more focus on hard examples)")
    args = parser.parse_args()

    # CUDA optimizations
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda":
        # Set CUDA memory fraction
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # Clear cache
        torch.cuda.empty_cache()

        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(
            f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Using GPU memory fraction: {args.gpu_memory_fraction}")
    else:
        print("Using CPU for training")

    os.makedirs(args.out_dir, exist_ok=True)

    # Choose training method based on arguments
    if args.walk_forward:
        return train_with_walk_forward_validation(args)

    # Load data - smart defaults for prepared datasets
    dataset_paths = [
        # Preferred: SMC-enhanced dataset
        "/kaggle/working/training_data/smc_lstm_training_dataset.csv",
        # Fallback: basic consolidated dataset
        "/kaggle/working/training_data/consolidated_dataset.csv"
    ]

    dataset_found = False
    for dataset_path in dataset_paths:
        if os.path.exists(dataset_path):
            print(f"Loading prepared dataset: {dataset_path}")
            df = load_csv(dataset_path)
            if 'symbol' in df.columns and 'timeframe' in df.columns:
                print(f"Loaded consolidated data with {len(df)} rows")
                print(f"Symbols: {df['symbol'].unique()}")
                print(f"Timeframes: {df['timeframe'].unique()}")
                if 'label' in df.columns:
                    print(
                        f"Label distribution: {df['label'].value_counts().to_dict()}")
            dataset_found = True
            break

    # Fallback to user-specified CSV or MT5 download
    if not dataset_found:
        if args.ticker.lower().endswith('.csv'):
            if os.path.exists(args.ticker):
                print(f"Loading user-specified dataset: {args.ticker}")
                df = load_csv(args.ticker)
            else:
                raise FileNotFoundError(f"Dataset not found: {args.ticker}")
        else:
            if not MT5_AVAILABLE:
                raise RuntimeError(
                    f"No prepared datasets found and MetaTrader5 not available. Please prepare your dataset first using consolidate_data.py and create_smc_training_dataset.py")
            print(
                f"No prepared datasets found. Downloading {args.ticker} from MT5...")
            initialize_mt5()
            save_path = f"DATA/{args.ticker}_{args.timeframe}_{args.bars}bars.csv"
            df = download_ticker(
                args.ticker, TIMEFRAME_MAP[args.timeframe], args.bars, save_path)

    scaler_to_save = None
    seqs = None
    labels = None
    quality_scores = None

    if dataset_found and 'label' in df.columns:
        print("Using pre-computed labels from dataset")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = {
            'label', 'quality_score', 'tbm_outcome', 'raw_signal_label',
            'signal_direction'
        }
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        if not feature_cols:
            raise ValueError(
                "No numeric feature columns available for training sequences.")

        feature_matrix = df[feature_cols].values
        feature_scaler = MinMaxScaler()
        scaled_features = feature_scaler.fit_transform(feature_matrix)
        scaler_to_save = feature_scaler

        seqs, _ = create_sequences(scaled_features, args.seq_len)

        all_labels = df['label'].values
        labels = all_labels[args.seq_len:args.seq_len + len(seqs)]

        if 'quality_score' in df.columns:
            quality_scores = df['quality_score'].values[args.seq_len:args.seq_len +
                                                        len(seqs)]
            if args.quality_threshold > 0:
                quality_mask = quality_scores >= args.quality_threshold
                print(f"Applying quality threshold {args.quality_threshold}: "
                      f"{np.sum(quality_mask)}/{len(quality_mask)} signals retained")
                seqs = seqs[quality_mask]
                labels = labels[quality_mask]
                quality_scores = quality_scores[quality_mask]
    else:
        series = prepare_ohlc_series(df)
        close_series = series[['Close']]
        scaled, scaler = scale_series(close_series)
        scaler_to_save = scaler

        seqs, _ = create_sequences(scaled, args.seq_len)

        # Generate enhanced SMC labels with trend filtering and triple barrier method
        if args.trend_filter or args.triple_barrier:
            print(
                "🎯 Generating enhanced SMC labels with trend filtering and triple barrier method...")
            labels, quality_scores = generate_enhanced_smc_labels(
                series, args.seq_len,
                use_trend_filter=args.trend_filter,
                use_triple_barrier=args.triple_barrier,
                min_adx=args.min_adx
            )
        else:
            print("Generating SMC-based labels with mitigation and quality scoring")
            labels, quality_scores = generate_smc_labels(series, args.seq_len)

        print(
            f"Generated {np.sum(labels > 0)} trading signals with average quality score: {quality_scores.mean():.3f}")

        # Apply quality threshold filtering
        if args.quality_threshold > 0:
            quality_mask = quality_scores >= args.quality_threshold
            seqs = seqs[quality_mask]
            labels = labels[quality_mask]
            print(
                f"After quality filtering ({args.quality_threshold}): {len(labels)} signals retained")

    # Ensure labels are suitable for CrossEntropyLoss (non-negative integer classes)
    if len(labels) == 0:
        raise ValueError(
            "No labels available after preprocessing. Check data and quality thresholds.")

    if labels.min() < 0:
        label_mapping = {-1: 0, 0: 1, 1: 2}
        labels = np.vectorize(lambda x: label_mapping.get(
            int(x), int(x)))(labels).astype(int)
    else:
        labels = labels.astype(int)

    # Convert to binary classification if requested (more robust for imbalanced data)
    if args.binary_classification:
        print("🔄 Converting to binary classification: Win (1) vs Not-Win (0)")
        # Map: class 2 (original win) → 1, classes 0,1 (loss/neutral) → 0
        labels = (labels == 2).astype(int)
        print(
            f"   Binary distribution: {np.sum(labels == 0)} Not-Win, {np.sum(labels == 1)} Win")

    # Check for extreme class imbalance before proceeding
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    min_class_samples = label_counts.min()
    max_class_samples = label_counts.max()
    imbalance_ratio = max_class_samples / min_class_samples

    if min_class_samples < 100:
        print(
            f"\n⚠️  WARNING: Smallest class has only {min_class_samples} samples!")
        print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1")
        print(f"   This may cause training instability. Consider:")
        print(
            f"   1. Lowering --quality-threshold (current: {args.quality_threshold})")
        print(f"   2. Using --no-class-weights flag")
        print(f"   3. Checking dataset quality scores distribution\n")

    # Use Walk-Forward Validation if requested
    if args.walk_forward:
        print("🔄 Using Walk-Forward Validation")
        return train_with_walk_forward_validation(args)

    # Traditional train / test split
    print("📊 Using traditional train/validation split")
    split = int(0.8 * len(seqs))
    X_train, X_val = seqs[:split], seqs[split:]
    y_train, y_val = labels[:split], labels[split:]

    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val)

    # CUDA-optimized data loading
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,  # Shuffle for better training
        pin_memory=(device == "cuda"),  # Pin memory for faster GPU transfer
        num_workers=0 if device == "cpu" else min(
            4, os.cpu_count() or 1),  # Use multiple workers for GPU
        persistent_workers=device == "cuda",
        prefetch_factor=2 if device == "cuda" else None
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        pin_memory=(device == "cuda"),
        num_workers=0 if device == "cpu" else min(4, os.cpu_count() or 1),
        persistent_workers=device == "cuda",
        prefetch_factor=2 if device == "cuda" else None
    )

    # Determine number of output classes
    num_classes = 2 if args.binary_classification else 3

    model = LSTMClassifier(
        input_size=X_train.shape[2],
        hidden_size=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=num_classes,
        bidirectional=args.bidirectional).to(device)

    # Move model to CUDA with optimizations
    if device == "cuda":
        model = model.cuda()
        # Use DataParallel for multiple GPUs if available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = nn.DataParallel(model)

    # Enhanced optimizer selection with momentum options
    if args.optimizer == "adam":
        opt = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_name = f"Adam(lr={args.lr}, weight_decay={args.weight_decay})"
    elif args.optimizer == "adamw":
        opt = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_name = f"AdamW(lr={args.lr}, weight_decay={args.weight_decay})"
    elif args.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay, nesterov=args.nesterov)
        optimizer_name = f"SGD(lr={args.lr}, momentum={args.momentum}, nesterov={args.nesterov}, weight_decay={args.weight_decay})"
    elif args.optimizer == "rmsprop":
        opt = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        optimizer_name = f"RMSprop(lr={args.lr}, momentum={args.momentum}, weight_decay={args.weight_decay})"

    # Enhanced learning rate scheduler options
    if args.lr_schedule:
        if args.optimizer in ["sgd", "rmsprop"]:
            # Cosine annealing works better with momentum-based optimizers
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=10, T_mult=2, eta_min=args.lr * 0.01)
            scheduler_name = f"CosineAnnealingWarmRestarts(T_0=10, eta_min={args.lr * 0.01:.1e})"
        else:
            # ReduceLROnPlateau for adaptive optimizers (more patient for smoother curves)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='max', factor=0.7, patience=10, min_lr=1e-7, verbose=False)
            scheduler_name = f"ReduceLROnPlateau(factor=0.7, patience=10, min_lr=1e-7)"
    else:
        scheduler = None
        scheduler_name = "None"

    # Mixed precision scaler
    scaler = GradScaler() if (args.mixed_precision and device == "cuda") else None

    # Calculate class weights for imbalanced dataset with capping
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)

    print(f"\n📊 Class distribution in training set:")
    for cls, count in zip(unique_classes, class_counts):
        print(
            f"   Class {cls}: {count} samples ({100*count/total_samples:.1f}%)")

    # Calculate class weights (used by both CE and Focal Loss)
    class_weights = None
    if not args.no_class_weights:
        class_weights = total_samples / (len(unique_classes) * class_counts)
        class_weights = np.clip(class_weights, 0.1, args.max_class_weight)
        class_weights = class_weights * \
            len(unique_classes) / class_weights.sum()
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)

        print("   Applied class weights (capped):")
        for cls, weight in zip(unique_classes, class_weights):
            print(f"      Class {cls} → weight: {weight:.3f}")
    else:
        print("   Using uniform class weights (no weighting)")
        class_weights_tensor = None

    # Select loss function
    if args.focal_loss:
        print(
            f"   Using Focal Loss (gamma={args.focal_gamma}, better for extreme imbalance)")
        loss_fn = FocalLoss(alpha=class_weights_tensor, gamma=args.focal_gamma)
    else:
        if class_weights_tensor is not None:
            loss_fn = nn.CrossEntropyLoss(
                weight=class_weights_tensor, label_smoothing=0.1)
        else:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Training tracking
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    training_history = []
    best_ckpt_path = None

    print(
        f"Starting training with {len(train_ds)} training samples, {len(val_ds)} validation samples")
    print(
        f"Model: {model.__class__.__name__}(input_size={X_train.shape[2]}, hidden_size={args.hidden}, num_layers={args.num_layers}, dropout={args.dropout}, bidirectional={args.bidirectional})")
    print(f"Optimizer: {optimizer_name}")
    print(f"Scheduler: {scheduler_name}")
    print(
        f"Early stopping patience: {args.patience} epochs (min_delta={args.min_delta})")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            opt.zero_grad()

            # Mixed precision training
            if scaler is not None:
                with autocast():
                    outputs = model(X_batch)
                    loss = loss_fn(outputs, y_batch)

                # Scale loss and backpropagate
                scaler.scale(loss).backward()

                # Gradient clipping (unscale first for proper clipping)
                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=args.grad_clip)

                # Optimizer step with scaler
                scaler.step(opt)
                scaler.update()
            else:
                # Standard precision training
                outputs = model(X_batch)
                loss = loss_fn(outputs, y_batch)
                loss.backward()

                # Gradient clipping
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=args.grad_clip)

                opt.step()

            train_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()

        train_loss /= len(train_ds)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                # Mixed precision inference
                if scaler is not None:
                    with autocast():
                        outputs = model(X_batch)
                        loss = loss_fn(outputs, y_batch)
                else:
                    outputs = model(X_batch)
                    loss = loss_fn(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        val_loss /= len(val_ds)
        val_acc = val_correct / val_total

        # Learning rate scheduling
        if args.lr_schedule:
            if args.optimizer in ["sgd", "rmsprop"]:
                # CosineAnnealingWarmRestarts steps every epoch
                scheduler.step()
            else:
                # ReduceLROnPlateau uses validation accuracy
                scheduler.step(val_acc)

        # Track history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': opt.param_groups[0]['lr']
        })

        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.4f} | LR: {opt.param_groups[0]['lr']:.6f}")

        # Checkpointing - save best model based on validation loss
        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0

            # Save best model
            best_ckpt_path = os.path.join(args.out_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'opt_state': opt.state_dict(),
                'scaler': scaler,
                'args': vars(args),
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'training_history': training_history
            }, best_ckpt_path)
            print(f"  → Saved best model (val_loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(
                f"\nEarly stopping triggered after {epoch} epochs (patience: {args.patience})")
            print(
                f"Best validation loss: {best_val_loss:.6f}, Best validation accuracy: {best_val_acc:.4f}")
            break

    # Load best model for final evaluation
    if os.path.exists(best_ckpt_path):
        map_location = torch.device(
            device) if isinstance(device, str) else device
        checkpoint = torch.load(
            best_ckpt_path, map_location=map_location, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")

    # Save final scaler
    scaler_path = os.path.join(
        args.out_dir, f"scaler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
    if scaler_to_save is not None:
        save_scaler(scaler_to_save, scaler_path)
    else:
        print("Warning: No scaler available to save (skipping scaler export).")

    print(f"\nTraining completed!")
    print(f"Best model saved: {best_ckpt_path}")
    print(f"Scaler saved: {scaler_path}")
    print(f"Final validation accuracy: {best_val_acc:.4f}")

    # Print training summary
    print("\nTraining Summary:")
    print(f"- Total epochs trained: {len(training_history)}")
    print(
        f"- Best epoch: {max(training_history, key=lambda x: x['val_acc'])['epoch']}")
    print(f"- Final learning rate: {training_history[-1]['lr']:.6f}")
    print(
        f"- Training time: ~{len(training_history) * 0.5:.1f} minutes (estimated)")

    # Generate training curves for overfitting detection
    plot_training_curves(training_history, args.out_dir)


def plot_training_curves(training_history, out_dir):
    """Plot training and validation loss/accuracy curves for overfitting detection."""
    epochs = [h['epoch'] for h in training_history]
    train_losses = [h['train_loss'] for h in training_history]
    val_losses = [h['val_loss'] for h in training_history]
    train_accs = [h['train_acc'] for h in training_history]
    val_accs = [h['val_acc'] for h in training_history]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training vs Validation Loss',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Add overfitting indicators
    if len(val_losses) > 10:
        # Check for overfitting: val loss increasing while train loss decreasing
        recent_train = train_losses[-10:]
        recent_val = val_losses[-10:]
        if recent_val[-1] > recent_val[0] and recent_train[-1] < recent_train[0]:
            ax1.axvspan(len(epochs)-10, len(epochs), alpha=0.2,
                        color='red', label='Potential Overfitting')
            ax1.legend(fontsize=11)

    # Accuracy curves
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training vs Validation Accuracy',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Add overfitting indicators for accuracy
    if len(val_accs) > 10:
        recent_train_acc = train_accs[-10:]
        recent_val_acc = val_accs[-10:]
        if recent_val_acc[-1] < recent_val_acc[0] and recent_train_acc[-1] > recent_train_acc[0]:
            ax2.axvspan(len(epochs)-10, len(epochs), alpha=0.2,
                        color='red', label='Potential Overfitting')
            ax2.legend(fontsize=11)

    plt.tight_layout()

    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = os.path.join(out_dir, f'training_curves_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training curves saved: {plot_path}")

    # Print overfitting analysis
    print("\nOverfitting Analysis:")
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    final_train_acc = train_accs[-1]
    final_val_acc = val_accs[-1]

    loss_gap = final_train_loss - final_val_loss
    acc_gap = final_train_acc - final_val_acc

    print(f"- Final train loss: {final_train_loss:.6f}")
    print(f"- Final val loss:   {final_val_loss:.6f}")
    print(f"- Loss gap (train - val): {loss_gap:.6f}")
    print(f"- Final train acc: {final_train_acc:.6f}")
    print(f"- Final val acc:   {final_val_acc:.6f}")
    print(f"- Acc gap (train - val): {acc_gap:.6f}")

    if loss_gap > 0.1:
        print("⚠️  WARNING: Large loss gap suggests potential overfitting!")
    elif loss_gap < -0.1:
        print(
            "⚠️  WARNING: Validation loss lower than training loss - possible underfitting!")

    if acc_gap > 0.1:
        print("⚠️  WARNING: Large accuracy gap suggests potential overfitting!")
    elif acc_gap < -0.1:
        print("⚠️  WARNING: Validation accuracy much lower - possible underfitting!")


def train_with_walk_forward_validation(args):
    """Train model using Walk-Forward Validation for time series."""
    print("🚀 Starting Walk-Forward Validation Training")
    print("=" * 50)

    # Load data - same logic as main function
    dataset_paths = [
        # Preferred: SMC-enhanced dataset
        "/kaggle/working/training_data/smc_lstm_training_dataset.csv",
        # Fallback: basic consolidated dataset
        "/kaggle/working/training_data/consolidated_dataset.csv"
    ]

    dataset_found = False
    for dataset_path in dataset_paths:
        if os.path.exists(dataset_path):
            print(f"📊 Loading prepared dataset: {dataset_path}")
            df = load_csv(dataset_path)
            if 'symbol' in df.columns and 'timeframe' in df.columns:
                print(f"Loaded consolidated data with {len(df)} rows")
                print(f"Symbols: {df['symbol'].unique()}")
                print(f"Timeframes: {df['timeframe'].unique()}")
                if 'label' in df.columns:
                    print(
                        f"Label distribution: {df['label'].value_counts().to_dict()}")
            dataset_found = True
            break

    if not dataset_found:
        raise FileNotFoundError(
            "No suitable training dataset found. Please ensure smc_lstm_training_dataset.csv or consolidated_dataset.csv exists.")

    # Check if dataset is already processed with SMC features and labels
    if 'label' in df.columns:
        print("🎯 Using pre-processed SMC dataset with existing labels...")
        # Use existing labels and quality scores
        labels = df['label'].iloc[args.seq_len:].values
        quality_series = df.get('quality_score', pd.Series(
            [1.0] * len(df))).iloc[args.seq_len:]
        quality_scores = quality_series.values

        if labels.min() < 0:
            label_mapping = {-1: 0, 0: 1, 1: 2}
            labels = np.vectorize(lambda x: label_mapping.get(
                int(x), int(x)))(labels).astype(int)
        else:
            labels = labels.astype(int)

        # Use all SMC and technical features
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'returns', 'log_returns',
            'hour', 'day_of_week', 'month',
            'atr', 'rsi', 'sma_20', 'sma_50', 'ema_20', 'ema_50',
            'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_lower', 'bb_middle', 'volume_sma',
            'trend_bias_indicator', 'volatility_state',
            'ob_bullish', 'ob_bearish', 'ob_high', 'ob_low', 'ob_displacement_atr',
            'ob_size_atr', 'ob_displacement_zscore', 'ob_entry_price', 'distance_to_ob_entry_atr',
            'risk_per_trade_atr',
            'fvg_bullish', 'fvg_bearish', 'fvg_top', 'fvg_bottom', 'fvg_depth_atr',
            'fvg_depth_zscore', 'fvg_entry_price', 'distance_to_fvg_entry_atr',
            'bos_bull_wick', 'bos_bull_close', 'bos_bear_wick', 'bos_bear_close',
            'bos_commitment_flag', 'bos_momentum_atr', 'displacement_mag_zscore',
            'choch_bull_close', 'choch_bear_close', 'choch_bull_wick', 'choch_bear_wick',
            'trend_state', 'recent_bull_break', 'recent_bear_break',
            'signal_direction', 'quality_score', 'trade_return', 'raw_signal_label'
        ]

        existing_feature_cols = [
            col for col in feature_cols if col in df.columns]
        features = df[existing_feature_cols].iloc[args.seq_len:]

    else:
        print("🎯 Processing raw data - generating SMC features and labels...")
        df = prepare_ohlc_series(df)

        # Add technical indicators
        from .data import compute_technical_indicators
        df = compute_technical_indicators(df)

        # Generate enhanced labels with trend filtering and triple barrier
        print("🎯 Generating enhanced SMC labels with trend filtering and triple barrier method...")
        labels, quality_scores = generate_enhanced_smc_labels(
            df,
            seq_len=args.seq_len,
            use_trend_filter=True,
            use_triple_barrier=True,
            min_adx=20
        )

        if labels.min() < 0:
            label_mapping = {-1: 0, 0: 1, 1: 2}
            labels = np.vectorize(lambda x: label_mapping.get(
                int(x), int(x)))(labels).astype(int)
        else:
            labels = labels.astype(int)

        # Create feature matrix
        feature_cols = [col for col in df.columns if col not in [
            'Open', 'High', 'Low', 'Close', 'Volume']]
        features = df[feature_cols].iloc[args.seq_len:]

    print(f"📈 Features shape: {features.shape}")
    print(f"🏷️  Labels shape: {labels.shape}")
    print(f"⭐ Quality scores shape: {quality_scores.shape}")

    # Filter for high-quality signals only
    quality_threshold = getattr(args, 'quality_threshold', 0.7)
    high_quality_mask = quality_scores > quality_threshold
    features_filtered = features[high_quality_mask]
    labels_filtered = labels[high_quality_mask]

    print(f"🎯 High-quality signals: {len(labels_filtered)}/{len(labels)} "
          f"({len(labels_filtered)/len(labels)*100:.1f}%)")

    if len(labels_filtered) < 1000:
        print("⚠️  WARNING: Very few high-quality signals found. Consider lowering quality threshold.")
        return

    # Perform Walk-Forward Validation
    print("🔄 Running Walk-Forward Validation...")
    wf_metrics = evaluate_walk_forward(
        LSTMClassifier,
        features_filtered,
        pd.Series(labels_filtered),
        n_splits=5,
        seq_len=args.seq_len,
        input_size=features_filtered.shape[1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=len(np.unique(labels_filtered)),
        dropout=args.dropout
    )

    print("\n📊 Walk-Forward Validation Results:")
    print("=" * 40)
    print(
        f"Mean Accuracy: {wf_metrics['mean_accuracy']:.4f} ± {wf_metrics['std_accuracy']:.4f}")
    print(
        f"Mean F1 Score: {wf_metrics['mean_f1']:.4f} ± {wf_metrics['std_f1']:.4f}")

    print("\n📋 Per-Split Performance:")
    for i, split in enumerate(wf_metrics['split_metrics']):
        print(f"Split {i+1}: Acc={split['accuracy']:.4f}, F1={split['f1']:.4f}, "
              f"Prec={split['precision']:.4f}, Rec={split['recall']:.4f}")

    # Save results
    results_file = f"walk_forward_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    import json
    with open(results_file, 'w') as f:
        json.dump(wf_metrics, f, indent=2)
    print(f"\n💾 Results saved to {results_file}")

    return wf_metrics


if __name__ == '__main__':
    main()
