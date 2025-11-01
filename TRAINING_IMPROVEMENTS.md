# Training Improvements for Smoother Convergence

## Summary of Changes

### 1. **Lower Quality Threshold** (0.7 → 0.3)
- **Previous**: Only 259 samples retained from 151,624 (0.17%)
- **Expected**: ~15,000-30,000 samples (10-20% retention)
- **Benefit**: Dramatically more training data while maintaining reasonable quality

### 2. **Class Imbalance Handling**
- **Added**: Automatic class weight calculation
  ```
  Class weights = total_samples / (num_classes × class_count)
  ```
- **Added**: Label smoothing (0.1) to prevent overconfidence
- **Benefit**: Model learns minority classes (-1, +1) better instead of defaulting to majority class (0)

### 3. **Improved Model Architecture**
- **Added**: Batch Normalization layers after each Linear layer
  - Normalizes activations → smoother gradients → faster convergence
  - Reduces internal covariate shift
- **Increased**: Hidden size 128 → 256 (more capacity for 60+ features)
- **Reduced**: Layers 4 → 3 (prevents overfitting with small dataset)
- **Increased**: Dropout 0.2 → 0.3 (better regularization)

### 4. **Better Hyperparameters**
- **Batch size**: 64 → 32 (smaller batches = noisier but more frequent updates)
- **Learning rate**: 0.001 → 0.0005 (smoother optimization)
- **Weight decay**: 0.0001 → 0.00001 (lighter L2 regularization)
- **Max epochs**: 100 → 200 (more training time)
- **Patience**: 15 → 30 (less premature stopping)

### 5. **Smoother Learning Rate Schedule**
- **ReduceLROnPlateau improvements**:
  - Factor: 0.5 → 0.7 (gentler reductions)
  - Patience: 5 → 10 (more epochs before reducing)
  - Min LR: 1e-6 → 1e-7 (can train longer at low LR)

### 6. **Gradient Clipping**
- **Added**: `--grad-clip` parameter (default: 1.0)
- **Benefit**: Prevents exploding gradients, stabilizes training
- **Configurable**: Set to 0 to disable, or adjust threshold

## Expected Improvements

### Training Curves
- ✅ **Smoother validation loss**: Class weights prevent collapse to majority class
- ✅ **Less oscillation**: Smaller LR, batch norm, gradient clipping stabilize updates
- ✅ **Better convergence**: More data, more epochs, patient scheduler
- ✅ **Higher accuracy**: Balanced learning across all 3 classes

### Before vs After
| Metric | Before | Expected After |
|--------|--------|----------------|
| Training samples | 207 | ~12,000-24,000 |
| Validation accuracy | 53.85% (stuck) | 60-75% (improving) |
| Training stability | High oscillation | Smooth curves |
| Class learning | Only class 0 | All 3 classes |
| Loss convergence | Plateaued at 0.69 | Converges < 0.5 |

## Usage

### On Kaggle
```bash
cd /kaggle/working/integrated-pipeline-2
git pull origin main

# Train with new defaults (already optimized)
python -m forex_lstm.train

# Or customize further
python -m forex_lstm.train \
    --quality-threshold 0.3 \
    --epochs 200 \
    --batch 32 \
    --lr 0.0005 \
    --hidden 256 \
    --dropout 0.3 \
    --grad-clip 1.0 \
    --patience 30
```

### Monitor Training
The training will now show:
```
📊 Class distribution in training set:
   Class 0: XXXX samples (XX.X%) → weight: X.XXX
   Class 1: XXXX samples (XX.X%) → weight: X.XXX
   Class 2: XXXX samples (XX.X%) → weight: X.XXX
```

Watch for:
- Validation accuracy should **steadily increase** (not flat)
- Training accuracy should be **smooth** (not wild swings)
- Loss curves should **converge together** (not diverge)

## Technical Details

### Class Weight Formula
```python
unique_classes, class_counts = np.unique(y_train, return_counts=True)
total_samples = len(y_train)
class_weights = total_samples / (len(unique_classes) * class_counts)
```

Example with 15,000 samples:
- Class 0 (neutral): 13,000 samples → weight = 15000/(3×13000) = 0.38
- Class 1 (loss): 1,200 samples → weight = 15000/(3×1200) = 4.17
- Class 2 (win): 800 samples → weight = 15000/(3×800) = 6.25

The loss function penalizes misclassifying minority classes **6-16x more** than majority class.

### Batch Normalization Effect
```python
# Before: x → Linear → ReLU → Dropout
# After:  x → Linear → BatchNorm → ReLU → Dropout
```

BatchNorm standardizes activations to mean=0, std=1, preventing:
- Vanishing gradients (values too small)
- Exploding gradients (values too large)
- Internal covariate shift (distribution changes between layers)

### Label Smoothing
Instead of hard targets [0, 0, 1], uses soft targets [0.033, 0.033, 0.933]:
- Prevents overconfidence
- Better calibration
- Improved generalization

## Troubleshooting

### If curves still oscillate
- Reduce learning rate: `--lr 0.0001`
- Increase grad clip: `--grad-clip 0.5`
- Reduce batch size: `--batch 16`

### If overfitting (train >> val)
- Increase dropout: `--dropout 0.4`
- Increase weight decay: `--weight-decay 0.0001`
- Reduce model size: `--hidden 128 --num-layers 2`

### If underfitting (both accuracies low)
- Lower quality threshold: `--quality-threshold 0.2`
- Increase model capacity: `--hidden 512`
- Train longer: `--epochs 300 --patience 50`

## Next Steps

1. **Run training with new configuration**
2. **Monitor class weights** - ensure they're reasonable (1-10x range)
3. **Check sample count** - should be 10,000+ after threshold 0.3
4. **Validate curves** - should be much smoother
5. **If successful**: Experiment with threshold 0.4-0.5 for even better quality

Good luck! 🚀
