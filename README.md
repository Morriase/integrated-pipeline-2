# Forex LSTM Classification (PyTorch) - SMC Strategy

This repository contains a comprehensive, production-ready LSTM neural network for forex price action classification (Buy, Sell, Hold) based on Smart Money Concepts (SMC) strategy. Features advanced architecture, CUDA optimization, and comprehensive evaluation tools.

## 🎯 Key Features

- **Advanced PyTorch LSTM**: Deep bidirectional architecture with 4 layers and enhanced classifier head
- **SMC Strategy Implementation**: Order Blocks, Fair Value Gaps, Break of Structure detection with ATR normalization
- **CUDA-Optimized Training**: Mixed precision FP16, GPU memory management, optimized data loading
- **Overfitting Detection**: Automatic training curves plotting with loss/accuracy visualization
- **MT5 Integration**: Real-time forex data from MetaTrader 5 across multiple timeframes
- **Comprehensive Training**: Early stopping, learning rate scheduling, L2 regularization, gradient clipping
- **Production Ready**: Checkpointing, evaluation metrics, confusion matrices, and deployment tools

## 📁 Project Structure

```
forex_lstm/
├── __init__.py
├── data.py          # SMC feature engineering & MT5 data processing
├── model.py         # Enhanced bidirectional LSTM classifier
├── train.py         # CUDA-optimized training with curves plotting
├── evaluate.py      # Comprehensive evaluation & metrics
└── utils.py         # Dataset utilities & helpers

Scripts:
├── download_data.py              # Batch MT5 data download
├── consolidate_data.py           # Data consolidation
├── create_smc_training_dataset.py # SMC feature engineering
└── requirements.txt              # Dependencies

Data:
├── DATA/                        # MT5 downloaded CSV files
├── consolidated_dataset.csv     # Consolidated training data
├── smc_lstm_training_dataset.csv # SMC-enhanced features
└── Resources/                   # Documentation & research
```

## 🚀 Quick Start (Windows PowerShell)

### 1. Setup Environment
```powershell
# Create virtual environment
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Forex Data
```powershell
# Batch download multiple symbols and timeframes
python download_data.py --symbols EURUSD GBPUSD USDJPY AUDUSD --timeframes M15 H1 H4 --bars 10000
```

### 3. Prepare SMC Training Dataset
```powershell
# Create consolidated dataset
python consolidate_data.py --wide-format

# Generate SMC features (Order Blocks, FVGs, BOS, ATR normalization)
python create_smc_training_dataset.py
```

### 4. Train Enhanced LSTM Model
```powershell
# Use smart defaults (automatically detects SMC dataset)
python -m forex_lstm.train
```

### 5. Evaluate Model Performance
```powershell
# Evaluate with confusion matrix and metrics
python -m forex_lstm.evaluate --checkpoint checkpoints/best_model.pt
```

## 🧠 Enhanced LSTM Architecture

### **Deep Bidirectional LSTM**
- **4 LSTM Layers** (vs 2 previously) for deeper temporal learning
- **Bidirectional Processing** for past + future context understanding
- **Enhanced Classifier Head**: Multi-layer dense network with ReLU activations
- **Advanced Initialization**: Xavier uniform weights for stable training

### **Architecture Details**
```
Input (seq_len=60, features=5) → Bidirectional LSTM (4 layers, hidden=128)
                                    ↓
Feature Extraction → Dense(64) → ReLU → Dropout(0.2)
                     ↓
Classification → Dense(32) → ReLU → Dropout(0.1) → Dense(3)
```

### **Model Capacity**: ~1.3M parameters for complex SMC pattern recognition

## ⚡ CUDA-Optimized Training

### **Performance Features**
- **Mixed Precision Training**: FP16 for 2x speed and 50% memory reduction
- **GPU Memory Management**: Configurable memory fraction usage
- **Optimized Data Loading**: Pinned memory, multi-worker prefetching
- **Multi-GPU Support**: Automatic DataParallel scaling
- **cuDNN Optimization**: Benchmark mode for faster convolutions

### **Training Curves & Overfitting Detection**
- **Automatic Plotting**: Loss and accuracy curves saved as PNG
- **Overfitting Analysis**: Gap detection with warning thresholds
- **Visual Indicators**: Red zones highlight potential overfitting regions

## 🎛️ Training Configuration

### **Default Parameters**
```python
epochs: 100 (early stopping)
hidden_size: 128
num_layers: 4 (bidirectional)
dropout: 0.2
batch_size: 64
learning_rate: 0.001 (scheduled)
weight_decay: 0.0001 (L2 regularization)
sequence_length: 60
```

### **Advanced Training Examples**

```powershell
# High-performance training with custom architecture
python -m forex_lstm.train --hidden 256 --num-layers 6 --lr 0.0005 --batch 128

# GPU optimization settings
python -m forex_lstm.train --device cuda --gpu-memory-fraction 0.9 --mixed-precision

# CPU training fallback
python -m forex_lstm.train --device cpu --no-bidirectional

# Custom dataset and timeframe
python -m forex_lstm.train --ticker EURUSD --timeframe H4 --bars 5000

# Disable bidirectional for faster training
python -m forex_lstm.train --no-bidirectional --num-layers 3
```

## 📊 SMC Strategy Implementation

### **Smart Money Concepts Features**
- **Order Blocks**: Bullish/bearish order block detection with volume validation
- **Fair Value Gaps**: Imbalance detection with gap-filling probability
- **Break of Structure**: Trend change identification with momentum confirmation
- **ATR Normalization**: Volatility-adjusted feature scaling
- **Multi-Timeframe Analysis**: Cross-timeframe signal confirmation

### **Label Generation**
- **Buy Signal**: Break above order block + FVG fill confirmation
- **Sell Signal**: Break below order block + FVG fill confirmation
- **Hold Signal**: No clear SMC setup or conflicting signals

## 🔧 Data Pipeline

### **MT5 Integration**
- Real-time forex data access
- Multiple timeframe support (M15, H1, H4, D1)
- Automatic data validation and cleaning

### **Feature Engineering**
- OHLCV price data
- SMC technical indicators
- Volatility measures (ATR)
- Volume-based confirmations

### **Dataset Formats**
- **Raw MT5 Data**: Individual CSV files per symbol/timeframe
- **Consolidated Data**: Single CSV with all symbols/timeframes
- **SMC Training Data**: Enhanced features with SMC labels

## 📈 Evaluation & Metrics

### **Comprehensive Metrics**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix visualization
- Per-class performance analysis
- Training curves and overfitting detection

### **Model Persistence**
- Checkpoint saving with full training state
- Scaler preservation for inference
- Timestamped model versions

## 🎯 Usage Examples

### **Basic Training**
```powershell
python -m forex_lstm.train
```

### **Custom Configuration**
```powershell
python -m forex_lstm.train --epochs 200 --hidden 256 --lr 0.0003 --patience 20
```

### **GPU Training**
```powershell
python -m forex_lstm.train --device cuda --mixed-precision --gpu-memory-fraction 0.8
```

### **Evaluation**
```powershell
python -m forex_lstm.evaluate --checkpoint checkpoints/best_model.pt --plot-confusion
```

## 📋 Requirements

- Python 3.8+
- MetaTrader 5 (for data download)
- PyTorch 1.13+ with CUDA support (recommended)
- 8GB+ RAM, NVIDIA GPU recommended for training

## 🔄 Next Steps & Improvements

- **Walk-Forward Validation**: Time-aware cross-validation for time series
- **Ensemble Methods**: Multiple model combination for robustness
- **Advanced SMC Features**: Triple Barrier Method, trend filtering
- **Real-time Inference**: Live trading integration with MT5
- **Hyperparameter Optimization**: Automated parameter tuning

## 📚 Documentation

- `Resources/`: Research papers and SMC strategy documentation
- Training curves automatically saved in `checkpoints/` directory
- Model checkpoints include full training history and configuration

---

**Built for production forex trading with institutional-grade SMC analysis.** 🚀