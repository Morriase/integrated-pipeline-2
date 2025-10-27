import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import MetaTrader5 as mt5
from sklearn.metrics import confusion_matrix

from .data import download_ticker, load_csv, prepare_ohlc_series, load_scaler, initialize_mt5
from .model import LSTMClassifier
from .utils import create_sequences, SequenceDataset
from torch.utils.data import DataLoader


TIMEFRAME_MAP = {
    'M15': mt5.TIMEFRAME_M15,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
    'D1': mt5.TIMEFRAME_D1,
}


def load_checkpoint(path, device):
    data = torch.load(path, map_location=device)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ticker", type=str, default="EURUSD")
    parser.add_argument("--timeframe", type=str,
                        default="H1", choices=TIMEFRAME_MAP.keys())
    parser.add_argument("--bars", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.checkpoint, device)
    scaler = ckpt.get('scaler', None)
    if scaler is None:
        # attempt to load accompanying scaler file in same folder
        folder = os.path.dirname(args.checkpoint)
        cand = None
        for f in os.listdir(folder):
            if f.startswith('scaler') and f.endswith('.pkl'):
                cand = os.path.join(folder, f)
                break
        if cand:
            scaler = load_scaler(cand)
        else:
            raise RuntimeError("No scaler found in checkpoint or folder")

    model_args = ckpt.get('args', {})
    input_size = 1
    model = LSTMClassifier(input_size=input_size)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    # Load data
    if args.ticker.lower().endswith('.csv'):
        df = load_csv(args.ticker)
    else:
        initialize_mt5()
        save_path = f"DATA/{args.ticker}_{args.timeframe}_{args.bars}bars.csv"
        df = download_ticker(
            args.ticker, TIMEFRAME_MAP[args.timeframe], args.bars, save_path)

    series = prepare_ohlc_series(df)
    close_series = series[['Close']]
    scaled = scaler.transform(close_series.values)
    seqs, targets = create_sequences(scaled, args.seq_len)

    ds = SequenceDataset(seqs, targets)
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    preds = []
    trues = []
    with torch.no_grad():
        for X, y in loader:
            out = model(X.to(device)).cpu().numpy()
            preds.append(out)
            trues.append(y.numpy())

    preds = np.vstack(preds)
    trues = np.vstack(trues)
    pred_classes = np.argmax(preds, axis=1)

    accuracy = np.mean(pred_classes == trues)
    print(f"Accuracy={accuracy:.6f}")

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(trues, pred_classes)
    print("Confusion Matrix:")
    print(cm)

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(trues, pred_classes)
    print("Confusion Matrix:")
    print(cm)


if __name__ == '__main__':
    main()
