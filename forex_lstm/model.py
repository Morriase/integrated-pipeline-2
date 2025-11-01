import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 128, num_layers: int = 4,
                 dropout: float = 0.2, num_classes: int = 3, bidirectional: bool = True):
        super().__init__()

        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Multi-layer Bidirectional LSTM with residual connections
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Calculate output size considering bidirectional
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # Enhanced classifier head with batch normalization for smoother training
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout / 2),  # Lighter dropout for final layer
            nn.Linear(hidden_size // 4, num_classes)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using Xavier uniform initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)

        # For bidirectional LSTM, we need to handle the last timestep differently
        if self.bidirectional:
            # Concatenate forward and backward outputs from last timestep
            last_forward = lstm_out[:, -1, :self.hidden_size]
            last_backward = lstm_out[:, 0, self.hidden_size:]
            last_hidden = torch.cat([last_forward, last_backward], dim=1)
        else:
            # Standard unidirectional case
            last_hidden = lstm_out[:, -1, :]

        # Pass through classifier head
        return self.classifier(last_hidden)
