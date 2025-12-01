import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    LSTM-based model for time series prediction.
    Predicts the probability of price moving UP (1) or DOWN (0).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 1, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Activation for binary classification (Sigmoid)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        
        # Apply activation
        out = self.sigmoid(out)
        return out

class TransformerModel(nn.Module):
    """
    Transformer-based model for time series prediction.
    Uses self-attention to capture long-range dependencies.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 1, dropout: float = 0.1, nhead: int = 4):
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 500, hidden_dim)) # Max seq len 500
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = self.embedding(x)
        
        # Add positional encoding (broadcasted)
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Transformer Encoder
        out = self.transformer_encoder(x)
        
        # Use the last time step
        out = self.fc(out[:, -1, :])
        
        return self.sigmoid(out)

def get_model(model_type: str, input_dim: int, **kwargs):
    """Factory method to get model by name."""
    if model_type.lower() == "lstm":
        return LSTMModel(input_dim, **kwargs)
    elif model_type.lower() == "transformer":
        return TransformerModel(input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
