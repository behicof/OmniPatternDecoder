import torch
import torch.nn as nn
import numpy as np

class CosmicDecoderNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, seq_length):
        super(CosmicDecoderNetwork, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=4,
            dim_feedforward=hidden_dim*2,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=2
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=4
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, seq_length, hidden_dim)
        
        # Prepare for transformer (needs seq_length, batch_size, hidden_dim)
        transformer_input = lstm_out.permute(1, 0, 2)
        
        # Apply transformer
        transformer_out = self.transformer_encoder(transformer_input)
        # transformer_out shape: (seq_length, batch_size, hidden_dim)
        
        # Apply attention
        attn_output, _ = self.attention(
            transformer_out, 
            transformer_out, 
            transformer_out
        )
        # attn_output shape: (seq_length, batch_size, hidden_dim)
        
        # Convert back to batch_first
        out = attn_output.permute(1, 0, 2)
        
        # Use only the last time step for prediction
        out = self.fc(out[:, -1, :])
        
        return out

class DataPreprocessor:
    def __init__(self):
        self.price_scaler = None
        self.astro_scaler = None
        
    def prepare_sequence_data(self, price_data, astro_data, seq_length=30, prediction_horizon=5):
        """
        Prepare sequence data for LSTM model
        
        Parameters:
        price_data: DataFrame of price data
        astro_data: Numpy array of astronomical data
        seq_length: Length of input sequences
        prediction_horizon: How many days ahead to predict
        
        Returns:
        X: Input sequences (price + astro data)
        y: Target values (future price changes)
        """
        # Extract features from price data
        close = price_data['Close'].values
        volume = price_data['Volume'].values
        
        # Calculate technical indicators
        rsi_period = 14
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.convolve(gain, np.ones(rsi_period)/rsi_period, mode='valid')
        avg_loss = np.convolve(loss, np.ones(rsi_period)/rsi_period, mode='valid')
        
        # Pad beginning to maintain length
        pad_len = len(close) - len(avg_gain)
        avg_gain = np.pad(avg_gain, (pad_len, 0), 'constant', constant_values=(0, 0))
        avg_loss = np.pad(avg_loss, (pad_len, 0), 'constant', constant_values=(0, 0))
        
        # Calculate RSI
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        
        # Combine all features
        price_features = np.column_stack([
            close,
            volume,
            rsi
        ])
        
        # Create sequences
        X = []
        y = []
        
        for i in range(len(price_features) - seq_length - prediction_horizon + 1):
            # Input sequence
            price_seq = price_features[i:i+seq_length]
            astro_seq = astro_data[i:i+seq_length]
            
            # Scale data
            if self.price_scaler is None:
                self.price_scaler = MinMaxScaler()
                self.astro_scaler = MinMaxScaler()
                price_seq_scaled = self.price_scaler.fit_transform(price_seq)
                astro_seq_scaled = self.astro_scaler.fit_transform(astro_seq)
            else:
                price_seq_scaled = self.price_scaler.transform(price_seq)
                astro_seq_scaled = self.astro_scaler.transform(astro_seq)
            
            # Combine price and astro data
            combined_seq = np.hstack([price_seq_scaled, astro_seq_scaled])
            
            # Target: future price change percentage
            current_price = close[i+seq_length-1]
            future_price = close[i+seq_length+prediction_horizon-1]
            price_change = ((future_price - current_price) / current_price) * 100
            
            X.append(combined_seq)
            y.append(price_change)
        
        return np.array(X), np.array(y)

# Example usage
if __name__ == "__main__":
    # Assuming you have price_data and astro_data from OmniPatternDecoder
    from omnipattern_decoder import OmniPatternDecoder
    
    decoder = OmniPatternDecoder()
    results = decoder.run_analysis('GC=F', '2020-01-01', '2023-01-01')
    
    # Prepare data
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_sequence_data(
        results['price_data'],
        results['astro_data']
    )
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).view(-1, 1)
    
    # Define model parameters
    input_dim = X.shape[2]  # Combined price and astro features
    hidden_dim = 64
    num_layers = 2
    output_dim = 1
    seq_length = 30
    
    # Create model
    model = CosmicDecoderNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        seq_length=seq_length
    )
    
    print(f"Model created with input dimension: {input_dim}")
    print(model)