import torch
import torch.nn as nn
import torch.nn.functional as F

class TIAM(nn.Module):
    def __init__(self, num_class=2):
        super(TIAM, self).__init__()
        
        # Low-level features processing
        self.fc_low = nn.Linear(312, 256)
        self.lstm = nn.LSTM(256, 128, bidirectional=True, batch_first=True)
        self.tanh = nn.Tanh()
        self.dropout1 = nn.Dropout(0.1)
        self.relu1 = nn.ReLU()
        
        # High-level features processing (Attention mechanism)
        self.attention = nn.Linear(256, 768)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        
        # Classification layers
        self.fc_concat = nn.Linear(256 + 768, num_class)  # Concatenation

    def forward(self, x_low, x_high):

        # Process low-level features
        x_low = self.fc_low(x_low)
        x_low, _ = self.lstm(x_low)
        x_low = self.tanh(x_low)
        x_low = self.dropout1(x_low)
        x_low = self.relu1(x_low)

        # Process high-level features through attention mechanism
        attn_weights = F.softmax(self.attention(x_low), dim=1)
        x_high_attn = torch.sum(attn_weights.unsqueeze(1) * x_high, dim=1)
        x_high_attn = self.relu2(x_high_attn)
        x_high_attn = self.dropout2(x_high_attn)
        
        # Concatenation
        x_concat = torch.cat((x_low, x_high_attn), dim=1)
        out_concat = self.fc_concat(x_concat)
        
        return out_concat
