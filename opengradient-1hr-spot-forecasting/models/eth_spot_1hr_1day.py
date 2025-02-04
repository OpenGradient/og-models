import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils.parametrizations import weight_norm

# Train on 1 1/2 yrs of data, backtest on 6 month, 3 month 
  # 6 mo: first part could go well, then be bad
  # if it does good on both, then it isn't time-invariant
  # do we need a retrain pipeline for this task?
# Sequential split (last 3 mo, last 6 mo)
# Shoot for 55% directional accuracy in the backtest
# Look for divergence of strategy vs buy & hold
# Think about adding Bitcoin data (experiment with it) 

class EnhancedFeatureLayer(nn.Module):
    """Engineered features from OHLCV over sequences."""
    def __init__(self, seq_length: int = 24):
        super().__init__()
        self.seq_length = seq_length
        # Despite the docstring claiming 32, we output 25 features.
        self.output_dim = 25

    def compute_ema(self, x: torch.Tensor, span: int) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        alpha = 2 / (span + 1)
        alpha_rev = 1 - alpha
        pows = torch.pow(alpha_rev, torch.arange(x.size(1), device=x.device, dtype=x.dtype))
        scale = 1 / torch.cumsum(pows, 0)
        weighted_sum = x * pows.view(1, -1)
        cumsum = torch.cumsum(weighted_sum, dim=1)
        return cumsum * scale.view(1, -1)
    
    def compute_rsi(self, prices, window=14):
        # Using slicing instead of torch.diff
        # prices assumed shape: [batch, seq_length]
        deltas = prices[:, 1:] - prices[:, :-1]
        gain = torch.where(deltas > 0, deltas, torch.zeros_like(deltas))
        loss = torch.where(deltas < 0, -deltas, torch.zeros_like(deltas))
        kernel = torch.ones(window, device=prices.device) / window
        gain_padded = F.pad(gain, (window - 1, 0), mode='replicate')
        loss_padded = F.pad(loss, (window - 1, 0), mode='replicate')
        avg_gain = F.conv1d(gain_padded.unsqueeze(1), kernel.view(1, 1, -1)).squeeze(1)
        avg_loss = F.conv1d(loss_padded.unsqueeze(1), kernel.view(1, 1, -1)).squeeze(1)
        rs = avg_gain / (avg_loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def compute_obv(self, close, volume):
        # Using slicing instead of torch.diff
        price_change = close[:, 1:] - close[:, :-1]
        multiplier = torch.where(price_change > 0, torch.ones_like(price_change), -torch.ones_like(price_change))
        multiplier = torch.where(price_change == 0, torch.zeros_like(price_change), multiplier)
        multiplier = F.pad(multiplier, (1, 0), value=0.0)
        obv = (volume * multiplier).cumsum(dim=1)
        return obv
    
    def forward(self, x):
        bsz, seqlen, _ = x.shape
        device = x.device
        open_, high, low, close, volume = x.unbind(-1)
        eps = 1e-8
        
        log_close = torch.log(close + eps)
        returns = torch.zeros_like(log_close)
        returns[:, 1:] = log_close[:, 1:] - log_close[:, :-1]
        
        hl_ratio = (high - low) / (close + eps)
        oc_abs   = torch.abs(open_ - close) / (close + eps)
        
        close_ema3  = self.compute_ema(close, 3)
        close_ema6  = self.compute_ema(close, 6)
        close_ema12 = self.compute_ema(close, 12)
        close_ema24 = self.compute_ema(close, 24)
        
        vol_ema6  = self.compute_ema(volume, 6)
        vol_ema12 = self.compute_ema(volume, 12)
        
        t = torch.arange(seqlen, device=device).float()
        hour_sin = torch.sin(2 * math.pi * (t % 24) / 24).expand(bsz, -1)
        hour_cos = torch.cos(2 * math.pi * (t % 24) / 24).expand(bsz, -1)
        
        momentum_3 = torch.zeros_like(log_close)
        momentum_6 = torch.zeros_like(log_close)
        momentum_3[:, 3:] = log_close[:, 3:] - log_close[:, :-3]
        momentum_6[:, 6:] = log_close[:, 6:] - log_close[:, :-6]
        
        rsi = F.pad(self.compute_rsi(close, 14), (1, 0), mode='replicate')
        obv = self.compute_obv(close, volume)
        
        features = torch.stack([
            open_, high, low, close, volume,
            returns, returns.abs(),
            hl_ratio, oc_abs,
            close_ema3 / close, close_ema6 / close,
            close_ema12 / close, close_ema24 / close,
            vol_ema6 / (volume + eps), vol_ema12 / (volume + eps),
            torch.log1p(volume),
            hour_sin, hour_cos,
            momentum_3, momentum_6,
            (close - close_ema12) / (close_ema12 + eps),
            rsi, obv,
            returns[:, -1:].expand(-1, seqlen),
            hl_ratio[:, -1:].expand(-1, seqlen)
        ], dim=-1)
        
        features = torch.nan_to_num(features, nan=0.0)
        features = F.instance_norm(features.permute(0, 2, 1)).permute(0, 2, 1)
        return features

class CombinedLoss(nn.Module):
    def __init__(self, cls_weight=0.3):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_criterion = nn.HuberLoss(delta=0.1)
        self.cls_criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, reg_pred, dir_pred, targets):
        # Regression loss
        reg_loss = self.reg_criterion(reg_pred.squeeze(), targets)
        
        # Classification loss (direction prediction)
        cls_target = (targets > 0).float()
        cls_loss = self.cls_criterion(dir_pred.squeeze(), cls_target)
        
        # Combined loss
        return reg_loss + self.cls_weight * cls_loss, reg_loss, cls_loss

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        # Using new weight_norm API
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.bn2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, 1),
            nn.BatchNorm1d(n_outputs)
        ) if n_inputs != n_outputs else None
        
        self.init_weights()
    
    def init_weights(self):
        for layer in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(layer.weight.data, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias.data, 0)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample[0].weight.data)
            if self.downsample[0].bias is not None:
                nn.init.constant_(self.downsample[0].bias.data, 0)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

class ETHSpot1hr1day(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        torch.set_default_dtype(torch.float32)
        
        self.input_dim = config.get('input_dim', 5)
        self.seq_length = config.get('seq_length', 24)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.tcn_channels = config.get('tcn_channels', 128)
        
        # Use tcn_channels instead of hidden_dim for TCN
        self.tcn = nn.Sequential(
            TemporalBlock(self.input_dim, self.tcn_channels, kernel_size=3, stride=1, dilation=2, dropout=config.get('dropout', 0.2)),
            TemporalBlock(self.tcn_channels, self.tcn_channels, kernel_size=3, stride=1, dilation=4, dropout=config.get('dropout', 0.2)),
            TemporalBlock(self.tcn_channels, self.tcn_channels, kernel_size=3, stride=1, dilation=8, dropout=config.get('dropout', 0.2)),
            TemporalBlock(self.tcn_channels, self.tcn_channels, kernel_size=3, stride=1, dilation=16, dropout=config.get('dropout', 0.2))
        )
        
        # Use explicit FC dimensions
        fc_dim = config.get('fc_dim', 64)
        self.fc1 = nn.Linear(self.tcn_channels, fc_dim)
        self.bn1 = nn.BatchNorm1d(fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)
        self.dropout = nn.Dropout(config.get('dropout', 0.2))
        
        # Remove dynamic residual connection
        self.res_scale = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        x = x.transpose(1, 2)
        
        # TCN with residual
        tcn_out = self.tcn(x)
        
        # Global average pooling with attention-like weighting
        weights = F.softmax(tcn_out.mean(1, keepdim=True), dim=2)
        pooled = (tcn_out * weights).sum(dim=2)
        
        # MLP with residual connection and batch norm
        out = self.fc1(pooled)
        out = self.bn1(out)
        out = F.silu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out