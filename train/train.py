import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import argparse
import os
import math
from tqdm import tqdm

# Import your dataset and collate function
from pretrain.data_preparation import OsuManiaDataset, make_collate_fn

# ==========================================
# Configuration / Defaults
# ==========================================

DATA_DIR = ["/mnt/p/dodosu/2024/","/mnt/p/dodosu/2023/","/mnt/p/dodosu/2022/","/mnt/p/dodosu/2021/","/mnt/p/dodosu/2020/","/mnt/p/dodosu/2015","/mnt/p/dodosu/2016","/mnt/p/dodosu/2017","/mnt/p/dodosu/2018","/mnt/p/dodosu/2019"]  # Can be a list of directories
MODEL_SAVE_DIR = "/mnt/code/elec/osu/model/"
KEYS = 4
EPOCHS = 30
BATCH_SIZE = 8  # Reduced for memory safety with attention layers
MAX_FRAMES = 2560  # 3000 # ~32 seconds max per sample (prevents OOM)
LR = 3e-4  # Lower LR for larger model
CACHE_PROCESSED = True  # Enable caching by default for faster training
NUM_CPU_THREADS = 24  # More workers for your 7950X
CUDA_AVAILABLE = True
USE_AMP = True  # Automatic Mixed Precision for ~50% memory reduction
AUGMENT_COL = 0 # Closed for now
CHORD_PEN = 0

# ==========================================
# Model Architecture - Enhanced RhythmNet
# ==========================================

class ResidualConvBlock(nn.Module):
    """Residual convolutional block with skip connection."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
        )
        
        # Skip connection with 1x1 conv if dimensions change
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.GELU()
        
    def forward(self, x):
        return self.activation(self.conv(x) + self.skip(x))


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for capturing long-range temporal dependencies."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [Batch, Time, Dim]
            mask: [Batch, Time] boolean mask (True = valid, False = padding)
        """
        B, T, C = x.shape
        residual = x
        x = self.norm(x)
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, T, head_dim]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, T, T]
        
        # Apply mask if provided
        if mask is not None:
            # mask: [B, T] -> [B, 1, 1, T]
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~attn_mask, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return residual + x


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return x + self.net(self.norm(x))


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward."""
    def __init__(self, dim, num_heads=8, ff_mult=4, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.ff = FeedForward(dim, ff_mult, dropout)
        
    def forward(self, x, mask=None):
        x = self.attn(x, mask)
        x = self.ff(x)
        return x


class RhythmNet(nn.Module):
    def __init__(self, n_mels=100, hidden_size=256, num_lstm_layers=3, num_attn_layers=2, 
                 num_heads=8, keys=4, dropout=0.15):
        """
        Enhanced CRNN architecture with attention for beatmap generation.
        
        Architecture:
        1. Deep CNN encoder with residual connections and multi-scale features
        2. Bidirectional LSTM for sequential modeling
        3. Transformer blocks with self-attention for long-range dependencies
        4. Deep classifier head
        
        Estimated parameters: ~8-10M (suitable for 16GB VRAM with batch_size=32-64)
        """
        super(RhythmNet, self).__init__()
        
        self.hidden_size = hidden_size
        
        # ===== 1. CNN Encoder with Residual Blocks =====
        # Multi-scale feature extraction with increasing receptive field
        self.cnn = nn.Sequential(
            # Initial projection
            nn.Conv1d(n_mels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            
            # Residual blocks with increasing dilation for multi-scale features
            ResidualConvBlock(64, 128, kernel_size=3, dilation=1),
            ResidualConvBlock(128, 128, kernel_size=3, dilation=2),
            ResidualConvBlock(128, 256, kernel_size=3, dilation=1),
            ResidualConvBlock(256, 256, kernel_size=3, dilation=2),
            ResidualConvBlock(256, hidden_size, kernel_size=3, dilation=1),
            
            nn.Dropout(dropout)
        )
        
        # ===== 2. Bidirectional LSTM =====
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Project LSTM output (bidirectional doubles the size)
        self.lstm_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # ===== 3. Self-Attention Layers =====
        self.attn_layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ff_mult=4, dropout=dropout)
            for _ in range(num_attn_layers)
        ])
        
        # ===== 4. Classifier Head =====
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 4, keys)
        )
        
        # BPM Projection
        self.bpm_proj = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.GELU()
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better training dynamics."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x, bpm, mask=None):
        """
        Args:
            x: [Batch, n_mels, Time] - Mel spectrogram input
            bpm: [Batch, 1] - BPM input
            mask: [Batch, Time] - Boolean mask (True = valid, False = padding)
        
        Returns:
            logits: [Batch, Time, Keys]
        """
        # CNN Pass: [Batch, n_mels, Time] -> [Batch, hidden_size, Time]
        x = self.cnn(x)
        
        # Inject BPM info (concatenate to features or project)
        # Simple approach: Project BPM to hidden_size and add to CNN features (broadcasting)
        # bpm: [Batch, 1] -> Project -> [Batch, hidden_size, 1]
        # Note: This requires a new projection layer in __init__
        if hasattr(self, 'bpm_proj'):
            bpm_emb = self.bpm_proj(bpm).unsqueeze(-1) # [Batch, Hidden, 1]
            x = x + bpm_emb # Broadcast add
        
        # Permute for RNN: [Batch, Time, hidden_size]
        x = x.permute(0, 2, 1)
        
        # LSTM Pass: [Batch, Time, hidden_size*2]
        x, _ = self.rnn(x)
        
        # Project back to hidden_size
        x = self.lstm_proj(x)
        
        # Self-Attention Layers
        for attn_layer in self.attn_layers:
            x = attn_layer(x, mask)
        
        # Classifier: [Batch, Time, Keys]
        logits = self.classifier(x)
        
        return logits
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==========================================
# Training Loop
# ==========================================

def create_mask(lengths, max_len, device):
    """Create boolean mask from lengths tensor."""
    batch_size = lengths.size(0)
    mask = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
    mask = mask < lengths.unsqueeze(1)
    return mask


def masked_bce_loss(logits, targets, mask, pos_weight):
    """
    Compute BCE loss only on valid (non-padded) positions.
    
    Args:
        logits: [B, T, K]
        targets: [B, T, K]
        mask: [B, T] boolean mask
        pos_weight: [K] positive class weights
    """
    # Expand mask for keys dimension
    mask_expanded = mask.unsqueeze(-1)  # [B, T, 1]
    
    # Compute per-element loss
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    loss = loss_fn(logits, targets)  # [B, T, K]
    
    # Apply mask
    loss = loss * mask_expanded
    
    # Average over valid elements
    num_valid = mask_expanded.sum() * targets.size(-1)
    return loss.sum() / (num_valid + 1e-8)


def chord_penalty_loss(logits, mask, max_expected=1.5):
    """
    Penalize predictions where too many keys are active simultaneously.
    
    This encourages the model to learn realistic note distributions where
    1-2 simultaneous notes are common, and 3-4 are rare.
    
    Args:
        logits: [B, T, K] - raw model outputs (before sigmoid)
        mask: [B, T] - boolean mask for valid positions
        max_expected: Maximum expected notes per frame before penalty applies
                     (1.5 means we expect ~1-2 notes on average)
    
    Returns:
        Scalar penalty loss
    """
    probs = torch.sigmoid(logits)  # [B, T, K]
    
    # Sum of probabilities per frame (expected number of notes)
    notes_per_frame = probs.sum(dim=-1)  # [B, T]
    
    # Penalize when expected notes > max_expected
    excess = torch.relu(notes_per_frame - max_expected)  # [B, T]
    
    # Apply mask to only count valid frames
    if mask is not None:
        excess = excess * mask.float()
        penalty = excess.sum() / (mask.sum() + 1e-8)
    else:
        penalty = excess.mean()
    
    return penalty


def train(args):
    # 1. Setup Device
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Using device: {device}")
    
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Setup AMP (Automatic Mixed Precision)
    use_amp = args.amp and use_cuda
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        print("Using Automatic Mixed Precision (AMP) for memory efficiency")
    
    # Handle data_dir being a list or string
    # If argparse nargs='+' is used, it's always a list. 
    # But if default is string, we might need to wrap it.
    data_dirs = args.data_dir
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
        
    for d in data_dirs:
        if not os.path.exists(d):
            print(f"Error: Data directory '{d}' does not exist.")
            return # Or continue/exit based on preference
            
    if not data_dirs:
        print("Error: No data directories provided.")
        return # or exit

    # 2. Dataset & Dataloader
    dataset = OsuManiaDataset(
        root_dir=data_dirs, 
        keys=args.keys,
        cache_processed=args.cache,
        augment_columns=args.augment_columns
    )
    
    if len(dataset) == 0:
        print(f"No valid {args.keys}K maps found in the provided data directories.")
        return

    # Create collate function with max_frames for memory control
    collate_fn = make_collate_fn(max_frames=args.max_frames)
    if args.max_frames:
        print(f"Max sequence length: {args.max_frames} frames (~{args.max_frames * 256 / 24000:.1f} seconds)")

    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=use_cuda,
        persistent_workers=args.workers > 0
    )
    
    # 3. Model Initialization
    model = RhythmNet(
        hidden_size=args.hidden_size,
        num_lstm_layers=args.num_lstm_layers,
        num_attn_layers=args.num_attn_layers,
        num_heads=args.num_heads,
        keys=args.keys,
        dropout=args.dropout
    ).to(device)
    
    num_params = model.count_parameters()
    print(f"Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    # 4. Loss & Optimizer
    pos_weight = torch.ones([args.keys]).to(device) * args.pos_weight
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # 5. Resume from checkpoint if exists
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # 6. Training Loop
    print(f"\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Dataset size: {len(dataset)} maps")
    print(f"  Batches per epoch: {len(train_loader)}")
    print()
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            # Move data to device
            mel = batch['mel'].to(device, non_blocking=True)       # [B, 100, T]
            targets = batch['labels'].to(device, non_blocking=True) # [B, T, K]
            bpm = batch['bpm'].to(device, non_blocking=True)       # [B, 1]
            lengths = batch['lengths'].to(device, non_blocking=True)
            
            # Create mask for valid positions
            max_len = mel.size(2)
            mask = create_mask(lengths, max_len, device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with AMP autocast
            with autocast(enabled=use_amp):
                logits = model(mel, bpm, mask)  # [B, T, K]
                bce_loss = masked_bce_loss(logits, targets, mask, pos_weight)
                
                # Chord penalty: discourage too many simultaneous notes
                chord_pen = chord_penalty_loss(logits, mask, max_expected=args.max_chord)
                
                # Combined loss
                loss = bce_loss + args.chord_penalty_weight * chord_pen
            
            # Backward with gradient scaling for AMP
            scaler.scale(loss).backward()
            
            # Gradient clipping (unscale first for correct norm)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        # Step scheduler
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save Checkpoint
        save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Save latest
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'args': vars(args)
        }
        torch.save(checkpoint, os.path.join(save_dir, "latest.pt"))
        
        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, os.path.join(save_dir, "best.pt"))
            print(f"  New best model saved! (loss: {best_loss:.4f})")
        
        # Save periodic checkpoints
        if (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(checkpoint, save_path)
            print(f"  Checkpoint saved to {save_path}")
    
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Models saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RhythmNet for osu!mania beatmap generation")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, nargs='+', default=DATA_DIR, 
                        help=f"Path(s) to osu! songs folder. Can specify multiple paths. (default: DATA_DIR in code)")
    parser.add_argument("--save_dir", type=str, default=MODEL_SAVE_DIR, 
                        help=f"Directory to save model checkpoints (default: {MODEL_SAVE_DIR})")
    parser.add_argument("--keys", type=int, default=KEYS, 
                        help=f"Key mode (4 or 7, default: {KEYS})")
    parser.add_argument("--cache", action="store_true", default=CACHE_PROCESSED, 
                        help="Cache processed tensors to disk (recommended)")
    parser.add_argument("--augment_columns", type=int, default=AUGMENT_COL, 
                        help="Number of additional column-permuted versions per song (default: 2, total 3 versions)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=EPOCHS, 
                        help=f"Number of epochs (default: {EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, 
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES, 
                        help=f"Max sequence length in frames, prevents OOM (default: {MAX_FRAMES}, ~32 sec)")
    parser.add_argument("--lr", type=float, default=LR, 
                        help=f"Learning rate (default: {LR})")
    parser.add_argument("--weight_decay", type=float, default=0.01, 
                        help="Weight decay for AdamW (default: 0.01)")
    parser.add_argument("--pos_weight", type=float, default=15.0, 
                        help="Positive class weight for BCE loss (default: 15.0)")
    parser.add_argument("--chord_penalty_weight", type=float, default=CHORD_PEN, 
                        help=f"Weight for chord penalty loss (default: {CHORD_PEN})")
    parser.add_argument("--max_chord", type=float, default=1.5, 
                        help="Max expected simultaneous notes before penalty (default: 1.5)")
    parser.add_argument("--workers", type=int, default=NUM_CPU_THREADS, 
                        help=f"Number of data loading workers (default: {NUM_CPU_THREADS})")
    parser.add_argument("--cuda", action="store_true", default=CUDA_AVAILABLE, 
                        help="Use CUDA if available")
    parser.add_argument("--amp", action="store_true", default=USE_AMP, 
                        help="Use Automatic Mixed Precision for ~50%% memory reduction (default: True)")
    parser.add_argument("--no_amp", action="store_false", dest="amp",
                        help="Disable Automatic Mixed Precision")
    
    # Model architecture arguments
    parser.add_argument("--hidden_size", type=int, default=256, 
                        help="Hidden size for LSTM and attention (default: 256)")
    parser.add_argument("--num_lstm_layers", type=int, default=3, 
                        help="Number of LSTM layers (default: 3)")
    parser.add_argument("--num_attn_layers", type=int, default=2, 
                        help="Number of transformer/attention layers (default: 2)")
    parser.add_argument("--num_heads", type=int, default=8, 
                        help="Number of attention heads (default: 8)")
    parser.add_argument("--dropout", type=float, default=0.15, 
                        help="Dropout rate (default: 0.15)")
    
    # Checkpointing
    parser.add_argument("--resume", type=str, default=None, 
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save_every", type=int, default=5, 
                        help="Save checkpoint every N epochs (default: 5)")
    
    args = parser.parse_args()
    
    # Check paths (optional, dataset class also checks)
    # args.data_dir is now a list
    
    train(args)
