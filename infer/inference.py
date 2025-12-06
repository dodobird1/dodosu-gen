import torch
import torchaudio
import argparse
import os
import sys
import numpy as np
import zipfile
from tqdm import tqdm

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from train.train import RhythmNet
    from reference.modules import MelSpec
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def load_audio(audio_path, sample_rate=24000):
    """Loads audio and converts to MelSpectrogram."""
    waveform, sr = torchaudio.load(audio_path)
    
    # Mix to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
        
    # Extract Mel
    mel_extractor = MelSpec(
        target_sample_rate=sample_rate,
        hop_length=256,
        n_mel_channels=100,
        normalize=True
    )
    
    with torch.no_grad():
        # MelSpec expects [Batch, Time] or [Batch, 1, Time]
        # We provide [1, Time]
        mel = mel_extractor(waveform)
        # Output: [1, Mels, Time]
        
    return mel

def generate_map(model, mel_spec, bpm, threshold=0.5, device='cpu',
                 max_conf_thresh=0.95, high_conf_thresh=0.90, mid_conf_thresh=0.75,
                 shuffle_prob=0.0):
    """
    Runs inference with probability-weighted simultaneous note limiting.
    
    Uses dynamic max simultaneous notes based on confidence:
    - Maximum confidence (>max_conf_thresh): allow up to 4 simultaneous (full chord)
    - Very high confidence (>high_conf_thresh): allow up to 3 simultaneous
    - High confidence (>mid_conf_thresh): allow up to 2 simultaneous  
    - Lower confidence: allow only 1 note
    
    Args:
        model: Trained RhythmNet model
        mel_spec: Mel spectrogram [1, Mels, Time]
        bpm: BPM of the song
        threshold: Base threshold for note placement
        device: torch device
        max_conf_thresh: Threshold for allowing 4 simultaneous notes (default: 0.95)
        high_conf_thresh: Threshold for allowing 3 simultaneous notes (default: 0.90)
        mid_conf_thresh: Threshold for allowing 2 simultaneous notes (default: 0.75)
        shuffle_prob: Probability of shuffling each note to a random unoccupied column (default: 0.0)
    """
    model.eval()
    num_keys = mel_spec.shape[1] if len(mel_spec.shape) == 2 else 4  # Default to 4 keys
    
    with torch.no_grad():
        # Input: [1, Mels, Time]
        # No mask needed for inference (single sample, no padding)
        
        # Expand BPM to batch size
        bpm_tensor = torch.tensor([bpm], dtype=torch.float32).to(device).unsqueeze(0)  # [1, 1]
        
        logits = model(mel_spec, bpm_tensor, mask=None)
        # Output: [1, Time, Keys]
        
        probs = torch.sigmoid(logits).cpu().numpy()[0]  # [Time, Keys]
        num_keys = probs.shape[1]
    
    # Probability-weighted note placement
    predictions = np.zeros_like(probs)
    
    for frame_idx in range(probs.shape[0]):
        frame_probs = probs[frame_idx]
        above_threshold = frame_probs > threshold
        
        if above_threshold.any():
            # Dynamic max simultaneous based on confidence level
            max_prob = frame_probs.max()
            if max_prob > max_conf_thresh:
                max_simul = 4  # Full chord allowed
            elif max_prob > high_conf_thresh:
                max_simul = 3
            elif max_prob > mid_conf_thresh:
                max_simul = 2
            else:
                max_simul = 1
            
            # Sort keys by probability (highest first)
            sorted_indices = np.argsort(frame_probs)[::-1]
            
            # Place notes for top keys up to max_simul
            count = 0
            for key in sorted_indices:
                if frame_probs[key] > threshold and count < max_simul:
                    predictions[frame_idx, key] = 1
                    count += 1
    
    # Apply column shuffle if enabled
    if shuffle_prob > 0:
        predictions = _shuffle_columns(predictions, shuffle_prob, num_keys)
    
    return predictions, probs


def _shuffle_columns(predictions, shuffle_prob, num_keys):
    """
    Randomly shuffle notes to unoccupied columns with given probability.
    
    For each note, there's a shuffle_prob chance it will be moved to a 
    random column that isn't already occupied in that frame.
    
    Args:
        predictions: [Time, Keys] binary note predictions
        shuffle_prob: Probability of shuffling each note (0.0 to 1.0)
        num_keys: Number of columns/keys
    
    Returns:
        Modified predictions with shuffled columns
    """
    result = predictions.copy()
    
    for frame_idx in range(result.shape[0]):
        # Find occupied columns in this frame
        occupied = np.where(result[frame_idx] == 1)[0]
        
        if len(occupied) == 0 or len(occupied) >= num_keys:
            continue  # No notes or all columns full
        
        # Find unoccupied columns
        all_cols = set(range(num_keys))
        occupied_set = set(occupied)
        
        # Process each note for potential shuffle
        for col in occupied:
            if np.random.random() < shuffle_prob:
                # Find available columns (unoccupied, excluding current)
                available = list(all_cols - occupied_set)
                
                if available:
                    # Pick a random available column
                    new_col = np.random.choice(available)
                    
                    # Move the note
                    result[frame_idx, col] = 0
                    result[frame_idx, new_col] = 1
                    
                    # Update occupied set
                    occupied_set.remove(col)
                    occupied_set.add(new_col)
    
    return result

def write_osu_file(predictions, audio_path, output_path, bpm, keys=4, hop_length=256, sample_rate=24000):
    """Writes the predictions to a valid .osu file."""
    
    filename = os.path.basename(audio_path)
    name_no_ext = os.path.splitext(filename)[0]
    
    # Calculate ms per beat
    beat_len = 60000 / bpm
    
    header = f"""osu file format v14

[General]
AudioFilename: {filename}
AudioLeadIn: 0
PreviewTime: -1
Countdown: 0
SampleSet: Normal
StackLeniency: 0.7
Mode: 3
LetterboxInBreaks: 0
WidescreenStoryboard: 0

[Editor]
DistanceSpacing: 1.2
BeatDivisor: 4
GridSize: 8
TimelineZoom: 2

[Metadata]
Title:{name_no_ext}
TitleUnicode:{name_no_ext}
Artist:(not set)
ArtistUnicode:(not set)
Creator:dodosu-gen!mania v0.0.1
Version:AI Difficulty (BPM {bpm}, Params: {args.high_conf}, {args.mid_conf}, {args.threshold})
Source:
Tags:AI generated rhythmnet
BeatmapID:0
BeatmapSetID:0

[Difficulty]
HPDrainRate:8
CircleSize:{keys}
OverallDifficulty:8
ApproachRate:5
SliderMultiplier:1.4
SliderTickRate:1

[Events]
//Background and Video events
//Break Periods
//Storyboard Layer 0 (Background)
//Storyboard Layer 1 (Fail)
//Storyboard Layer 2 (Pass)
//Storyboard Layer 3 (Foreground)
//Storyboard Sound Samples

[TimingPoints]
0,{beat_len},4,2,1,60,1,0


[HitObjects]
"""
    
    hit_objects = []
    frame_time = hop_length / sample_rate
    
    # predictions: [TimeFrames, Keys]
    for frame_idx in range(predictions.shape[0]):
        for key in range(keys):
            if predictions[frame_idx, key] == 1:
                # Time in ms
                time_ms = int(frame_idx * frame_time * 1000)
                
                # X position for osu!mania keys
                # Using standard formula mapping
                x = int((key * 512) / keys + (512/keys)/2)
                y = 192  # Standard Y
                
                # Type: 1 (Circle)
                # HitSound: 0
                # Extras: 0:0:0:0:
                line = f"{x},{y},{time_ms},1,0,0:0:0:0:"
                hit_objects.append(line)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write("\n".join(hit_objects))
        
    print(f"Generated {len(hit_objects)} notes to {output_path}")


def create_osz(osu_path, audio_path, output_path=None):
    """
    Package an .osu file and audio into a playable .osz file.
    
    Args:
        osu_path: Path to the generated .osu file
        audio_path: Path to the source audio file
        output_path: Output .osz path (default: same name as osu file in same directory)
    
    Returns:
        Path to the created .osz file
    """
    if output_path is None:
        output_path = osu_path.rsplit('.', 1)[0] + '.osz'
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add the .osu file
        zf.write(osu_path, arcname=os.path.basename(osu_path))
        # Add the audio file
        zf.write(audio_path, arcname=os.path.basename(audio_path))
    
    print(f"Created .osz package: {output_path}")
    return output_path


def load_model(model_path, keys, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model args from checkpoint if available
    if 'args' in checkpoint:
        saved_args = checkpoint['args']
        model = RhythmNet(
            hidden_size=saved_args.get('hidden_size', 256),
            num_lstm_layers=saved_args.get('num_lstm_layers', 3),
            num_attn_layers=saved_args.get('num_attn_layers', 2),
            num_heads=saved_args.get('num_heads', 8),
            keys=keys,
            dropout=0.0  # No dropout during inference
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Legacy checkpoint format
        model = RhythmNet(keys=keys).to(device)
        model.load_state_dict(checkpoint)
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate osu!mania beatmap from audio using trained RhythmNet")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--model", type=str, default='model/best.pt', help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--output", type=str, default="output.osu", help="Path to output .osu file")
    parser.add_argument("--keys", type=int, default=4, help="Key count (default: 4)")
    parser.add_argument("--threshold", type=float, default=0.75, help="Threshold for note placement (default: 0.7)")
    parser.add_argument("--bpm", type=float, default=120.0, help="BPM of the song (default: 120.0)")
    parser.add_argument("--max_conf", type=float, default=0.97, 
                        help="Confidence threshold to allow 4 simultaneous notes (default: 0.95)")
    parser.add_argument("--high_conf", type=float, default=0.90, 
                        help="Confidence threshold to allow 3 simultaneous notes (default: 0.90)")
    parser.add_argument("--mid_conf", type=float, default=0.80, 
                        help="Confidence threshold to allow 2 simultaneous notes (default: 0.80)")
    parser.add_argument("--shuffle", type=float, default=0.0, 
                        help="Probability of shuffling each note to a random unoccupied column (default: 0.0)")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for inference")
    parser.add_argument("--osz", action="store_true", help="Create .osz package (includes audio, ready to import into osu!)")

    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
        print("Audio file not found.")
        sys.exit(1)
        
    if not os.path.exists(args.model):
        print("Model checkpoint not found.")
        sys.exit(1)
        
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if not args.cpu else 'cpu'
    print(f"Using device: {device}")
    
    model = load_model(args.model, args.keys, device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Process
    print("Processing audio...")
    mel = load_audio(args.audio).to(device)  # [1, Mels, Time]
    print(f"Audio length: {mel.shape[2] * 256 / 24000:.1f} seconds ({mel.shape[2]} frames)")
    
    print("Generating chart...")
    print(f"  Confidence thresholds: max(4)={args.max_conf}, high(3)={args.high_conf}, mid(2)={args.mid_conf}")
    if args.shuffle > 0:
        print(f"  Column shuffle probability: {args.shuffle:.0%}")
    # Create a dummy tqdm bar for inference to show "progress" (though it's instantaneous for small files)
    # For real-time progress on long files, we'd need to chunk inference, but for now we just show it's happening
    with tqdm(total=1, desc="Inference") as pbar:
        predictions, probs = generate_map(
            model, mel, args.bpm, args.threshold, device,
            max_conf_thresh=args.max_conf, high_conf_thresh=args.high_conf, mid_conf_thresh=args.mid_conf,
            shuffle_prob=args.shuffle
        )
        pbar.update(1)
    
    print("Writing file...")
    write_osu_file(predictions, args.audio, args.output, args.bpm, keys=args.keys)
    
    # Create .osz package if requested
    if args.osz:
        create_osz(args.output, args.audio)
    
    # Stats
    total_notes = predictions.sum()
    duration_sec = mel.shape[2] * 256 / 24000
    nps = total_notes / duration_sec
    print(f"\nStats:")
    print(f"  Duration: {duration_sec:.1f}s")
    print(f"  Total notes: {int(total_notes)}")
    print(f"  Notes per second: {nps:.2f}")
