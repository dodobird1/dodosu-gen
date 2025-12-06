import os
import sys
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json

# Add the project root to sys.path to allow importing from reference/
# Assuming this file is located in /mnt/code/elec/osu/pretrain/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from reference.osu_mania_parser import parse_beatmap, Beatmap
    from reference.modules import MelSpec
except ImportError as e:
    print(f"Error importing reference modules: {e}")
    print("Please ensure 'reference/osu_mania_parser.py' and 'reference/modules.py' exist.")
    sys.exit(1)

class OsuManiaDataset(Dataset):
    def __init__(self, 
                 root_dir: str | list[str], 
                 keys: int = 4, 
                 sample_rate: int = 24000, 
                 hop_length: int = 256,
                 max_duration: int = None,
                 cache_processed: bool = False,
                 augment_columns: int = 0):
        """
        Dataset for Osu!Mania maps.
        
        Args:
            root_dir: Directory or list of directories containing mapsets.
            keys: Number of keys to filter for (default 4).
            sample_rate: Target sample rate for audio (must match MelSpec config).
            hop_length: Hop length for MelSpectrogram (must match MelSpec config).
            max_duration: Max duration in seconds to load (truncates). None for full song.
            cache_processed: If True, saves/loads .pt tensors to speed up subsequent loads.
            augment_columns: Number of additional column-permuted versions per song (default 0).
                            Set to 2 to get 3 total versions (original + 2 permutations).
        """
        self.root_dirs = [root_dir] if isinstance(root_dir, str) else root_dir
        self.keys = keys
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_duration = max_duration
        self.cache_processed = cache_processed
        self.augment_columns = augment_columns
        
        # Initialize MelSpec extractor (on CPU for data loading)
        self.mel_extractor = MelSpec(
            target_sample_rate=sample_rate,
            hop_length=hop_length,
            n_mel_channels=100,
            normalize=True
        )
        
        self.map_entries = self._scan_maps()
        
        # Create augmented entries (each song appears augment_columns+1 times)
        self.num_versions = 1 + augment_columns
        self.total_entries = len(self.map_entries) * self.num_versions
        
        if augment_columns > 0:
            print(f"Found {len(self.map_entries)} valid {self.keys}K maps across {len(self.root_dirs)} directories.")
            print(f"  Column augmentation: {self.num_versions} versions per song ({self.total_entries} total)")
        else:
            print(f"Found {len(self.map_entries)} valid {self.keys}K maps across {len(self.root_dirs)} directories.")

    def _scan_maps(self):
        """Scans all root_dirs for valid .osu files and pairs them with audio."""
        entries = []
        
        for root_dir in self.root_dirs:
            if not os.path.exists(root_dir):
                print(f"Warning: Directory not found: {root_dir}")
                continue
                
            for root, dirs, files in os.walk(root_dir):
                # First, find valid audio files in this directory
                audio_files = [f for f in files if f.lower().endswith(('.mp3', '.wav', '.ogg'))]
                if not audio_files:
                    continue
                
                # Find the largest audio file (assumption: song is the main content)
                try:
                    main_audio = max(audio_files, key=lambda f: os.path.getsize(os.path.join(root, f)))
                except OSError:
                    continue # Skip if file access fails
                    
                # Iterate through .osu files and pair them with the main audio
                for file in files:
                    if file.endswith(".osu"):
                        full_path = os.path.join(root, file)
                        try:
                            entry = self._validate_map(full_path, main_audio)
                            if entry:
                                entries.append(entry)
                        except Exception as e:
                            continue
        return entries

    def _validate_map(self, osu_path: str, audio_filename: str):
        """
        Validates if the map matches criteria and pairs it with the provided audio file.
        Returns a dict with paths and metadata if valid, else None.
        """
        audio_path = os.path.join(os.path.dirname(osu_path), audio_filename)
        
        if not os.path.exists(audio_path):
            return None

        # Use the reference parser
        try:
            beatmap = parse_beatmap(osu_path)
        except ValueError:
             # Wrong mode
            return None

        if beatmap.key_count != self.keys:
            return None
            
        # Extract BPM
        bpm = (beatmap.min_bpm + beatmap.max_bpm) / 2.0
            
        return {
            "osu_path": osu_path,
            "audio_path": audio_path,
            "beatmap_obj": beatmap,
            "bpm": bpm
        }

    def __len__(self):
        return self.total_entries

    def __getitem__(self, idx):
        # Determine which base entry and which augmentation version
        base_idx = idx // self.num_versions
        version = idx % self.num_versions  # 0 = original, 1+ = permuted
        
        entry = self.map_entries[base_idx]
        
        # cache path (only for original version)
        cache_path = entry["osu_path"] + ".pt"
        
        if self.cache_processed and os.path.exists(cache_path):
            try:
                data = torch.load(cache_path)
                # Apply column permutation if this is an augmented version
                if version > 0:
                    data = self._apply_column_permutation(data, version)
                return data
            except Exception:
                pass  # Fallback to reprocessing
        
        # 1. Process Audio
        mel_spec = self._process_audio(entry["audio_path"])
        
        # 2. Process Chart (Labels)
        # mel_spec shape: [n_mels, time_frames]
        # We need labels of shape [time_frames, keys]
        num_frames = mel_spec.shape[1]
        labels = self._process_chart(entry["beatmap_obj"], num_frames)
        
        # 3. BPM Feature
        bpm_tensor = torch.tensor([entry["bpm"]], dtype=torch.float32)
        
        data = {
            "mel": mel_spec,      # [Channels, Time]
            "labels": labels,     # [Time, Keys]
            "bpm": bpm_tensor,    # [1]
            "path": entry["osu_path"]
        }
        
        if self.cache_processed:
            torch.save(data, cache_path)
        
        # Apply column permutation if this is an augmented version
        if version > 0:
            data = self._apply_column_permutation(data, version)
            
        return data
    
    def _apply_column_permutation(self, data, version):
        """
        Apply a deterministic column permutation based on version number.
        
        Args:
            data: Dictionary with 'labels' key of shape [Time, Keys]
            version: Which permutation to apply (1, 2, etc.)
        
        Returns:
            Data dictionary with permuted labels
        """
        # Create a copy to avoid modifying cached data
        data = data.copy()
        labels = data["labels"].clone()
        
        # Generate a deterministic permutation based on version
        # Using a simple rotation + shuffle pattern
        num_keys = labels.shape[1]
        
        if version == 1:
            # Reverse column order
            perm = list(range(num_keys - 1, -1, -1))
        elif version == 2:
            # Rotate by half
            half = num_keys // 2
            perm = list(range(half, num_keys)) + list(range(half))
        else:
            # For additional versions, use a seeded random permutation
            rng = np.random.default_rng(seed=version * 42)
            perm = rng.permutation(num_keys).tolist()
        
        # Apply permutation
        data["labels"] = labels[:, perm]
        data["path"] = data["path"] + f"_perm{version}"
        
        return data

    def _process_audio(self, audio_path):
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            # Create dummy audio if load fails (shouldn't happen if validated)
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros((100, 1000))

        # Mix to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Compute MelSpec
        # MelSpec expects [Batch, 1, Time] or [Batch, Time]
        # We add a batch dim: [1, Time]
        with torch.no_grad():
            # Use the extractor from modules.py
            # It expects shape [Batch, Length] usually, but let's check modules.py
            # forward(self, inp): if len(inp.shape) == 3: squeeze...
            # So we can pass [1, Time]
            mel = self.mel_extractor(waveform)
            # Output is [Batch, Mels, Time] -> [1, 100, T]
            mel = mel.squeeze(0) # -> [100, T]
            
        return mel

    def _process_chart(self, beatmap: Beatmap, num_frames: int):
        """
        Converts beatmap hit objects into a binary grid.
        Shape: [num_frames, keys]
        """
        labels = torch.zeros((num_frames, self.keys), dtype=torch.float32)
        
        # Time per frame in seconds
        frame_time = self.hop_length / self.sample_rate
        
        for obj in beatmap.hit_objects:
            # Convert ms to seconds then to frame index
            t_sec = obj.time / 1000.0
            frame_idx = int(t_sec / frame_time)
            
            if frame_idx < num_frames:
                # Determine column (0 to keys-1)
                # formula: floor(x * keys / 512)
                col = int(obj.x * self.keys / 512)
                col = min(max(col, 0), self.keys - 1) # Clamp just in case
                
                labels[frame_idx, col] = 1.0
                
                # Optional: Handle Hold notes (body)
                # If you want to represent hold bodies, you can set them to 1 or another value here.
                # For simple onset detection, we only mark the start.
                
        return labels

def collate_fn(batch, max_frames=None):
    """
    Collate function to pad sequences to the max length in the batch.
    Optionally crops sequences to max_frames to limit memory usage.
    
    Args:
        batch: List of samples from dataset
        max_frames: Maximum sequence length. If provided, sequences longer than this
                   will be randomly cropped. None = no limit (use full length).
                   Recommended: 4000 (~42 sec) for 16GB VRAM, 2000 (~21 sec) for 8GB
    """
    # Sort by length for packed_sequence compatibility if using RNNs (optional but good practice)
    batch.sort(key=lambda x: x['mel'].shape[1], reverse=True)
    
    mels = [x['mel'] for x in batch]
    labels = [x['labels'] for x in batch]
    bpms = [x['bpm'] for x in batch]
    paths = [x['path'] for x in batch]
    
    # Apply random cropping if max_frames is set
    if max_frames is not None:
        cropped_mels = []
        cropped_labels = []
        actual_lengths = []
        
        for mel, label in zip(mels, labels):
            seq_len = mel.shape[1]
            
            if seq_len > max_frames:
                # Random crop
                max_start = seq_len - max_frames
                start = torch.randint(0, max_start + 1, (1,)).item()
                mel = mel[:, start:start + max_frames]
                label = label[start:start + max_frames, :]
                actual_lengths.append(max_frames)
            else:
                actual_lengths.append(seq_len)
            
            cropped_mels.append(mel)
            cropped_labels.append(label)
        
        mels = cropped_mels
        labels = cropped_labels
    
    # Pad Mels [Channels, Time] -> pad Time dimension
    # Pad Labels [Time, Keys] -> pad Time dimension
    
    max_len = max(mel.shape[1] for mel in mels)
    
    # Pad Mels
    padded_mels = torch.zeros(len(batch), mels[0].shape[0], max_len)
    for i, mel in enumerate(mels):
        length = mel.shape[1]
        padded_mels[i, :, :length] = mel
        
    # Pad Labels
    padded_labels = torch.zeros(len(batch), max_len, labels[0].shape[1])
    for i, label in enumerate(labels):
        length = label.shape[0]
        padded_labels[i, :length, :] = label
        
    padded_bpms = torch.stack(bpms)
    
    # Create lengths tensor for masking
    lengths = torch.tensor([mel.shape[1] for mel in mels], dtype=torch.long)
    
    return {
        "mel": padded_mels,
        "labels": padded_labels,
        "bpm": padded_bpms,
        "lengths": lengths,
        "paths": paths
    }


def make_collate_fn(max_frames=None):
    """
    Factory function to create a collate_fn with a specific max_frames setting.
    
    Args:
        max_frames: Maximum sequence length for random cropping.
                   Recommended values:
                   - 4000 frames (~42 sec) for 16GB VRAM with batch_size=16
                   - 3000 frames (~32 sec) for 16GB VRAM with batch_size=32
                   - 2000 frames (~21 sec) for 8GB VRAM
    
    Returns:
        A collate function with the specified max_frames
    """
    def _collate(batch):
        return collate_fn(batch, max_frames=max_frames)
    return _collate

if __name__ == "__main__":
    # Example usage
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--keys", type=int, default=4, help="Key count (4 or 7)")
    args = parser.parse_args()
    
    dataset = OsuManiaDataset(root_dir=args.data_dir, keys=args.keys, cache_processed=False)
    
    if len(dataset) > 0:
        sample = dataset[0]
        print("Sample processed:")
        print(f"Mel shape: {sample['mel'].shape}")
        print(f"Labels shape: {sample['labels'].shape}")
        
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(loader))
        print("Batch shape (Mel):", batch['mel'].shape)
        print("Batch shape (Labels):", batch['labels'].shape)
    else:
        print("No valid maps found.")

