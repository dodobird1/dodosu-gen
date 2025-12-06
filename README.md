# dodosu-gen!mania v0.0.1

A CNN/RNN-based generator of osu!mania 4K beatmaps powered by **RhythmNet** created by dodobird1. 

> ⚠️ **USE THIS MODEL RESPONSIBLY**  
> Disclose any use of AI in the creation of beatmaps. The creator of this model is not responsible for any consequences caused by using this model, especially for plagiarism or any kind of violation of copyright.

---

## Features

- 🎵 Generates osu!mania 4K beatmaps from audio files
- 🧠 **RhythmNet** architecture: CNN encoder + Bidirectional LSTM + Self-Attention
- ⚡ Mixed precision training (AMP) for efficient GPU utilization
- 📦 Direct `.osz` export for easy import into osu!
- 🎛️ Configurable note density and chord complexity

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dodosu-gen-mania.git
cd dodosu-gen-mania

# Install dependencies
pip install torch torchaudio numpy tqdm x-transformers
```

---

## Usage

### Training

```bash
python -m train.train \
    --data_dir /path/to/osu/songs \
    --batch_size 8 \
    --epochs 30 \
    --cache
```

### Inference

```bash
python -m infer.inference \
    --audio song.mp3 \
    --model model/best.pt \
    --output song.osu \
    --bpm 180 \
    --threshold 0.7 \
    --osz
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--audio` | Path to input audio file | Required |
| `--model` | Path to trained model checkpoint | Required |
| `--output` | Output .osu file path | `output.osu` |
| `--bpm` | Song BPM | `120.0` |
| `--threshold` | Note detection threshold | `0.7` |
| `--high_conf` | Threshold for 3 simultaneous notes | `0.90` |
| `--mid_conf` | Threshold for 2 simultaneous notes | `0.75` |
| `--osz` | Create .osz package | `False` |

---

## Project Structure

```
dodosu-gen-mania/
├── train/
│   └── train.py          # Training script with RhythmNet model
├── infer/
│   └── inference.py      # Inference and .osu/.osz generation
├── pretrain/
│   └── data_preparation.py  # Dataset and data loading
├── reference/
│   ├── modules.py        # Neural network modules
│   └── osu_mania_parser.py  # .osu file parser
└── model/                # Saved checkpoints
```

---

## Roadmap

- [x] Core model and training pipeline (current data: ranked 2024, part of? ranked 2023)
- [ ] Data augmentation (time stretching, cropping)
- [ ] More data and Loved maps
- [ ] Non-4K formats (5K, 7K, etc.)
- [ ] GUI application
- [ ] osu!taiko support
- [ ] osu!standard support
- [ ] osu!catch support

---

## Acknowledgements

*No meaning implied by the order of listing.*

- **osu!** — For keeping such a nice, warm, open-source community
- **Salty Mermaid** — From the osu! community, who provided a list of all 2024 ranked and loved beatmaps which served as the training set
- **DiffRhythm & Tencent Music Entertainment (TME) Group** — For introducing me to Music+AI and all its possibilities
- **Mr. Xinning Zhang** — For his excellent AI class!
- **PerseverantDT** — For their JS-based parser of .osu files on GitHub

---

## License

See [LICENSE.md](LICENSE.md) for details.
