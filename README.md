# dodosu-gen!mania v0.0.1

#### NOTE: A renewed version with a techinically better architecture is presented at https://github.com/dodobird1/dododual and is still under development. Currently, dododual doesn't work and this one does though. 
A CNN/RNN-based generator with self-attention of osu!mania 4K beatmaps powered by **RhythmNet** created by dodobird1. 

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

## Citing

If you find this code useful in your research or projects, even just a quick "thank you" or a link back to this repository is very helpful for me and **greatly appreciated** (but not required under the lisence)! 

**Recommended Citation:**
```bibtex
@misc{dodosu-gen2025,
  author = {dodobird1},
  title = {dodosu-gen!mania v0.0.1},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/dodobird1/dodosu-gen}}
}
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/dodobird1/dodosu-gen.git
cd dodosu-gen-mania

# It is recommended to create a new python environment, e.g. using Anaconda
conda create --name dodosu python==3.10

# Install dependencies
pip install torch torchaudio numpy tqdm x-transformers
```

---

## Usage

*Currently the installation and training processes are only tested on Ubuntu 24.04 for Osu!Mania 4K only. The inference is tested to work on Ubuntu and for CPU on MacOS under conda.*

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

- [x] Core model and training pipeline
- [ ] Data augmentation (time stretching, cropping)
- [ ] More data (waiting for 2025 to end)
- [ ] Non-4K formats (5K, 7K, etc.)
- [ ] GUI application
- [ ] osu!taiko support
- [ ] osu!standard support
- [ ] osu!catch support
- [ ] More powerful model when I have more computational power

---

## Acknowledgements

*No meaning implied by the order of listing.*

- **osu!** — For keeping such a nice, warm, open-source community
- **Salty Mermaid** — From the osu! community, who provided a list of all 2024 ranked and loved beatmaps which served as the training set; they are also working on the 2025 set, which I am also hoping to use. 
- **DiffRhythm & Tencent Music Entertainment (TME) Group** — For introducing me to Music+AI and all its possibilities
- **Mr. Xinning Zhang** — For his excellent AI class!
- **PerseverantDT** — For their JS-based parser of .osu files on GitHub

---

## License

See [LICENSE](LICENSE) for details.
