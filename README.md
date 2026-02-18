# AlphaZero for Togyz Kumalak

A PyTorch implementation of DeepMind's AlphaZero applied to **Togyz Kumalak** (Тоғыз Құмалақ) — a traditional Central Asian mancala board game.

Based on the AlphaZero framework from [The Art of Reinforcement Learning](https://link.springer.com/book/10.1007/978-1-4842-9606-6) by Michael Hu, adapted and extended for Togyz Kumalak.

---

## Table of Contents

- [About the Game](#about-the-game)
- [Model Architecture](#model-architecture)
- [Play Against the AI](#play-against-the-ai)
- [Train From Scratch](#train-from-scratch)
- [Resume Training](#resume-training)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [License](#license)

---

## About the Game

Togyz Kumalak is a two-player strategy game originating from Central Asia (Kazakhstan, Kyrgyzstan). The rules:

- **Board**: 2 rows of 9 pits (otau), each player owns one row, plus a scoring pit (kazan)
- **Stones**: 162 total (9 per pit initially)
- **Sowing**: Counter-clockwise. Pick up all stones from a pit and distribute one per pit. If picking up more than 1 stone, leave one in the origin pit
- **Capture**: If the last stone lands in an opponent's pit making the count **even**, capture all stones there
- **Tuzdyk (Tuz)**: If the last stone lands in an opponent's pit making the count exactly **3**, that pit becomes your permanent tuzdyk — all stones that land there go to your kazan. Rules: max 1 tuzdyk per player, can't be pit 9, both players can't tuzdyk the same pit
- **Win condition**: First player to collect **82+ stones** (out of 162) wins

---

## Model Architecture

The neural network follows the AlphaZero architecture: a shared ResNet backbone with separate policy and value heads.

```
Input: (25, 2, 9) tensor
  - 4 frames of history x 6 channels each + 1 color plane
  - Each frame: own pits, opponent pits, own kazan, opponent kazan, own tuzdyk, opponent tuzdyk

         ┌─────────────────────┐
         │   Conv2d 3x3, 128   │  Initial convolution block
         │   BatchNorm + ReLU  │
         └────────┬────────────┘
                  │
         ┌────────▼────────────┐
         │   ResNet Block x6   │  6 residual blocks
         │  Conv-BN-ReLU-Conv  │  Each: 2 x Conv2d 3x3, 128 filters
         │   + Skip Connection │  Optional: SE attention (--use_se)
         └──┬──────────────┬───┘
            │              │
   ┌────────▼───────┐  ┌──▼──────────────┐
   │  Policy Head   │  │   Value Head     │
   │ Conv 1x1 → 2ch │  │ Conv 1x1 → 1ch  │
   │ BN → ReLU      │  │ BN → ReLU       │
   │ FC → 9 actions │  │ FC → 256 → FC 1 │
   │ (logits)       │  │ Tanh → [-1, 1]  │
   └────────────────┘  └─────────────────┘
```

**Parameters**: ~1.8M (with 128 filters, 6 blocks)

### Enhancements over original AlphaZero (2017)

| Feature | Original AlphaZero | This Implementation |
|---------|-------------------|---------------------|
| Attention | None | Squeeze-and-Excitation (SE) channel attention (`--use_se`) |
| Precision | FP32 on TPU v1 | Auto BF16 on H100/Ampere+, FP16 fallback |
| LR Schedule | Step decay | Step decay or Cosine Annealing (`--cosine_lr`) |
| MCTS | Sequential leaf eval | Batched parallel leaf evaluation (`--num_parallel`) |
| Compilation | N/A | `torch.compile` for inference acceleration |
| Game logic | Pure Python | Numba JIT-compiled core (5x faster self-play) |

---

## Play Against the AI

### GUI Mode (recommended)

```bash
python play_gui.py --ckpt checkpoints/training_steps_4800.ckpt
```

Options:
```
--human_color black     # Play as Player 1 (moves first), default
--human_color white     # Play as Player 2
--num_simulations 400   # Stronger AI (slower), default 200
--num_simulations 50    # Weaker AI (faster)
```

**Controls**: Click on a pit to make your move. Legal moves are highlighted on hover.

### Terminal Mode

```bash
python eval_play/eval_agent_toguz_cmd.py \
    --num_res_blocks=6 --num_filters=128 --num_fc_units=256 \
    --black_ckpt=checkpoints/training_steps_4800.ckpt \
    --white_ckpt=checkpoints/training_steps_4800.ckpt \
    --human_vs_ai --human_color=black \
    --num_simulations=200
```

### AI vs AI

```bash
python eval_play/eval_agent_toguz_cmd.py \
    --num_res_blocks=6 --num_filters=128 --num_fc_units=256 \
    --black_ckpt=checkpoints/training_steps_4800.ckpt \
    --white_ckpt=checkpoints/training_steps_4800.ckpt \
    --nohuman_vs_ai
```

---

## Train From Scratch

### Basic training

```bash
python -m alpha_zero.training_toguz \
    --num_res_blocks=6 \
    --num_filters=128 \
    --num_fc_units=256 \
    --batch_size=512 \
    --num_simulations=200 \
    --num_parallel=32 \
    --num_actors=8 \
    --init_lr=0.02 \
    --lr_milestones 15000 25000 \
    --max_training_steps=30000 \
    --min_games=300 \
    --games_per_ckpt=300 \
    --replay_capacity=300000 \
    --ckpt_dir=./checkpoints/toguz \
    --save_replay_interval=5000
```

### H100/Ampere+ optimized training

```bash
python -m alpha_zero.training_toguz \
    --num_res_blocks=10 \
    --num_filters=256 \
    --num_fc_units=256 \
    --batch_size=2048 \
    --num_simulations=400 \
    --num_parallel=32 \
    --use_se \
    --cosine_lr \
    --max_training_steps=100000 \
    --replay_capacity=500000
```

BF16 mixed precision is **automatically enabled** on supported GPUs (H100, A100, RTX 3090+).

### Training Pipeline

The AlphaZero training pipeline runs three components in parallel:

1. **Self-Play Actors** (8 processes) — play games against themselves using MCTS + neural network, generating training data
2. **Learner** (1 process) — trains the neural network on collected game data
3. **Evaluator** (1 process) — plays matches between the latest and previous checkpoints to track improvement

Checkpoints are saved every 300-500 training steps to `./checkpoints/toguz/`.

---

## Resume Training

If training was interrupted (e.g., Kaggle 12-hour GPU limit), resume from the last checkpoint:

```bash
python -m alpha_zero.training_toguz \
    --load_ckpt=./checkpoints/toguz/training_steps_4800.ckpt \
    --load_replay=./checkpoints/toguz/replay_state.ckpt \
    ... (same flags as original run)
```

`--load_ckpt` restores network weights, optimizer state, and LR scheduler.
`--load_replay` restores the replay buffer (requires `--save_replay_interval` during training).

---

## Project Structure

```
alpha_zero_toguz_Qumalaq/
├── play_gui.py                     # GUI to play against the AI
├── alpha_zero/
│   ├── training_toguz.py           # Training entry point
│   ├── core/
│   │   ├── network.py              # AlphaZeroNet (ResNet + SE + policy/value heads)
│   │   ├── pipeline.py             # Training loop, self-play, evaluation
│   │   ├── mcts_v2.py              # Monte Carlo Tree Search with parallel leaf eval
│   │   ├── replay.py               # Uniform replay buffer
│   │   └── rating.py               # Elo rating tracker
│   ├── envs/
│   │   ├── base.py                 # Base BoardGameEnv (OpenAI Gym)
│   │   └── toguz.py                # Togyz Kumalak environment (Numba JIT)
│   └── utils/
│       ├── csv_writer.py           # Training statistics logger
│       ├── transformation.py       # Data augmentation utilities
│       ├── upload_weights.py       # Upload to HuggingFace / Git
│       └── util.py                 # Common utilities
├── eval_play/
│   └── eval_agent_toguz_cmd.py     # Terminal-based play / AI vs AI
├── checkpoints/                    # Saved model weights
├── unit_tests/                     # Test suite
└── requirements.txt
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy
- gym 0.25.2
- absl-py
- numba (optional, ~5x faster self-play)
- tkinter (included with Python, needed for GUI)

```bash
pip install torch numpy gym absl-py numba python-snappy
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

Based on the AlphaZero implementation from *The Art of Reinforcement Learning* by Michael Hu.
