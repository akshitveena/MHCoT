# Running Track A + Track B on Colab (A100)

Two parallel data-generation jobs, both persistent (clone-into-Drive so nothing
vanishes on disconnect). Track A secures the calibration contribution
(replication + distractor + ambiguity); Track B tests the higher-capability
hypothesis (Game of 24, multi-path).

---

## ONE-TIME — persistent clone into Drive (do once, ever)

```python
from google.colab import drive
drive.mount('/content/drive')

GH_TOKEN = "github_pat_..."          # your fine-grained PAT
USER, REPO = "akshitveena", "MHCoT"
!git clone https://{USER}:{GH_TOKEN}@github.com/{USER}/{REPO}.git \
    /content/drive/MyDrive/MHCoT
```

## EVERY session — mount + pull + deps (the only startup cell)

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/MHCoT
!git pull
!pip -q install "transformers>=4.45,<5" "huggingface_hub>=0.24,<0.30" \
    "datasets>=3.0,<4" "accelerate>=0.34,<1" sentencepiece tqdm scipy \
    scikit-learn "openai<1.0" "graph_of_thoughts==0.0.2"
import torch; print('GPU:', torch.cuda.get_device_name(0))
```

The 8 MB GoT cache travels with the repo; large hidden caches are regenerated
here and live in Drive (persistent).

---

## TRACK A — scale GSM8K (replication of the calibration finding)

Generates more GoT data on the GSM8K **train** split. Resumable — re-run the
same cell across sessions; it skips done problems.

```python
# ~30-60s per problem on A100; resumable. Target ~2000 for a solid val set.
!caffeinate 2>/dev/null; python main/preprocess_got.py --split train --limit 2000
```

When enough has accumulated, encode + replicate the calibration result across
seeds (this is the #1 rigor gate):

```python
!python main/encoder.py --seq      # per-token sequences (uses train cache too)
# re-run the calibration ablations on the bigger, less-noisy val set:
!python main/train.py --epochs 40 --batch_size 16 --max_seq_len 512
!python experiments/exp5_ablation.py --epochs 40
!python experiments/exp_calibration.py
!python experiments/exp_n1_ablation.py
```

> Replication target: the ordering **real ≈ N1 (ECE ~0.23) ≫ N2 (ECE ~0.07)**
> holds on the bigger val set and across seeds → the calibration finding is
> bulletproof.

## TRACK A (cont.) — #2 Distractor data

(Distractor generator is a small follow-up: inject an irrelevant sentence with
a number into each GSM8K problem, then run the same GoT pipeline. Build once
Track-A scaling is running.)

---

## TRACK B — Game of 24 (the multi-path frontier test)

```python
# generate solvable puzzles + run GoT on each (~30-60s/puzzle, resumable)
!python main/preprocess_game24.py --n 600
```

This caches `data/got_cache/game24.jsonl` with candidate expressions and
PROGRAMMATIC validity labels (verify() — exact ground truth, no LLM needed).

The MHCoT test on Game-24 (built next): does interference between competing
solution paths predict solvability / give a structural edge over a real model
on a genuinely multi-path task? This is where MHCoT's anti-collapse should help
in a way it can't on single-answer GSM8K.

---

## Push results back (end of session)

```python
%cd /content/drive/MyDrive/MHCoT
!git add -A && git commit -m "Colab: Track A/B data + results" && git pull && git push
```

(Always `git pull` before `git push` — a stale Colab push is what reverted
complex_ops.py before.)
