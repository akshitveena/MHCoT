# Colab — persistent setup (clone ONCE into Drive, never re-clone)

The problem: code cloned into `/content/` is **wiped** every time the Colab
runtime disconnects. The fix: put the repo in **Google Drive**, which persists
across runtimes. Clone once; every future session just mounts Drive and pulls.

The 8 MB GoT cache (`data/got_cache/`) is committed to the repo, so it travels
with every clone — you never re-run preprocessing. Only the large hidden-state
caches (`data/hidden_cache/`, gitignored) are regenerated on Colab.

---

## ONE-TIME setup (first session only)

```python
# 1. Mount Drive (persists across runtimes)
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone the repo INTO Drive (once). Use your PAT for a private repo.
GH_TOKEN = "github_pat_..."          # your fine-grained PAT
USER, REPO = "akshitveena", "MHCoT"
!git clone https://{USER}:{GH_TOKEN}@github.com/{USER}/{REPO}.git \
    /content/drive/MyDrive/MHCoT
```

That's it — the repo (code + 8 MB GoT cache) now lives in your Drive
permanently.

---

## EVERY session after (the only cell you need at the start)

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/MHCoT
!git pull                      # get any code changes you pushed from the Mac
!pip -q install transformers accelerate datasets scikit-learn  # deps (fast)
import torch; print('GPU:', torch.cuda.get_device_name(0))
```

Your code, GoT data, results, and checkpoints are all there — no re-cloning.

---

## Run training on Colab (A100 / T4)

```python
# 1. Regenerate the per-token sequence cache ON Colab (lives in Drive, ~15 min)
!python main/encoder.py --seq

# 2. Train (A100 has 40 GB — can use bigger batch / longer seqs than the Mac)
!python main/train.py --epochs 40 --batch_size 16 --max_seq_len 512
```

Because the repo is in Drive, `best_model.pt`, `training_log.json`, and the
hidden caches all persist after the runtime disconnects.

---

## Push changes back to GitHub from Colab

```python
%cd /content/drive/MyDrive/MHCoT
!git config user.email "akshitveena.07@gmail.com"
!git config user.name "Akshit Veena"
!git add -A && git commit -m "Update from Colab" && git push
```

(The PAT is already embedded in the remote URL from the clone, so push works.)

---

## Notes / gotchas

- **Drive I/O is slower than local disk.** For the ~1.8 GB hidden caches this
  is fine (written once, read during training). If training feels I/O-bound,
  copy the seq cache to local `/content/` at session start and point the
  loader there — but for our sizes it's not necessary.
- **Don't put the 1.8 GB caches in git.** They're gitignored and regenerable
  by `encoder.py`. Only the 8 MB GoT cache is committed.
- **Session limits:** free Colab ~90 min idle / ~12 h; Pro longer. Training
  checkpoints to Drive each time val AUC improves, so a disconnect costs
  minutes, not the run.
