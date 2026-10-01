## 2. Dataset Acquisition

1. Go to https://xview2.org → DATASET tab → "Datasets from the Challenge" section.
2. Right-click "Download Challenge training set" → Copy link address (the direct
   download link is a time-limited signed URL generated per page load — do not
   reuse an old one).
3. Download via curl rather than the browser to avoid resumable-download failures
   against the signed URL's expiry window:
```cmd
   curl -L -o train_images_labels_targets.tar "PASTE_COPIED_URL"
```
   Expected size: ~7.8 GB. Expected SHA1: `b37a4ef4ee9c909e2b19d046e49d42ee3965714b`
4. Verify integrity before extracting:
```cmd
   certutil -hashfile train_images_labels_targets.tar SHA1
```
5. Extract and move into place:
```cmd
   mkdir data\raw\xbd_extract_temp
   tar -xf train_images_labels_targets.tar -C data\raw\xbd_extract_temp
   mkdir data\raw\xbd
   move data\raw\xbd_extract_temp\train data\raw\xbd\train
   rmdir data\raw\xbd_extract_temp
   del train_images_labels_targets.tar
```
6. Verified resulting structure:
data/raw/xbd/train/images/ — 5,598 PNGs (pre + post disaster tiles combined)
data/raw/xbd/train/labels/ — 5,598 JSONs (building polygon + damage annotations)
data/raw/xbd/train/targets/ — 5,598 PNGs (pre-rendered damage mask targets)
   Filename convention: `{disaster-event}_{tile-id:08d}_{pre|post}_disaster.png`,
   e.g. `guatemala-volcano_00000000_pre_disaster.png`. 5,598 files = 2,799 pre/post
   pairs across the disaster events included in the Challenge training split.
7. Do **not** commit this data to git — excluded via `.gitignore`, and its
   CC BY-NC-SA 3.0 license does not permit redistribution here.
   **Note on PyTorch:** Install separately with the CPU-specific index to avoid
pulling unnecessary CUDA dependencies:
```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
Verified working: torch 2.14.0+cpu on Python 3.13.15, Windows.
      
## 3. Preprocessing

After acquiring the dataset (Section 2), generate masks and splits:

```powershell
python src\preprocessing\generate_masks.py
python scripts\event_type_mapping.py
python scripts\generate_splits.py
python scripts\validate_pairs.py
```

Expected: 2,799 masks generated in `data/processed/masks/`, splits created
in `data/interim/splits/` (train 1782 / val 440 / test 577 tiles), and
"VALIDATION PASSED" from the final script. Verified reproducible across
4 independent runs (1 local, 3 separate Colab sessions) with identical
class distributions every time.

Compute normalization statistics (only needs running once; values are
already saved in `configs/config.yaml`, but to regenerate):

```powershell
python scripts\compute_dataset_stats.py
```

## 4. Training

**Local CPU training is not practical for the full U-Net** (measured at
~4 hours/epoch). Training was performed on Google Colab's free T4 GPU tier
(~7 minutes/epoch). To reproduce:

1. Open a new Google Colab notebook, set runtime to T4 GPU
   (Runtime → Change runtime type → T4 GPU)
2. Clone this repository and install dependencies:
```python
   !git clone https://github.com/ali79850/intelligent-disaster-response.git
   %cd intelligent-disaster-response
   !pip install -q torch torchvision numpy pandas pillow pyyaml scipy scikit-learn
```
3. Download and prepare the dataset inside Colab (same commands as
   Section 2-3 above, run via `!` prefix)
4. Train:
```python
   import torch
   from torch.utils.data import DataLoader
   from src.data.xbd_dataset import XBDDataset
   from src.models.unet import UNetResNet18
   from src.training.losses import CombinedLoss

   device = torch.device("cuda")
   # ... see DECISIONS.md Phase 5 for the full training loop and
   # class-weight calculation used
```
5. Save checkpoints after every epoch (not just at the end) — a Colab
   runtime disconnect mid-training is a real, encountered risk; see
   DECISIONS.md Phase 5 for the full incident and the per-epoch-save fix.

Trained checkpoints are not included in this repository (too large for
git). The final model used throughout this project is
`models/unet_epoch10.pt`, trained for 10 epochs total (5 initial + 5
resumed), batch_size=4, Adam lr=1e-4.

## 5. Evaluation

```powershell
python scripts\evaluate_unet.py
```

Expected: per-class precision/recall/F1 on val and test sets. Test set
macro F1 should be approximately 0.338 for the epoch-10 checkpoint.

## 6. Running the Application

Requires a trained model checkpoint at `models/unet_epoch10.pt` (see
Section 4) and, optionally, a Groq API key for the LLM summary feature.

**Backend:**
```powershell
cd app\backend
uvicorn main:app --reload
```
Expect: `Model loaded: unet_epoch10 on cpu` and `Application startup
complete`, server at `http://127.0.0.1:8000`.

**Frontend** (separate terminal):
```powershell
cd app\frontend
npm install
npm run dev
```
Expect: Vite dev server at `http://localhost:5173/`.

**Optional: LLM summary feature.** Requires a free Groq API key
(console.groq.com). Create a `.env` file at the project root:
GROQ_API_KEY=your_key_here
Without this, core damage assessment (`/api/analyze`) still works fully;
only the optional `/api/summarize` endpoint requires it.

## 7. Running Tests

```powershell
pytest tests\ -v
```
Expected: 17 tests passing. Model-dependent tests skip gracefully if
`models/unet_epoch10.pt` is not present (e.g. on a fresh clone before
training).