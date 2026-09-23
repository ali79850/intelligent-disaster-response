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
      