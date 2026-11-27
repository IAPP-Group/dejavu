# DejaVu

This repository contains the official code for the paper "Mapping the Landscape of Frozen Pretrained Backbones for Image Forensics" (TMLR 2026). 

![Frozen pretrained backbones pipeline](images/pipeline.png)

Most synthetic image detectors follow a common paradigm, combining a **frozen pretrained backbone** for feature extraction with a **lightweight classifier** for prediction. 
They primarily differ in implementation choiches, including the selection of the backbone, classifier architecture, preprocessing strategy, and evaluation protocol.

In this work, we systematically investigate these detectors across three tasks:

| task | classes |
|---|---|
| `detection` | real vs. synthetic |
| `attribution` | the source generators that produced a synthetic image |
| `detection+attribution` | real vs. the source generators |

**DejaVu** is the best configuration found on the validation set, combining MetaCLIP2-ViT-bigG/14 with a 3-layer MLP.

The pipeline has two stages: features are extracted once per backbone and split (`extract_features.py`), then classifiers are trained and evaluated on them (`train.py`, `evaluate.py`).

## Contents

```
extract_features.py   feature extraction (backbones, preprocessing, layers, post-processing)
train.py              train classifiers on GenImage train, evaluate on any evaluation set (validation or test)
evaluate.py           evaluate trained classifiers (OOD sets, post-processed images)
run_dejavu.py         run DejaVu (or any trained classifier) on your own images
summarize.py          average the per-run metrics (mean ± std)
app.py                graphical interface over the scripts above
notebooks/            paper_tables.ipynb: the tables of the paper (DataFrame + LaTeX) from the metrics
checkpoints/          trained DejaVu classifiers for the three tasks
dejavu/               library code
data/splits/          split CSVs (image paths, labels, generators, prompts)
configs/              dataset locations
scripts/              one script per experiment of the paper
```

## Installation

```bash
conda create -n dejavu python=3.10 -y && conda activate dejavu
pip install -r requirements.txt 
```

⚠️ DINOv3 weights are gated: accept the license on their HF pages first. Then:

```bash
huggingface-cli login 
```

The paper's experiments ran on NVIDIA RTX 4090 (24 GB) and RTX 5090 (32 GB) GPUs.
DINOv3-ViT-7B/16 in float32 needs ~28 GB of GPU memory; on smaller GPUs use
`--dtype bfloat16` (results may then differ slightly from the paper).

## Datasets Preparation

| dataset key | used for | what to download | download link |
|---|---|---|---|
| `GenImage` | training, in-domain evaluation (fake) | GenImage, evaluation partition (`val/ai`) of the 8 generators; the CSVs expect `<generator>_imagenet/ai/` (e.g. `adm_imagenet/ai/`), or set `genimage_layout: official` to read the official folders | https://github.com/GenImage-Dataset/GenImage |
| `ImageNet` | training, in-domain evaluation (real) | ILSVRC2012 validation images, in one folder and named `ILSVRC2012_val_XXXXXXXX_<synset>.JPEG` (original name + WordNet id of the class) | https://huggingface.co/datasets/ILSVRC/imagenet-1k |
| `Synthbuster` | OOD | Synthbuster (9 diffusion generators) | https://zenodo.org/records/10066460?preview_file=synthbuster.zip |
| `RAISE` | OOD (real images of Synthbuster) | RAISE-1k (`r*.TIF`) | https://loki.disi.unitn.it/RAISE/download.html |
| `AIGIBench` | OOD | AIGIBench test set (`test/<subset>/{0_real,1_fake}`) | https://huggingface.co/datasets/HorizonTEL/AIGIBench | 
| `CO-SPY` | OOD | CO-SPY-Bench (`Co-Spy-Bench/{real_image_examples,synthetic}`); we use a portion of the CC3M, SBU and TextCaps subsets | https://github.com/Megum1/CO-SPY |
| `CommunityForensics` | OOD | Community Forensics evaluation set | https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval |

Within `configs/data_paths.yaml`, set the root of each dataset (or a single `data_root` with one sub-folder per key) and, if needed, `genimage_layout`.

### Splits (`data/splits/`)

Paths in the CSVs are relative (`<dataset key>/...`), so features and results do not
depend on where the data is stored.

| split | images | notes |
|---|---|---|
| `genimage_train` | 79,950 (39,950 real) | 800 ImageNet classes ("prompts") |
| `genimage_validation` | 10,050 (5,050 real) | 100 other classes; all design choices are made here |
| `genimage_test` | 10,000 (5,000 real) | 100 other classes |
| `{synthbuster,aigibench,cospy,communityforensics}_{validation,test}` | | OOD benchmarks, split by content, all generators in both halves |
| `ood_attribution_{validation,test}` | 8,312 / 8,181 | OOD images of GLIDE, Midjourney, SDv4, SDv5 (Section 4.2.1) |
| `ood_detection_attribution_{validation,test}` | 73,734 / 73,678 | the same + the real images of the OOD benchmarks (Section 4.3.1) |

GenImage splits are class-disjoint: each prompt appears in one split only. The
`ood_*` splits reuse the features of the four OOD benchmarks, so they need no
extraction of their own.

In training, fake images are JPEG-compressed at QF 96 before feature extraction (the real ImageNet images are JPEGs); the OOD benchmarks are used as distributed. Use `--fake_qf96` to change this.

## Usage

All commands accept `-h`. The `--backbones` option of `extract_features.py`, `train.py` and `evaluate.py` takes one or more values among:

| value | selects |
|---|---|
| a backbone id, e.g. `MetaCLIP2-ViT-bigG-14` | that backbone (see the table below) |
| a paper name, e.g. `"MetaCLIP2-ViT-bigG/14"` | that backbone (quote names containing `/`) |
| `DejaVu` | `MetaCLIP2-ViT-bigG-14` |
| `clip` | the 9 OpenAI CLIP backbones |
| `metaclip2` | the 4 MetaCLIP 2 backbones |
| `dino` | the 9 DINOv2 and DINOv3 backbones |
| `all` | all 22 backbones |

Values can be combined, e.g. `--backbones clip DINOv3-ViT-7B-16`; duplicates are removed.

| family | id (`--backbones`) | paper name |
|---|---|---|
| `clip` | `ViT-B-32` | ViT-B/32 | 
| `clip` | `ViT-B-16` | ViT-B/16 | 
| `clip` | `ViT-L-14` | ViT-L/14 | 
| `clip` | `ViT-L-14@336px` | ViT-L/14@336px | 
| `clip` | `RN50` | RN50 | 
| `clip` | `RN101` | RN101 | 
| `clip` | `RN50x4` | RN50×4 | 
| `clip` | `RN50x16` | RN50×16 | 
| `clip` | `RN50x64` | RN50×64 | 
| `metaclip2` | `MetaCLIP2-ViT-S-16` | MetaCLIP2-ViT-S/16 | 
| `metaclip2` | `MetaCLIP2-ViT-M-16` | MetaCLIP2-ViT-M/16 |
| `metaclip2` | `MetaCLIP2-ViT-H-14` | MetaCLIP2-ViT-H/14 | 
| `metaclip2` | `MetaCLIP2-ViT-bigG-14` | MetaCLIP2-ViT-bigG/14 (DejaVu) | 
| `dino` | `DINOv2-ViT-S-14` | DINOv2-ViT-S/14 | 
| `dino` | `DINOv2-ViT-B-14` | DINOv2-ViT-B/14 | 
| `dino` | `DINOv2-ViT-L-14` | DINOv2-ViT-L/14 | 
| `dino` | `DINOv2-ViT-g-14` | DINOv2-ViT-g/14 |
| `dino` | `DINOv3-ViT-S-16` | DINOv3-ViT-S/16 | 
| `dino` | `DINOv3-ViT-B-16` | DINOv3-ViT-B/16 |
| `dino` | `DINOv3-ViT-L-16` | DINOv3-ViT-L/16 | 
| `dino` | `DINOv3-ViT-Hplus-16` | DINOv3-ViT-H+/16 | 
| `dino` | `DINOv3-ViT-7B-16` | DINOv3-ViT-7B/16 |

The id is used in feature and output folder names.

### 1. Extract features

```bash
# GenImage train/validation/test and the OOD benchmarks (validation + test), all 22 backbones
python extract_features.py --backbones all --splits genimage ood --data_config configs/data_paths.yaml
```

Features go to `<features_dir>/<backbone>/<split>[__<variant>].pt` (default `features/`). 

Options: `--splits` (`genimage`, `ood`, `ood_validation`, `ood_test` or split names), `--preprocess {default,openai,defake,univfd}`, `--layers block22 ...` or `--all_layers`, one post-processing among `--jpeg QF`, `--blur SIGMA`, `--resize R`, `--crop C`, `--center_crop C`. 
Existing files are skipped unless `--overwrite`.

### 2. Train and evaluate

```bash
# DejaVu on the three tasks, evaluated on GenImage validation and test, in-domain and OOD
python train.py --backbones DejaVu --classifiers 3-MLP \
    --tasks detection attribution detection+attribution --eval_sets genimage ood

# any of the 6 classifiers (or `all`) on any backbone
python train.py --backbones DINOv3-ViT-7B-16 RN50x64 --classifiers LR SVM kNN --tasks attribution

# training-set size: 1 ... 800 prompts (800 = whole training set)
python train.py --backbones all --tasks detection --num_prompts 1 5 10 50 100 300 500 800

# alternative preprocessing: first extract the features with that preprocessing,
# then train/evaluate passing the same --preprocess
python extract_features.py --backbones DejaVu --splits genimage --preprocess defake --data_config configs/data_paths.yaml
python train.py --backbones DejaVu --tasks attribution --preprocess defake

# intermediate layer: first extract the features of that layer (or --all_layers),
# then train/evaluate passing the same --layers
python extract_features.py --backbones DejaVu --splits genimage --layers block28 --data_config configs/data_paths.yaml
python train.py --backbones DejaVu --tasks attribution --layers block28

# cross-model generalization: train on chosen generators (+ all real images)
python train.py --backbones DejaVu --tasks detection --train_generators vqdm wukong --eval_sets genimage
```

Every combination of `--backbones`, `--tasks`, `--classifiers`, `--layers` and `--num_prompts` is trained. MLPs are trained `--runs 10` times; LR/SVM/kNN once, with theirhyper-parameters chosen by 5-fold cross-validation on the training set. 
With `--num_prompts`, each run draws its own random prompts (seed 42 + run), for every classifier. `--eval_sets`: `genimage` (validation + test, the default), `ood`, `ood_validation`, `ood_test` (the OOD sets of each task) or split names.

### 3. Evaluate saved classifiers

```bash
# OOD evaluation of classifiers trained earlier
python evaluate.py --backbones DejaVu --tasks detection attribution detection+attribution --eval_sets ood

# robustness: features of post-processed test images, then evaluation
python extract_features.py --backbones DejaVu --splits genimage_test --jpeg 75 --data_config configs/data_paths.yaml
python evaluate.py --backbones DejaVu --eval_sets genimage_test --jpeg 75
```

The configuration is identified by the same options as in `train.py`
(e.g. `--num_prompts 1`, `--train_generators vqdm wukong`).

### 4. Outputs and summaries

```
<out_dir>/metrics/<task>.csv                 one row per configuration x evaluation set x run
<out_dir>/metrics/detection_per_generator.csv detection metrics per generator (each paired with all real images)
<out_dir>/<task>/<backbone>/<config>/
    config.json
    classifiers/run<r>.pth|.pkl
    predictions/<eval_set>[__<post-processing>]__run<r>.npz   image paths, labels, predictions, scores
    confusion_matrices/<eval_set>.{csv,png,pdf} attribution and joint task: run-averaged, row-normalized
    confusion_matrices/<eval_set>_counts.csv  counts summed over runs
```

`<config>` is e.g. `3-MLP`, `3-MLP__prompts10`, `3-MLP__defake`, `3-MLP__block22`, `3-MLP__train-vqdm+wukong`. 
Metrics: balanced accuracy (`balanced_accuracy`, "Acc." in the paper), support-weighted F1, AP and AUC (one-vs-rest, macro-averaged over the classes present, for the multi-class tasks), plus plain accuracy, precision and recall.
Predicted labels are the argmax of the class scores, with no threshold tuning.

```bash
python summarize.py results/metrics/detection.csv --filter eval_set=genimage_validation --output summary.csv
```

### 5. Run DejaVu on your own images

`checkpoints/` holds the trained DejaVu classifiers (MetaCLIP2-ViT-bigG/14 + 3-MLP, trained on the whole GenImage train set; for each task, the run whose GenImage test accuracy is closest to the mean of the 10 runs):

| file | task | classes (output order) |
|---|---|---|
| `checkpoints/dejavu_det.pth` | `detection` | real, fake |
| `checkpoints/dejavu_attr.pth` | `attribution` | adm, biggan, glide, midjourney, sdv4, sdv5, vqdm, wukong |
| `checkpoints/dejavu_joint.pth` | `detection+attribution` | real, adm, biggan, glide, midjourney, sdv4, sdv5, vqdm, wukong |

The JSON next to each checkpoint records its configuration, training options, training CSV and GenImage test results.

`run_dejavu.py` applies a checkpoint to any images, with no need for split CSVs or precomputed features: the features are extracted on the fly with the checkpoint's backbone (read from its JSON) and the classifier predicts every image.

```bash
# unlabelled images: files and/or folders (searched recursively)
python run_dejavu.py --checkpoint checkpoints/dejavu_det.pth --images my_images/ other/image.png

# labelled images: a CSV with image_path and, optionally, label (real/fake or 0/1) and model (generator)
python run_dejavu.py --checkpoint checkpoints/dejavu_joint.pth --csv my_dataset.csv --out predictions/my_dataset
```

In the CSV, `image_path` is absolute or relative to `--image_root` (default: the CSV's folder). The ground truth is used for the metrics: `label` for detection, `model` for attribution (only fakes from the 8 GenImage generators count; others are predicted but left out), both for the joint task.

Outputs in `--out` (default `predictions/<checkpoint name>/`):

```
predictions.csv   image_path, [label, model,] prediction, score_<class> for every class
run_info.json     checkpoint, backbone, task, classes and input of the run
metrics.json      only with ground truth: the metrics of section 4 (+ per-generator detection metrics if `model` is given)
```

Any classifier trained with `train.py` can be used the same way, e.g. `--checkpoint results/detection/RN50x64/3-MLP/classifiers/run0.pth` (its configuration is read from the `config.json` of its folder). For a checkpoint without a JSON, pass `--backbone` and `--task`.
In the paper, GenImage fake images (PNG) are JPEG-compressed at QF=96 before feature extraction to match the JPEG real images; `--fake_qf96` does the same to reproduce that setting (it needs the `label` column). Leave it off for other data, as for the OOD benchmarks.

### 6. Graphical interface

`app.py` is a web interface for the same pipeline. Backbones, classifiers, tasks, preprocessing, layers, number of training prompts, training generators, evaluation sets and post-processing are chosen from forms instead of command-line options. The interface only builds and runs the commands of sections 1-3 and 5, so its results are identical to those of the scripts.

#### Starting it

```bash
conda activate dejavu
streamlit run app.py        # then open http://localhost:8501
```

On a remote server, forward port 8501 to your machine: VS Code does it automatically, otherwise use `ssh -L 8501:localhost:8501 <server>`. Use `--server.port <port>` to change the port. Streamlit is listed in `requirements.txt`; usage statistics are disabled in `.streamlit/config.toml`.

#### Sidebar

| setting | meaning |
|---|---|
| Data config (YAML) / Data root | dataset locations, as `--data_config` / `--data_root` (only needed for extraction) |
| Features folder | `--features_dir` (default `features`) |
| Output folder | `--out_dir` (default `results`): classifiers, predictions, metrics, confusion matrices |
| GPUs | none selected: one job that sees all the GPUs; one: the job runs on it; several, or *Use all GPUs*: the work is split into one parallel job per GPU (see below) |

#### Tabs

| tab | what it does |
|---|---|
| Feature extraction | runs `extract_features.py`: backbones, splits, preprocessing, final embedding / selected layers / all layers, post-processing |
| Training | runs `train.py`: backbones, classifiers, tasks, preprocessing, layers, number of training prompts, training generators (cross-model), evaluation sets, hyperparameters (defaults = paper) |
| Evaluation | runs `evaluate.py` on saved classifiers: same configuration options as Training, plus evaluation sets, post-processing and runs |
| Run DejaVu | runs `run_dejavu.py`: a checkpoint (the DejaVu ones in `checkpoints/` or any classifier in the output folder), images / folders or a CSV, output folder; then shows the metrics (with ground truth) and the predictions, as a table or as images (a single image is shown large, with its verdict and scores; several images as a gallery with filters, sorting by P(fake) and, with ground truth, *only wrong predictions*), with CSV download |
| Jobs | launched jobs with status and live log; a running job can be stopped, finished ones deleted |
| Results | mean (± std) over runs of `<out_dir>/metrics/*.csv`, with filters and CSV download; optionally the confusion matrices |

#### Typical workflow

1. **Feature extraction**: select the backbones (★ marks DejaVu's, MetaCLIP2-ViT-bigG/14) and the splits (`genimage`, `ood`), then press **▶ Run**.
2. **Jobs**: follow the extraction log; progress bars update in place.
3. **Training**: select the same backbones and preprocessing / layers used for extraction, the classifiers and tasks, and press **▶ Run**. The tab warns if some feature files have not been extracted yet.
4. **Evaluation** (optional): evaluate the saved classifiers on other sets (e.g. `ood`) or on post-processed images, whose features must be extracted first with the same post-processing.
5. **Results**: choose a metrics file (`detection.csv`, `attribution.csv`, ...), filter by backbone, classifier, evaluation set, etc., and download the averaged table. Turn on *Show confusion matrices* to browse the matrices of attribution and detection+attribution; they are off by default because listing them is slow with many results.

#### Good to know

- Every form shows the exact command it will run, so a configuration made in the interface can be copied into a script.
- Preprocessing, layer and post-processing options select different feature files: extract with an option first, then train / evaluate with the same option.
- Several post-processing values run one after the other in a single job.
- With several GPUs, extraction is split by backbone and training / evaluation by configuration (backbone × task × classifier × layer × number of prompts), and the pieces are distributed over one job per GPU. The plan is shown before running.
- Jobs run in the background, independently of the browser, with their commands and log in `gui_jobs/<job>/`. Closing the page does not stop them, and new jobs can be started while others are running (jobs on the same GPU share its memory).
- Deleting a job in the Jobs tab removes only its folder in `gui_jobs/`, never features or results.

## Citation
@article{
  bertazzini2026mapping,
  title={Mapping the Landscape of Frozen Pretrained Backbones for Image Forensics},
  author={Giulia Bertazzini and Daniele Baracchi and Dasara Shullani and Alessandro Piva},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2026},
  url={https://openreview.net/forum?id=G2ksWOJ4Qa},
  note={}
}

```
