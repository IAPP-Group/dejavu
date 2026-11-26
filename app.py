"""Graphical interface for the DejaVu pipeline.

    streamlit run app.py              # then open http://localhost:8501

The interface only builds and launches the same commands documented in the README
(extract_features.py, train.py, evaluate.py, predict.py): every form shows the exact command it
will run. Jobs run in the background, independently of the browser, with their log in
gui_jobs/<job>/log.txt, so a long extraction or training survives closing the page.
"""

import glob
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dejavu.backbones import BACKBONES, DEJAVU_BACKBONE, FAMILIES, expand_backbones
from dejavu.classifiers import CLASSIFIERS
from dejavu.data import EXTRACTION_GROUPS, GENERATORS, SPLITS, TASKS, feature_path, variant_tag
from dejavu.transforms import DEGRADATIONS, PREPROCESSINGS, degradation_tag
from summarize import summarize

ROOT = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.environ.get("DEJAVU_GUI_JOBS", os.path.join(ROOT, "gui_jobs"))
PYTHON = sys.executable

# options split into separate commands when the work is spread over several GPUs
TRAIN_SPLIT_KEYS = ("backbones", "tasks", "classifiers", "layers", "num_prompts")
PROMPT_CHOICES = [1, 5, 10, 50, 100, 300, 500, 800]
EVAL_SET_CHOICES = ["genimage", "genimage_validation", "genimage_test", "ood", "ood_validation", "ood_test"]
DEGRADATION_LABELS = {"none": "none", "jpeg": "JPEG (QF)", "blur": "Gaussian blur (σ)",
                      "resize": "resize (factor)", "crop": "random crop (kept fraction)",
                      "center_crop": "center crop (kept fraction)"}

# Number of transformer blocks of each ViT (CLIP ResNets expose layer1..layer4).
NUM_BLOCKS = {
    "ViT-B-32": 12, "ViT-B-16": 12, "ViT-L-14": 24, "ViT-L-14@336px": 24,
    "MetaCLIP2-ViT-S-16": 12, "MetaCLIP2-ViT-M-16": 12, "MetaCLIP2-ViT-H-14": 32, "MetaCLIP2-ViT-bigG-14": 48,
    "DINOv2-ViT-S-14": 12, "DINOv2-ViT-B-14": 12, "DINOv2-ViT-L-14": 24, "DINOv2-ViT-g-14": 40,
    "DINOv3-ViT-S-16": 12, "DINOv3-ViT-B-16": 12, "DINOv3-ViT-L-16": 24, "DINOv3-ViT-Hplus-16": 32,
    "DINOv3-ViT-7B-16": 40,
}
SPEC = {b.id: b for b in BACKBONES}


def layer_names(bid):
    if bid in NUM_BLOCKS:
        return [f"block{i}" for i in range(NUM_BLOCKS[bid])]
    return ["layer1", "layer2", "layer3", "layer4"]


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def launch(kind, commands, gpu=None):
    """Run `commands` one after the other in a detached shell; return the job folder.
    `gpu` (e.g. "0" or "0,1") sets CUDA_VISIBLE_DEVICES; None leaves it unchanged."""
    base = f"{datetime.now():%Y%m%d-%H%M%S}_{kind}" + (f"_gpu{gpu.replace(',', '+')}" if gpu else "")
    job, i = base, 1
    while os.path.exists(os.path.join(JOBS_DIR, job)):
        i += 1
        job = f"{base}-{i}"
    jdir = os.path.join(JOBS_DIR, job)
    os.makedirs(jdir)
    script = "set -e\n" + "\n".join(commands) + "\n"
    with open(os.path.join(jdir, "commands.sh"), "w") as f:
        f.write(script)
    # `python` in the commands is the interpreter running this app
    env = {**os.environ, "PYTHONUNBUFFERED": "1",
           "PATH": os.path.dirname(PYTHON) + os.pathsep + os.environ.get("PATH", "")}
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    wrapper = f"bash {shlex.quote(os.path.join(jdir, 'commands.sh'))}; echo $? > {shlex.quote(os.path.join(jdir, 'exit_code'))}"
    with open(os.path.join(jdir, "log.txt"), "w") as log:
        proc = subprocess.Popen(["bash", "-c", wrapper], cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
    with open(os.path.join(jdir, "pid"), "w") as f:
        f.write(str(proc.pid))
    return job


def _alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # a finished child we started is a zombie until reaped
        return os.waitpid(pid, os.WNOHANG) == (0, 0)
    except ChildProcessError:
        return True


def job_status(jdir):
    code_file = os.path.join(jdir, "exit_code")
    if os.path.exists(code_file):
        code = open(code_file).read().strip()
        return "✅ done" if code == "0" else f"❌ failed (exit {code})"
    try:
        pid = int(open(os.path.join(jdir, "pid")).read())
    except (OSError, ValueError):
        return "?"
    return "⏳ running" if _alive(pid) else "⛔ stopped"


def stop_job(jdir):
    pid = int(open(os.path.join(jdir, "pid")).read())
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def tail(path, n=300):
    try:
        # newline="" keeps the carriage returns (universal newlines would turn them into \n)
        with open(path, errors="replace", newline="") as f:
            text = f.read()
    except OSError:
        return ""
    # tqdm redraws its bar with carriage returns: keep only the last state of each line
    # (the last non-blank state: a finished bar with leave=False is cleared with blanks)
    lines = [next((p for p in reversed(l.split("\r")) if p.strip()), "") for l in text.split("\n")]
    return "\n".join(lines[-n:])


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def cmd(script, args):
    parts = ["python", script]
    for k, v in args:
        if v is None or v is False or v == []:
            continue
        parts.append(f"--{k}")
        if v is True or v == "":
            continue
        parts += [str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)]
    return " ".join(shlex.quote(p) for p in parts)


def backbone_label(bid):
    label = f"{bid}  ({SPEC[bid].paper_name})"
    return f"★ DejaVu: {label}" if bid == DEJAVU_BACKBONE else label


def backbone_picker(key, default=(DEJAVU_BACKBONE,)):
    c1, c2 = st.columns([1, 3])
    family = c1.selectbox("Family", ["—", "all"] + list(FAMILIES), key=f"{key}_fam",
                          help="Select all the backbones of a family (dino = DINOv2 + DINOv3).")
    # DejaVu's backbone first, then the others in the paper's order
    ids = [DEJAVU_BACKBONE] + [b.id for b in BACKBONES if b.id != DEJAVU_BACKBONE]
    if family != "—":
        chosen = [b.id for b in BACKBONES] if family == "all" else FAMILIES[family]
        c2.multiselect("Backbone", ids, default=chosen, disabled=True, key=f"{key}_bb_fixed",
                       format_func=backbone_label)
        return chosen, [family]
    chosen = c2.multiselect("Backbone", ids, default=list(default), key=f"{key}_bb", format_func=backbone_label,
                            help="★ DejaVu = MetaCLIP2-ViT-bigG/14, the backbone of DejaVu "
                                 "(use it with the 3-MLP classifier).")
    return chosen, chosen


def common_layers(backbones):
    if not backbones:
        return []
    sets = [layer_names(b) for b in backbones]
    return [l for l in sets[0] if all(l in s for s in sets[1:])]


def degradation_picker(key):
    name = st.selectbox("Post-processing (robustness)", list(DEGRADATION_LABELS), key=f"{key}_deg",
                        format_func=DEGRADATION_LABELS.get)
    if name == "none":
        return [None]
    paper_values = DEGRADATIONS[name][2]
    values = st.multiselect("Values (one or more, run one after the other)", paper_values, default=paper_values[:1],
                            key=f"{key}_degv")
    extra = st.text_input("Other values (space-separated)", key=f"{key}_dege")
    try:
        values += [int(v) if name == "jpeg" else float(v) for v in extra.split()]
    except ValueError:
        st.error(f"Invalid value in '{extra}'.")
    return [(name, v) for v in values] or [None]


def config_section(key):
    """Options shared by train.py and evaluate.py; returns the argument list."""
    backbones, bb_args = backbone_picker(key)
    c1, c2 = st.columns(2)
    classifiers = c1.multiselect("Classifiers", CLASSIFIERS, default=["3-MLP"], key=f"{key}_clf")
    tasks = c2.multiselect("Task", TASKS, default=["detection"], key=f"{key}_task",
                           help="detection+attribution = joint task (real + the 8 generators).")
    c1, c2 = st.columns(2)
    preprocess = c1.selectbox("Preprocessing", PREPROCESSINGS, key=f"{key}_pre",
                              help="The features must have been extracted with the same preprocessing.")
    avail = common_layers(backbones)
    layers = c2.multiselect("Intermediate layers (empty = final embedding)", avail, key=f"{key}_layers",
                            help="Only layers present in all the selected backbones. The features of the "
                                 "layer must have been extracted first.")
    c1, c2 = st.columns(2)
    prompts = c1.multiselect("Training prompts", PROMPT_CHOICES, default=[800], key=f"{key}_np",
                             help="800 = the whole training set. With fewer prompts every run draws a "
                                  "different subset (seed 42 + run).")
    only_detection = tasks == ["detection"]
    gens = c2.multiselect("Training generators (cross-model)", GENERATORS, key=f"{key}_gens",
                          disabled=not only_detection,
                          help="Detection only. Empty = all 8 generators.")
    args = [("backbones", bb_args), ("classifiers", classifiers), ("tasks", tasks),
            ("preprocess", None if preprocess == "default" else preprocess),
            ("layers", layers), ("num_prompts", [] if prompts == [800] else prompts),
            ("train_generators", gens if only_detection else [])]
    errors = []
    if not backbones:
        errors.append("Select at least one backbone.")
    if not classifiers:
        errors.append("Select at least one classifier.")
    if not tasks:
        errors.append("Select at least one task.")
    if not prompts:
        errors.append("Select at least one number of prompts.")
    return args, errors, backbones, preprocess, layers


def missing_features(backbones, splits, preprocess, layers, dtags=("",)):
    missing = []
    for b in backbones:
        for s in splits:
            for src in SPLITS[s].sources:
                for l in (layers or [None]):
                    for d in dtags:
                        p = feature_path(st.session_state.features_dir, b, src, variant_tag(preprocess, l, d))
                        if not os.path.exists(os.path.join(ROOT, p)):  # no-op join if p is absolute
                            missing.append(p)
    return list(dict.fromkeys(missing))


def eval_splits(eval_sets, tasks):
    from dejavu.data import expand_eval_sets
    return list(dict.fromkeys(s for t in tasks for s in expand_eval_sets(eval_sets, t)))


def split_units(units, keys):
    """Split each (script, args) into one command per value of the list options in `keys`
    (backbone families are expanded), so the pieces can run in parallel."""
    out = []
    for script, args in units:
        combos = [[]]
        for k, v in args:
            if k == "backbones":
                v = expand_backbones(v)
            if k in keys and isinstance(v, list) and len(v) > 1:
                combos = [c + [(k, [x])] for c in combos for x in v]
            else:
                combos = [c + [(k, v)] for c in combos]
        out += [(script, c) for c in combos]
    return out


def plan_jobs(units, split_keys):
    """[(gpu, [commands])]: one job on the selected GPU (or all visible ones), or, with several
    GPUs selected, the work split into independent commands distributed round-robin, one job per GPU."""
    gpus = st.session_state.gpus
    if len(gpus) < 2:
        return [(gpus[0] if gpus else None, [cmd(*u) for u in units])]
    pieces = split_units(units, split_keys)
    queues = [(g, [cmd(*u) for u in pieces[i::len(gpus)]]) for i, g in enumerate(gpus)]
    return [q for q in queues if q[1]]


def launch_button(kind, units, errors, warnings=(), split_keys=("backbones",)):
    for e in errors:
        st.error(e)
    for w in warnings:
        st.warning(w)
    plan = plan_jobs(units, split_keys) if units and not errors else []
    if len(plan) > 1:
        st.caption(f"Work split into {len(plan)} parallel jobs, one per GPU:")
        for g, commands in plan:
            st.code(f"# GPU {g}\n" + "\n".join(commands), language="bash")
    else:
        if len(st.session_state.gpus) > 1 and plan:
            st.info("Nothing to split: a single command runs on the first selected GPU.")
            plan = [(st.session_state.gpus[0], plan[0][1])]
        st.caption("Command that will be run:")
        st.code("\n".join(plan[0][1]) if plan else "—", language="bash")
    if st.button("▶ Run", key=f"{kind}_run", type="primary", disabled=bool(errors)):
        jobs = [launch(kind, commands, g) for g, commands in plan]
        st.success(f"Started {', '.join(f'**{j}**' for j in jobs)}. Follow {'them' if len(jobs) > 1 else 'it'} "
                   f"in the *Jobs* tab.")


def warn_missing(missing, what):
    if not missing:
        return []
    shown = ", ".join(missing[:4]) + (f" … (+{len(missing) - 4})" if len(missing) > 4 else "")
    return [f"{len(missing)} feature file(s) missing for {what}: {shown}. Extract them first in the "
            f"*Feature extraction* tab with the same options."]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_extract():
    st.subheader("Feature extraction")
    st.caption("Extracts the features of the frozen backbones. Each preprocessing, layer and post-processing "
               "gives separate feature files: train and evaluate with the same options afterwards.")
    backbones, bb_args = backbone_picker("ex")
    groups = list(EXTRACTION_GROUPS)
    single = [s for s in SPLITS if not s.startswith("ood_")]
    splits = st.multiselect("Split", groups + single, default=["genimage"], key="ex_splits",
                            help="genimage = train/validation/test; ood = the four OOD benchmarks "
                                 "(also enough for the ood_attribution / ood_detection_attribution sets).")
    c1, c2 = st.columns(2)
    preprocess = c1.selectbox("Preprocessing", PREPROCESSINGS, key="ex_pre")
    layer_mode = c2.radio("Feature", ["final embedding", "selected layers", "all layers"], key="ex_lmode",
                          horizontal=True)
    layers = []
    if layer_mode == "selected layers":
        layers = st.multiselect("Layer", common_layers(backbones), key="ex_layers")
    degs = degradation_picker("ex")
    with st.expander("Other options"):
        c1, c2, c3, c4 = st.columns(4)
        fake_qf96 = c1.selectbox("QF96 on fakes", ["auto", "yes", "no"], key="ex_qf96",
                                 help="auto = as in the paper (all GenImage splits, never OOD).")
        dtype = c2.selectbox("Precision (HF)", ["float32", "float16", "bfloat16"], key="ex_dtype",
                             help="The paper uses float32; bfloat16 fits DINOv3-7B on 24 GB GPUs.")
        bs = c3.number_input("Batch size", 1, 1024, 32, key="ex_bs")
        nw = c4.number_input("Workers", 0, 64, 4, key="ex_nw")
        overwrite = st.checkbox("Overwrite existing features", key="ex_ow")
    errors = []
    if not backbones:
        errors.append("Select at least one backbone.")
    if not splits:
        errors.append("Select at least one split.")
    if layer_mode == "selected layers" and not layers:
        errors.append("Select at least one layer.")
    units = []
    for d in degs:
        dflag = [(d[0], d[1])] if d else []
        units.append(("extract_features.py", [
            ("backbones", bb_args), ("splits", splits),
            ("data_config", st.session_state.data_config or None),
            ("data_root", st.session_state.data_root or None),
            ("features_dir", st.session_state.features_dir),
            ("preprocess", None if preprocess == "default" else preprocess),
            ("layers", layers), ("all_layers", layer_mode == "all layers"), *dflag,
            ("fake_qf96", None if fake_qf96 == "auto" else fake_qf96),
            ("dtype", None if dtype == "float32" else dtype),
            ("batch_size", None if bs == 32 else bs), ("num_workers", None if nw == 4 else nw),
            ("overwrite", overwrite)]))
    warnings = []
    if not st.session_state.data_config and not st.session_state.data_root:
        warnings.append("No data config and no data root: set the dataset paths in the sidebar.")
    # in parallel: one piece per backbone (and post-processing value); the splits of a
    # backbone stay together so that the model is loaded once
    launch_button("extract", units, errors, warnings, split_keys=("backbones",))


def page_train():
    st.subheader("Training")
    st.caption("Trains classifiers on the GenImage train features and evaluates them on the chosen sets. "
               "Every combination of backbone × task × classifier × layer × number of prompts is a configuration.")
    args, errors, backbones, preprocess, layers = config_section("tr")
    eval_sets = st.multiselect("Evaluate after training on", EVAL_SET_CHOICES, default=["genimage"],
                               key="tr_eval", help="Empty = no evaluation. ood = the OOD sets of each task.")
    with st.expander("Hyperparameters (defaults = paper)"):
        c1, c2, c3, c4 = st.columns(4)
        runs = c1.number_input("Runs (MLP)", 1, 100, 10, key="tr_runs")
        epochs = c2.number_input("Epochs", 1, 1000, 50, key="tr_ep")
        bs = c3.number_input("Batch size", 1, 4096, 64, key="tr_bs")
        lr = c4.number_input("Learning rate", 0.0, 1.0, 3e-4, format="%.5f", key="tr_lr")
        c1, c2, c3 = st.columns(3)
        seed = c1.number_input("Seed", 0, 10**6, 0, key="tr_seed")
        pseed = c2.number_input("Prompt seed", 0, 10**6, 42, key="tr_pseed")
        n_jobs = c3.number_input("sklearn n_jobs", -1, 256, -1, key="tr_nj")
        skip = st.checkbox("Skip already trained configurations", key="tr_skip")
    tasks = dict(args)["tasks"]
    unit = ("train.py", args + [
        ("features_dir", st.session_state.features_dir), ("out_dir", st.session_state.out_dir),
        ("eval_sets", eval_sets if eval_sets else ""),
        ("runs", None if runs == 10 else runs), ("epochs", None if epochs == 50 else epochs),
        ("batch_size", None if bs == 64 else bs), ("lr", None if lr == 3e-4 else lr),
        ("seed", None if seed == 0 else seed), ("prompt_seed", None if pseed == 42 else pseed),
        ("n_jobs", None if n_jobs == -1 else n_jobs), ("skip_existing", skip)])
    warnings = []
    if not errors:
        warnings += warn_missing(missing_features(backbones, ["genimage_train"], preprocess, layers), "training")
        warnings += warn_missing(missing_features(backbones, eval_splits(eval_sets, tasks), preprocess, layers),
                                 "evaluation")
    launch_button("train", [unit], errors, warnings, split_keys=TRAIN_SPLIT_KEYS)


def page_evaluate():
    st.subheader("Evaluate trained classifiers")
    st.caption("Select the configuration used for training: its saved runs are evaluated, "
               "e.g. on the OOD sets or on post-processed images (robustness).")
    args, errors, backbones, preprocess, layers = config_section("ev")
    eval_sets = st.multiselect("Evaluation sets", EVAL_SET_CHOICES, default=["genimage_test"], key="ev_eval")
    degs = degradation_picker("ev")
    runs = st.text_input("Runs to evaluate (empty = all saved runs, e.g. 0 1 2)", key="ev_runs")
    if not eval_sets:
        errors.append("Select at least one evaluation set.")
    tasks = dict(args)["tasks"]
    units, dtags = [], []
    for d in degs:
        dflag = [(d[0], d[1])] if d else []
        dtags.append(degradation_tag(d))
        units.append(("evaluate.py", args + [
            ("features_dir", st.session_state.features_dir), ("out_dir", st.session_state.out_dir),
            ("eval_sets", eval_sets), ("runs", runs.split() or None), *dflag]))
    warnings = []
    if not errors:
        warnings += warn_missing(missing_features(backbones, eval_splits(eval_sets, tasks), preprocess, layers,
                                                  dtags), "evaluation")
    launch_button("evaluate", units, errors, warnings, split_keys=TRAIN_SPLIT_KEYS)


def checkpoint_choices():
    """Released checkpoints first, then the classifiers saved in the output folder."""
    order = {"dejavu_det": 0, "dejavu_attr": 1, "dejavu_joint": 2}
    found = sorted(glob.glob(os.path.join(ROOT, "checkpoints", "*.pth")) + glob.glob(os.path.join(ROOT, "checkpoints", "*.pkl")),
                   key=lambda f: (order.get(os.path.splitext(os.path.basename(f))[0], 9), f))
    found += sorted(glob.glob(os.path.join(st.session_state.out_dir, "*", "*", "*", "classifiers", "run*.*")))
    return [os.path.relpath(f, ROOT) if f.startswith(ROOT) else f for f in found]


def page_predict():
    st.subheader("Predict on your own images")
    st.caption("Applies a trained classifier (e.g. the released DejaVu checkpoints in checkpoints/) to any images: "
               "the features are extracted on the fly with the classifier's backbone. With a CSV holding the "
               "ground truth (label: real/fake, model: generator) the metrics are computed too.")
    choices = checkpoint_choices()
    c1, c2 = st.columns([3, 2])
    ckpt = c1.selectbox("Checkpoint", choices + ["other…"], key="pr_ckpt",
                        help="checkpoints/dejavu_{det,attr,joint}.pth = DejaVu for detection, attribution and "
                             "the joint task. Backbone and task are read from the JSON next to the checkpoint.")
    if ckpt == "other…":
        ckpt = c2.text_input("Checkpoint path (.pth / .pkl)", key="pr_ckpt_other")
    elif ckpt:
        stem = os.path.splitext(os.path.join(ROOT, ckpt))[0]
        cfg_file = next((f for f in (stem + ".json", os.path.join(os.path.dirname(os.path.dirname(stem)), "config.json"))
                         if os.path.isfile(f)), None)
        if cfg_file:
            import json
            cfg = json.load(open(cfg_file))["config"]
            c2.caption(f"**Backbone:** {cfg['backbone']}  \n**Task:** {cfg['task']}  \n"
                       f"**Preprocessing:** {cfg.get('preprocess') or 'default'}  \n**Layer:** {cfg.get('layer') or 'final'}")
    mode = st.radio("Input", ["images / folders", "CSV"], horizontal=True, key="pr_mode")
    errors, src = [], []
    fake_qf96 = False
    if mode == "CSV":
        c1, c2 = st.columns(2)
        csv = c1.text_input("CSV file", key="pr_csv",
                            help="Columns: image_path (absolute, or relative to the image root), optional "
                                 "label (real/fake or 0/1) and model (generator of the fakes).")
        root = c2.text_input("Image root (optional, default: the CSV's folder)", key="pr_root")
        fake_qf96 = st.checkbox("JPEG QF=96 on the fake images (GenImage protocol of the paper)", key="pr_qf96",
                                help="Only to reproduce the GenImage setting, whose PNG fakes were compressed to "
                                     "match the JPEG real images. Leave off for other data (as for the OOD benchmarks).")
        src = [("csv", csv or None), ("image_root", root or None)]
        if not csv:
            errors.append("Give the CSV file.")
        elif not os.path.isfile(csv if os.path.isabs(csv) else os.path.join(ROOT, csv)):
            errors.append(f"CSV not found: {csv}")
    else:
        paths = st.text_area("Image files and/or folders (one per line; folders are searched recursively)",
                             key="pr_images")
        paths = [l.strip() for l in paths.splitlines() if l.strip()]
        src = [("images", paths)]
        if not paths:
            errors.append("Give at least one image or folder.")
    name = os.path.splitext(os.path.basename(ckpt))[0] if ckpt else "predictions"
    out = st.text_input("Output folder", os.path.join("predictions", name), key=f"pr_out_{name}")
    with st.expander("Other options"):
        c1, c2, c3 = st.columns(3)
        dtype = c1.selectbox("Precision (HF)", ["float32", "float16", "bfloat16"], key="pr_dtype")
        bs = c2.number_input("Batch size", 1, 1024, 32, key="pr_bs")
        nw = c3.number_input("Workers", 0, 64, 4, key="pr_nw")
    if not ckpt:
        errors.append("Select a checkpoint.")
    unit = ("predict.py", [("checkpoint", ckpt or None), *src, ("out", out), ("fake_qf96", fake_qf96),
                           ("dtype", None if dtype == "float32" else dtype),
                           ("batch_size", None if bs == 32 else bs), ("num_workers", None if nw == 4 else nw)])
    launch_button("predict", [unit], errors, split_keys=())

    st.divider()
    st.subheader("Predictions")
    out_abs = out if os.path.isabs(out) else os.path.join(ROOT, out)
    pred_csv = os.path.join(out_abs, "predictions.csv")
    if not os.path.isfile(pred_csv):
        st.info(f"No predictions in {out} yet: run the job and follow it in the *Jobs* tab.")
        return
    metrics_file = os.path.join(out_abs, "metrics.json")
    if os.path.isfile(metrics_file):
        import json
        m = json.load(open(metrics_file))
        cols = st.columns(4)
        for col, (k, label) in zip(cols, [("balanced_accuracy", "Acc."), ("f1_score", "F1"), ("ap", "AP"), ("auc", "AUC")]):
            col.metric(label, f"{m[k]:.3f}")
        if m.get("per_generator"):
            pg = pd.DataFrame(m["per_generator"])[["model", "n_samples", "balanced_accuracy", "f1_score", "ap", "auc"]]
            st.caption("Detection per generator (each paired with all the real images)")
            st.dataframe(pg.style.format(precision=3), hide_index=True, width="stretch")
    df = pd.read_csv(pred_csv)
    st.caption(f"{len(df)} images · " + ", ".join(f"{k}: {v}" for k, v in df.prediction.value_counts().items()))
    st.dataframe(df, hide_index=True, width="stretch")
    st.download_button("Download predictions.csv", df.to_csv(index=False), file_name="predictions.csv", mime="text/csv")


def page_jobs():
    st.subheader("Jobs")
    jobs = sorted(glob.glob(os.path.join(JOBS_DIR, "*")), reverse=True)
    if not jobs:
        st.info("No jobs started yet.")
        return
    rows = [{"job": os.path.basename(j), "status": job_status(j)} for j in jobs]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=min(250, 40 + 35 * len(rows)))
    finished = [j for j, r in zip(jobs, rows) if not r["status"].startswith("⏳")]
    if finished and st.button(f"🗑 Delete all finished or stopped jobs ({len(finished)})",
                              help="Removes only their commands and logs (gui_jobs/), never features or results."):
        for j in finished:
            shutil.rmtree(j, ignore_errors=True)
        st.rerun()
    job = st.selectbox("Job log", [r["job"] for r in rows])
    jdir = os.path.join(JOBS_DIR, job)
    c1, c2 = st.columns([1, 5])
    auto = c2.toggle("Auto-refresh (5 s)", value=True)
    status = job_status(jdir)
    if status.startswith("⏳"):
        if c1.button("■ Stop", type="secondary"):
            stop_job(jdir)
            time.sleep(1)
            st.rerun()
    elif c1.button("🗑 Delete", help="Removes the job's commands and log (gui_jobs/), not its features or results."):
        shutil.rmtree(jdir, ignore_errors=True)
        st.rerun()
    with st.expander("Commands"):
        st.code(open(os.path.join(jdir, "commands.sh")).read(), language="bash")

    @st.fragment(run_every=5 if auto and status.startswith("⏳") else None)
    def show_log():
        st.markdown(f"**Status:** {job_status(jdir)}")
        st.code(tail(os.path.join(jdir, "log.txt")) or "(empty log)", language="text")
    show_log()


def page_results():
    st.subheader("Results")
    mdir = os.path.join(st.session_state.out_dir, "metrics")
    csvs = sorted(glob.glob(os.path.join(mdir, "*.csv")))
    if not csvs:
        st.info(f"No metrics files in {mdir}.")
    else:
        name = st.selectbox("Metrics file", [os.path.basename(c) for c in csvs])
        df = pd.read_csv(os.path.join(mdir, name))
        st.caption(f"{len(df)} rows (one per configuration × set × run), averaged over runs.")
        filt_cols = [c for c in ["backbone", "classifier", "preprocess", "layer", "num_prompts", "train_generators",
                                 "eval_set", "eval_degradation", "model"] if c in df.columns]
        cols = st.columns(3)
        for i, c in enumerate(filt_cols):
            vals = sorted(df[c].astype(str).unique())
            if len(vals) > 1:
                sel = cols[i % 3].multiselect(c, vals, key=f"res_{name}_{c}")
                if sel:
                    df = df[df[c].astype(str).isin(sel)]
        if df.empty:
            st.warning("No rows match the filters.")
        else:
            out = summarize(df)
            show_std = st.checkbox("Show standard deviations")
            keep = [c for c in out.columns if not c.endswith("_std") or show_std]
            keep = [c for c in keep if not c.startswith(("accuracy", "precision", "recall"))]
            out = out[keep]
            # hide key columns that are constant after filtering
            out = out[[c for c in out.columns if not (c in filt_cols + ["task"] and out[c].nunique() == 1
                                                      and len(out) > 1)]]
            st.dataframe(out.style.format(precision=3), hide_index=True, width="stretch")
            st.download_button("Download CSV", out.to_csv(index=False), file_name=f"summary_{name}", mime="text/csv")

    st.divider()
    st.subheader("Confusion matrices")
    if not st.toggle("Show confusion matrices", value=False, key="show_cms",
                     help="Attribution and detection+attribution only. Off by default: with many results "
                          "listing them takes a while."):
        return
    pngs = sorted(glob.glob(os.path.join(st.session_state.out_dir, "*", "*", "*", "confusion_matrices", "*.png")))
    if not pngs:
        st.info("No confusion matrices yet (saved for attribution and detection+attribution).")
        return
    rel = [os.path.relpath(p, st.session_state.out_dir) for p in pngs]
    choice = st.selectbox("Matrix", rel, format_func=lambda r: r.replace("/confusion_matrices/", " · ")[:-4])
    path = os.path.join(st.session_state.out_dir, choice)
    c1, c2 = st.columns([3, 2])
    c1.image(path)
    csv = path[:-4] + ".csv"
    if os.path.exists(csv):
        c2.caption("Row-normalized, averaged over runs")
        c2.dataframe(pd.read_csv(csv, index_col=0).style.format(precision=3).background_gradient(cmap="Blues", axis=None))


# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="DejaVu", layout="wide")
    # hide Streamlit's animated "running" indicator (swimmer, cyclist, ...) in the top-right corner
    st.html("<style>[data-testid='stStatusWidget'] {visibility: hidden; height: 0;}</style>")
    st.title("DejaVu")
    with st.sidebar:
        st.header("Paths")
        st.text_input("Data config (YAML)", "configs/data_paths.yaml" if os.path.exists(
            os.path.join(ROOT, "configs", "data_paths.yaml")) else "", key="data_config",
            help="YAML with the dataset roots (see configs/data_paths.yaml). Only needed for extraction.")
        st.text_input("Data root (optional)", "", key="data_root",
                      help="Folder containing all the datasets; overrides data_root of the YAML.")
        st.text_input("Features folder", "features", key="features_dir")
        st.text_input("Output folder", "results", key="out_dir")
        st.header("Execution")
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
                                  "--format=csv,noheader"], capture_output=True, text=True, timeout=5).stdout
            gpus = [l.split(",")[0].strip() for l in out.strip().splitlines()]
            st.caption("\n\n".join(out.strip().splitlines()))
        except (OSError, subprocess.SubprocessError):
            gpus = []
        if st.button("Use all GPUs", disabled=not gpus):
            st.session_state.gpus = list(gpus)
        st.multiselect("GPUs", gpus, key="gpus",
                       help="Empty: one job that sees all the GPUs (it uses the first). One GPU: the job runs "
                            "there. Several GPUs: the work is split by backbone (extraction) or by "
                            "configuration (training, evaluation) into one parallel job per GPU.")
        st.caption("Jobs are independent: you can start new ones while others are running "
                   "(jobs on the same GPU share its memory).")
        st.caption(f"Jobs and logs: {os.path.relpath(JOBS_DIR, ROOT)}/")

    tabs = st.tabs(["Feature extraction", "Training", "Evaluation", "Predict", "Jobs", "Results"])
    with tabs[0]:
        page_extract()
    with tabs[1]:
        page_train()
    with tabs[2]:
        page_evaluate()
    with tabs[3]:
        page_predict()
    with tabs[4]:
        page_jobs()
    with tabs[5]:
        page_results()


main()
