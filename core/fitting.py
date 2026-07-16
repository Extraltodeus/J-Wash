import hashlib
import json
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

from jlens.lens import JacobianLens

import config
from core.gpus import gpu_stats

FITS_DIR = config.DATA_DIR / "fits"
WORKER = config.ROOT / "scripts" / "fit_worker.py"

# Fit corpus. Any HuggingFace dataset id works as-is: wikitext is the default
# and keeps a dedicated streamed path, every other id goes through the generic
# loader below. Several ids = an equal-parts mix.
DATASET_WIKITEXT = "Salesforce/wikitext-103-raw-v1"

# Seed shared by the row sampling and the mix shuffle: a continued fit that asks
# for skip+n rows deterministically extends the sequence it drew the first n
# from (sample(skip+n) then drop the head).
_SAMPLE_SEED = 1729


def _slug(dataset):
    """Short, filename-safe tag from a dataset id's last path segment, e.g.
    ``heretic-org/Semantic-Harmless`` -> ``semantic-harmless``."""
    tail = dataset.rstrip("/").split("/")[-1].lower()
    return re.sub(r"[^a-z0-9]+", "-", tail).strip("-")[:24] or "dataset"


def _text_column(features):
    """Column to fit on: prefer ``text``, else the first string-valued column."""
    from datasets import Value

    if "text" in features:
        return "text"
    for name, feat in features.items():
        if isinstance(feat, Value) and feat.dtype == "string":
            return name
    raise ValueError(
        "dataset exposes no text column to fit on "
        f"(columns: {', '.join(features) or 'none'})"
    )


def _pack(texts, count, target=350):
    """Pack ``texts`` into ~``target``-char sequences, stopping as soon as
    ``count`` sequences are ready. jlens skips the first 16 positions of every
    sequence as attention sinks, so short unpacked prompts would almost all be
    dropped as too short. Returns fewer than ``count`` only if ``texts`` runs
    out (the caller decides whether that is an error)."""
    packs, cur = [], ""
    for text in texts:
        text = (text or "").strip()
        if not text:
            continue
        cur = f"{cur}\n\n{text}" if cur else text
        if len(cur) >= target:
            packs.append(cur)
            cur = ""
            if len(packs) >= count:
                return packs
    if cur and len(packs) < count:
        # trailing remainder: keep it so a just-large-enough dataset still fills
        # its quota (jlens tolerates a slightly-short final sequence)
        packs.append(cur)
    return packs


def _load_split(dataset):
    """``dataset``'s ``train`` split, or its first split if it has no ``train``."""
    from datasets import load_dataset

    try:
        return load_dataset(dataset, split="train")
    except ValueError:
        dd = load_dataset(dataset)
        return dd[next(iter(dd))]


def _load_one(dataset, n, skip):
    """``n`` training SEQUENCES from a single ``dataset`` id, skipping the first
    ``skip`` (continue-from: the new sequences must not overlap the base lens's).

    wikitext keeps its historical path (first records >=600 chars, streamed —
    one record already is one sequence). Any other HF dataset is loaded whole,
    its text column shuffled with a fixed seed, then PACKED into ~350-char
    sequences until skip+n are ready (median instruct prompt ~10 tokens, so
    several rows per sequence). ``n``/``skip`` count OUTPUT sequences, so the
    number the user asks for is exactly what the fit iterates over — not source
    rows, whose count varies per dataset."""
    if n <= 0:
        return []
    if dataset == DATASET_WIKITEXT:
        from jlens.examples import load_wikitext_prompts

        # load skip + n then keep the tail: the new sequences don't overlap the
        # base lens's
        return load_wikitext_prompts(skip + n)[skip:]
    import random

    ds = _load_split(dataset)
    col = _text_column(ds.features)
    texts = [r[col] for r in ds]
    random.Random(_SAMPLE_SEED).shuffle(texts)
    packs = _pack(texts, skip + n)
    if len(packs) < skip + n:
        raise ValueError(
            f"{dataset}: {len(texts)} rows pack into only {len(packs)} sequences, "
            f"{skip + n} requested — lower n_prompts"
        )
    return packs[skip:skip + n]


def _load_corpus(datasets, n, skip=0):
    """``n`` training SEQUENCES drawn from ``datasets`` (a list of HF dataset
    ids). A single id loads that dataset; several are mixed in EQUAL parts — n
    and skip are each split across them (the first datasets take the rounding
    remainder) and the union is shuffled so multi-GPU slices stay mixed. Because
    the count is in sequences, ``n`` is exactly what the fit iterates over."""
    datasets = list(datasets)
    if len(datasets) == 1:
        return _load_one(datasets[0], n, skip)
    import random

    k = len(datasets)
    prompts = []
    for i, ds in enumerate(datasets):
        prompts += _load_one(ds, n // k + int(i < n % k), skip // k + int(i < skip % k))
    random.Random(_SAMPLE_SEED).shuffle(prompts)
    return prompts

def _default_dim_batch(device):
    """Default dim_batch scaled to the device's VRAM.
    Measured on a 4B bf16 fit: 8 fits in 16 GB, 4 in 12 GB."""
    try:
        total = gpu_stats()[int(device.split(":")[1])]["vram_total"]
        return 4 if total >= 15 * 2**30 else 2
    except Exception:
        return 1


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class FitManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._procs = []
        self.state = {"state": "idle"}
        self.on_progress = None

    def _emit(self):
        if self.on_progress:
            self.on_progress(dict(self.state))

    def start(self, *, model_id, source, n_prompts=100, dtype="bf16", quant=None,
              devices=("cuda:0",), name=None, dim_batch=None,
              max_seq_len=128, source_layers=None, model_revision=None,
              continue_from=None, datasets=(DATASET_WIKITEXT,)):
        with self._lock:
            if self.state.get("state") == "running":
                raise ValueError("a fitting is already in progress")
            if not devices:
                raise ValueError("at least one device required")
            datasets = [d.strip() for d in datasets if d and d.strip()]
            if not datasets:
                raise ValueError("at least one dataset required")
            skip_prompts = 0
            base_lens = None
            if continue_from:
                base_lens = JacobianLens.load(continue_from)
                # new prompts: skip those already seen by the base lens
                skip_prompts = base_lens.n_prompts
                if source_layers is None:
                    source_layers = list(base_lens.source_layers)
            if name is None:
                base = model_id.split("/")[-1]
                if len(datasets) > 1:
                    base += "_mixed"
                elif datasets[0] != DATASET_WIKITEXT:
                    base += "_" + _slug(datasets[0])
                total = n_prompts + skip_prompts
                name = f"{base}_n{total}" if continue_from else f"{base}_n{n_prompts}"
            params = {
                "model_id": model_id,
                "source": source,
                "model_revision": model_revision,
                "dtype": dtype,
                "quant": quant,
                "n_prompts": n_prompts,
                "datasets": datasets,
                "devices": list(devices),
                "dim_batch": dim_batch,
                "max_seq_len": max_seq_len,
                "source_layers": source_layers,
                "continue_from": continue_from,
                "skip_prompts": skip_prompts,
            }
            self.state = {
                "state": "running",
                "name": name,
                "phase": "corpus",
                "total": n_prompts,
                "done": 0,
                "workers": [],
                "eta_seconds": None,
                "started_at": _now(),
                "params": params,
                "error": None,
            }
            self._procs = []
        threading.Thread(target=self._run, args=(name, params), daemon=True).start()
        return dict(self.state)

    def stop(self):
        with self._lock:
            for proc in self._procs:
                if proc.poll() is None:
                    proc.terminate()
            if self.state.get("state") == "running":
                self.state["state"] = "stopping"
        self._emit()
        return dict(self.state)

    def _run(self, name, params):
        try:
            job_dir = FITS_DIR / name
            job_dir.mkdir(parents=True, exist_ok=True)
            corpus_path = job_dir / "corpus.json"
            if corpus_path.exists():
                prompts = json.loads(corpus_path.read_text(encoding="utf-8"))
            else:
                prompts = _load_corpus(
                    params.get("datasets", [DATASET_WIKITEXT]),
                    params["n_prompts"],
                    params.get("skip_prompts", 0),
                )
                corpus_path.write_text(
                    json.dumps(prompts, ensure_ascii=False), encoding="utf-8"
                )
            devices = params["devices"]
            n = len(prompts)
            if len(devices) == 2:
                cut = int(n * 0.65)
                slices = [prompts[:cut], prompts[cut:]]
            else:
                slices = [prompts]

            self.state.update(phase="fitting", total=n)
            workers = []
            started = time.perf_counter()
            for i, (device, chunk) in enumerate(zip(devices, slices)):
                slice_path = job_dir / f"slice{i}.json"
                if not slice_path.exists():
                    slice_path.write_text(json.dumps(chunk, ensure_ascii=False), encoding="utf-8")
                dim_batch = params["dim_batch"] or _default_dim_batch(device)
                cmd = [
                    sys.executable, "-X", "utf8", str(WORKER),
                    "--model", params["source"],
                    "--device", device,
                    "--dtype", params["dtype"],
                    "--prompts", str(slice_path),
                    "--checkpoint", str(job_dir / f"ckpt{i}.pt"),
                    "--out", str(job_dir / f"lens{i}.pt"),
                    "--dim-batch", str(dim_batch),
                    "--max-seq-len", str(params["max_seq_len"]),
                ]
                if params["quant"]:
                    cmd += ["--quant", params["quant"]]
                if params["source_layers"]:
                    cmd += ["--source-layers", json.dumps(params["source_layers"])]
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    cwd=str(config.ROOT),
                )
                self._procs.append(proc)
                worker_state = {
                    "device": device,
                    "done": 0,
                    "total": len(chunk),
                    "dim_batch": dim_batch,
                    "state": "loading",
                    "elapsed": 0.0,
                    # [done, elapsed] of the last 10 updates: the ETA follows the
                    # RECENT pace (throughput can degrade mid-fit, e.g. VRAM
                    # saturated — a global average would then freeze the ETA)
                    "hist": [],
                }
                workers.append(worker_state)
                threading.Thread(
                    target=self._read_worker, args=(proc, worker_state, started), daemon=True
                ).start()
            self.state["workers"] = workers
            self._emit()

            stderr_tails = [""] * len(self._procs)

            def drain_err(index, proc):
                data = proc.stderr.read()
                stderr_tails[index] = (data or "")[-2000:]

            drainers = [
                threading.Thread(target=drain_err, args=(i, p), daemon=True)
                for i, p in enumerate(self._procs)
            ]
            for t in drainers:
                t.start()
            for proc in self._procs:
                proc.wait()
            for t in drainers:
                t.join()
            failed = [i for i, p in enumerate(self._procs) if p.returncode != 0]
            if self.state.get("state") == "stopping":
                self.state.update(state="stopped")
                self._emit()
                return
            if failed:
                detail = " | ".join(stderr_tails[i].strip().splitlines()[-1] if stderr_tails[i].strip() else "?" for i in failed)
                raise RuntimeError(f"worker(s) {failed} failed: {detail}")

            self.state.update(phase="merge")
            self._emit()
            partials = [
                JacobianLens.load(str(job_dir / f"lens{i}.pt"))
                for i in range(len(slices))
            ]
            merged = JacobianLens.merge(partials) if len(partials) > 1 else partials[0]
            if params.get("continue_from"):
                base_lens = JacobianLens.load(params["continue_from"])
                if base_lens.source_layers != merged.source_layers:
                    raise RuntimeError(
                        "cannot continue: the source layers differ from the base lens "
                        f"({base_lens.source_layers[0]}..{base_lens.source_layers[-1]} vs "
                        f"{merged.source_layers[0]}..{merged.source_layers[-1]})"
                    )
                # weighted average by n_prompts = equivalent to a fit over the union
                merged = JacobianLens.merge([base_lens, merged])
            out_dir = config.LENSES_DIR / name
            out_dir.mkdir(parents=True, exist_ok=True)
            lens_path = out_dir / "lens.pt"
            merged.save(str(lens_path))
            meta = {
                "name": name,
                "model_id": params["model_id"],
                "model_revision": params["model_revision"],
                "model_source": params["source"],
                "d_model": merged.d_model,
                "source_layers": [merged.source_layers[0], merged.source_layers[-1]],
                "dtype": params["dtype"],
                "quant": params["quant"],
                "n_prompts": merged.n_prompts,
                "corpus": (
                    "mixed: " + " + ".join(params["datasets"]) + " (equal parts)"
                    if len(params["datasets"]) > 1
                    else params["datasets"][0]
                ),
                "max_seq_len": params["max_seq_len"],
                "devices": params["devices"],
                "continued_from": params.get("continue_from"),
                "config_hash": hashlib.sha1(
                    json.dumps(params, sort_keys=True).encode()
                ).hexdigest()[:16],
                "created_at": _now(),
                "fit_seconds": round(time.perf_counter() - started, 1),
            }
            (out_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            self.state.update(
                state="done",
                phase="done",
                lens_path=str(lens_path),
                meta=meta,
                eta_seconds=0,
            )
            self._emit()
        except Exception as exc:
            self.state.update(state="error", error=str(exc))
            self._emit()

    def _read_worker(self, proc, worker_state, started):
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event["event"] == "loading":
                worker_state["state"] = "loading"
            elif event["event"] in ("progress", "resume"):
                worker_state["state"] = "fitting"
                worker_state["done"] = event["done"]
                worker_state["total"] = event["total"]
                worker_state["elapsed"] = round(time.perf_counter() - started, 1)
                hist = worker_state.setdefault("hist", [])
                hist.append([worker_state["done"], worker_state["elapsed"]])
                del hist[:-10]
            elif event["event"] == "done":
                worker_state["state"] = "done"
                worker_state["done"] = worker_state["total"]
            self._refresh_totals(started)
            self._emit()

    def _refresh_totals(self, started):
        workers = self.state.get("workers", [])
        self.state["done"] = sum(w["done"] for w in workers)
        etas = []
        for w in workers:
            if not (w["done"] > 0 and w["elapsed"] > 0 and w["done"] < w["total"]):
                continue
            hist = w.get("hist") or []
            if len(hist) >= 2 and hist[-1][1] > hist[0][1] and hist[-1][0] > hist[0][0]:
                # pace over the last 10 updates (sliding window)
                rate = (hist[-1][0] - hist[0][0]) / (hist[-1][1] - hist[0][1])
            else:
                rate = w["done"] / w["elapsed"]
            etas.append((w["total"] - w["done"]) / rate)
        # multi-GPU: the fit ETA = the slowest worker
        self.state["eta_seconds"] = round(max(etas), 0) if etas else None
        self.state["vram"] = [
            {"index": g["index"], "used_gb": round(g["vram_used"] / 2**30, 1)}
            for g in gpu_stats()
        ]
