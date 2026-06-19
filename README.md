# on-DeviceNLP

Research code for exporting MarianMT translation models to on-device formats, with a focus on ONNX and CoreML deployment.

The repository is organized around a few related experiments:

- Baseline MarianMT evaluation in PyTorch.
- Exporting full MarianMT models to ONNX.
- Applying dynamic quantization and weight quantization to ONNX exports.
- Reducing vocabulary size before export and then evaluating the compressed models.
- Exporting MarianMT models to CoreML for Apple-device inference.

This is not a polished library package. It is an experiment-driven repo with runnable scripts, custom runtime wrappers, benchmark helpers, and several branch-specific variants.

## What This Repo Does

The codebase works primarily with Hugging Face MarianMT models such as `Helsinki-NLP/opus-mt-en-fr` and `Helsinki-NLP/opus-mt-en-de`.

The main workflows are:

1. Evaluate a baseline PyTorch translation model.
2. Download a MarianMT model from Hugging Face.
3. Export encoder and decoder graphs to ONNX or CoreML.
4. Quantize exported ONNX models.
5. Prune large shared embeddings from ONNX graphs in some experiment tracks.
6. Compare quality and runtime using BLEU, ChrF, TER, METEOR, and simple benchmark scripts.

## Repository Layout

### Top level

- `translation/baseline/`: baseline MarianMT evaluation in PyTorch.
- `Baseline_to_ONNX/`: ONNX export pipelines for baseline MarianMT models.
- `VocabRed_to_ONNX/`: vocabulary-reduction experiments plus ONNX export and evaluation.
- `Baseline_to_CoreML/`: CoreML export, verification, and BLEU evaluation scripts.
- `bleu_wmt.py`: standalone BLEU scoring utility for model outputs against WMT14.
- `bashrun.sh`: an old SLURM launcher stub.

### ONNX experiment families

Inside both `Baseline_to_ONNX/` and `VocabRed_to_ONNX/`, there are multiple variants:

- `Onnx_dynamic_Quant/`: export plus dynamic quantization support.
- `Onnx_dynamic_Quant_pruned/`: export plus a pruning path for large shared embeddings.
- `Weights_Quantized/`: weight-quantization experiments and comparisons.
- `Onnx_dynamic_Quant_pruning/`: additional pruning-oriented experiments in the vocab-reduction track.

Each variant typically contains:

- `convert.py` for download and export.
- `core/` for custom ONNX runtime wrappers, generation code, quantization helpers, and benchmarks.
- evaluation or BLEU scripts.
- shell scripts used to run batches of experiments.

## Key Scripts

### Baseline PyTorch evaluation

- `translation/baseline/baseline_test.py`

What it does:

- Loads a MarianMT model from Hugging Face.
- Streams a subset of WMT14.
- Generates translations in PyTorch.
- Computes SacreBLEU through Hugging Face `evaluate`.

Example:

```bash
python translation/baseline/baseline_test.py
```

### Baseline ONNX export

- `Baseline_to_ONNX/Onnx_dynamic_Quant/convert.py`

What it does:

- Downloads a MarianMT model locally.
- Exports encoder and decoder ONNX graphs.
- Optionally applies quantization.
- Runs a correctness check comparing ONNX Runtime output with PyTorch output.
- Archives the exported artifacts.

Example:

```bash
python Baseline_to_ONNX/Onnx_dynamic_Quant/convert.py \
	Helsinki-NLP/opus-mt-en-fr \
	--output ./artifacts
```

Disable quantization:

```bash
python Baseline_to_ONNX/Onnx_dynamic_Quant/convert.py \
	Helsinki-NLP/opus-mt-en-fr \
	--output ./artifacts \
	--no-quantize
```

### Pruning large ONNX shared embeddings

- `Baseline_to_ONNX/Onnx_dynamic_Quant_pruned/prune_shared_embedding.py`
- `VocabRed_to_ONNX/Onnx_dynamic_Quant_pruned/prune_shared_embedding.py`

What it does:

- Scans ONNX initializers.
- Removes very large tensors above a threshold.
- Re-inserts them as graph inputs so they can be provided at runtime.

Example:

```bash
python Baseline_to_ONNX/Onnx_dynamic_Quant_pruned/prune_shared_embedding.py \
	--input-model ./encoder.onnx \
	--output-model ./encoder_pruned.onnx
```

### Vocabulary-reduced model evaluation

- `VocabRed_to_ONNX/eval.py`

What it does:

- Loads a locally saved MarianMT model.
- Runs a sample translation.
- Evaluates BLEU, ChrF, TER, and METEOR on a WMT14 split.

Example:

```bash
python VocabRed_to_ONNX/eval.py \
	--model-dir ./path/to/model_dir \
	--device cpu \
	--src fr \
	--tgt en \
	--split 'validation[:100]'
```

### CoreML export

- `Baseline_to_CoreML/convert_coreml.py`

What it does:

- Downloads a MarianMT model locally.
- Builds CoreML encoder and decoder packages.
- Writes output artifacts and archives them.

Example:

```bash
python Baseline_to_CoreML/convert_coreml.py \
	Helsinki-NLP/opus-mt-en-fr \
	--output ./coreml_out
```

### CoreML BLEU evaluation

- `Baseline_to_CoreML/bleu_score_coreml.py`

Example:

```bash
python Baseline_to_CoreML/bleu_score_coreml.py \
	./coreml_out/Helsinki-NLP_opus-mt-en-fr \
	./source.txt \
	./reference.txt
```

## Setup

### Python

Use Python 3.10 or newer. A virtual environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### Install dependencies

The checked-in `requirements.txt` is currently empty, so install the packages used by the scripts directly:

```bash
pip install \
	torch \
	transformers \
	datasets \
	evaluate \
	sacrebleu \
	sentencepiece \
	tqdm \
	numpy \
	onnx \
	onnxruntime \
	onnxsim \
	onnxoptimizer \
	coremltools
```

Notes:

- `coremltools` is only needed for the CoreML pipeline.
- CUDA is optional, but several scripts will use GPU automatically when available.
- Marian tokenizers often require `sentencepiece`.

## Typical Workflows

### 1. Run the PyTorch baseline

```bash
python translation/baseline/baseline_test.py
```

### 2. Export a Hugging Face MarianMT model to ONNX

```bash
python Baseline_to_ONNX/Onnx_dynamic_Quant/convert.py \
	Helsinki-NLP/opus-mt-en-fr \
	--output ./artifacts
```

Expected outputs usually include:

- `encoder.onnx`
- `decoder.onnx`
- tokenizer and config files
- a zip archive of the output directory

### 3. Evaluate ONNX quality or speed

The ONNX experiment folders include custom runtime wrappers under `core/` and benchmark helpers such as:

- `Baseline_to_ONNX/Onnx_dynamic_Quant/core/benchmark.py`
- `VocabRed_to_ONNX/Onnx_dynamic_Quant/core/benchmark.py`

These compare PyTorch and ONNX Runtime outputs and report simple CPU or GPU timings.

### 4. Export to CoreML for Apple-device experiments

```bash
python Baseline_to_CoreML/convert_coreml.py \
	Helsinki-NLP/opus-mt-en-fr \
	--output ./coreml_out
```

### 5. Evaluate vocab-reduced models

```bash
python VocabRed_to_ONNX/eval.py \
	--model-dir ./path/to/model_dir \
	--device cpu
```

## Current Caveats

This repo contains research code, so a few rough edges matter if you are trying to reuse it:

- Several scripts contain hardcoded `sys.path.append(...)` values pointing to an older UMass filesystem layout. Those paths should be replaced with repo-relative imports before treating the code as portable.
- `requirements.txt` is empty even though the code depends on PyTorch, Transformers, ONNX Runtime, datasets, evaluation libraries, and CoreML tools.
- Some shell scripts are cluster-specific and use SLURM directives.
- Output directories and local model caches are created by scripts on demand and are not standardized across all folders.
- A few experiment directories duplicate similar logic with small variations, so not every branch of the repo is equally maintained.

## Data and Models

The code relies on:

- Hugging Face MarianMT checkpoints, especially the `Helsinki-NLP/opus-mt-*` family.
- WMT14 evaluation splits loaded through the Hugging Face `datasets` library.

Network access is generally required the first time you download a model or dataset.

## Output Artifacts

Depending on the workflow, the repo can generate:

- Hugging Face model snapshots in local model directories.
- ONNX encoder and decoder graphs.
- Quantized ONNX graphs.
- Pruned ONNX graphs with large embeddings externalized as inputs.
- CoreML `.mlpackage` exports.
- ZIP archives of exported assets.
- Benchmark and metric reports.

## Suggested Cleanup If You Plan To Extend This Repo

If you want to turn this into a reusable project instead of a research snapshot, these are the highest-value next steps:

1. Replace hardcoded absolute paths with package-relative imports.
2. Populate `requirements.txt` or move to `pyproject.toml`.
3. Deduplicate the repeated ONNX export code across experiment folders.
4. Add a single entrypoint CLI for export, quantization, pruning, and evaluation.
5. Document expected artifact layouts for each experiment path.

## Branches

The upstream repository includes multiple experiment branches such as `Pytorch_onnx`, `Deployment_test`, `evaluation_test`, `onnx-export-code`, and others. If you cloned your own fork first, add the upstream remote to access those branches:

```bash
git remote add upstream https://github.com/SchrOdinger11/on-DeviceNLP.git
git fetch upstream
git branch -a
```

Then create a local branch from any upstream branch you need:

```bash
git checkout -b Pytorch_onnx upstream/Pytorch_onnx
```

## Status

The repository already contains useful export and evaluation code, but it should be treated as a research prototype rather than a turnkey package.

