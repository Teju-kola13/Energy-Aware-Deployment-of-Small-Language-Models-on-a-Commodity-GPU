# Energy-Aware-Deployment-of-Small-Language-Models-on-a-Commodity-GPU
Benchmark and optimize energy consumption, power draw, and inference throughput for Small Language Models running on consumer and commodity NVIDIA GPUs
# Energy-Aware Deployment of Small Language Models on a Commodity GPU

An end-to-end benchmarking framework to profile, optimize, and analyze the energy efficiency, throughput, and accuracy tradeoffs of Small Language Models (SLMs) during inference on commodity hardware (e.g., NVIDIA T4, RTX series).

---

## Overview

Deploying language models on consumer-grade or budget data-center GPUs introduces strict constraints on memory capacity and thermal envelopes. This repository provides the tooling to measure hardware-level energy consumption alongside task accuracy across quantization precisions and batch configurations.

* **Hardware-Level Energy Tracking**: Real-time power telemetry using NVIDIA Management Library (`pynvml`) counters to compute Joules per prompt and Joules per token.
* **Quantization Benchmarking**: Direct performance comparisons across native **FP16**, **INT8**, and **NF4 (4-bit)** quantization formats using `bitsandbytes`.
* **Pareto Efficiency Profiling**: Automated generation of Pareto front plots mapping accuracy retention vs. total energy consumed.
* **Batch & Latency Profiling**: Throughput analysis across varying batch sizes and context lengths on memory-constrained GPUs.

---

## Directory Structure

```text
├── configs/                  # Benchmark configurations (batch sizes, precisions, datasets)
├── data/                     # Dataset loaders (GSM8K, Alpaca, etc.)
├── results/                  # Generated CSV metrics and benchmark logs
│   ├── figures/              # Output Pareto plots and energy comparison charts
│   └── tables/               # Formatted latency, throughput, and energy summary tables
├── src/
│   ├── energy_meter.py       # Background GPU power tracking thread using pynvml
│   ├── evaluator.py          # Downstream task accuracy evaluation pipelines
│   ├── inference_engine.py   # Model loading and batched text generation runner
│   └── visualize.py          # Plotting utilities for energy vs. accuracy trade-offs
├── run_benchmark.py          # Main CLI entrypoint to execute experiments
├── requirements.txt
└── README.md

```

---

## Installation

### Prerequisites

* Python 3.10+
* NVIDIA GPU with CUDA support (e.g., NVIDIA Tesla T4 16GB, RTX 3060/4060 or higher)
* NVIDIA Driver with NVML support

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/energy-aware-slm-deployment.git
cd energy-aware-slm-deployment

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

```

### Dependencies (`requirements.txt`)

```text
torch>=2.2.0
transformers>=4.40.0
accelerate>=0.28.0
bitsandbytes>=0.43.0
pynvml>=11.5.0
datasets>=2.18.0
pandas>=2.2.0
matplotlib>=3.8.0
seaborn>=0.13.0

```

---

## Quickstart

### 1. Run a Single Benchmark

Measure energy consumption, latency, and throughput for a specific model and quantization format:

```bash
python run_benchmark.py \
  --model_name "meta-llama/Llama-3.2-3B-Instruct" \
  --precision "nf4" \
  --dataset "gsm8k" \
  --batch_size 4 \
  --samples 100 \
  --output_dir "./results"

```

### 2. Multi-Precision Grid Search

Run comparative profiling across FP16, INT8, and NF4 precisions:

```bash
python run_benchmark.py \
  --model_name "meta-llama/Llama-3.2-3B-Instruct" \
  --run_all_precisions \
  --dataset "gsm8k" \
  --batch_sizes 1 2 4 8 \
  --output_dir "./results"

```

---

## Metrics Tracked

| Metric | Unit | Measurement Method |
| --- | --- | --- |
| **Average Power Draw** | Watts (W) | Sampled via `nvmlDeviceGetPowerUsage` at 100ms intervals |
| **Total Energy** | Joules (J) | Numerical integration of power over inference duration ($\int P \, dt$) |
| **Energy per Token** | Joules/Token (J/tok) | Total Energy divided by total output tokens generated |
| **Energy per Prompt** | Joules/Prompt (J/prompt) | Total Energy divided by batch size and processed prompts |
| **Token Throughput** | Tokens/sec | Total generated tokens over total wall-clock time |
| **Task Accuracy** | % | Benchmark-specific metric (e.g., exact match on GSM8K) |

---

## Generating Figures

After benchmark completion, generate Pareto trade-off curves and latency breakdowns:

```bash
python -m src.visualize --data_path "./results/benchmark_metrics.csv" --output_dir "./results/figures"

```

The output plots include:

* **Accuracy vs. Joules/Prompt** (Pareto efficiency frontier)
* **Power Draw vs. Batch Size** across quantization levels
* **Memory Footprint (VRAM) vs. Throughput**

