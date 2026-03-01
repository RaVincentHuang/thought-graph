# Thought Data

## Directory Structure

```
thought_data/
├── raw_data/           # Raw data files before processing
├── processed_data/     # Cleaned and processed data
└── src/                # Source code
```

## Usage

### Evaluation of different LLMs with multiple metrics
Raw thought data is stored in the raw_data directory. Five bash scripts with identical names are located in src/blocksworld, game24, and gsm8k directories. Script functions and usage notes:
1. Raw thought data is stored in the raw_data directory. Five bash scripts with identical names are located in src/blocksworld, game24, and gsm8k directories. Script functions and usage notes:
2. basic_process.sh: Must run first. Preprocesses raw_data and performs cluster encoding to generate data for $NIS$ calculation. Configure: EXP_NAME (filename in raw_data), BENCHMARK_MODEL (reference model), and cluster range (NUM_CLUSTERS_MIN, NUM_CLUSTERS_MAX).
3. aggregation_calculation.sh: Calculates $P_w$ for each model. Configure: EXP_NAME, BENCHMARK_MODEL, and CLUSTER_ID.
ctrs_calculation.sh: Calculates $CTRS$ for each model. Configure: EXP_NAME and BENCHMARK_MODEL.
4. graph_generation.sh: Generates thought data graphs for frequent subgraph analysis. Configure: EXP_NAME, BENCHMARK_MODEL, and CLUSTER_ID.
5. kl_calculation.sh: Calculates $KL$ divergence metrics. Configure: EXP_NAME, BENCHMARK_MODEL, and CLUSTER_ID.

For reasoning efficiency metrics using frequent subgraphs, follow instructions in src/frequent_pattern/run_experiment.sh.

## Prompt Optimization
Navigate to src/prompt_optimization/case_study. Configure the bash scripts (download open-source model parameters or input API keys for closed-source models), then execute them

## RL Optimization
Navigate to src/RL, configure the bash scripts according to the README, and then run them.

## Environment

To set up the environment, ensure you have the following installed:

- **Python**: Version 3.11
- **Java SDK**: Version 1.8.0

To install the required Python packages, run:

```bash
pip install -r requirements.txt
```





