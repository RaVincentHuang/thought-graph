#!/bin/bash
SUPPORT_START=20
SUPPORT_END=30
CONFIG_PATH="src/frequent_pattern/config.json"

show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -n, --exp_name         Experiment name"
    echo "  -s, --support_start    Start value for support range (default: 20)"
    echo "  -e, --support_end      End value for support range (default: 30)"
    echo "  -d, --dataset          Dataset name"
    echo "  -x, --exp_num          Experiment number"
    echo "  -b, --benchmark        Benchmark model name"
    echo "  -c, --cluster_num      Cluster number"
    echo "  -h, --help            Show this help message"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        -s|--support_start)
            SUPPORT_START="$2"
            shift 2
            ;;
        -e|--support_end)
            SUPPORT_END="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -x|--exp_num)
            EXP_NUM="$2"
            shift 2
            ;;
        -b|--benchmark)
            BENCHMARK="$2"
            shift 2
            ;;
        -c|--cluster_num)
            CLUSTER_NUM="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

if [ -z "$EXP_NAME" ] || [ -z "$DATASET" ] || [ -z "$EXP_NUM" ] || [ -z "$BENCHMARK" ] || [ -z "$CLUSTER_NUM" ]; then
    echo "Error: Missing required parameters"
    show_help
    exit 1
fi

python src/frequent_pattern/main.py \
    --exp_name "$EXP_NAME" \
    --support_start "$SUPPORT_START" \
    --support_end "$SUPPORT_END" \
    --dataset "$DATASET" \
    --exp_num "$EXP_NUM" \
    --benchmark_model "$BENCHMARK" \
    --cluster_num "$CLUSTER_NUM" \
    --config "$CONFIG_PATH" 