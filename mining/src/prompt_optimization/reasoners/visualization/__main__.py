def main():
    import argparse

    from reasoners.visualization import VisualizerClient

    parser = argparse.ArgumentParser()
    # parser.add_argument('tree_log', type=str)
    parser.add_argument('--base_url', type=str)
    args = parser.parse_args()

    if args.base_url is None:
        client = VisualizerClient()
    else:
        client = VisualizerClient(args.base_url)
    tree_log = r"/home/jt_ws/llm-reasoners/logs/exp2/bfs_v1_step2_r_gpt4/algo_output/1.pkl"
    # with open(args.tree_log) as f:
    #     data = f.read()
    with open(tree_log,'r',encoding="utf-8") as f:
        data = f.read()
    result = client.post_log(data)
    print(result.access_url)

if __name__ == '__main__':
    main()
