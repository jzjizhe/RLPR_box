import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="../../datasets/math_gold")
    parser.add_argument("--save_path", default="/data0/jzzhang/datasets/GRM/rlpr_train_debug.parquet")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "general-reasoner"
    train_dataset = pd.read_parquet("/data0/jzzhang/datasets/RLPR/WebInstruct/rlpr_train_debug.parquet")
    instruction = "A conversation between User and Assistant. The user asks a question, and the Assistant should solve it step by step and present the final answer in the following format: '\\boxed{X}', where X is the final answer."
    # instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."
    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(row):
            idx = row.name
            example = row.to_dict()
            question_raw = example["prompt"][1]['content']
            # question = question_raw + ' ' + instruction_following
            solution = example["reward_model"]["ground_truth"] 
            ability = example["ability"]
            data = {
                "data_source": data_source,
                "prompt": [
                            {"role":"system","content":instruction},
                            {"role": "user", "content": question_raw}
                        ],
                "ability": ability,
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, 
                "index": idx,
                "answer":solution,
                "question":question_raw
                },
            }
            return data

        return process_fn

    processed_data = train_dataset.apply(make_map_fn("train"), axis=1).tolist()
    train_dataset = pd.DataFrame(processed_data)
    train_dataset.to_parquet(args.save_path)
