import os
import json

def clean(input_path, splits):
    for split in splits:
        input_file = os.path.join(input_path, f'{split}.json')
        output_file = os.path.join(input_path, f'{split}_cleaned.json')
        print("processing: ", input_file)

        # delete output_file if it exists
        if os.path.exists(output_file):
            os.remove(output_file)
        if not os.path.exists(input_file):
            continue

        with open(input_file, "r") as f:
            lines = f.readlines()
            prev_json_obj = {"instance_id": -1, "nfeatures": -1}
            for line in lines:
                json_obj = json.loads(line)
                # clean duplicate data
                if (json_obj["instance_id"] == prev_json_obj["instance_id"] 
                    and json_obj["nfeatures"] == prev_json_obj["nfeatures"]):
                    continue
                else:
                    with open(output_file, "a+") as f1:
                        f1.write(json.dumps(json_obj) + "\n")
                    prev_json_obj = json_obj
        print("done:", output_file)
        
        # delete original file
        os.remove(input_file)
        # rename new file
        os.rename(output_file, input_file)

def merge(input_path, splits):
    for split in splits:
        file = os.path.join(input_path, f'{split}.json')
        print("precossing: ", file)
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                json_obj = json.loads(line)
                # train_x -> x
                split_int = int(split[6:])
                json_obj["instance_id"] = json_obj["instance_id"] + split_int * 1000
                with open(os.path.join(input_path, f'train.json'), "a") as f:
                    f.write(json.dumps(json_obj) + "\n")
    print("merge done")
    return os.path.join(input_path, f'train.json')

def main():
    # input_path is a directory containing multiple json files (valid.json, test.json, train_0.json, train_1.json, ...)
    # each train_x.json contains 1000 samples here
    input_path = "/home2/tsun/code/vac_s2t/data/p14t/pami/data_mu/rw_cls/best_4500/remove6_re"
    splits = ["test", "valid"] + [f'train_{i}' for i in range(19)]

    # clean
    clean(input_path, splits)

    # merge train
    if len(splits) <= 2: return
    splits = splits[2:]
    if os.path.exists(os.path.join(input_path, f'train.json')):
        print("remove old train.json")
        os.remove(os.path.join(input_path, f'train.json'))
    merged_path = merge(input_path, splits)
    print("merged_path: ", merged_path)
    

if __name__ == '__main__':
    main()