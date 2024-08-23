import os, json, gzip, pickle, tqdm, time, requests, argparse

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

def dump_dataset_file(filename, data):
    with gzip.open(filename, "wb") as f:
        pickle.dump(data, f)

def main(ratio: int):
    splits = ["test", "valid", "train"]
    for split in splits:
        input_data_path = os.path.join(input_data_root, split)
        write_data_path = os.path.join(write_data_root, f'{split}.json')
        output_data_path = os.path.join(output_data_root, split)

        # read input_data(sign and ref)
        input_data = load_dataset_file(input_data_path)

        # read write_data(sign prefix and hypo prefix), construct
        output_data = []
        with open(write_data_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                json_obj = json.loads(line)
                instance_id = json_obj["instance_id"]
                nfeatures = json_obj["nfeatures"]
                translation = json_obj["translation"]
                translation_ori = json_obj["translation_ori"]
                label_rw = json_obj["label_rw"]
                # only write pairs with label_rw == 1
                if label_rw == 0: continue
                if len(translation) == 0: continue
                one_input_data = input_data[instance_id]
                prefix_prefix = {
                    "name": one_input_data["name"],
                    "signer": one_input_data["signer"],
                    "gloss": one_input_data["gloss"],
                    "text": translation,
                    "sign": one_input_data["sign"][:nfeatures],
                }
                output_data.append(prefix_prefix)

        print("-"*10)
        print(f'num of prefix pairs: {len(output_data)}')
        # add full pairs
        for i in range(ratio):
            output_data += input_data 
        # dump
        print(f'num of full pairs: {len(input_data) * ratio}')
        print(f'num of output data: {len(output_data)}')
        print(f'dumping: {output_data_path}')
        dump_dataset_file(output_data_path, output_data)
        print("-"*10)
    

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=int, required=True, help="How many times to repeat full pairs")
    args = parser.parse_args()

    ratio = args.ratio
    
    dataset="CSL-Daily"
    data="csl"
    feat="smkd2d"
    # input_data_root: PATH_TO_DATABIN
    input_data_root=f'/home2/tsun/data/sign/vacs2t/{dataset}/{feat}'
    # write_data_root: PATH_TO_CTL_DATA
    write_data_root=f'/home2/tsun/code/vac_s2t/data/{data}/{feat}/data_mu/len_regression/avg_3000'
    # output_data_root: PATH_TO_NEW_DATABIN
    output_data_root=f'/home2/tsun/data/sign/vacs2t/{dataset}/{feat}_stage2/avg_3000/translation_rw1/prefix1_full{ratio}'
    # ratio is 10 for PHOENIX2014T and 20 for CSL-Daily
    main(ratio)
