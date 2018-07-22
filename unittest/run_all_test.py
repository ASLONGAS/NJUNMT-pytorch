import os
import subprocess
import argparse
from src.utils.logging import INFO


parser = argparse.ArgumentParser()
parser.add_argument("--use_gpu", action="store_true",
                    help="Test on GPU. Default is disable.")

def get_model_name(path):
    return os.path.basename(path).strip().split(".")[0]


def clean_tmp_dir(path):
    subprocess.run("rm -rf {0}/*".format(path), shell=True)


def rm_tmp_dir(path):
    subprocess.run("rm -rf {0}".format(path), shell=True)


def test_data_io(test_dir):
    import yaml
    from src.data import TextLineDataset, DataIterator
    from src.data.vocabulary import Vocabulary

    config_path = "./unittest/configs/test_transformer.yaml"

    with open(config_path) as f:
        configs = yaml.load(f)

    data_configs = configs['data_configs']

    vocab_src = Vocabulary(**data_configs["vocabularies"][0])

    with open(data_configs["train_data"][0]) as fin:
        test_data_real = fin.readlines()
    test_data_real = [line.strip().split() for line in test_data_real]

    test_dataset = TextLineDataset(data_path=data_configs["train_data"][0],
                                   vocabulary=vocab_src)

    test_iterator = DataIterator(dataset=test_dataset,
                                 batch_size=3,
                                 use_bucket=False)

    test_iter = test_iterator.build_generator()

    test_data = []

    for batch in test_iter:
        test_data += batch

    if any(map(lambda pred, truth: len(pred) != len(truth), test_data, test_data_real)):
        print("The order of data io is broken!")
        exit(1)


def test_transformer_train(test_dir, use_gpu=False):
    from src.bin import train

    config_path = "./unittest/configs/test_transformer.yaml"
    model_name = get_model_name(config_path)

    saveto = os.path.join(test_dir, "save")
    log_path = os.path.join(test_dir, "log")
    valid_path = os.path.join(test_dir, "valid")

    train.run(model_name=model_name,
              config_path=config_path,
              saveto=saveto,
              log_path=log_path,
              valid_path=valid_path,
              debug=True,
              reload=True,
              use_gpu=use_gpu)


def test_transformer_inference(test_dir, use_gpu=False):
    from src.bin import translate
    from src.utils.common_utils import GlobalNames
    config_path = "./unittest/configs/test_transformer.yaml"

    saveto = os.path.join(test_dir, "save")
    model_name = get_model_name(config_path)
    model_path = os.path.join(saveto, model_name + GlobalNames.MY_BEST_MODEL_SUFFIX)
    source_path = "./unittest/data/dev/zh.0"
    batch_size = 3
    beam_size = 3

    translate.run(model_name=model_name,
                  source_path=source_path,
                  batch_size=batch_size,
                  beam_size=beam_size,
                  model_path=model_path,
                  use_gpu=use_gpu,
                  config_path=config_path,
                  saveto=saveto,
                  max_steps=20)


def test_dl4mt_train(test_dir, use_gpu=False):
    from src.bin import train

    config_path = "./unittest/configs/test_dl4mt.yaml"
    model_name = get_model_name(config_path)

    saveto = os.path.join(test_dir, "save")
    log_path = os.path.join(test_dir, "log")
    valid_path = os.path.join(test_dir, "valid")

    train.run(model_name=model_name,
              config_path=config_path,
              saveto=saveto,
              log_path=log_path,
              valid_path=valid_path,
              debug=True,
              use_gpu=use_gpu)


def test_dl4mt_inference(test_dir, use_gpu=False):
    from src.bin import translate
    from src.utils.common_utils import GlobalNames

    config_path = "./unittest/configs/test_dl4mt.yaml"

    saveto = os.path.join(test_dir, "save")
    model_name = get_model_name(config_path)
    model_path = os.path.join(saveto, model_name + GlobalNames.MY_BEST_MODEL_SUFFIX)
    source_path = "./unittest/data/dev/zh.0"
    batch_size = 3
    beam_size = 3

    translate.run(model_name=model_name,
                  source_path=source_path,
                  batch_size=batch_size,
                  beam_size=beam_size,
                  model_path=model_path,
                  use_gpu=use_gpu,
                  config_path=config_path,
                  saveto=saveto,
                  max_steps=20)


def test_all(use_gpu=False):
    test_dir = "./tmp"

    if not os.path.exists(test_dir):
        os.makedirs(test_dir, exist_ok=True)

    INFO("=" * 20)
    INFO("Test data io...")
    test_data_io(test_dir)
    INFO("Done.")
    INFO("=" * 20)

    clean_tmp_dir(test_dir)

    INFO("=" * 20)
    INFO("Test transformer training...")
    test_transformer_train(test_dir, use_gpu=use_gpu)
    INFO("Done.")
    INFO("=" * 20)

    INFO("=" * 20)
    INFO("Test transformer inference...")
    test_transformer_inference(test_dir, use_gpu=use_gpu)
    INFO("Done.")
    INFO("=" * 20)

    INFO("=" * 20)
    INFO("Test transformer reload...")
    test_transformer_train(test_dir, use_gpu=use_gpu)
    INFO("Done.")
    INFO("=" * 20)

    clean_tmp_dir(test_dir)

    INFO("=" * 20)
    INFO("Test DL4MT training...")
    test_dl4mt_train(test_dir, use_gpu=use_gpu)
    INFO("Done.")
    INFO("=" * 20)

    INFO("=" * 20)
    INFO("Test DL4MT inference...")
    test_dl4mt_inference(test_dir, use_gpu=use_gpu)
    INFO("Done.")
    INFO("=" * 20)

    rm_tmp_dir(test_dir)


if __name__ == "__main__":
    args = parser.parse_args()
    test_all(use_gpu=args.use_gpu)
