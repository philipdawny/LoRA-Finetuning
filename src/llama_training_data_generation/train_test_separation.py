import pickle
import random
import json


def load_config(config_path):
    """Loads configuration parameters from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config



def load_data(lora_training_data):
    with open(lora_training_data, 'rb') as f1:
        data = pickle.load(f1)

    data = list(zip(data['question'], data['answer']))

    return data




def main():

    config = load_config("llamaconfig.json")

    lora_training_data = config["paths"]["output_path"]
    train_save_path = config["paths"]["train_save_path"]
    test_save_path = config["paths"]["test_save_path"]

    data = load_data(lora_training_data)

    train_set = random.sample(data, 1000)
    test_set = random.sample([i for i in data if i not in train_set], 10)

    with open(train_save_path, 'wb') as f1:
        pickle.dump(train_set, f1)

    with open(test_save_path, 'wb') as f1:
        pickle.dump(test_set, f1)

    print(r">>> Training and Test data saved to paths")


if __name__ == "__main__":

    main()