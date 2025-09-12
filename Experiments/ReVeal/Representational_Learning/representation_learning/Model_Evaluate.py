import json
from tqdm import tqdm
import numpy as np
import torch
from representation_learning_api import RepresentationLearningModel
from models import MetricLearningModel

if __name__ == "__main__":
    path = "First_Run/input_dir/test_GGNNinput_graph.json"
    lambda1 = 0.5
    lambda2 = 0.001
    num_layers = 1
    hidden_dim=256
    dropout=0.2
    alpha=0.5
    
    json_data_file = open(path)
    data = json.load(json_data_file)
    json_data_file.close()
    features = []
    targets = []
    CUDA = True
    for d in data:
        features.append(d['graph_feature'])
        targets.append(d['target'])
    del data
    # X = np.array(features)
    # Y = np.array(targets)
    input_dim =len(features[0])
    base_model = model = MetricLearningModel(
            input_dim=input_dim, hidden_dim=hidden_dim, aplha=alpha, lambda1=lambda1,
            lambda2=lambda2, dropout_p=dropout, num_layers=num_layers
        )
    base_model.load_state_dict(torch.load(
                    "Vuld_SySe/representation_learning/TrainedModels/rep_model.bin", map_location='cuda:0' if CUDA else "cpu"))
    predicted_value = []
    with torch.no_grad():
        for item in tqdm(features):
            value, _, _ = base_model(torch.tensor(item))
            value  = np.argmax(
                value, axis=-1).tolist()
            print(value)
