# Models
import torch
import models.network as network
import models.loss as loss
import torch.nn.functional as F

def load_cfg():
    
    cfg = {
        "name": "twostream_LSTM",
        "mode": "train",
        "resume": False,
        "best" : False,
        "log": True,
        "bucket": "",
        "cnn": "resnet18",
        "hyperparameters": {
            "epochs": 50,
            "learning_rate": 0.0001,
            "sgd_momentum": 0.9,
            "regularisation": 0.01,
            "dropout": 0
        },
        "loss": {
            "function" : "focal",
            "cross_entropy": {
                "weighted" : False
            },
            "focal": {
                "alpha": 1,
                "gamma": 1
            }
        },
        "lstm" : {
            "layers": 1,
            "hidden_units": 512,
            "dropout": 0
        }
    }

    return cfg

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device

def load_model(cfg, device):
    return network.CNN(cfg=cfg, loss='focal', num_classes=9, device=device)

def load_weights(model, weights_path):
    model_path = weights_path
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.model.load_state_dict(checkpoint['state_dict'])
    model.optimiser.load_state_dict(checkpoint['optimiser'])
    
    epoch = checkpoint['epoch']
    acc = checkpoint['accuracy']

    return model




def main():
     
    cfg = load_cfg()
    device = get_device()
    model = load_model(cfg, device)
    fitted_model = load_weights(model, 'weights/twostream_LSTM/model')

if __name__ == '__main__':
    main()
