# Models
import torch
import models.network as network
import models.loss as loss
import torch.nn.functional as F

# Pandas
import pandas as pd

# Data
from dataset.dataset import GreatApeDataset
from torch.utils.data import DataLoader

# Progress bar
from tqdm import tqdm

# Saving things
import pickle



def load_dataset_cfg():
 
    cfg = {
         'dataset': 
            {
                'sample_interval': 20,
                'sequence_length': 20,
                'activity_duration_threshold':72
            },
          'paths': 
            {
                'frames':'/home/dl18206/Desktop/phd/code/personal/ape-behaviour-triplet-network/data/frames',
                'annotations':'/home/dl18206/Desktop/phd/code/personal/ape-behaviour-triplet-network/data/annotations',

            },
            'augmentation':
            {
                "probability": 0,
                "spatial":
                {
                    "colour_jitter": False,
                    "horizontal_flip": False,
                    "rotation": False
                },
                "temporal": 
                {
                    "horizontal_flip": False
                }
            }
        }

    return cfg


def load_model_cfg():

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

def logits2label(logits, classes):
    index = logits.max(1).indices
    return classes[index]

def logits2conf(logits):
    probs = torch.nn.functional.softmax(logits, dim=1)
    conf, _ = torch.max(probs, 1)
    return conf.item()

def process_metadata(metadata, seq_length, label, pred, conf):

    # Need to return bounding box too...
    # metadata = {"ape_id": ape_id, "start_frame": start_frame, "video": video, "bboxes": bboxes}
    
    # To store dict
    pred_frames = []
    gt_frames = []

    start = int(metadata['start_frame'].item())
    end = start + seq_length # exclusive of final frame
    
    assert(len(metadata['video'])==1)
    video = metadata['video'][0]

    bboxes = metadata['bboxes']
    assert(len(bboxes)==seq_length)

    for i, index in enumerate(range(start, end)):

        # Instatiate 'new' dict
        pframe = {}
        gtframe = {}
        
        # Add gt info
        gtframe['file'] = f"{video}_frame_{index}.jpg"
        gtframe['detections'] = {}
        gtframe['detections']['label'] = label
        gtframe['detections']['bbox'] = [float(x) for x in bboxes[i]]

        # Add pred info
        pframe['file'] = f"{video}_frame_{index}.jpg"
        pframe['detections'] = {}
        pframe['detections']['label'] = pred
        pframe['detections']['conf'] = conf
        pframe['detections']['bbox'] = [float(x) for x in bboxes[i]]
        
        gt_frames.append(gtframe)
        pred_frames.append(pframe)
        
    assert(len(pred_frames)==len(gt_frames))
    assert(len(pred_frames)==seq_length)
    
    return gt_frames, pred_frames

def get_results(loader, model, classes, device):

    results = {'gts': [], 'preds': []}

    model.model.eval()

    with torch.no_grad():
        for spatial_data, temporal_data, labels, metadata in tqdm(loader):
            # Set gradients to zero
            model.optimiser.zero_grad()

            spatial_data = spatial_data.to(device)
            temporal_data = temporal_data.to(device)
            labels = labels.to(device)

            # Compute the forward pass of the model
            logits = model.model(spatial_data, temporal_data)
            
            # Get text label
            label = classes[labels.item()]

            # Process metadata
            pred = logits2label(logits, classes) 
            conf = logits2conf(logits)
            gt, pred = process_metadata(metadata, 20, label, pred, conf)
            
            # Add to results
            results['gts'].extend(gt)
            results['preds'].extend(pred)

    return results 

def log_results(loader, model, classes, device):

    results = {'preds': [], 'conf': [], 'labels': []}

    model.model.eval()

    with torch.no_grad():
        for spatial_data, temporal_data, labels, metadata in tqdm(loader):
            # Set gradients to zero
            model.optimiser.zero_grad()

            spatial_data = spatial_data.to(device)
            temporal_data = temporal_data.to(device)
            labels = labels.to(device)

            # Compute the forward pass of the model
            logits = model.model(spatial_data, temporal_data)
            
            # Process metadata
            pred = logits2label(logits, classes) 
            conf = logits2conf(logits)
            
            probs = torch.nn.functional.softmax(logits, dim=1)
            conf, index = torch.max(probs, 1)
            
            results['preds'].append(index.item())
            results['conf'].append(probs.tolist())
            results['labels'].append(labels.item())
    
    return results

def xyxy_to_normalised_xywh(xyxy, dims=(720, 404)):
    """Converts [xmin, ymin, xmax, ymax] to normalised [x, y, w, h] """
    dw = 1./dims[0]
    dh = 1./dims[1]
    x = xyxy[0]
    y = xyxy[1] 
    w = xyxy[2] - xyxy[0]
    h = xyxy[3] - xyxy[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [x,y,w,h]


def format_results(results):

    results = pd.DataFrame(results).groupby('file').agg(list).reset_index().to_dict(orient='records')
    
    for r in results:
        for d in r['detections']:
            d['bbox'] = xyxy_to_normalised_xywh(d['bbox'])
    
    results = {'images': results}
    
    return results

def format_result_dict(results):
    gt_results = format_results(results['gts'])
    pred_results = format_results(results['preds'])
    return gt_results, pred_results

def main():
     
    cfg = load_model_cfg()
    device = get_device()
    model = load_model(cfg, device)
    fitted_model = load_weights(model, 'weights/twostream_LSTM/model')

    video_names = '/home/dl18206/Desktop/phd/code/personal/ape-behaviour-triplet-network/splits/valdata.txt'
    classes_path = '/home/dl18206/Desktop/phd/code/personal/ape-behaviour-triplet-network/classes.txt'
    classes = open(classes_path).read().strip().split()
    
    # Load dataset
    dataset_cfg = load_dataset_cfg()
    dataset = GreatApeDataset(dataset_cfg, 'validation', video_names, classes, device)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, sampler=None)
    
    mode='log'

    if(mode=='log'):
        logged_results = log_results(loader, fitted_model, classes, device)
        
        with open('logged_results_20frames.pkl', 'wb') as handle:
                pickle.dump(logged_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        results = get_results(loader, fitted_model, classes, device)
        gts, preds = format_result_dict(results)

        with open('gt.pkl', 'wb') as handle:
            pickle.dump(gts, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('pred.pkl', 'wb') as handle:
            pickle.dump(preds, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
