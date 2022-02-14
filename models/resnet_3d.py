import random
import torch
import argparse
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import pytorchvideo.models.resnet
from pytorchvideo.models.head import create_res_basic_head
from pytorchvideo.transforms import MixUp, AugMix, CutMix
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers
from pytorch_lightning.plugins import DDPPlugin
from kornia.losses import FocalLoss

class VideoClassificationLightningModule(pl.LightningModule):
  
    def __init__(self, model_name):
      super().__init__()
      
      self.model_name = model_name
    
      # Load pretrained model
      pretrained_model = torch.hub.load("facebookresearch/pytorchvideo:main", model=self.model_name, pretrained=True)
      
      # Strip the head from backbone  
      self.backbone = nn.Sequential(*list(pretrained_model.children())[0][:-1])

      # Attach a new head with specified class number (hard coded for now...)
      self.res_head = create_res_basic_head(
              in_features=2048, out_features=500
      )

      self.fc = nn.Linear(in_features=500, out_features=9)
      
      # Dropout hardcoded 0 for now
      self.dropout = nn.Dropout(p=0)

    def forward(self, x):
        output = self.dropout(self.res_head(self.backbone(x)))
        return self.fc(output)

    def training_step(self, batch, batch_idx):
        pass
    def training_epoch_end(self, outputs):
        pass

    def get_result(self, batch_idx, meta, pred, label):

        result = {batch_idx: {'video': meta['video'],
                              'start_frame': meta['start_frame'],
                              'ape_id': meta['ape_id'],
                              'activity': meta['activity'],
                              'pred': pred.topk(1).indices,
                              'label': label}}
        return result

    def validation_step(self, batch, batch_idx):
  
      data, label, meta = batch
      pred = self(data)
      loss = F.cross_entropy(pred, label)
     
      if(self.save_results):
        result = self.get_result(batch_idx, meta, pred, label)
     
      return {"result": result}

    def validation_epoch_end(self, outputs):

        # Process results
        if(self.save_results):
            results = [x["result"] for x in outputs]
            
            # Save results
            with open(self.results_name, 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def configure_optimizers(self):
        pass
    
    def get_lr(self):
        pass

