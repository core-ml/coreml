#!/usr/bin/env python
# coding: utf-8

# In[18]:


from os.path import join, splitext, basename
from os import listdir
import torch
from torch.optim.swa_utils import AveragedModel, update_bn
import numpy as np
import librosa
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import mode
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
from natsort import natsorted
from sklearn.metrics import confusion_matrix, roc_auc_score
from coreml.utils.io import read_yml
from coreml.config import Config
from coreml.data.dataloader import get_dataloader
from coreml.models import factory as model_factory
from coreml.utils.logger import set_logger


# In[2]:


config_name = 'competitions/2020/melanoma-classification/configs/effb5/best-1cycle-wd4e-1-384'


# In[3]:


whole_train_2020 = pd.read_csv('/data/siim-isic-melanoma/raw/2020/train.csv')
whole_train_2019 = pd.read_csv('/data/siim-isic-melanoma/raw/2019/train.csv')
test_2020 = pd.read_csv('/data/siim-isic-melanoma/raw/2020/test.csv')


# In[4]:


train = whole_train_2019[['image_name', 'target']].append(whole_train_2020[['image_name', 'target']])


# In[5]:


train.head()


# In[6]:


data_config_path = f'/data/siim-isic-melanoma/processed/versions/v3.0.0.yml'
print(f'Reading data config: {data_config_path}')
data_config = read_yml(data_config_path)


# In[7]:


prediction_val = pd.read_csv(join('/output', config_name, 'logs/evaluation/val.csv'))


# In[8]:


val = pd.DataFrame(data_config['val'])
val['image_name'] = val['file'].apply(lambda x: splitext(basename(x))[0])
val['label'] = val['label'].apply(lambda x: x['classification'])
val = val.drop(columns=['file'])

print('Shapes:')
print(len(prediction_val), len(val))
print()
    
val = pd.merge(prediction_val, val)
    
print('Performance without using SWA')
val_preds = val['target'].values
val_labels = val['label'].values
roc = roc_auc_score(val_labels, val_preds)
print(roc)


# In[9]:


config = Config(join('/workspace/coreml', config_name + '.yml'))


# In[19]:


set_logger(join(config.log_dir, 'debug.log'))


# In[10]:


val_dataloader, _ = get_dataloader(
        config.data, 'val',
        config.model['batch_size'],
        num_workers=10,
        shuffle=False,
        drop_last=False)


# In[16]:


# set epoch
config.model['load']['version'] = config_name
config.model['load']['load_best'] = True


# In[39]:


config.checkpoint_dir = '/output/' + config_name + '/checkpoints'


# In[40]:


model = model_factory.create(config.model['name'], **{'config': config})


# # SWA

# In[21]:


swa_model = AveragedModel(model.network)


# In[41]:


# all checkpoints available
available_ckpts = natsorted(glob(join(config.checkpoint_dir, '*')))[::-1]


# In[42]:


available_ckpts


# In[45]:


swa_epochs = np.arange(5, 15)

for epoch in tqdm(swa_epochs):
    config.model['load']['epoch'] = epoch
    model = model_factory.create(config.model['name'], **{'config': config})
    swa_model.update_parameters(model.network)


# In[ ]:





# In[46]:


# load the train data loader for doing a forward pass on the model
train_dataloader, _ = get_dataloader(
        config.data, 'train',
        config.model['batch_size'],
        num_workers=10,
        shuffle=False,
        drop_last=False)


# In[47]:


# update batch norm params
for batch in tqdm(train_dataloader):
    swa_model(batch['signals'].cuda())


# In[49]:


# set the SWA model as the network
model.network = swa_model


# In[ ]:


# compute the new results
results = model.process_epoch(val_dataloader, mode='val', use_wandb=False)


# In[ ]:


# results['auc-roc']


# In[ ]:




