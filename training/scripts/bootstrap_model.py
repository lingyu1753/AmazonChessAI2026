import sys
import os
sys.path.insert(0, r'g:\KataGomo\python')
from katago.train.modelconfigs import config_of_name
from katago.train.model_pytorch import Model
import torch

model_kind = sys.argv[1]
model_dir = sys.argv[2]

config = config_of_name[model_kind]
model = Model(config, 10)
os.makedirs(model_dir, exist_ok=True)
ckpt_path = os.path.join(model_dir, 'model.ckpt')
torch.save({'model': model.state_dict(), 'config': model.config}, ckpt_path)
print(f'Initial model saved to: {ckpt_path}')
