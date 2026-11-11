import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


class FERDataset(Dataset):
    def __init__(self, directory, split_name, quadruplets=False):
        self.directory = Path(directory)
        self.split = json.loads((self.directory/'split.json').read_text(encoding='utf-8'))
        self.rows = [json.loads(l) for l in (self.directory/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
        self.indices = self.split[split_name]
        self.images = np.load(self.directory/'images.npy', mmap_mode='r')
        self.labels = np.array([self.rows[i]['label'] for i in self.indices])
        self.quadruplets = quadruplets
        self.by_class = {int(c):np.where(self.labels == c)[0].tolist() for c in np.unique(self.labels)}
        if quadruplets and split_name != 'train':
            raise ValueError('Quadruplet partners must come exclusively from the training split.')

    def __len__(self):
        return len(self.indices)

    def get_image(self, position):
        return torch.from_numpy(self.images[self.indices[position]].copy())

    def __getitem__(self, position):
        label = int(self.labels[position])
        if not self.quadruplets:
            return self.get_image(position), label, self.indices[position]
        positive = random.choice(self.by_class[label])
        while positive == position:
            positive = random.choice(self.by_class[label])
        negative_label = random.choice([c for c in self.by_class if c != label])
        negative, negative2 = random.sample(self.by_class[negative_label], 2)
        return (torch.stack([self.get_image(i) for i in (position, positive, negative, negative2)]),
                label, negative_label)


class ImagePipeline:
    """Per-image GPU augmentation; defaults preserve the first comparison."""
    def __init__(self, device, size=224, augmentation=None):
        self.size = size
        self.augmentation = augmentation or {}
        self.mean = torch.tensor([.485,.456,.406], device=device).view(1,3,1,1)
        self.std = torch.tensor([.229,.224,.225], device=device).view(1,3,1,1)

    @torch.no_grad()
    def __call__(self, raw, training=False):
        x = raw.unsqueeze(1).float().div_(255.)
        batch = x.shape[0]
        if training:
            degrees = self.augmentation.get('degrees',10.)
            scale_range = self.augmentation.get('scale_range',.05)
            translation = self.augmentation.get('translation',.03)
            jitter = self.augmentation.get('jitter',.08)
            angle = (torch.rand(batch, device=x.device)*2*degrees-degrees) * math.pi / 180.
            scale = torch.rand(batch, device=x.device)*2*scale_range + 1-scale_range
            flip = torch.where(torch.rand(batch, device=x.device)<.5, -1., 1.)
            theta = torch.zeros(batch,2,3,device=x.device)
            theta[:,0,0] = torch.cos(angle)*scale*flip
            theta[:,0,1] = -torch.sin(angle)*scale
            theta[:,1,0] = torch.sin(angle)*scale*flip
            theta[:,1,1] = torch.cos(angle)*scale
            theta[:,:,2] = (torch.rand(batch,2,device=x.device)-.5)*2*translation
            grid = F.affine_grid(theta, [batch,1,self.size,self.size], align_corners=False)
            x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
            brightness = torch.rand(batch,1,1,1,device=x.device)*2*jitter + 1-jitter
            contrast = torch.rand(batch,1,1,1,device=x.device)*2*jitter + 1-jitter
            mean = x.mean((2,3), keepdim=True)
            x = ((x-mean)*contrast+mean).mul_(brightness).clamp_(0,1)
            erase_prob = self.augmentation.get('erase_probability',0.)
            if erase_prob:
                apply = torch.rand(batch,1,1,1,device=x.device)<erase_prob
                area = torch.rand(batch,1,1,1,device=x.device)*.06+.02
                ratio = torch.rand(batch,1,1,1,device=x.device)*1.5+.5
                height = (area*ratio).sqrt()*self.size
                width = (area/ratio).sqrt()*self.size
                top = torch.rand(batch,1,1,1,device=x.device)*(self.size-height)
                left = torch.rand(batch,1,1,1,device=x.device)*(self.size-width)
                yy = torch.arange(self.size,device=x.device).view(1,1,-1,1)
                xx = torch.arange(self.size,device=x.device).view(1,1,1,-1)
                mask = apply & (yy>=top) & (yy<top+height) & (xx>=left) & (xx<left+width)
                x = torch.where(mask,x.mean((2,3),keepdim=True),x)
        else:
            x = F.interpolate(x, size=(self.size,self.size), mode='bilinear', align_corners=False)
        return ((x.expand(-1,3,-1,-1)-self.mean)/self.std).contiguous(memory_format=torch.channels_last)
