"""Two-stage EfficientNet-B0 training; test evaluation is a separate explicit command."""
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image, ImageOps
from data import PalaceDataset

class Letterbox:
    def __init__(self, size): self.size = size
    def __call__(self, image):
        return ImageOps.pad(image, (self.size, self.size), method=Image.Resampling.BICUBIC,
                            color=(124, 116, 104))

def transform(size, train=False):
    ops = [Letterbox(size)]
    if train:
        ops += [T.RandomAffine(degrees=4, translate=(0.025, 0.025), scale=(0.95, 1.05),
                               interpolation=T.InterpolationMode.BILINEAR, fill=(124,116,104)),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)]
    return T.Compose(ops + [T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed); random.seed(seed)

def metrics(truth, predicted, n):
    cm = np.zeros((n,n), dtype=int)
    for y,p in zip(truth,predicted): cm[y,p] += 1
    tp = np.diag(cm)
    precision = tp / np.maximum(cm.sum(0),1)
    recall = tp / np.maximum(cm.sum(1),1)
    f1 = 2*precision*recall / np.maximum(precision+recall,1e-12)
    return {'accuracy': float(tp.sum()/max(cm.sum(),1)), 'macro_f1': float(f1.mean()),
            'per_class_recall': recall.tolist(), 'support': cm.sum(1).tolist(),
            'confusion_matrix_true_rows': cm.tolist()}

def build_model(n, pretrained):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n)
    return model

def run_epoch(model, loader, device, criterion, optimizer=None, head_only=False):
    model.train(optimizer is not None)
    if head_only: model.features.eval()
    truth=[]; predicted=[]; total=0.; count=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        if optimizer is not None: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            logits=model(x); loss=criterion(logits,y)
            if optimizer is not None:
                loss.backward(); optimizer.step()
        total += loss.item()*len(y); count += len(y)
        truth.extend(y.detach().cpu().tolist()); predicted.extend(logits.argmax(1).detach().cpu().tolist())
    result=metrics(truth,predicted,len(loader.dataset.classes))
    result['loss']=total/count
    return result,truth,predicted

def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['train','evaluate','smoke'])
    p.add_argument('--manifest',default='manifest.json')
    p.add_argument('--out',default='runs/efficientnet_b0_baseline')
    p.add_argument('--checkpoint')
    p.add_argument('--device',default='cuda')
    p.add_argument('--size',type=int,default=224)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--head-epochs',type=int,default=5)
    p.add_argument('--finetune-epochs',type=int,default=20)
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--seed',type=int,default=20261011)
    p.add_argument('--class-weights',action='store_true')
    a=p.parse_args()
    if a.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; verify GPU environment. CPU smoke: --device cpu')
    if min(a.head_epochs,a.finetune_epochs,a.patience,a.batch_size,a.size)<=0:
        raise ValueError('Epochs, patience, batch size and size must be positive')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device(a.device)
    checkpoint=None
    if a.mode=='evaluate':
        if not a.checkpoint: p.error('evaluate requires --checkpoint')
        checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
        a.size=checkpoint['size']
    split='test' if a.mode=='evaluate' else 'train'
    ds=PalaceDataset(split,a.manifest,transform=transform(a.size,a.mode=='train'))
    if checkpoint and (checkpoint['manifest_sha256']!=ds.manifest_sha256 or checkpoint['classes']!=ds.class_mapping):
        raise ValueError('Checkpoint/data split or class mapping mismatch')
    model=build_model(len(ds.classes),a.mode=='train').to(device)
    if checkpoint: model.load_state_dict(checkpoint['model'])
    generator=torch.Generator().manual_seed(a.seed)
    loader=DataLoader(ds,batch_size=a.batch_size,shuffle=a.mode=='train',num_workers=a.workers,
                      worker_init_fn=seed_worker,generator=generator)
    criterion=nn.CrossEntropyLoss()
    if a.mode=='smoke':
        x,y=next(iter(loader)); model.train()
        logits=model(x.to(device)); loss=criterion(logits,y.to(device)); loss.backward()
        print(json.dumps({'status':'forward_backward_passed','batch':list(x.shape),'output':list(logits.shape),
                          'note':'random initialization; not pretrained accuracy or completed training'}))
        return
    out=Path(a.out)
    if a.mode=='evaluate':
        out.mkdir(parents=True,exist_ok=True)
        with torch.no_grad(): result,truth,predicted=run_epoch(model,loader,device,criterion)
        result.update(classes=ds.class_mapping,manifest_sha256=ds.manifest_sha256,
                      checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest())
        (out/'test_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        with (out/'test_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f); w.writerow(['image','true','predicted'])
            w.writerows((r['image'],ds.classes[y],ds.classes[z]) for r,y,z in zip(ds.samples,truth,predicted))
        print(json.dumps(result,ensure_ascii=False,indent=2)); return
    out.mkdir(parents=True,exist_ok=False)
    val=PalaceDataset('validation',a.manifest,transform=transform(a.size))
    vl=DataLoader(val,batch_size=a.batch_size,num_workers=a.workers)
    if a.class_weights:
        counts=np.bincount([ds.class_to_index[r['class']] for r in ds.samples],minlength=len(ds.classes))
        criterion=nn.CrossEntropyLoss(weight=torch.tensor(counts.sum()/(len(counts)*counts),dtype=torch.float32,device=device))
    config=vars(a)|{'torch':str(torch.__version__),'manifest_sha256':ds.manifest_sha256,
                    'classes':ds.class_mapping,'preprocessing':'aspect-preserving letterbox, ImageNet normalization'}
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    history=[]; best=-1.; best_loss=float('inf'); started=time.time()
    for stage,epochs in [('head',a.head_epochs),('finetune',a.finetune_epochs)]:
        if stage=='finetune':
            model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True)['model'])
        for param in model.features.parameters(): param.requires_grad=stage!='head'
        groups=[{'params':model.classifier.parameters(),'lr':1e-3 if stage=='head' else 1e-4}]
        if stage=='finetune': groups.append({'params':model.features.parameters(),'lr':1e-5})
        optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=2,factor=0.5)
        stale=0
        for epoch in range(1,epochs+1):
            train,_,_=run_epoch(model,loader,device,criterion,optimizer,stage=='head')
            with torch.no_grad(): valid,_,_=run_epoch(model,vl,device,nn.CrossEntropyLoss())
            row={'stage':stage,'epoch':epoch,'train':train,'validation':valid,'elapsed_seconds':time.time()-started}
            history.append(row)
            (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
            score=valid['macro_f1']
            if score>best+1e-8 or (abs(score-best)<=1e-8 and valid['loss']<best_loss):
                best=score; best_loss=valid['loss']; stale=0
                torch.save({'model':model.state_dict(),'classes':ds.class_mapping,'size':a.size,
                            'manifest_sha256':ds.manifest_sha256,'stage':stage,'epoch':epoch,
                            'validation':valid},out/'best.pt')
            else: stale+=1
            scheduler.step(score)
            print(f'{stage} {epoch}: train_loss={train["loss"]:.4f} val_acc={valid["accuracy"]:.4f} val_f1={score:.4f}',flush=True)
            if stale>=a.patience: break
    print('Training finished. Best validation Macro F1:',best,'Test has not been evaluated.')

if __name__=='__main__': main()
