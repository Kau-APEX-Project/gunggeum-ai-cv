"""Additional logging for future runs; EXP001 snapshots remain unchanged."""
import hashlib,json,platform,shutil,subprocess,sys
from pathlib import Path
import torch
import torchvision

def initialize(out,model,config):
    out=Path(out)
    source=Path(__file__).parent
    snapshots=out/'source';snapshots.mkdir(exist_ok=True)
    hashes={}
    for name in ['finetune.py','data.py','experiment_logging.py']:
        path=source/name;shutil.copy2(path,snapshots/name)
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    runtime={'python':platform.python_version(),'platform':platform.platform(),'torch':str(torch.__version__),
             'torchvision':str(torchvision.__version__),'cuda_runtime':torch.version.cuda,
             'cudnn':torch.backends.cudnn.version(),
             'device':torch.cuda.get_device_name(0) if config['device']=='cuda' else 'cpu',
             'source_sha256':hashes,'total_parameters':sum(p.numel() for p in model.parameters()),
             'head_parameters':sum(p.numel() for p in model.classifier.parameters()),
             'optimizer':'AdamW','weight_decay':1e-4,'betas':[0.9,0.999],'eps':1e-8,
             'initial_lr':{'head_classifier':1e-3,'finetune_classifier':1e-4,'finetune_backbone':1e-5},
             'scheduler':{'name':'ReduceLROnPlateau','mode':'max','factor':0.5,'patience':2},
             'pretrained_weights':'EfficientNet_B0_Weights.IMAGENET1K_V1','loss':'CrossEntropyLoss',
             'train_augmentation':{'rotation_degrees':4,'translate':[0.025,0.025],'scale':[0.95,1.05],
                                   'brightness':0.5,'contrast':0.15,'saturation':0.1,'flip':False},
             'normalization':{'mean':[0.485,0.456,0.406],'std':[0.229,0.224,0.225]}}
    (out/'runtime_and_hyperparameters.json').write_text(json.dumps(runtime,ensure_ascii=False,indent=2),encoding='utf-8')
    proc=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True)
    (out/'pip_freeze.txt').write_text(proc.stdout if proc.returncode==0 else proc.stderr,encoding='utf-8')

def save_epoch(out,model,optimizer,scheduler,row,classes,size,manifest_sha256):
    out=Path(out);folder=out/'checkpoints';folder.mkdir(exist_ok=True)
    state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
           'stage':row['stage'],'epoch':row['epoch'],'validation':row['validation'],
           'classes':classes,'size':size,'manifest_sha256':manifest_sha256,
           'lr_next_epoch':[g['lr'] for g in optimizer.param_groups]}
    path=folder/f'{row["stage"]}_{row["epoch"]:03d}.pt'
    torch.save(state,path)
    entry={'file':path.relative_to(out).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
           'stage':row['stage'],'epoch':row['epoch'],'validation':row['validation']}
    with (out/'checkpoint_index.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
