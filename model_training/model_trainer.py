from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch, torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torcheval.metrics import MulticlassAccuracy  # built-in metric
from PIL import Image
import torchvision.transforms as T
import os, torch
from pathlib import Path

#Choose GPU iv avaliable
device = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 224         # matches EfficientNet-B0 default
BATCH    = 64
ROOT_PATH = "./data"   
EPOCHS = 15

class EarlyStopping:
    """
    Stop training when the monitored metric has stopped improving.

    Args
    ----
    patience   : int   – epochs with no (meaningful) improvement before stopping
    min_delta  : float – minimum change regarded as an improvement
    """
    def __init__(self, patience: int = 5, min_delta: float = 1e-3):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = -float("inf")
        self.counter    = 0
        self.should_stop = False

    def __call__(self, current_score: float) -> bool:
        # True  -> keep training
        # False -> trigger early stop
        if current_score - self.best_score > self.min_delta:
            self.best_score = current_score
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True          # set flag once
        return not self.should_stop

def save_ckpt(name: str, epoch: int, val_acc: float) -> None:
    """
    Save a full training checkpoint.

    Parameters
    ----------
    name : str
        File name to create inside CKPT_DIR.
    epoch : int
        Epoch index completed *before* the save - stored so training can resume exactly where it left off.
    val_acc : float
        Current best validation accuracy.  

    Side effects
    ------------
    Writes a file ``CKPT_DIR / name`` containing a dictionary with:
        - ``"epoch"``        : int
        - ``"val_acc"``      : float
        - ``"model_state"``  : model.state_dict()
        - ``"optim_state"``  : optimizer.state_dict()
        - ``"sched_state"``  : scheduler.state_dict()
        - ``"classes"``      : list of class names (train_set.classes)

    Notes
    """
    torch.save({
        "epoch": epoch,
        "val_acc": val_acc,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "sched_state": scheduler.state_dict(),
        "classes": ds_weak.classes
    }, os.path.join(CKPT_DIR, name))


'''
Use Compose to apply transofmation to picture set. 

For each photo from training data set we apply weak and strong transformation.
Then results from both are treated as new data point to double number of examples.
'''
weak_tfms  = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])
strong_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.4,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomApply(
        [transforms.ColorJitter(0.4,0.4,0.4,0.2)], p=0.8),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

'''
For test dataset we only use transofmration that will normalise images.
There are no transformation that could altered images.
This is to ensure model is validated on 'normal' food photos. 
'''
test_tfms = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

root = Path(ROOT_PATH)  
'''
Create datasets for both training and test. 
A torch.utils.data.Dataset is just a Python-list-like wrapper around your files.
'''
ds_weak  = datasets.Food101(root, split="train", transform=weak_tfms, download=True)
ds_strong= datasets.Food101(root, split="train", transform=strong_tfms, download=True)
train_set = ConcatDataset([ds_weak, ds_strong])  
 
test_set  = datasets.Food101(root, split="test",  transform=test_tfms, download=True)

'''
Create two dataloasers that will be used during model training.
torch.utils.data.DataLoader wraps a Dataset to create an iterator that:
-- assembles samples into batches (batch_size)
-- shuffles indices (shuffle / sampler)
-- fetches data in parallel (num_workers, pre-fetching, pin-memory)
-- lets you decide how samples become a batch (collate_fn)
'''
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,  num_workers=8, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=BATCH, shuffle=False, num_workers=8, pin_memory=True)


weights = EfficientNet_B0_Weights.IMAGENET1K_V1 #Load EfficientNet Weights
model   = efficientnet_b0(weights=weights) #Load model with pre-trainer weights for Convolutional layers
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 101) # Reaplce last layer (classification) with one suitable for our datased
model.to(device) 

criterion  = nn.CrossEntropyLoss() # computes the cross entropy loss between input logits and target. It is useful when training a classification problem with C classes. 
optimizer  = AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
scheduler  = CosineAnnealingLR(optimizer, T_max=EPOCHS) # Starts training with aggresive LR, then smoothly decays it to tiny steps
metric_val = MulticlassAccuracy(num_classes=101).to(device) #Compute accuracy score, which is the frequency of input matching target. Its functional version is 


CKPT_DIR = "checkpoints"; os.makedirs(CKPT_DIR, exist_ok=True) # Ensure folder for checkpoints exists. 
best_acc = 0.0                                                 #  Reset accuracy to 0 


early_stop = EarlyStopping(patience=4, min_delta=1e-3)
# ----------------------- training loop ---------------------------------------
for epoch in range(1, EPOCHS + 1):
    model.train(); train_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward(); optimizer.step()
        train_loss += loss.item() * y.size(0)
    scheduler.step()

    # validation --------------------------------------------------------------
    model.eval(); metric_val.reset()
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x.to(device))
            metric_val.update(preds, y.to(device))
    val_acc = metric_val.compute().item()
    print(f"epoch {epoch:02d} | "
          f"train {train_loss/len(train_set):.4f} | "
          f"val {val_acc:.4f}")

    # save last + best --------------------------------------------------------
    save_ckpt("last.pt", epoch, val_acc)
    if val_acc > best_acc:
        best_acc = val_acc
        save_ckpt("best.pt", epoch, val_acc)

    if not early_stop(val_acc):
        print(f"\nEarly stopping triggered (no improvement in "
              f"{early_stop.patience} epochs).")
        break  

# ----------------------- TorchScript bundle -----------------------------------
scripted = torch.jit.script(model.cpu())   # CPU first → portable  :contentReference[oaicite:0]{index=0}
scripted.save(os.path.join(CKPT_DIR, "model_scripted.pt"))

print(f"\nFinished. Checkpoints in: {os.path.abspath(CKPT_DIR)}")

