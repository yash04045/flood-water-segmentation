import os, cv2, numpy as np, torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE=384  # change to 512 if GPU allows

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.Affine(translate_percent=(-0.1,0.1),scale=(0.8,1.2),rotate=(-20,20),p=0.5),
        A.RandomBrightnessContrast(0.2,0.2,p=0.5),
        A.HueSaturationValue(20,30,20,p=0.3),
        # Fixed GaussNoise parameter
        A.GaussNoise(variance_limit=(10.0,50.0),p=0.2),
        A.Blur(blur_limit=3,p=0.2),
        A.RandomCrop(height=IMG_SIZE,width=IMG_SIZE,p=0.5),
        # Fixed CoarseDropout parameters
        A.CoarseDropout(num_holes=8, max_h_size=IMG_SIZE//8, max_w_size=IMG_SIZE//8, p=0.4), 
        A.Resize(IMG_SIZE,IMG_SIZE),
        A.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

def get_val_transform():
    return A.Compose([
        A.Resize(IMG_SIZE,IMG_SIZE),
        A.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

class FloodDataset(Dataset):
    def __init__(self,img_dir,mask_dir,transforms=None):
        self.img_dir=img_dir; self.mask_dir=mask_dir; self.transforms=transforms
        self.images=sorted([f for f in os.listdir(img_dir) if not f.startswith('.')])
        self.masks=sorted([f for f in os.listdir(mask_dir) if not f.startswith('.')])
        assert len(self.images)==len(self.masks), f"images/masks mismatch {len(self.images)} vs {len(self.masks)}"
    def __len__(self): return len(self.images)
    def __getitem__(self,idx):
        img=cv2.cvtColor(cv2.imread(os.path.join(self.img_dir,self.images[idx])),cv2.COLOR_BGR2RGB)
        mask=cv2.imread(os.path.join(self.mask_dir,self.masks[idx]),cv2.IMREAD_GRAYSCALE).astype(np.int64)
        if self.transforms:
            aug=self.transforms(image=img,mask=mask); img,mask=aug["image"],aug["mask"]
        return img, torch.from_numpy(mask).long() if isinstance(mask,np.ndarray) else mask.long()

def get_sampler(dataset):
    counts={}
    for i in range(len(dataset)):
        _,mask=dataset[i]; labs,cnt=torch.unique(mask,return_counts=True)
        for l,c in zip(labs.tolist(),cnt.tolist()): counts[l]=counts.get(l,0)+c
    total=sum(counts.values()); cw={k: total/v for k,v in counts.items()}
    weights=[]
    for i in range(len(dataset)):
        _,mask=dataset[i]; labs=torch.unique(mask)
        w=float(np.mean([cw[int(l)] for l in labs.tolist()])); weights.append(w)
    from torch.utils.data import WeightedRandomSampler
    return WeightedRandomSampler(torch.DoubleTensor(weights),num_samples=len(weights),replacement=True)