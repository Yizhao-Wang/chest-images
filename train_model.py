#!/usr/bin/env python
# coding: utf-8

import os,sys
sys.path.insert(0,"..")
import os,sys,inspect
from glob import glob
from os.path import exists, join
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torchvision, torchvision.transforms
import skimage.transform
import sklearn, sklearn.model_selection

import random
import train_utils
import torchxrayvision as xrv
from newDesnet import Dense_Nonlocal
from models import op_norm


parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default="", help='')
parser.add_argument('weights_filename', type=str,)
parser.add_argument('-mdtable', action='store_true', help='')
parser.add_argument('-name', type=str)
parser.add_argument('--output_dir', type=str, default="/scratch/users/joecohen/output/")
parser.add_argument('--dataset', type=str, default="google")
parser.add_argument('--dataset_dir', type=str, default="/home/groups/akshaysc/joecohen/")
parser.add_argument('--model', type=str, default="resnet50")
parser.add_argument('--seed', type=int, default=0, help='')
parser.add_argument('--cuda', type=bool, default=True, help='')
parser.add_argument('--num_epochs', type=int, default=400, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--shuffle', type=bool, default=True, help='')
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--threads', type=int, default=4, help='')
parser.add_argument('--taskweights', type=bool, default=True, help='')
parser.add_argument('--featurereg', type=bool, default=False, help='')
parser.add_argument('--weightreg', type=bool, default=False, help='')
parser.add_argument('--data_aug', type=bool, default=True, help='')
parser.add_argument('--data_aug_rot', type=int, default=45, help='')
parser.add_argument('--data_aug_trans', type=float, default=0.15, help='')
parser.add_argument('--data_aug_scale', type=float, default=0.15, help='')
parser.add_argument('--label_concat', type=bool, default=False, help='')
parser.add_argument('--label_concat_reg', type=bool, default=False, help='')
parser.add_argument('--labelunion', type=bool, default=False, help='')

cfg = parser.parse_args()
print(cfg)

data_aug = None
if cfg.data_aug:
    data_aug = torchvision.transforms.Compose([
        xrv.datasets.ToPILImage(),
        torchvision.transforms.RandomAffine(cfg.data_aug_rot, 
                                            translate=(cfg.data_aug_trans, cfg.data_aug_trans), 
                                            scale=(1.0-cfg.data_aug_scale, 1.0+cfg.data_aug_scale)),
        torchvision.transforms.ToTensor()
    ])
    print(data_aug)

transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(512)])

datas = []
datas_names = []
if "nih" in cfg.dataset:
    #  change the file path
    dataset = xrv.datasets.NIH_Dataset(
        imgpath=cfg.dataset_dir + "/images-NIH-224", 
        transform=transforms, data_aug=data_aug, unique_patients=False, views=["PA","AP"])
    datas.append(dataset)
    datas_names.append("nih")
if "pc" in cfg.dataset:
    dataset = xrv.datasets.PC_Dataset(
        imgpath=cfg.dataset_dir + "/images-512-PC", 
        transform=transforms, data_aug=data_aug, unique_patients=False, views=["PA","AP"])
    datas.append(dataset)
    datas_names.append("pc")
if "chex" in cfg.dataset:
    dataset = xrv.datasets.CheX_Dataset(
        imgpath=cfg.dataset_dir + "/CheXpert-v1.0-small",
        csvpath=cfg.dataset_dir + "/CheXpert-v1.0-small/train.csv",
        transform=transforms, data_aug=data_aug, unique_patients=False)
    datas.append(dataset)
    datas_names.append("chex")
if "google" in cfg.dataset:
    dataset = xrv.datasets.NIH_Google_Dataset(
        imgpath=cfg.dataset_dir + "/images-NIH-224",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("google")
if "mimic_ch" in cfg.dataset:
    dataset = xrv.datasets.MIMIC_Dataset(
        imgpath="/scratch/users/joecohen/data/MIMICCXR-2.0/files/",
        csvpath=cfg.dataset_dir + "/MIMICCXR-2.0/mimic-cxr-2.0.0-chexpert.csv.gz",
        metacsvpath=cfg.dataset_dir + "/MIMICCXR-2.0/mimic-cxr-2.0.0-metadata.csv.gz",
        transform=transforms, data_aug=data_aug, unique_patients=False, views=["PA","AP"])
    datas.append(dataset)
    datas_names.append("mimic_ch")
if "openi" in cfg.dataset:
    dataset = xrv.datasets.Openi_Dataset(
        imgpath=cfg.dataset_dir + "/OpenI/images/",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("openi")
if "rsna" in cfg.dataset:
    dataset = xrv.datasets.RSNA_Pneumonia_Dataset(
        imgpath=cfg.dataset_dir + "/kaggle-pneumonia-jpg/stage_2_train_images_jpg",
        transform=transforms, data_aug=data_aug, unique_patients=False, views=["PA","AP"])
    datas.append(dataset)
    datas_names.append("rsna")
if "siim" in cfg.dataset:
    dataset = xrv.datasets.SIIM_Pneumothorax_Dataset(
        imgpath=cfg.dataset_dir + "/SIIM_TRAIN_TEST/dicom-images-train",
        csvpath=cfg.dataset_dir + "/SIIM_TRAIN_TEST/train-rle.csv",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("siim")
if "vin" in cfg.dataset:
    dataset = xrv.datasets.VinBrain_Dataset(
        imgpath=cfg.dataset_dir + "vinbigdata-chest-xray-abnormalities-detection/train",
        csvpath=cfg.dataset_dir + "vinbigdata-chest-xray-abnormalities-detection/train.csv",
        transform=transform, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("vin")


print("datas_names", datas_names)

if cfg.labelunion:
    newlabels = set()
    for d in datas:
        newlabels = newlabels.union(d.pathologies)
    newlabels.remove("Support Devices")
    print(list(newlabels))
    for d in datas:
        xrv.datasets.relabel_dataset(list(newlabels), d)
else:
    for d in datas:
        xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, d)

#cut out training sets
train_datas = []
test_datas = []
for i, dataset in enumerate(datas):
    
    # give patientid if not exist
    if "patientid" not in dataset.csv:
        dataset.csv["patientid"] = ["{}-{}".format(dataset.__class__.__name__, i) for i in range(len(dataset))]
        
    gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.8,test_size=0.2, random_state=cfg.seed)
    
    train_inds, test_inds = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))
    train_dataset = xrv.datasets.SubsetDataset(dataset, train_inds)
    test_dataset = xrv.datasets.SubsetDataset(dataset, test_inds)
    
    train_datas.append(train_dataset)
    test_datas.append(test_dataset)
    
if len(datas) == 0:
    raise Exception("no dataset")
elif len(datas) == 1:
    train_dataset = train_datas[0]
    test_dataset = test_datas[0]
else:
    print("merge datasets")
    train_dataset = xrv.datasets.Merge_Dataset(train_datas)
    test_dataset = xrv.datasets.Merge_Dataset(test_datas)


# Setting the seed
np.random.seed(cfg.seed)
random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
if cfg.cuda:
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print("train_dataset.labels.shape", train_dataset.labels.shape)
print("test_dataset.labels.shape", test_dataset.labels.shape)
print("train_dataset",train_dataset)
print("test_dataset",test_dataset)
    
# create models
if "densenet" in cfg.model:
    
    # model = xrv.models.DenseNet(num_classes=train_dataset.labels.shape[1], in_channels=1, 
    #                             **xrv.models.get_densenet_params(cfg.model)) 
    # model = Dense_Nonlocal(num_classes=train_dataset.labels.shape[1],in_channels=1)
    model = Dense_Nonlocal(weights=cfg.weights_filename, num_classes=train_dataset.labels.shape[1], in_channels=1,apply_sigmoid=True,args=cfg)
    # model = xrv.models.DenseNet(num_classes=train_dataset.labels.shape[1], in_channels=1, 
                                # weights="densenet121-res224-all")
    model.op_threshs = None
elif "resnet101" in cfg.model:
    model = torchvision.models.resnet101(num_classes=train_dataset.labels.shape[1], pretrained=False)
    #patch for single channel
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
elif "resnet50" in cfg.model:
    model = torchvision.models.resnet50(num_classes=train_dataset.labels.shape[1], pretrained=False)
    #patch for single channel
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
elif "shufflenet_v2_x2_0" in cfg.model:
    model = torchvision.models.shufflenet_v2_x2_0(num_classes=train_dataset.labels.shape[1], pretrained=False)
    #patch for single channel
    model.conv1[0] = torch.nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
elif "squeezenet1_1" in cfg.model:
    model = torchvision.models.squeezenet1_1(num_classes=train_dataset.labels.shape[1], pretrained=False)
    #patch for single channel
    model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
else:
    raise Exception("no model")


# train_utils.train(model, train_dataset, cfg)


print("Done")


#####
test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=cfg.batch_size,
                                           shuffle=cfg.shuffle,
                                           num_workers=cfg.threads, pin_memory=cfg.cuda)


filename = "results_" + os.path.basename(cfg.weights_filename).split(".")[0] + "_" + "-".join(datas_names) + ".pkl"
print(filename)
# if os.path.exists(filename):#####change
#     print("Results already computed")
#     results = pickle.load(open(filename, "br"))
# else:
print("Results are being computed")
if cfg.cuda:
    model = model.cuda()
results = train_utils.valid_test_epoch("test", 0, model, "cuda", test_loader, torch.nn.BCEWithLogitsLoss(), limit=99999999)
# pickle.dump(results, open(filename, "bw"))

print("Model pathologies:",model.pathologies)
print("Dataset pathologies:",test_dataset.pathologies)

perf_dict = {}
all_threshs = []
all_min = []
all_max = []
all_ppv80 = []
for i, patho in enumerate(test_dataset.pathologies):
    opt_thres = np.nan
    opt_min = np.nan
    opt_max = np.nan
    ppv80_thres = np.nan
    if (len(results[3][i]) > 0) and (len(np.unique(results[3][i])) == 2):
        
        #sigmoid
        all_outputs = 1.0/(1.0 + np.exp(-results[2][i]))
        
        fpr, tpr, thres = sklearn.metrics.roc_curve(results[3][i], all_outputs)
        pente = tpr - fpr
        opt_thres = thres[np.argmax(pente)]
        opt_min = all_outputs.min()
        opt_max = all_outputs.max()
        
        ppv, recall, thres = sklearn.metrics.precision_recall_curve(results[3][i], all_outputs)
        ppv80_thres_idx = np.where(ppv > 0.8)[0][0]
        ppv80_thres = thres[ppv80_thres_idx-1]
        
        auc = sklearn.metrics.roc_auc_score(results[3][i], all_outputs)
        
        print(patho, auc)
        perf_dict[patho] = str(round(auc,2))
        
    else:
        perf_dict[patho] = "-"
        
    all_threshs.append(opt_thres)
    all_min.append(opt_min)
    all_max.append(opt_max)
    all_ppv80.append(ppv80_thres)

    
print("pathologies",test_dataset.pathologies)
    
print("op_threshs",str(all_threshs).replace("nan","np.nan"))
    
print("min",str(all_min).replace("nan","np.nan"))
    
print("max",str(all_max).replace("nan","np.nan"))

print("ppv80",str(all_ppv80).replace("nan","np.nan"))
    
    
if cfg.mdtable:
    print("|Model Name|" + "|".join(orj_dataset_pathologies) + "|")
    print("|---|" + "|".join(("-"*len(orj_dataset_pathologies))) + "|")
    
    accs = [perf_dict[patho] if (patho in perf_dict) else "-" for patho in orj_dataset_pathologies]
    print("|"+str(model)+"|" + "|".join(accs) + "|")
    
print("Done")



