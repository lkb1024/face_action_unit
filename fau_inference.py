import os
import numpy as np
import torch
import torch.nn as nn
import logging
from dataset import pil_loader
from utils import *
from conf import get_config,set_logger,set_outdir,set_env, edict

def initialize_fau_model():
    conf_dict={'dataset': 'hybrid', 'batch_size': 64, 'learning_rate': 1e-05, 'epochs': 20, 'num_workers': 4, 'weight_decay': 0.0005, 'optimizer_eps': 1e-08, 'crop_size': 224, 'evaluate': False, 'arc': 'resnet50', 'metric': 'dots', 'lam': 0.001, 'gpu_ids': '0', 'seed': 0, 'exp_name': 'demo', 'resume': 'checkpoints/OpenGprahAU-ResNet50_first_stage.pth', 'input': 'demo_imgs/1014.jpg', 'draw_text': True, 'stage': 1, 'dataset_path': 'data/Hybrid', 'num_main_classes': 27, 'num_sub_classes': 14, 'neighbor_num': 4}
    conf = edict(conf_dict)
    conf.evaluate = True
    set_env(conf)

    if conf.stage == 1:
        from model.ANFL import MEFARG
        net = MEFARG(num_main_classes=conf.num_main_classes, num_sub_classes=conf.num_sub_classes, backbone=conf.arc, neighbor_num=conf.neighbor_num, metric=conf.metric)
    else:
        from model.MEFL import MEFARG
        net = MEFARG(num_main_classes=conf.num_main_classes, num_sub_classes=conf.num_sub_classes, backbone=conf.arc)
    
    # resume
    if conf.resume != '':
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)

    net.eval()
    return net

def fau_inference(net, img):
    img_size=256
    crop_size=224

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size),
        transforms.CenterCrop(crop_size),
        normalize
    ])
    img_ = transform(img).unsqueeze(0)

    if torch.cuda.is_available():
        net = net.cuda()
        img_ = img_.cuda()

    with torch.no_grad():
        pred = net(img_)
        pred = pred.squeeze().cpu().numpy()

    dataset_info = hybrid_prediction_infolist
    infostr_probs,  infostr_aus = dataset_info(pred, 0.5)
    return infostr_probs,  infostr_aus


