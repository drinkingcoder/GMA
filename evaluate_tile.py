from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datasets

from utils.utils import InputPadder, forward_interpolate
import itertools
from evaluate import compute_epe

TRAIN_SIZE = [400, 768]
# 436 1024
def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
  if min_overlap >= TRAIN_SIZE[0] or min_overlap >= TRAIN_SIZE[1]:
    raise ValueError(
        f"Overlap should be less than size of patch (got {min_overlap}"
        f"for patch size {patch_size}).")
  hs = list(range(0, image_shape[0], TRAIN_SIZE[0] - min_overlap))
  ws = list(range(0, image_shape[1], TRAIN_SIZE[1] - min_overlap))
  # Make sure the final patch is flush with the image boundary
  hs[-1] = image_shape[0] - patch_size[0]
  ws[-1] = image_shape[1] - patch_size[1]
  return [(h, w) for h in hs for w in ws]

@torch.no_grad()
def create_sintel_submission(model, output_path='sintel_submission_multi8_011_768'):
    """ Create submission for the Sintel leaderboard """

    IMAGE_SIZE = [436, 1024]

    hws = compute_grid_indices(IMAGE_SIZE)
    weights_h, weights_w = torch.meshgrid(torch.arange(TRAIN_SIZE[0]), torch.arange(TRAIN_SIZE[1]))
    weights_h = torch.min(weights_h + 1, TRAIN_SIZE[0] - weights_h)
    weights_w = torch.min(weights_w + 1, TRAIN_SIZE[1] - weights_w)
    weights = torch.min(weights_h, weights_w)[None, None, :, :].cuda()
    results = {}

    model.eval()
    for dstype in ['final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        epe_list = []
    
        for test_id in range(len(test_dataset)):
            print(test_id)
            image1, image2, (sequence, frame) = test_dataset[test_id]
            
            image1, image2 = image1[None].cuda(), image2[None].cuda()

            flows = 0
            flow_count = 0

            for h, w in hws:
                image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
                image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]    
                flow_low, flow_pre = model(image1_tile, image2_tile, iters=32, flow_init=None, test_mode=True)
                padding = (w, IMAGE_SIZE[1]-w-TRAIN_SIZE[1], h, IMAGE_SIZE[0]-h-TRAIN_SIZE[0], 0, 0)
                flows += F.pad(flow_pre * weights, padding)
                flow_count += F.pad(weights, padding)

            flow_pre = flows / flow_count
            flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

            sequence_prev = sequence

            epe = compute_epe(sequence, frame, flow)
            epe_list.append(epe)

        epe_all = np.concatenate(epe_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

@torch.no_grad()
def create_sintel_submission_warm_start(model, output_path='sintel_submission_multi8_768_warm_start'):
    """ Create submission for the Sintel leaderboard """

    IMAGE_SIZE = [436, 1024]

    hws = compute_grid_indices(IMAGE_SIZE)
    weights_h, weights_w = torch.meshgrid(torch.arange(TRAIN_SIZE[0]), torch.arange(TRAIN_SIZE[1]))
    weights_h = torch.min(weights_h + 1, TRAIN_SIZE[0] - weights_h)
    weights_w = torch.min(weights_w + 1, TRAIN_SIZE[1] - weights_w)
    weights = torch.min(weights_h, weights_w)[None, None, :, :].cuda()

    model.eval()
    for dstype in ['final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        flow_prev, sequence_prev = None, None
        epe_list = []
        for test_id in range(len(test_dataset)):
            print(test_id)
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            image1, image2 = image1[None].cuda(), image2[None].cuda()

            flows = 0
            flow_count = 0

            this_flow_low = []
            for idx, (h, w) in enumerate(hws):
                image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
                image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
                if flow_prev != None:
                    flow_prev_tile = flow_prev[idx]
                else:
                    flow_prev_tile = None
                flow_low, flow_pre = model(image1_tile, image2_tile, flow_init=flow_prev_tile, test_mode=True)
                this_flow_low.append(forward_interpolate(flow_low[0])[None].cuda())
                
                padding = (w, IMAGE_SIZE[1]-w-TRAIN_SIZE[1], h, IMAGE_SIZE[0]-h-TRAIN_SIZE[0], 0, 0)
                flows += F.pad(flow_pre * weights, padding)
                flow_count += F.pad(weights, padding)

            flow_pre = flows / flow_count
            flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

            epe = compute_epe(sequence, frame, flow)
            epe_list.append(epe)

            sequence_prev = sequence
            flow_prev = this_flow_low
        epe_all = np.concatenate(epe_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        flow_pre = model(image1, image2)

        epe = torch.sum((flow_pre[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=6):
    """ Peform validation using the Sintel (train) split """

    IMAGE_SIZE = [436, 1024]

    hws = compute_grid_indices(IMAGE_SIZE)
    weights_h, weights_w = torch.meshgrid(torch.arange(TRAIN_SIZE[0]), torch.arange(TRAIN_SIZE[1]))
    weights_h = torch.min(weights_h + 1, TRAIN_SIZE[0] - weights_h)
    weights_w = torch.min(weights_w + 1, TRAIN_SIZE[1] - weights_w)
    weights = torch.min(weights_h, weights_w)[None, None, :, :].cuda()

    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            print(val_id)
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda() 

            flows = 0
            flow_count = 0
            
            for h, w in hws:
                image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
                image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]    
                _, flow_pre = model(image1_tile, image2_tile, iters=iters, test_mode=True)
                padding = (w, IMAGE_SIZE[1]-w-TRAIN_SIZE[1], h, IMAGE_SIZE[0]-h-TRAIN_SIZE[0], 0, 0)
                flows += F.pad(flow_pre * weights, padding)
                flow_count += F.pad(weights, padding)

            flow_pre = flows / flow_count
            flow_pre = flow_pre[0].cpu()

            epe = torch.sum((flow_pre - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results

@torch.no_grad()
def validate_kitti(model):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_pre = model(image1, image2)

        flow_pre = padder.unpad(flow_pre[0]).cpu()

        epe = torch.sum((flow_pre - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    cfg = get_cfg()
    cfg.update(vars(args))

    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()

    create_sintel_submission_warm_start(model.module)
    # create_sintel_submission(model.module)
    # create_kitti_submission(model.module)
    exit()
    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'sintel':
            validate_sintel(model.module)

        elif args.dataset == 'kitti':
            validate_kitti(model.module)

