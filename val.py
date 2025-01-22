"""
Validate a trained YOLOv5 model accuracy on a custom dataset
Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""
import argparse
import json
import os
import string
import sys
from pathlib import Path
from threading import Thread
import numpy as np
import torch
import yaml
from tqdm import tqdm
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  
from models.experimental import attempt_load
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.general import (LOGGER, box_iou, check_dataset, check_img_size, check_requirements, check_suffix, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, fitness
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1)) 
RANK = int(os.getenv('RANK', -1))             
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))  
def save_one_txt(predn, save_conf, shape, file):
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')
def save_one_json(predn, jdict, path, class_map):
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])
    box[:, :2] -= box[:, 2:] / 2
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})
def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5] 
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]] 
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True 
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)
@torch.no_grad()
def run(data,
        weights=None,  
        batch_size=32,  
        imgsz=640,  
        conf_thres=0.4,  
        iou_thres=0.2,  
        task='val',  
        device='1',  
        single_cls=False,  
        augment=False,  
        verbose=True,  
        save_txt=False,  
        save_hybrid=False,  
        save_conf=False,  
        save_json=False,  
        project=ROOT / 'runs/val',  
        name='exp',  
        exist_ok=False,  
        half=False,  
        model=None,  
        dataloader=None, 
        save_dir=Path(''), 
        plots=True,  
        callbacks=Callbacks(),
        compute_loss=None, 
        ):
    global gs, ap50
    training = model is not None
    if training:  
        device = next(model.parameters()).device
    else:  
        device = select_device(device, batch_size=batch_size)
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
        check_suffix(weights, '.pt')
        model = attempt_load(weights, map_location=device)
        gs = max(int(model.stride.max()), 32)
        imgsz = check_img_size(imgsz, s=gs)
        data = check_dataset(data)
    half &= device.type != 'cpu'  
    model.half() if half else model.float()
    model.eval()
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  
    nc = 1 if single_cls else int(data['nc'])
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  
        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('train', 'val', 'test') else 'val'  
        hyps = check_yaml(ROOT / 'data/hyps/hyp.VisDrone.yaml')
        with open(hyps, errors='ignore') as f:
            hyps = yaml.safe_load(f)  
            if 'anchors' not in hyps:  
                hyps['anchors'] = 4
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       gs,
                                       single_cls,
                                       hyp=hyps,
                                       pad=pad,
                                       rank=-1,  
                                       rect=True,
                                       prefix=colorstr(f'{task}: '))[0]
    
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        t1 = time_sync() 
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  
        img /= 255  
        targets = targets.to(device)
        nb, _, height, width = img.shape
        t2 = time_sync()
        dt[0] += t2 - t1
        out, train_out= model(img, augment=augment)  
        dt[1] += time_sync() - t2
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  
        t3 = time_sync()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        dt[2] += time_sync() - t3
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1
            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])  
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  
                scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            if save_json:
                save_one_json(predn, jdict, path, class_map)  
            callbacks.run('on_val_image_end', pred, predn, path, names, img[si])
        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  
            Thread(target=plot_images, args=(img, targets, paths, f), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f), daemon=True).start()
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  
    else:
        nt = torch.zeros(1)
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
    t = tuple(x / seen * 1E3 for x in dt)  
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  
        pred_json = str(save_dir / f"{w}_predictions.json")  
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)
        try:  
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            anno = COCO(anno_json)  
            pred = anno.loadRes(pred_json)  
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')
    
    model.float()  
    if not training:
        
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/VisDrone.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default='weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=48, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold 0.001  0.4')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='NMS IoU threshold 0.6  0.2')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class') 
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='yolov5_640', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt
def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    if opt.task in ('train', 'val', 'test'):  
        results, maps, _ = run(**vars(opt))
        isfi = fitness(np.array(results).reshape(1, -1))  
        LOGGER.info('%fi' % isfi)
    elif opt.task == 'speed':
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=opt.imgsz, conf_thres=.25, iou_thres=.45,
                device=opt.device, save_json=False, plots=False)
    elif opt.task == 'study':
        x = list(range(512, 640 + 128, 128))  
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  
            y = []  
            for i in x:  
                LOGGER.info(f'\nRunning {f} point {i}...')
                r, _, t = run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=i, conf_thres=opt.conf_thres,
                              iou_thres=opt.iou_thres, device=opt.device, save_json=opt.save_json, plots=False)
                y.append(r + t)  
            np.savetxt(f, y, fmt='%10.4g')  
        os.system('zip -r study.zip study_*.txt')
        plot_val_study(x=x)  
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
