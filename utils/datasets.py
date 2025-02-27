 
"""
Dataloaders and dataset utils
"""

import glob
import hashlib
import json
import logging
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from zipfile import ZipFile

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective, \
    copy_reduce_paste
from utils.general import (LOGGER, check_dataset, check_requirements, check_yaml, clean_str, segments2boxes, xyn2xy,
                           xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first

 
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']   
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']   
NUM_THREADS = min(8, os.cpu_count())   

 
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(paths):
     
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))   
    h = hashlib.md5(str(size).encode())   
    h.update(''.join(paths).encode())   
    return h.hexdigest()   
def exif_size(img):
     
    s = img.size   
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:   
            s = (s[1], s[0])
        elif rotation == 8:   
            s = (s[1], s[0])
    except:
        pass

    return s


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)   
    if orientation > 1:
        method = {2: Image.FLIP_LEFT_RIGHT,
                  3: Image.ROTATE_180,
                  4: Image.FLIP_TOP_BOTTOM,
                  5: Image.TRANSPOSE,
                  6: Image.ROTATE_270,
                  7: Image.TRANSVERSE,
                  8: Image.ROTATE_90,
                  }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image

 
def create_dataloader(path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False, pad=0.0,
                      rect=False, rank=-1, workers=0, image_weights=False, quad=False, prefix=''):

     
     
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,   
                                      hyp=hyp,   
                                      rect=rect,   
                                      cache_images=cache,
                                      single_cls=single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])   
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
     
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        drop_last=True,
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn)
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:
     
    def __init__(self, path, img_size=640, stride=32, auto=True):
        p = str(Path(path).resolve())   
         
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))   
         
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))   
         
        elif os.path.isfile(p):
            files = [p]   
        else:
            raise Exception(f'ERROR: {p} does not exist')
         
         
        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
         
        ni, nv = len(images), len(videos)
         
        self.img_size = img_size
        self.stride = stride
         
        self.files = images + videos
         
        self.nf = ni + nv   
         
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
         
        if any(videos):
            self.new_video(videos[0])   
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
         
        if self.count == self.nf:
            raise StopIteration
         
        path = self.files[self.count]
         
        if self.video_flag[self.count]:
             
            self.mode = 'video'
             
            ret_val, img0 = self.cap.read()
             
            if not ret_val:
                self.count += 1
                 
                self.cap.release()
                 
                if self.count == self.nf:   
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()
             
            self.frame += 1
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
             
            self.count += 1
            img0 = cv2.imread(path)   
            assert img0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

         
         
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

         
         
        img = img.transpose((2, 0, 1))[::-1]   
         
        img = np.ascontiguousarray(img)
         
         
        return path, img, img0, self.cap, s

    def new_video(self, path):
         
        self.frame = 0
        self.cap = cv2.VideoCapture(path)   
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))   

    def __len__(self):
        return self.nf   


class LoadWebcam:   
     
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe
        self.cap = cv2.VideoCapture(self.pipe)   
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)   

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):   
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

         
        ret_val, img0 = self.cap.read()
        img0 = cv2.flip(img0, 1)   

         
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        s = f'webcam {self.count}: '

         
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

         
        img = img.transpose((2, 0, 1))[::-1]   
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None, s

    def __len__(self):
        return 0


class LoadStreams:
     
    def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True):
        self.mode = 'stream'    
        self.img_size = img_size
        self.stride = stride
         
         
        if os.path.isfile(sources):
            with open(sources) as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]
         
        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]   
        self.auto = auto
        for i, s in enumerate(sources):   
             
            st = f'{i + 1}/{n}: {s}... '
            if 'youtube.com/' in s or 'youtu.be/' in s:   
                check_requirements(('pafy', 'youtube_dl'))
                import pafy
                s = pafy.new(s).getbest(preftype="mp4").url   
             
            s = eval(s) if s.isnumeric() else s   
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'{st}Failed to open {s}'
             
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
             
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0   
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')   
             
            _, self.imgs[i] = cap.read()   
             
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            LOGGER.info(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        LOGGER.info('')   

         
         
        s = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1   
        if not self.rect:
            LOGGER.warning('WARNING: Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap, stream):
         
        n, f, read = 0, self.frames[i], 1   
        while cap.isOpened() and n < f:
            n += 1
             
            cap.grab()
             
            if n % read == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    LOGGER.warn('WARNING: Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] *= 0
                    cap.open(stream)   
            time.sleep(1 / self.fps[i])   

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):   
            cv2.destroyAllWindows()
            raise StopIteration

         
        img0 = self.imgs.copy()
         
        img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]

         
         
        img = np.stack(img, 0)

         
        img = img[..., ::-1].transpose((0, 3, 1, 2))   
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None, ''

    def __len__(self):
        return len(self.sources)   


def img2label_paths(img_paths):
     
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep   
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


class LoadImagesAndLabels(Dataset):
     
    cache_version = 0.6   

    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp    
        self.image_weights = image_weights    
        self.rect = False if image_weights else rect   
         
        self.mosaic = self.augment and not self.rect   
        self.mosaic_border = [-img_size // 2, -img_size // 2]   
        self.stride = stride   
        self.path = path
        self.albumentations = Albumentations() if augment else None

        try:
            f = []   
            for p in path if isinstance(path, list) else [path]:
                 
                 
                p = Path(p)   
                if p.is_dir():   
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                     
                 
                elif p.is_file():   
                    with open(p) as t:
                         
                        t = t.read().strip().splitlines()
                         
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]   
                         
                else:
                    raise Exception(f'{prefix}{p} does not exist')
             
            self.img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
             
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')

         
         
        self.label_files = img2label_paths(self.img_files)   
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True   
            assert cache['version'] == self.cache_version   
            assert cache['hash'] == get_hash(self.label_files + self.img_files)   
        except:
            cache, exists = self.cache_labels(cache_path, prefix), False   

         
        nf, nm, ne, nc, n = cache.pop('results')   
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)   
            if cache['msgs']:
                logging.info('\n'.join(cache['msgs']))   
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {HELP_URL}'

         
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]   
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())   
        self.label_files = img2label_paths(cache.keys())   
        n = len(shapes)   
        bi = np.floor(np.arange(n) / batch_size).astype(int)   
        nb = bi[-1] + 1   
        self.batch = bi   
        self.n = n
        self.indices = range(n)

         
        include_class = []   
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = segment[j]
            if single_cls:   
                self.labels[i][:, 0] = 0
                if segment:
                    self.segments[i][:, 0] = 0

         
        if self.rect:
             
            s = self.shapes   
             
            ar = s[:, 1] / s[:, 0]   
            irect = ar.argsort()
             
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]   
            ar = ar[irect]

             
             
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                 
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                 
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride

         
         
        self.imgs, self.img_npy = [None] * n, [None] * n
        if cache_images:
            if cache_images == 'disk':
                self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + '_npy')
                self.img_npy = [self.im_cache_dir / Path(f).with_suffix('.npy').name for f in self.img_files]
                self.im_cache_dir.mkdir(parents=True, exist_ok=True)
            gb = 0   
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(NUM_THREADS).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))

            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                if cache_images == 'disk':
                    if not self.img_npy[i].exists():
                        np.save(self.img_npy[i].as_posix(), x[0])
                    gb += self.img_npy[i].stat().st_size
                else:
                    self.imgs[i], self.img_hw0[i], self.img_hw[i] = x   
                    gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB {cache_images})'
            pbar.close()

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
         
        x = {}   
         
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []   
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label, zip(self.img_files, self.label_files, repeat(prefix))),
                        desc=desc, total=len(self.img_files))
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [l, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        pbar.close()
        if msgs:
            logging.info('\n'.join(msgs))
        if nf == 0:
            logging.info(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, len(self.img_files)
        x['msgs'] = msgs   
        x['version'] = self.cache_version   
        try:
            np.save(path, x)   
            path.with_suffix('.cache.npy').rename(path)   
            logging.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            logging.info(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')   
        return x

    def __len__(self):
        return len(self.img_files)

     
     
     
     
     

    def __getitem__(self, index):
        index = self.indices[index]   

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
             
            img, labels = load_mosaic(self, index)
            shapes = None

             
            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *load_mosaic(self, random.randint(0, self.n - 1)))

        else:
             
             
            img, (h0, w0), (h, w) = load_image(self, index)

             
             
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size   
             
             
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)   

            labels = self.labels[index].copy()
            if labels.size:   
                 
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                 
                img, labels = random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        nl = len(labels)   
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
             
            img, labels = self.albumentations(img, labels)
            nl = len(labels)   

             
             
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

             
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]
             
             
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

             
             
         
        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

         
        img = img.transpose((2, 0, 1))[::-1]   
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)   
        for i, l in enumerate(label):
            l[:, 0] = i   
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)   
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])   
        for i in range(n):   
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i   

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4


 
def load_image(self, i):
     
    im = self.imgs[i]
    if im is None:   
        npy = self.img_npy[i]
        if npy and npy.exists():   
            im = np.load(npy)
        else:   
            path = self.img_files[i]
            im = cv2.imread(path)   
            assert im is not None, f'Image Not Found {path}'
        h0, w0 = im.shape[:2]   
        r = self.img_size / max(h0, w0)   
         
        if r != 1:   
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        return im, (h0, w0), im.shape[:2]   
    else:
        return self.imgs[i], self.img_hw0[i], self.img_hw[i]   


def load_mosaic(self, index):
     
    labels4, segments4 = [], []
    s = self.img_size
     
    yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)   
     
    indices = [index] + random.choices(self.indices, k=3)   
    random.shuffle(indices)
    for i, index in enumerate(indices):
         
        img, _, (h, w) = load_image(self, index)

         
        if i == 0:   
             
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)   
             
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc   
             
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h   
        elif i == 1:   
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:   
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:   
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
         
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]   
         
        padw = x1a - x1b
        padh = y1a - y1b

         
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
         
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)   
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)

        segments4.extend(segments)

     
     
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)   
     

     
     
     
     
    img4, labels4, segments4 = copy_reduce_paste(img4, labels4, segments4, paste_prob=self.hyp['copy_paste'])
    img4, labels4 = random_perspective(img4, labels4, segments4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)   

    return img4, labels4


def load_mosaic9(self, index):
     
    labels9, segments9 = [], []
    s = self.img_size
    indices = [index] + random.choices(self.indices, k=8)   
    random.shuffle(indices)
    for i, index in enumerate(indices):
         
        img, _, (h, w) = load_image(self, index)

         
        if i == 0:   
            img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)   
            h0, w0 = h, w
            c = s, s, s + w, s + h   
        elif i == 1:   
            c = s, s - h, s + w, s
        elif i == 2:   
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:   
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:   
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:   
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:   
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:   
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:   
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]
        x1, y1, x2, y2 = (max(x, 0) for x in c)   

         
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)   
            segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
        labels9.append(labels)
        segments9.extend(segments)

         
        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]   
        hp, wp = h, w   

     
    yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)   
    img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

     
    labels9 = np.concatenate(labels9, 0)
    labels9[:, [1, 3]] -= xc
    labels9[:, [2, 4]] -= yc
    c = np.array([xc, yc])   
    segments9 = [x - c for x in segments9]

    for x in (labels9[:, 1:], *segments9):
        np.clip(x, 0, 2 * s, out=x)   
     

     
    img9, labels9 = random_perspective(img9, labels9, segments9,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)   

    return img9, labels9


def create_folder(path='./new'):
     
    if os.path.exists(path):
        shutil.rmtree(path)   
    os.makedirs(path)   


def flatten_recursive(path='datasets/coco128'):
     
    new_path = Path(path + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path='datasets/coco128'):   
     
    path = Path(path)   
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None   
    files = list(path.rglob('*.*'))
    n = len(files)   
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS:
             
            im = cv2.imread(str(im_file))[..., ::-1]   
            h, w = im.shape[:2]

             
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file) as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)   

                for j, x in enumerate(lb):
                    c = int(x[0])   
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'   
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]   
                     
                    b[2:] = b[2:] * 1.2 + 3   
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)   
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path='datasets/coco128/images', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.datasets import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)   
    files = sorted(x for x in path.rglob('*.*') if x.suffix[1:].lower() in IMG_FORMATS)   
    n = len(files)   
    random.seed(0)   
    indices = random.choices([0, 1, 2], weights=weights, k=n)   

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']   
    [(path.parent / x).unlink(missing_ok=True) for x in txt]   

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():   
            with open(path.parent / txt[i], 'a') as f:
                f.write('./' + img.relative_to(path.parent).as_posix() + '\n')   


def verify_image_label(args):
     
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []   
    try:
         
        im = Image.open(im_file)
        im.verify()   
        shape = exif_size(im)   
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':   
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING: {im_file}: corrupt JPEG restored and saved'

         
        if os.path.isfile(lb_file):
            nf = 1   
            with open(lb_file) as f:
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any([len(x) > 8 for x in l]):   
                    classes = np.array([x[0] for x in l], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]   
                    l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)   
                l = np.array(l, dtype=np.float32)
            nl = len(l)
            if nl:
                assert l.shape[1] == 5, f'labels require 5 columns, {l.shape[1]} columns detected'
                assert (l >= 0).all(), f'negative label values {l[l < 0]}'
                assert (l[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {l[:, 1:][l[:, 1:] > 1]}'
                l = np.unique(l, axis=0)   
                if len(l) < nl:
                    segments = np.unique(segments, axis=0)
                    msg = f'{prefix}WARNING: {im_file}: {nl - len(l)} duplicate labels removed'
            else:
                ne = 1   
                l = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1   
            l = np.zeros((0, 5), dtype=np.float32)
        return im_file, l, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]


def dataset_stats(path='coco128.yaml', autodownload=False, verbose=False, profile=False, hub=False):
    """ Return dataset statistics dictionary with images and instances counts per split per class
    To run in parent directory: export PYTHONPATH="$PWD/yolov5"
    Usage1: from utils.datasets import *; dataset_stats('coco128.yaml', autodownload=True)
    Usage2: from utils.datasets import *; dataset_stats('../datasets/coco128_with_yaml.zip')
    Arguments
        path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
        autodownload:   Attempt to download dataset if not found locally
        verbose:        Print stats dictionary
    """

    def round_labels(labels):
         
        return [[int(c), *(round(x, 4) for x in points)] for c, *points in labels]

    def unzip(path):
         
        if str(path).endswith('.zip'):   
            assert Path(path).is_file(), f'Error unzipping {path}, file not found'
            ZipFile(path).extractall(path=path.parent)   
            dir = path.with_suffix('')   
            return True, str(dir), next(dir.rglob('*.yaml'))   
        else:   
            return False, None, path

    def hub_ops(f, max_dim=1920):
         
        f_new = im_dir / Path(f).name   
        try:   
            im = Image.open(f)
            r = max_dim / max(im.height, im.width)   
            if r < 1.0:   
                im = im.resize((int(im.width * r), int(im.height * r)))
            im.save(f_new, quality=75)   
        except Exception as e:   
            print(f'WARNING: HUB ops PIL failure {f}: {e}')
            im = cv2.imread(f)
            im_height, im_width = im.shape[:2]
            r = max_dim / max(im_height, im_width)   
            if r < 1.0:   
                im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(f_new), im)

    zipped, data_dir, yaml_path = unzip(Path(path))
    with open(check_yaml(yaml_path), errors='ignore') as f:
        data = yaml.safe_load(f)   
        if zipped:
            data['path'] = data_dir   
    check_dataset(data, autodownload)   
    hub_dir = Path(data['path'] + ('-hub' if hub else ''))
    stats = {'nc': data['nc'], 'names': data['names']}   
    for split in 'train', 'val', 'test':
        if data.get(split) is None:
            stats[split] = None   
            continue
        x = []
        dataset = LoadImagesAndLabels(data[split])   
        for label in tqdm(dataset.labels, total=dataset.n, desc='Statistics'):
            x.append(np.bincount(label[:, 0].astype(int), minlength=data['nc']))
        x = np.array(x)   
        stats[split] = {'instance_stats': {'total': int(x.sum()), 'per_class': x.sum(0).tolist()},
                        'image_stats': {'total': dataset.n, 'unlabelled': int(np.all(x == 0, 1).sum()),
                                        'per_class': (x > 0).sum(0).tolist()},
                        'labels': [{str(Path(k).name): round_labels(v.tolist())} for k, v in
                                   zip(dataset.img_files, dataset.labels)]}

        if hub:
            im_dir = hub_dir / 'images'
            im_dir.mkdir(parents=True, exist_ok=True)
            for _ in tqdm(ThreadPool(NUM_THREADS).imap(hub_ops, dataset.img_files), total=dataset.n, desc='HUB Ops'):
                pass

     
    stats_path = hub_dir / 'stats.json'
    if profile:
        for _ in range(1):
            file = stats_path.with_suffix('.npy')
            t1 = time.time()
            np.save(file, stats)
            t2 = time.time()
            x = np.load(file, allow_pickle=True)
            print(f'stats.npy times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

            file = stats_path.with_suffix('.json')
            t1 = time.time()
            with open(file, 'w') as f:
                json.dump(stats, f)   
            t2 = time.time()
            with open(file) as f:
                x = json.load(f)   
            print(f'stats.json times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

     
    if hub:
        print(f'Saving {stats_path.resolve()}...')
        with open(stats_path, 'w') as f:
            json.dump(stats, f)   
    if verbose:
        print(json.dumps(stats, indent=2, sort_keys=False))
    return stats
