 
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path
import torch.nn.functional as F
from ultralytics.utils.tal import make_anchors, dist2bbox
from models.common import *
from models.common import ASFF, ODConv_3rd, AutoShape, C3STR, C3SPP, ASPP, BAM, SCDown, PSA, C3CR, C2fCBAM, C2fBAM, \
    SPPELAN, SPPCSPC, SPPCSPCS, se_block, RepVGGBlock, ADown, C3_CBAMS, C3_CBAMS_DWC, C3_CBAM_DWC, \
    CoordConv, SimConv, C3_CBAM, C3_BAM, C3_CA, C3_SCBAM, C3GAM, C3CPCA, Conv2Former, Involution, BasicRFB, \
    BasicRFB_a, ACmix, SimAM, ODConv, CNeB, ConvMix, CSPCM, DownSimper, CARAFE, BiFPN_Add2, BiFPN_Add3, BiFPN, BiFPNs, \
    CAM, BiFPNSDI, \
    S2Attention, CPCA, NAMAttention, DySample, BiFusion, Zoom_cat, attention_model, ScalSeq, SF, SKAttention, CShortcut, \
    space_to_depth, SPD, CoorAttention, Contract, eca_block, Expand, C3CBAM

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]   
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))   
 
from ultralytics.nn.modules import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (copy_attr, fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device,
                               time_sync)

 
try:
    import thop   
except ImportError:
    thop = None


 
 
class Detect(nn.Module):
    stride = None   
    onnx_dynamic = False
    def __init__(self, nc=10, anchors=(), ch=(), inplace=False):
        super().__init__()
        self.nc = nc   
        self.no = nc + 5   
        self.nl = len(anchors)   
        self.na = len(anchors[0]) // 2   
        self.grid = [torch.zeros(1)] * self.nl   
        self.anchor_grid = [torch.zeros(1)] * self.nl   
         
         
         
         
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))   
         
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)   
         
        self.inplace = inplace   

    def forward(self, x):
        z = []
         
         
        for i in range(self.nl):
            x[i] = self.m[i](x[i])   
            bs, _, ny, nx = x[i].shape   

            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:   
                 
                 
                 
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                 
                y = x[i].sigmoid()
                if self.inplace:
                     
                     
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]   
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]   
                else:   
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]   
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]   
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                 
                z.append(y.view(bs, -1, self.no))
                 
        return x if self.training else (torch.cat(z, 1), x)
         

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):   
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class DetectODConv(nn.Module):
    stride = None   
    onnx_dynamic = False   

     
    def __init__(self, nc=80, anchors=(), ch=(), inplace=False):   
        super().__init__()
        self.nc = nc   
        self.no = nc + 5   
        self.nl = len(anchors)   
        self.na = len(anchors[0]) // 2   
        self.grid = [torch.zeros(1)] * self.nl   
        self.anchor_grid = [torch.zeros(1)] * self.nl   
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
         
        self.m = nn.ModuleList(
            ODConv2d_3rd(x, self.na * (5 + self.nc), kernel_size=1, stride=1) for x in ch)   
         
        self.inplace = inplace   

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])   
            bs, _, ny, nx = x[i].shape   

            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:   
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                 
                y = x[i].sigmoid()
                if self.inplace:
                     
                     
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]   
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]   
                else:   
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]   
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]   
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                 
                z.append(y.view(bs, -1, self.no))
                 
        return x if self.training else (torch.cat(z, 1), x)
         

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):   
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class ASFF_Detect(Detect):
     
    def __init__(self, nc=80, anchors=(), ch=(), inplace=False):   
        super().__init__(nc, anchors, ch, inplace)
        self.nl = len(anchors)
        self.asffs = nn.ModuleList(ASFF(i) for i in range(self.nl))
        self.detect = Detect.forward

    def forward(self, x):   
        x = x[::-1]
        for i in range(self.nl):
            x[i] = self.asffs[i](*x)
        return self.detect(self, x[::-1])


class DetectYOLO8Head(nn.Module):
    """YOLOv8 Detect head for detection models."""

    dynamic = False   
    export = False   
    shape = None
    anchors = torch.empty(0)   
    strides = torch.empty(0)   

    def __init__(self, nc=80, width=0.5, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc   
        self.nl = len(ch)   
        self.reg_max = 16   
        self.no = nc + self.reg_max * 4   
        self.stride = torch.zeros(self.nl)   
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)   
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:   
            return x

         
        shape = x[0].shape   
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:   
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
             
             
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self   
         
         
        for a, b, s in zip(m.cv2, m.cv3, m.stride):   
            a[-1].bias.data[:] = 1.0   
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)   

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)


class CLLA(nn.Module):
    def __init__(self, range, c):
        super().__init__()
        self.c_ = c
        self.q = nn.Linear(self.c_, self.c_)
        self.k = nn.Linear(self.c_, self.c_)
        self.v = nn.Linear(self.c_, self.c_)
        self.range = range
        self.attend = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        b1, c1, w1, h1 = x1.shape
        b2, c2, w2, h2 = x2.shape
        assert b1 == b2 and c1 == c2

        x2_ = x2.permute(0, 2, 3, 1).contiguous().unsqueeze(3)
        pad = int(self.range / 2 - 1)
        padding = nn.ZeroPad2d(padding=(pad, pad, pad, pad))
        x1 = padding(x1)

        local = []
        for i in range(int(self.range)):
            for j in range(int(self.range)):
                tem = x1
                tem = tem[..., i::2, j::2][..., :w2, :h2].contiguous().unsqueeze(2)
                local.append(tem)
        local = torch.cat(local, 2)

        x1 = local.permute(0, 3, 4, 2, 1)

        q = self.q(x2_)
        k, v = self.k(x1), self.v(x1)

        dots = torch.sum(q * k / self.range, 4)
        irr = torch.mean(dots, 3).unsqueeze(3) * 2 - dots
        att = self.attend(irr)

        out = v * att.unsqueeze(4)
        out = torch.sum(out, 3)
        out = out.squeeze(3).permute(0, 3, 1, 2).contiguous()
         
        return (out + x2) / 2
         


class CLLABlock(nn.Module):
    def __init__(self, range=2, ch=256, ch1=128, ch2=256, out=0):
        super().__init__()
        self.range = range
        self.c_ = ch
        self.cout = out
        self.conv1 = nn.Conv2d(ch1, self.c_, 1)
        self.conv2 = nn.Conv2d(ch2, self.c_, 1)

        self.att = CLLA(range=range, c=self.c_)

        self.det = nn.Conv2d(self.c_, out, 1)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        f = self.att(x1, x2)

        return self.det(f)


class CLLADetect(nn.Module):
    stride = None   
    onnx_dynamic = False   

    def __init__(self, nc=80, anchors=(), ch=(), inplace=False):   
        super().__init__()
        self.nc = nc   
        self.no = nc + 5   
        self.nl = len(anchors)   
        self.na = len(anchors[0]) // 2   
        self.grid = [torch.zeros(1)] * self.nl   
        self.anchor_grid = [torch.zeros(1)] * self.nl   
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))   
        self.det = CLLABlock(range=2, ch=ch[0], ch1=ch[0], ch2=ch[1], out=self.no * self.na)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[2:])   
        self.inplace = inplace   

    def forward(self, x):
        z = []   
        p = []
        for i in range(self.nl):
            if i == 0:
                p.append(self.det(x[0], x[1]))
            else:
                p.append(self.m[i - 1](x[i + 1]))   
            bs, _, ny, nx = p[i].shape   

            p[i] = p[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:   
                if self.onnx_dynamic or self.grid[i].shape[2:4] != p[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = p[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]   
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]   
                else:   
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]   
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]   
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))
        return p if self.training else (torch.cat(z, 1), p)
         

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):   
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class IDetect(nn.Module):
     
    stride = None   
    dynamic = False   
    export = False   
    include_nms = False
    end2end = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=(), inplace=False):   
        super().__init__()
        self.nc = nc   
        self.no = nc + 5   
        self.nl = len(anchors)   
        self.na = len(anchors[0]) // 2   
        self.grid = [torch.zeros(1)] * self.nl   
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)   
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))   

        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)   

        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

    def forward(self, x):

        z = []   
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))   
            bs, _, ny, nx = x[i].shape   
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:   
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]   
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]   

                else:   
                    xy, wh, conf = y.split((2, 2, self.no - 4), 4)   
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]   
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]   
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def fuseforward(self, x):
         
        z = []   
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])   
            bs, _, ny, nx = x[i].shape   
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:   
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]   
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]   
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)   
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))   
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)   
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z,)
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out

    def fuse(self):
         
         
        for i in range(len(self.m)):
            with torch.no_grad():
                c1, c2, _, _ = self.m[i].weight.shape
                c1_, c2_, _, _ = self.ia[i].implicit.shape
                self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1, c2),
                                               self.ia[i].implicit.reshape(c2_, c1_)).squeeze(1)

         
        for i in range(len(self.m)):
            with torch.no_grad():
                c1, c2, _, _ = self.im[i].implicit.shape
                self.m[i].bias *= self.im[i].implicit.reshape(c2)
                self.m[i].weight *= self.im[i].implicit.transpose(0, 1)

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                      dtype=torch.float32,
                                      device=z.device)
        box @= convert_matrix
        return (box, score)


class IAuxDetect(nn.Module):
    stride = None   
    export = False   
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=()):   
        super(IAuxDetect, self).__init__()
        self.nc = nc   
        self.no = nc + 5   
        self.nl = len(anchors)   
        self.na = len(anchors[0]) // 2   
        self.grid = [torch.zeros(1)] * self.nl   
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)   
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))   
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[:self.nl])   
        self.m2 = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[self.nl:])   

        self.ia = nn.ModuleList(ImplicitA(x) for x in ch[:self.nl])
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch[:self.nl])

    def forward(self, x):
         
        z = []   
        self.training |= self.export
        for i in range(self.nl):   
            x[i] = self.m[i](self.ia[i](x[i]))   
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape   
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            x[i + self.nl] = self.m2[i](x[i + self.nl])
            x[i + self.nl] = x[i + self.nl].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:   
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]   
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]   
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)   
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))   
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)   
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x[:self.nl])

    def fuseforward(self, x):
         
        z = []   
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])   
            bs, _, ny, nx = x[i].shape   
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:   
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]   
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]   
                else:
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]   
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].data   
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z,)
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out

    def fuse(self):
        print("IAuxDetect.fuse")
         
        for i in range(len(self.m)):
            with torch.no_grad():
                c1, c2, _, _ = self.m[i].weight.shape
                c1_, c2_, _, _ = self.ia[i].implicit.shape
                self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1, c2),
                                               self.ia[i].implicit.reshape(c2_, c1_)).squeeze(1)

         
        for i in range(len(self.m)):
            with torch.no_grad():
                c1, c2, _, _ = self.im[i].implicit.shape
                self.m[i].bias *= self.im[i].implicit.reshape(c2)
                self.m[i].weight *= self.im[i].implicit.transpose(0, 1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                      dtype=torch.float32,
                                      device=z.device)
        box @= convert_matrix
        return (box, score)


class TSCODE_Detect(nn.Module):
     
     
    stride = None   
    dynamic = False   
    export = False   

    def __init__(self, nc=80, anchors=(), ch=(), inplace=False):   
        super().__init__()
        self.nc = nc   
        self.no = nc + 5   
        self.nl = len(anchors)   
        self.na = len(anchors[0]) // 2   
        self.grid = [torch.empty(0) for _ in range(self.nl)]   
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]   
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))   
        self.m_sce = nn.ModuleList(SCE(ch[id:id + 2]) for id in range(1, len(ch) - 1))
        self.m_dpe = nn.ModuleList(DPE(ch[id - 1:id + 2], ch[id]) for id in range(1, len(ch) - 1))

        self.m_cls = nn.ModuleList(nn.Sequential(Conv(sum(ch[id:id + 2]), ch[id], 1), Conv(ch[id], ch[id], 3),
                                                 nn.Conv2d(ch[id], self.na * self.nc * 4, 1)) for id in
                                   range(1, len(ch) - 1))   
        self.m_reg_conf = nn.ModuleList(nn.Sequential(*[Conv(ch[id], ch[id], 3) for i in range(2)]) for id in
                                        range(1, len(ch) - 1))   
        self.m_reg = nn.ModuleList(nn.Conv2d(ch[id], self.na * 4, 1) for id in range(1, len(ch) - 1))   
        self.m_conf = nn.ModuleList(nn.Conv2d(ch[id], self.na * 1, 1) for id in range(1, len(ch) - 1))   
        self.ph, self.pw = 2, 2

        self.inplace = inplace   

    def forward(self, x_):
        x, z = [], []   
        for i, idx in enumerate(range(1, self.nl + 1)):
            bs, _, ny, nx = x_[idx].shape

            x_sce, x_dpe = self.m_sce[i](x_[idx:idx + 2]), self.m_dpe[i](x_[idx - 1:idx + 2])
            x_cls = rearrange(self.m_cls[i](x_sce), 'bs (nl ph pw nc) h w -> bs nl nc (h ph) (w pw)', nl=self.nl,
                              ph=self.ph, pw=self.pw, nc=self.nc)
            x_cls = x_cls.permute(0, 1, 3, 4, 2).contiguous()

            x_reg_conf = self.m_reg_conf[i](x_dpe)
            x_reg = self.m_reg[i](x_reg_conf).view(bs, self.na, 4, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x_conf = self.m_conf[i](x_reg_conf).view(bs, self.na, 1, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x.append(torch.cat([x_reg, x_conf, x_cls], dim=4))

            if not self.training:   
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):   
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]   
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]   
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:   
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]   
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]   
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2   
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)   
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5   
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


 
class LightweightConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=1):
        super(LightweightConvBlock, self).__init__()
         
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
                                   bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.depthwise(x)))
        out = self.relu(self.bn2(self.pointwise(out)))
        return out




class DetectYOLOv8(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False   
    export = False   
    shape = None
    anchors = torch.empty(0)   
    strides = torch.empty(0)   

    def __init__(self, nc=80, ch=()):   
        super().__init__()
        self.nc = nc   
        self.nl = len(ch)   
        self.reg_max = 16   
        self.no = nc + self.reg_max * 4   
        self.stride = torch.zeros(self.nl)   

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)   
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape   
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):   
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self   
         
         
        for a, b, s in zip(m.cv2, m.cv3, m.stride):   
            a[-1].bias.data[:] = 1.0   
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)   


class DetectYolov11(nn.Module):
    """YOLOv8 Detect head for detection models."""

    dynamic = False   
    export = False   
    end2end = False   
    max_det = 300   
    shape = None
    anchors = torch.empty(0)   
    strides = torch.empty(0)   

    def __init__(self, nc=80, ch=()):

        super().__init__()
        self.nc = nc   
        self.nl = len(ch)   
        self.reg_max = 16   
        self.no = nc + self.reg_max * 4   
        self.stride = torch.zeros(self.nl)   
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))   
         
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3),
                SEAM(c3, c3, 1, 16),
                nn.Conv2d(c3, self.nc, 1))
            for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()   

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:   
            return x
         
        y = self._inference(x)
         
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:   
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
         
        shape = x[0].shape   
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
         
         
         
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:   
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
             
             
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
         
        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self   
         
         
        for a, b, s in zip(m.cv2, m.cv3, m.stride):   
            a[-1].bias.data[:] = 1.0   
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)   
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):   
                a[-1].bias.data[:] = 1.0   
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)   

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape   
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]   
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


class DecoupledDetect(nn.Module):
    stride = None   
    onnx_dynamic = False   
    export = False   

    def __init__(self, nc=10, anchors=(), ch=(), inplace=False):   
        super().__init__()

        self.nc = nc   
        self.no = nc + 5   
        self.nl = len(anchors)   
        self.na = len(anchors[0]) // 2   
        self.grid = [torch.zeros(1)] * self.nl   
        self.anchor_grid = [torch.zeros(1)] * self.nl   
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))   
        self.m = nn.ModuleList(
            Decouple(x, self.nc, self.na) for x in ch)   
        self.inplace = False   
    def forward(self, x):
        z = []   

        for i in range(self.nl):   
            x[i] = self.m[i](x[i])   
            bs, _, ny, nx = x[i].shape   
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:   
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                xy, wh, conf = y.split((2, 2, self.nc + 1), 4)   
                xy = (xy * 2 + self.grid[i]) * self.stride[i]   
                wh = (wh * 2) ** 2 * self.anchor_grid[i]   
                y = torch.cat((xy, wh, conf), 4)

                z.append(y.view(bs, -1, self.no))


        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)


    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
         
        shape = 1, self.na, ny, nx, 2   
         
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
         
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)   
         
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5   
         
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class DecoupledDetect1(nn.Module):
    stride = None   
    onnx_dynamic = False   
    export = False   

    def __init__(self, nc=80, anchors=(), ch=(), inplace=False):   
        super().__init__()

        self.nc = nc   
        self.no = nc + 5   
        self.nl = len(anchors)   
        self.na = len(anchors[0]) // 2   
        self.grid = [torch.zeros(1)] * self.nl   
        self.anchor_grid = [torch.zeros(1)] * self.nl   
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))   
        self.m = nn.ModuleList(
            Decouple1(x, self.nc, self.na) for x in ch)   
        self.inplace = inplace   

    def forward(self, x):
        z = []   
        for i in range(self.nl):   
            x[i] = self.m[i](x[i])   
            bs, _, ny, nx = x[i].shape   
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:   
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:   
                    y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]   
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]   
                else:   
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)   
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]   
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]   
                    y = torch.cat((xy, wh, conf), 4)

                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
         
        shape = 1, self.na, ny, nx, 2   
         
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
         
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)   
         
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5   
         
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid

class Decouple(nn.Module):
     
    def __init__(self, c1, nc=80, na=3):   
        super().__init__()
        c_ = min(c1, 256)   
        self.na = na   
        self.nc = nc   
        self.a = Conv(c1, c_, 1)   
         
        c = [int(x + na * 5) for x in (c_ - na * 5) * torch.linspace(1, 0, 4)]   
         
         
         
         
        self.b1 = Conv(c_, c[1], 3)
        self.b2 = Conv(c[1], c[2], 3)
        self.b3 = nn.Conv2d(c[2], na * 5, 1)
         
         
        self.c1 = Conv(c_, c_, 1)
        self.c2 = Conv(c_, c_, 1)
        self.c3 = nn.Conv2d(c_, na * nc, 1)

    def forward(self, x):
        bs, nc, ny, nx = x.shape   
        x = self.a(x)
         
        b = self.b3(self.b2(self.b1(x)))
         
        c = self.c3(self.c2(self.c1(x)))
         
        return torch.cat((b.view(bs, self.na, 5, ny, nx), c.view(bs, self.na, self.nc, ny, nx)), 2).view(bs, -1, ny, nx)


class Decouple1(nn.Module):
     
    def __init__(self, c1, nc=80, na=3):   
        super().__init__()
        c_ = min(c1, 256)   
        self.na = na   
        self.nc = nc   
        self.a = Conv(c1, c_, 1)   
        c = [int(x + na * 5) for x in (c_ - na * 5) * torch.linspace(1, 0, 4)]   
        self.b1 = Conv(c_, c[1], 3)
        self.b2 = Conv(c[1], c[2], 3)
        self.reg_preds = nn.Conv2d(c[2], 4 * self.na, 1)   
        self.obj_preds = nn.Conv2d(c[2], 1 * self.na, 1)   
        self.c1 = Conv(c_, c_, 1)
        self.c2 = Conv(c_, c_, 1)
        self.c3 = nn.Conv2d(c_, na * nc, 1)

    def forward(self, x):
        bs, nc, ny, nx = x.shape   
        x = self.a(x)
        x1 = self.b2(self.b1(x))
        b_reg = self.reg_preds(x1)
        b_obj = self.obj_preds(x1)
        c = self.c3(self.c2(self.c1(x)))
        return torch.cat((b_reg.view(bs, self.na, 4, ny, nx), b_obj.view(bs, self.na, 1, ny, nx),
                          c.view(bs, self.na, self.nc, ny, nx)), 2).view(bs, -1, ny, nx)

class Decoupled_Detect(nn.Module):
     
    stride = None   
    dynamic = False   
    export = False   

    def __init__(self, nc=80, anchors=(), ch=(), inplace=False):   
        super().__init__()
        self.nc = nc   
        self.no = nc + 5   
        self.nl = len(anchors)   
        self.na = len(anchors[0]) // 2   
        self.grid = [torch.empty(0) for _ in range(self.nl)]   
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]   
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))   

        self.m_stem = nn.ModuleList(Conv(x, x, 1) for x in ch)   
        self.m_cls = nn.ModuleList(nn.Sequential(Conv(x, x, 3), nn.Conv2d(x, self.na * self.nc, 1)) for x in ch)   

        self.m_reg_conf = nn.ModuleList(Conv(x, x, 3) for x in ch)   
        self.m_reg = nn.ModuleList(nn.Conv2d(x, self.na * 4, 1) for x in ch)   
        self.m_conf = nn.ModuleList(nn.Conv2d(x, self.na * 1, 1) for x in ch)   

        self.inplace = inplace   

    def forward(self, x):
        z = []   
        for i in range(self.nl):
            x[i] = self.m_stem[i](x[i])   

            bs, _, ny, nx = x[i].shape
            x_cls = self.m_cls[i](x[i]).view(bs, self.na, self.nc, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x_reg_conf = self.m_reg_conf[i](x[i])
            x_reg = self.m_reg[i](x_reg_conf).view(bs, self.na, 4, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x_conf = self.m_conf[i](x_reg_conf).view(bs, self.na, 1, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x[i] = torch.cat([x_reg, x_conf, x_cls], dim=4)

            if not self.training:   
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]   
                wh = (wh * 2) ** 2 * self.anchor_grid[i]   
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))
        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2   
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)   
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5   
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


 


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):   
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg   
        else:   
            import yaml   
            self.yaml_file = Path(cfg).name
             
            with open(cfg, errors='ignore') as f:
                 
                self.yaml = yaml.safe_load(f)   

         
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)   
         
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc   
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)   
         
         
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])   
         
        self.names = [str(i) for i in range(self.yaml['nc'])]   
         
        self.inplace = self.yaml.get('inplace', False)
         
         
        m = self.model[-1]   
        if isinstance(m, (Detect, TSCODE_Detect, CLLADetect, ASFF_Detect, IDetect,Decoupled_Detect)):
            s = 256   
            m.inplace = self.inplace
             
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s), visualize=False)])   
             
            m.anchors /= m.stride.view(-1, 1, 1)
             
            check_anchor_order(m)
            self.stride = m.stride
             
            self._initialize_biases()   
        elif isinstance(m, (DecoupledDetect,DecoupledDetect1)):
            s = 256   
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])   
            check_anchor_order(m)   
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_dh_biases()   
        if isinstance(m, DetectODConv):
            s = 256   
            m.inplace = self.inplace
             
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s), visualize=False)])   
             
            m.anchors /= m.stride.view(-1, 1, 1)
             
            check_anchor_order(m)
            self.stride = m.stride
        elif isinstance(m, IAuxDetect):
            s = 256   
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))[:4]])   
             
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_aux_biases()   
             

         
         
        initialize_weights(self)
         
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False,save_dir='runs/detect/exp'):
         
         
        if augment:
            return self._forward_augment(x)   
         
        return self._forward_once(x, profile, visualize,save_dir)   

    def _forward_augment(self, x):
        img_size = x.shape[-2:]   
        s = [1, 1, 0.83, 0.83, 0.67, 0.67]   
        f = [None, 3, None, 3, None, 3]   
        y = []   
        for si, fi in zip(s, f):
             
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]   
             
             
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)   
        return torch.cat(y, 1), None   

    def _forward_once(self, x, profile=False, visualize=False,save_dir='runs/detect/exp'):
        y, dt = [], []   
        for m in self.model:
            if m.f != -1:   
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
             
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, nn.Upsample):
                m.recompute_scale_factor = False
             
            if isinstance(m, ODConv_3rd):
                x = x.contiguous()
            x = m(x)   
             
            y.append(x if m.i in self.save else None)   

            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=Path(save_dir))


        return x

    def _descale_pred(self, p, flips, scale, img_size):
        if self.inplace:
            p[..., :4] /= scale   
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]   
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]   
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale   
            if flips == 2:
                y = img_size[0] - y   
            elif flips == 3:
                x = img_size[1] - x   
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
         
        nl = self.model[-1].nl   
        g = sum(4 ** x for x in range(nl))   
        e = 1   
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))   
        y[0] = y[0][:, :-i]   
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))   
        y[-1] = y[-1][:, i:]   
        return y

    def _profile_one_layer(self, m, x, dt):
         
        c = isinstance(m,
                       (Detect, DetectODConv, TSCODE_Detect, ASFF_Detect))   
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0   
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_dh_biases(self, cf=None):   
         
         
        m = self.model[-1]   
        if isinstance(m, DecoupledDetect):
            for mi, s in zip(m.m, m.stride):   
                b = mi.b3.bias.view(m.na, -1)
                b.data[:, 4] += math.log(8 / (640 / s) ** 2)   
                mi.b3.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
                b = mi.c3.bias.data
                b += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())   
                mi.c3.bias = torch.nn.Parameter(b, requires_grad=True)
        elif isinstance(m, DecoupledDetect1):
            for mi, s in zip(m.m, m.stride):   
                b = mi.obj_preds.bias.view(m.na, -1)
                b.data += math.log(8 / (640 / s) ** 2)   
                mi.obj_preds.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

                b1 = mi.c3.bias.view(m.na, -1)
                b1.data += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())   
                mi.c3.bias = torch.nn.Parameter(b1.view(-1), requires_grad=True)

    def _initialize_biases(self, cf=None):   
         
         
        m = self.model[-1]   

        if isinstance(m, (Detect, DetectODConv)) or isinstance(m, ASFF_Detect):
            for mi, s in zip(m.m, m.stride):   
                b = mi.bias.view(m.na, -1)   
                b.data[:, 4] += math.log(8 / (640 / s) ** 2)   
                b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())   
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        elif isinstance(m, TSCODE_Detect):
            for mi, s in zip(m.m_conf, m.stride):   
                b = mi.bias.view(m.na, -1)   
                b.data += math.log(8 / (640 / s) ** 2)   
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

            for mi, s in zip(m.m_cls, m.stride):   
                b = mi[-1].bias.view(m.na, -1)   
                b.data += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())   
                mi[-1].bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        elif isinstance(m, Decoupled_Detect):
            for mi, s in zip(m.m_conf, m.stride):   
                b = mi.bias.view(m.na, -1)   
                b.data += math.log(8 / (640 / s) ** 2)   
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

            for mi, s in zip(m.m_cls, m.stride):   
                b = mi[-1].bias.view(m.na, -1)   
                b.data += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())   
                mi[-1].bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        else:
            for mi, s in zip(m.m, m.stride[1:]):   
                b = mi.bias.view(m.na, -1)   
                b.data[:, 4] += math.log(8 / (640 / s) ** 2)   
                b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())   
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            b = m.det.det.bias.view(m.na, -1)   
            b.data[:, 4] += math.log(8 / (640 / m.stride[0]) ** 2)   
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())   
            m.det.det.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
         
        m = self.model[-1]   
        if getattr(m.m, 'bias', False):
            for mi in m.m:   
                b = mi.bias.detach().view(m.na, -1).T   
                LOGGER.info(
                    ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))
        else:
            for mi in m.m:   
                b1 = mi.cls.bias.detach().view(m.na, -1).T   
                b2 = mi.bbox.bias.detach().view(m.na, -1).T   
                LOGGER.info(
                    ('%6g Conv2d.bias and %6g Conv2d.bias:' + '%10.3g' * 6) % (
                        mi.bbox.weight.shape[1], mi.cls.weight.shape[1], *b2[:].mean(1).tolist(), b1[:].mean()))
    def fuse(self):
        """用在detect.py、val.py
                fuse model Conv2d() + BatchNorm2d() layers
                调用torch_utils.py中的fuse_conv_and_bn函数和common.py中Conv模块的fuseforward函数
                """
        LOGGER.info('Fusing layers... ')
         
        for m in self.model.modules():
             
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)   
                delattr(m, 'bn')   
                 
                m.forward = m.forward_fuse   
        self.info()   
        return self

    def autoshape(self):   
         
        LOGGER.info('Adding AutoShape... ')
        m = AutoShape(self)   
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())   
        return m

    def info(self, verbose=False, img_size=640):   
         
        model_info(self, verbose, img_size)

    def _apply(self, fn):
         
        self = super()._apply(fn)
        m = self.model[-1]   
        if isinstance(m, (Detect, DetectODConv, ASFF_Detect, DecoupledDetect,Decoupled_Detect,DecoupledDetect1)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


def parse_model(d, ch):   
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
     
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
     
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors   
     
    no = na * (nc + 5)   
    layers, save, c2 = [], [], ch[-1]
     
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):   
         
        m = eval(m) if isinstance(m, str) else m   
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a   
            except NameError:
                pass
        n = n_ = max(round(n * gd), 1) if n > 1 else n
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Conv_SWS,
                 MixConv2d, Focus, CrossConv, SCDown, PSA, C3_CBAMS, C3_CBAMS_DWC, GSConv,
                 C3_CBAM_DWC, CoordConv, SimConv, CoordConvd, C2f_DWR, VoVGSCSPCBAM,
                 PSContextAggregation, ContextAggregation, SEAM, SPPF_improve, MultiSEAM,
                 BottleneckCSP, C3, C3TR, C3STR, C3SPP, C3Ghost, ASPP, C3CBAM, BAM, nn.ConvTranspose2d,
                 DWConvTranspose2d, C3_CBAM, C3_BAM, C3_CA, C3_SCBAM, C3GAM, C3CPCA, C2fCIB,
                 C3CR, C2f, C2fCBAM, C2fBAM, SPPELAN, SPPCSPC, SPPCSPCS, se_block,
                 RepVGGBlock, ADown, C3x, RepC3, Conv2Former, Involution, BasicRFB, BasicRFB_a]:
            c1, c2 = ch[f], args[0]
             
            if c2 != no:   
                 
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
             
            if m in [BottleneckCSP, C3, C3TR, C3STR, C3Ghost, C3CR,
                     C2f, C2fBAM, C2fCBAM, Conv2Former, C2fCIB,
                     SPPCSPC, SPPCSPCS, C3x, RepC3, C3_CBAM, C3_CBAMS, C2f_DWR, VoVGSCSPCBAM,
                     C3_BAM, C3_SCBAM, C3GAM, C3CPCA, C3_CBAMS_DWC, C3_CBAM_DWC]:
                args.insert(2, n)   
                n = 1   
        elif m is nn.BatchNorm2d:
             
            args = [ch[f]]
        elif m in [ACmix]:
             
            c1, c2 = ch[f], args[0]
            if c2 != no:   
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
        elif m in [SimAM]:
            args = [*args[:]]
        elif m is SimAMWithSlicing:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, *args[1:]]
        elif m is SimAMWithFlexibleSlicing:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, *args[1:]]
        elif m is SDI:
            args = [[ch[x] for x in f]]
        elif m in [ODConv_3rd, ODConv]:
             
             
            c1, c2 = ch[f], args[0]
            if c2 != no:   
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
        elif m is CNeB:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m is CNeB:
                args.insert(2, n)
                n = 1
        elif m in [ConvMix, CSPCM]:
            c1, c2 = ch[f], args[0]
            if c2 != no:   
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
        elif m is DownSimper:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2]
        elif m is CARAFE:
             
            c2 = ch[f]
            args = [c2, *args]
         
        elif m in [BiFPN_Add2, BiFPN_Add3]:
            c2 = max([ch[x] for x in f])
        elif m in {BiFPN}:
            length = len([ch[x] for x in f])
            args = [length]
        elif m in {BiFPNSDI}:
            length = len([ch[x] for x in f])
            c2 = args[0]
            args = [c2, [ch[x] for x in f], length]
        elif m in {BiFPNs}:
            c1 = args[0]   
            c2 = args[1]
        elif m is CAM:
             
            c1, c2 = ch[f], (ch[f] * 3 if args[0] == 'concat' else ch[f])
            args = [c1, args[0]]
        elif m is BAM:
            args = [ch[f]]
        elif m in [S2Attention]:
            c1 = ch[f]
            args = [c1]
        elif m in [CPCA]:
            c1 = ch[f]
            args = [c1]
        elif m in [NAMAttention]:
            c1 = ch[f]
            args = [c1]
        elif m in [DySample]:
            args.insert(0, ch[f])
        elif m is BiFusion:
            c2 = args[3]
            print(c2)
        elif m is Zoom_cat:
            c2 = 3 * args[0]
        elif m is attention_model:
            c2 = args[0]
        elif m is ScalSeq:
            c2 = args[0]
        elif m is SF:
            c2 = sum(ch[x] for x in f)
            print(c2)
        elif m in [SKAttention]:   
            c2 = ch[f]
            args = [c2, *args[0:]]
        elif m is Concat:
             
            c2 = sum(ch[x] for x in f)
        elif m is CShortcut:
            c2 = sum(ch[x] for x in f)
        elif m is space_to_depth:
            c2 = 4 * ch[f]
             
             
             
        elif m is SPD:
            c2 = 4 * ch[f]
             
             
         
         
         
        elif m in {Detect, DetectODConv}:   
             
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):   
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is ASFF_Detect:   
             
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):   
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m in {IDetect, IAuxDetect, DecoupledDetect,DecoupledDetect1,Decoupled_Detect}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):   
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)
        elif m is CLLADetect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):   
                args[1] = [list(range(args[1] * 2))] * (len(f) - 1)
        elif m in {TSCODE_Detect}:
            args.append([ch[x] for x in f])   
            if isinstance(args[1], int):   
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is CoorAttention:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is eca_block:
            args = [*args[:]]
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        elif m is ChannelAttention_HSFPN:
            c2 = ch[f]
            args = [c2, *args]
        elif m is Multiply:
            c2 = ch[f[0]]
        elif m is Add:
            c2 = ch[f[-1]]
        else:
            c2 = ch[f]
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)   
         
        t = str(m)[8:-2].replace('__main__.', '')   
        np = sum(x.numel() for x in m_.parameters())   
        m_.i, m_.f, m_.type, m_.np = i, f, t, np   
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')   
         
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)   
         
        layers.append(m_)
        if i == 0:
            ch = []   
         
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)   
    print_args(FILE.stem, opt)
    device = select_device(opt.device)

     
    model = Model(opt.cfg).to(device)
    model.train()

     
     
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_Savailable() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

     
    from torch.utils.tensorboard import SummaryWriter

    tb_writer = SummaryWriter('.')
    LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])   
