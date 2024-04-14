from utils_querymatch.distributed import *
import torch.multiprocessing as mp
from utils_querymatch.ckpt import *
from torch.nn.parallel import DistributedDataParallel as DDP
from utils_querymatch.logging import *
import argparse
import time
from utils_querymatch import config
from datasets_querymatch.dataloader import loader, RefCOCODataSet
from tensorboardX import SummaryWriter
from utils_querymatch.utils import *
from importlib import import_module
import torch.nn.functional as F
import torch.optim as Optim
from test_querymatch import validate
from utils_querymatch.utils import EMA
import torch.nn as nn
from torch.utils.data import Subset
import sys
sys.path.append('../')  
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
import detectron2.utils.comm as comm
import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")
from functools import partial
import numpy as np


class ModelLoader:
    def __init__(self, __C):
        self.model_use = __C.MODEL
        model_module_path = 'models_querymatch.' + self.model_use + '.net'
        self.model_module = import_module(model_module_path)

    def Net(self, __arg1, __arg2, __arg3, cfg):
        return self.model_module.Net(__arg1, __arg2, __arg3, cfg)


def train_one_epoch(__C,
                    net,
                    optimizer,
                    scheduler,
                    loader,
                    scalar,
                    writer,
                    epoch,
                    rank,
                    ema=None):
    net.train()
    if __C.MULTIPROCESSING_DISTRIBUTED:
        loader.sampler.set_epoch(epoch)
    batches = len(loader)
    batch_time = AverageMeter('Time', ':6.5f')
    data_time = AverageMeter('Data', ':6.5f')
    losses = AverageMeter('Loss', ':.4f')
    lr = AverageMeter('lr', ':.5f')
    meters = [batch_time, data_time, losses, lr]
    meters_dict = {meter.name: meter for meter in meters}
    progress = ProgressMeter(__C.VERSION, __C.EPOCHS, len(loader), meters, prefix='Train: ')
    end = time.time()
    input_shape = __C.INPUT_SHAPE
    stat_sim_dict = {
                "pos_sim_mean": 0,
                "neg_sim_top1_mean": 0, 
                "sim_hq_mean": 0,
                "num":0,
            }
    for ith_batch, data in enumerate(loader):
        data_time.update(time.time() - end)
        idx, ref_iter,image_iter,mask_iter,box_iter,gt_box_iter,mask_id,info_iter,padded_images = data
        ref_iter = ref_iter.cuda(non_blocking=True)
        image_iter = image_iter.cuda(non_blocking=True)
        mask_iter = mask_iter.cuda(non_blocking=True)
        box_iter = box_iter.cuda( non_blocking=True)
        if scalar is not None:
            with th.cuda.amp.autocast():
                loss, stat_sim_dict = net(image_iter, ref_iter, input_shape, None, stat_sim_dict)  # , idx
        else:
            loss, stat_sim_dict = net(image_iter, ref_iter, input_shape, None, stat_sim_dict)  # , idx
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        if scalar is not None:
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            if __C.GRAD_NORM_CLIP > 0:
                nn.utils.clip_grad_norm_(
                    net.parameters(),
                    __C.GRAD_NORM_CLIP
                )
            scalar.update()
        else:
            loss.backward()
            if __C.GRAD_NORM_CLIP > 0:
                nn.utils.clip_grad_norm_(
                    net.parameters(),
                    __C.GRAD_NORM_CLIP
                )
            optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update_params()
        losses.update(loss.item(), image_iter.size(0))
        lr.update(optimizer.param_groups[0]["lr"], -1)

        reduce_meters(meters_dict, rank, __C)
        if main_process(__C, rank):
            global_step = epoch * batches + ith_batch
            writer.add_scalar("loss/train", losses.avg_reduce, global_step=global_step)
            if ith_batch % __C.PRINT_FREQ == 0 or ith_batch == len(loader):
                progress.display(epoch, ith_batch)
                try:
                    print(" "*4+"pos_sim:{:.3f}, neg_top1:{:.3f}, sim_hq_mean:{:.3f}".format(
                        stat_sim_dict["pos_sim_mean"]/stat_sim_dict['num'],
                        stat_sim_dict['neg_sim_top1_mean']/stat_sim_dict['num'],
                        stat_sim_dict['sim_hq_mean']/stat_sim_dict['num'],
                    ))
                except Exception as err:
                    print(err)
        batch_time.update(time.time() - end)
        end = time.time()
    torch.cuda.empty_cache()


def main_worker(gpu, __C, cfg):
    global best_det_acc, best_seg_acc
    best_det_acc, best_seg_acc = 0., 0.
    if __C.MULTIPROCESSING_DISTRIBUTED:
        if __C.DIST_URL == "env://" and __C.RANK == -1:
            __C.RANK = int(os.environ["RANK"])
        if __C.MULTIPROCESSING_DISTRIBUTED:
            __C.RANK = __C.RANK * len(__C.GPU) + gpu
        dist.init_process_group(backend=dist.Backend('NCCL'), init_method=__C.DIST_URL, world_size=__C.WORLD_SIZE,
                                rank=__C.RANK)
    train_set = RefCOCODataSet(__C, cfg, split='train')
    train_loader = loader(__C, train_set, gpu, shuffle=(not __C.MULTIPROCESSING_DISTRIBUTED), drop_last=True)  # , collate_fn=custom_collate_fn

    val_set = RefCOCODataSet(__C, cfg, split='val')
    val_loader = loader(__C, val_set, gpu, shuffle=False)

    net = ModelLoader(__C).Net(
        __C,
        train_set.pretrained_emb,
        train_set.token_size,
        cfg
    )
    # optimizer
    params = filter(lambda p: p.requires_grad, net.parameters()) 
    std_optim = getattr(Optim, __C.OPT)

    eval_str = 'params, lr=%f' % __C.LR
    for key in __C.OPT_PARAMS:
        eval_str += ' ,' + key + '=' + str(__C.OPT_PARAMS[key])
    optimizer = eval('std_optim' + '(' + eval_str + ')')

    ema = None

    if __C.MULTIPROCESSING_DISTRIBUTED:
        torch.cuda.set_device(gpu)
        net = DDP(net.cuda(), device_ids=[gpu], find_unused_parameters=True)
    elif len(gpu) == 1:
        net.cuda()
    else:
        net = DP(net.cuda())


    if main_process(__C, gpu):
        print(__C)
        # print(net)
        total = sum([param.nelement() for param in net.parameters()])
        print('  + Number of all params: %.2fM' % (total / 1e6))  
        total = sum([param.nelement() for param in net.parameters() if param.requires_grad])
        print('  + Number of trainable params: %.2fM' % (total / 1e6))  

    scheduler = get_lr_scheduler(__C, optimizer, len(train_loader))

    start_epoch = 0

    if os.path.isfile(__C.RESUME_PATH):
        net.cuda()
        checkpoint = torch.load(__C.RESUME_PATH, map_location=lambda storage, loc: storage.cuda())
        new_dict = {}
        for k in checkpoint['state_dict']:
            if 'module.' in k:
                new_k = k.replace('module.', '')
                new_dict[new_k] = checkpoint['state_dict'][k]
        if len(new_dict.keys()) == 0:
            new_dict = checkpoint['state_dict']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        if main_process(__C, gpu):
            print("==> loaded checkpoint from {}\n".format(__C.RESUME_PATH) +
                  "==> epoch: {} lr: {} ".format(checkpoint['epoch'], checkpoint['lr']))

    if __C.AMP:
        assert th.__version__ >= '1.6.0', \
            "Automatic Mixed Precision training only supported in PyTorch-1.6.0 or higher"
        scalar = th.cuda.amp.GradScaler()
    else:
        scalar = None

    if main_process(__C, gpu):
        writer = SummaryWriter(log_dir=os.path.join(__C.LOG_PATH, str(__C.VERSION)))
    else:
        writer = None

    save_ids = np.random.randint(1, len(val_loader) * __C.BATCH_SIZE, 100) if __C.LOG_IMAGE else None
    for ith_epoch in range(start_epoch, __C.EPOCHS):
        if __C.USE_EMA and ema is None:
            ema = EMA(net, 0.9997)
        train_one_epoch(__C, net, optimizer, scheduler, train_loader, scalar, writer, ith_epoch, gpu, ema)

        _, mask_ap=validate(__C,net,val_loader,writer,ith_epoch,gpu,val_set.ix_to_token,save_ids=save_ids,ema=ema)

        if main_process(__C, gpu):
            if ema is not None:
                ema.apply_shadow()
            torch.save({'epoch': ith_epoch + 1, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'lr': optimizer.param_groups[0]["lr"], },
                       os.path.join(__C.LOG_PATH, str(__C.VERSION), 'ckpt', 'last.pth'))
            if mask_ap>best_seg_acc:
                best_seg_acc=mask_ap
                torch.save({'epoch': ith_epoch + 1, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),'lr':optimizer.param_groups[0]["lr"],},
                           os.path.join(__C.LOG_PATH, str(__C.VERSION),'ckpt', 'seg_best.pth'))
                
            if ema is not None:
                ema.restore()
    if __C.MULTIPROCESSING_DISTRIBUTED:
        cleanup_distributed()


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, default='./config/refcoco.yaml')
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
        Modify config options at the end of the command. For Yacs configs, use
        space-separated "PATH.KEY VALUE" pairs.
        For python-based LazyConfig, use "path.key=value".
                """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    assert args.config is not None
    __C = config.load_cfg_from_cfg_file(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in __C.GPU)
    setup_unique_version(__C)
    seed_everything(__C.SEED)
    N_GPU = len(__C.GPU)

    if not os.path.exists(os.path.join(__C.LOG_PATH, str(__C.VERSION))):
        os.makedirs(os.path.join(__C.LOG_PATH, str(__C.VERSION), 'ckpt'), exist_ok=True)

    if N_GPU == 1:
        __C.MULTIPROCESSING_DISTRIBUTED = False
    else:
        # turn on single or multi node multi gpus training
        __C.MULTIPROCESSING_DISTRIBUTED = True
        __C.WORLD_SIZE *= N_GPU
        __C.DIST_URL = f"tcp://127.0.0.1:{find_free_port()}"
    cfg = setup(args)
    if __C.MULTIPROCESSING_DISTRIBUTED:
        mp.spawn(main_worker, args=(__C,cfg), nprocs=N_GPU, join=True)
    else:
        main_worker(__C.GPU, __C, cfg)


if __name__ == '__main__':
    main()
