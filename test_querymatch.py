# import os
# os.environ["CUDA_DEVICES"]="1"
from utils_querymatch.distributed import *
import torch.multiprocessing as mp
from utils_querymatch.ckpt import *
from torch.nn.parallel import DistributedDataParallel as DDP
from utils_querymatch.logging import *
import argparse
import time
from utils_querymatch import config
from datasets_querymatch.dataloader import loader,RefCOCODataSet
from tensorboardX import SummaryWriter
from utils_querymatch.utils import *
import torch.optim as Optim
from importlib import import_module
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
import cv2
import random


def generate_random_color():
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)
    return red, green, blue


def draw_masks_fromList(image, masks_generated) :
    masked_image = image.copy()
    for i in range(len(masks_generated)) :
        random_color = generate_random_color()
        masked_image = np.where(np.repeat(masks_generated[i][:, :, np.newaxis], 3, axis=2),
                                np.asarray(random_color, dtype='uint8'),
                                masked_image)

        masked_image = masked_image.astype(np.uint8)

    return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)


class ModelLoader:
    def __init__(self, __C):

        self.model_use = __C.MODEL
        model_module_path = 'models_querymatch.' + self.model_use + '.net'
        self.model_module = import_module(model_module_path)

    def Net(self, __arg1, __arg2, __arg3,cfg):
        return self.model_module.Net(__arg1, __arg2, __arg3,cfg)


def validate(__C,
             net,
             loader,
             writer,
             epoch,
             rank,
             ix_to_token,
             save_ids=None,
             prefix='Val',
             ema=None,
             whether_test=False):
    if ema is not None:
        ema.apply_shadow()
    net.eval()
    input_shape = __C.INPUT_SHAPE
    batches = len(loader)
    batch_time = AverageMeter('Time', ':6.5f')
    data_time = AverageMeter('Data', ':6.5f')
    losses = AverageMeter('Loss', ':.4f')
    mask_ap = AverageMeter('MaskIoU', ':6.2f')
    inconsistency_error = AverageMeter('IE', ':6.2f')
    mask_aps={}
    for item in np.arange(0.5, 1, 0.05):
        mask_aps[item]=[]
    meters = [batch_time, data_time, losses, mask_ap, inconsistency_error]
    meters_dict = {meter.name: meter for meter in meters}
    progress = ProgressMeter(__C.VERSION, __C.EPOCHS, len(loader), meters, prefix=prefix+': ')
    with th.no_grad():
        end = time.time()
        for ith_batch, data in enumerate(loader):
            idx, ref_iter, image_iter, mask_iter, box_iter,gt_box_iter, mask_id, info_iter, padded_image = data
            ref_iter = ref_iter.cuda( non_blocking=True)
            image_iter = image_iter.cuda( non_blocking=True)
            box_iter = box_iter.cuda( non_blocking=True)
            mask, all_masks= net(image_iter, ref_iter, input_shape)
            padded_image = padded_image.cpu().numpy()
            all_masks = all_masks.cpu().numpy()
            info_iter=info_iter.cpu().numpy()

            #predictions to gt
            seg_iou=[]
            mask=mask.cpu().numpy()
            for i, mask_pred in enumerate(mask):
                mask_gt=np.load(os.path.join(__C.MASK_PATH[__C.DATASET],'%d.npy'%mask_id[i]))
                mask_pred=mask_processing(mask_pred,info_iter[i])
                if writer is not None:
                    # view gt masks and pred masks
                    if whether_test:
                        writer.add_image('image/' + str(ith_batch * __C.BATCH_SIZE + i) + '_orig', (padded_image[i]).astype(np.uint8), dataformats='HWC')
                        masked_image = draw_masks_fromList(padded_image[i],all_masks[i])
                        writer.add_image('image/' + str(ith_batch * __C.BATCH_SIZE + i) + '_allmasks', (masked_image).astype(np.uint8), dataformats='HWC')
                        ixs=ref_iter[i].cpu().numpy()
                        words=[]
                        for ix in ixs:
                            if ix >0:
                                words.append(ix_to_token[ix])
                        sent=' '.join(words)
                        writer.add_text('image/' + str(ith_batch * __C.BATCH_SIZE + i) + '_text',str(sent))
                    writer.add_image('image/' + str(ith_batch * __C.BATCH_SIZE + i) + '_gt-seg', (mask_gt[None]*255).astype(np.uint8))
                    writer.add_image('image/' + str(ith_batch * __C.BATCH_SIZE + i) + '_pred-seg', (mask_pred[None]*255).astype(np.uint8))

                single_seg_iou,single_seg_ap=mask_iou(mask_gt,mask_pred)
                for item in np.arange(0.5, 1, 0.05):
                    mask_aps[item].append(single_seg_ap[item]*100.)
                seg_iou.append(single_seg_iou)
            seg_iou=np.array(seg_iou).astype(np.float32)

            mask_ap.update(seg_iou.mean()*100., seg_iou.shape[0])

            reduce_meters(meters_dict, rank, __C)
            if (ith_batch % __C.PRINT_FREQ == 0 or ith_batch==(len(loader)-1)) and main_process(__C,rank):
                progress.display(epoch, ith_batch)
            batch_time.update(time.time() - end)
            end = time.time()

        if main_process(__C,rank) and writer is not None:
            writer.add_scalar("Acc/MaskIoU", mask_ap.avg_reduce, global_step=epoch)
            for item in mask_aps:
                writer.add_scalar("Acc/MaskIoU@%.2f"%item, np.array(mask_aps[item]).mean(), global_step=epoch)
    if ema is not None:
        ema.restore()
    return mask_ap.avg_reduce


def main_worker(gpu,__C,cfg):
    global best_det_acc,best_seg_acc
    best_det_acc,best_seg_acc=0.,0.
    if __C.MULTIPROCESSING_DISTRIBUTED:
        if __C.DIST_URL == "env://" and __C.RANK == -1:
            __C.RANK = int(os.environ["RANK"])
        if __C.MULTIPROCESSING_DISTRIBUTED:
            __C.RANK = __C.RANK* len(__C.GPU) + gpu
        dist.init_process_group(backend=dist.Backend('NCCL'), init_method=__C.DIST_URL, world_size=__C.WORLD_SIZE, rank=__C.RANK)

    train_set=RefCOCODataSet(__C,cfg,split='train')
    # train_loader=loader(__C,train_set,gpu,shuffle=(not __C.MULTIPROCESSING_DISTRIBUTED))
    loaders=[]
    prefixs=['val']
    val_set=RefCOCODataSet(__C,cfg,split='val')
    val_loader=loader(__C,val_set,gpu,shuffle=False, drop_last=False)
    loaders.append(val_loader)
    if __C.DATASET=='refcoco' or __C.DATASET=='refcoco+':
        testA=RefCOCODataSet(__C,cfg,split='testA')
        testA_loader=loader(__C,testA,gpu,shuffle=False, drop_last=False)
        testB=RefCOCODataSet(__C,cfg,split='testB')
        testB_loader=loader(__C,testB,gpu,shuffle=False, drop_last=False)
        prefixs.extend(['testA','testB'])
        loaders.extend([testA_loader,testB_loader])
    elif __C.DATASET=='referit':
        test=RefCOCODataSet(__C,cfg,split='test')
        test_loader=loader(__C,test,gpu,shuffle=False, drop_last=False)
        prefixs.append('test')
        loaders.append(test_loader)

    net= ModelLoader(__C).Net(
        __C,
        train_set.pretrained_emb,
        train_set.token_size,
        cfg
    )

    #optimizer
    std_optim = getattr(Optim, __C.OPT)
    params = filter(lambda p: p.requires_grad, net.parameters()) 
    eval_str = 'params, lr=%f'%__C.LR
    for key in __C.OPT_PARAMS:
        eval_str += ' ,' + key + '=' + str(__C.OPT_PARAMS[key])
    optimizer=eval('std_optim' + '(' + eval_str + ')')


    if __C.MULTIPROCESSING_DISTRIBUTED:
        torch.cuda.set_device(gpu)
        net = DDP(net.cuda(), device_ids=[gpu],find_unused_parameters=True)
    elif len(gpu)==1:
        net.cuda()
    else:
        net = DP(net.cuda())
    if main_process(__C, gpu):
        print(__C)
        total = sum([param.nelement() for param in net.parameters()])
        print('  + Number of all params: %.2fM' % (total / 1e6))  
        total = sum([param.nelement() for param in net.parameters() if param.requires_grad])
        print('  + Number of trainable params: %.2fM' % (total / 1e6)) 


    if os.path.isfile(__C.RESUME_PATH):
        checkpoint = torch.load(__C.RESUME_PATH,map_location=lambda storage, loc: storage.cuda() )
        new_dict = {}
        for k in checkpoint['state_dict']:
            if 'module.' in k:
                new_k = k.replace('module.', '')
                new_dict[new_k] = checkpoint['state_dict'][k]
        if len(new_dict.keys()) == 0:
            new_dict = checkpoint['state_dict']
        net.load_state_dict(new_dict,strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])

        if main_process(__C,gpu):
            print("==> loaded checkpoint from {}\n".format(__C.RESUME_PATH) +
                  "==> epoch: {} lr: {} ".format(checkpoint['epoch'],checkpoint['lr']))

    if __C.AMP:
        assert th.__version__ >= '1.6.0', \
            "Automatic Mixed Precision training only supported in PyTorch-1.6.0 or higher"
        scalar = th.cuda.amp.GradScaler()
    else:
        scalar = None

    if main_process(__C,gpu):
        writer = SummaryWriter(log_dir=os.path.join(__C.LOG_PATH,str(__C.VERSION)))
    else:
        writer = None

    save_ids=np.random.randint(1, len(val_loader) * __C.BATCH_SIZE, 100) if __C.LOG_IMAGE else None
    for loader_,prefix_ in zip(loaders,prefixs):
        print()
        mask_ap=validate(__C,net,loader_,writer,0,gpu,val_set.ix_to_token,save_ids=save_ids,prefix=prefix_,whether_test=False)
        print(mask_ap)


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
    parser = argparse.ArgumentParser(description="QueryMatch")
    parser.add_argument('--config', type=str, default='config/refcoco.yaml')
    parser.add_argument('--eval-weights', type=str, default='')
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
    __C.RESUME_PATH=args.eval_weights
    if not os.path.exists(os.path.join(__C.LOG_PATH,str(__C.VERSION))):
        os.makedirs(os.path.join(__C.LOG_PATH,str(__C.VERSION),'ckpt'),exist_ok=True)

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
        main_worker(__C.GPU,__C,cfg)


if __name__ == '__main__':
    main()