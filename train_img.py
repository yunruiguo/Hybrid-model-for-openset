import argparse
import time
import math
import os
import os.path
import numpy as np
from tqdm import tqdm
import gc
from cifar_dataset import CIFARDataset
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score
from lib.resflow import ACT_FNS, ResidualFlow
import lib.datasets as datasets
import lib.optimizers as optim
import lib.utils as utils
import lib.layers as layers
import lib.layers.base as base_layers
from lib.lr_scheduler import CosineAnnealingWarmRestarts
import pickle
# Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--data', type=str, default='cifar10', choices=[
        'mnist',
        'cifar10',
        'svhn',
    ]
)
parser.add_argument('--dataroot', type=str, default='../data')
parser.add_argument('--split', type=str, default='split0')
parser.add_argument('--imagesize', type=int, default=32)
parser.add_argument('--nbits', type=int, default=8)  # Only used for celebahq.

parser.add_argument('--block', type=str, choices=['resblock', 'coupling'], default='resblock')

parser.add_argument('--coeff', type=float, default=0.98)
parser.add_argument('--vnorms', type=str, default='2222')
parser.add_argument('--n-lipschitz-iters', type=int, default=None)
parser.add_argument('--sn-tol', type=float, default=1e-3)
parser.add_argument('--learn_p', type=eval, choices=[True, False], default=True)

parser.add_argument('--n-power-series', type=int, default=None)
parser.add_argument('--factor-out', type=eval, choices=[True, False], default=False)
parser.add_argument('--n-dist', choices=['geometric', 'poisson'], default='poisson')
parser.add_argument('--n-samples', type=int, default=1)
parser.add_argument('--n-exact-terms', type=int, default=2)
parser.add_argument('--var-reduc-lr', type=float, default=0)
parser.add_argument('--neumann-grad', type=eval, choices=[True, False], default=True)
parser.add_argument('--mem-eff', type=eval, choices=[True, False], default=True)

parser.add_argument('--act', type=str, choices=ACT_FNS.keys(), default='swish')
parser.add_argument('--idim', type=int, default=128)
parser.add_argument('--nblocks', type=str, default='6-4')
parser.add_argument('--squeeze-first', type=eval, default=False, choices=[True, False])
parser.add_argument('--actnorm', type=eval, default=True, choices=[True, False])
parser.add_argument('--fc-actnorm', type=eval, default=False, choices=[True, False])
parser.add_argument('--batchnorm', type=eval, default=False, choices=[True, False])
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--fc', type=eval, default=False, choices=[True, False])
parser.add_argument('--kernels', type=str, default='3-1-3')
parser.add_argument('--quadratic', type=eval, choices=[True, False], default=False)
parser.add_argument('--fc-end', type=eval, choices=[True, False], default=True)
parser.add_argument('--fc-idim', type=list, default=[256, 128])
parser.add_argument('--preact', type=eval, choices=[True, False], default=True)
parser.add_argument('--padding', type=int, default=0)
parser.add_argument('--first-resblock', type=eval, choices=[True, False], default=True)
parser.add_argument('--cdim', type=int, default=256)

parser.add_argument('--scheduler', type=eval, choices=[True, False], default=False)
parser.add_argument('--nepochs', help='Number of epochs for training', type=int, default=1000)
parser.add_argument('--batchsize', help='Minibatch size', type=int, default=128)
parser.add_argument('--lr', help='Learning rate', type=float, default=1e-2)
parser.add_argument('--wd', help='Weight decay', type=float, default=0)
parser.add_argument('--warmup-iters', type=int, default=1000)
parser.add_argument('--annealing-iters', type=int, default=0)
parser.add_argument('--save', help='directory to save results', type=str, default='cifar10split0_v2')
parser.add_argument('--val-batchsize', help='minibatch size', type=int, default=64)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--ema-val', type=eval, choices=[True, False], default=True)
parser.add_argument('--update_freq', type=int, default=1)

parser.add_argument('--scale-dim', type=eval, choices=[True, False], default=False)
parser.add_argument('--rcrop-pad-mode', type=str, choices=['constant', 'reflect'], default='reflect')
parser.add_argument('--padding-dist', type=str, choices=['uniform', 'gaussian'], default='uniform')

parser.add_argument('--resume', type=str, default=None, help='./cifar10split0_v2/models/most_recent.pth')
parser.add_argument('--begin-epoch', type=int, default=0)

parser.add_argument('--print-freq', help='Print progress every so iterations', type=int, default=20)
parser.add_argument('--vis-freq', help='Visualize progress every so iterations', type=int, default=500)
args = parser.parse_args()

# Random seed
if args.seed is None:
    args.seed = np.random.randint(100000)

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info(args)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
    logger.info('Found {} CUDA devices.'.format(torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info('{} \t Memory: {:.2f}GB'.format(props.name, props.total_memory / (1024**3)))
else:
    logger.info('WARNING: Using device {}'.format(device))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(args.seed)


def geometric_logprob(ns, p):
    return torch.log(1 - p + 1e-10) * (ns - 1) + torch.log(p + 1e-10)


def standard_normal_sample(size):
    return torch.randn(size)


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def normal_logprob(z, mean, log_std):
    mean = mean + torch.tensor(0.)
    log_std = log_std + torch.tensor(0.)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reduce_bits(x):
    if args.nbits < 8:
        x = x * 255
        x = torch.floor(x / 2**(8 - args.nbits))
        x = x / 2**args.nbits
    return x


def update_lr(optimizer, itr):
    iter_frac = min(float(itr + 1) / max(args.warmup_iters, 1), 1.0)
    lr = args.lr * iter_frac
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def add_padding(x, nvals=256):
    # Theoretically, padding should've been added before the add_noise preprocessing.
    # nvals takes into account the preprocessing before padding is added.
    if args.padding > 0:
        if args.padding_dist == 'uniform':
            u = x.new_empty(x.shape[0], args.padding, x.shape[2], x.shape[3]).uniform_()
            logpu = torch.zeros_like(u).sum([1, 2, 3]).view(-1, 1)
            return torch.cat([x, u / nvals], dim=1), logpu
        elif args.padding_dist == 'gaussian':
            u = x.new_empty(x.shape[0], args.padding, x.shape[2], x.shape[3]).normal_(nvals / 2, nvals / 8)
            logpu = normal_logprob(u, nvals / 2, math.log(nvals / 8)).sum([1, 2, 3]).view(-1, 1)
            return torch.cat([x, u / nvals], dim=1), logpu
        else:
            raise ValueError()
    else:
        return x, torch.zeros(x.shape[0], 1).to(x)


def remove_padding(x):
    if args.padding > 0:
        return x[:, :im_dim, :, :]
    else:
        return x

dataset_folder_path = os.path.join(args.dataroot, args.data, args.split)
with open(dataset_folder_path + '/train_obj.pkl', 'rb') as f:
    train_obj = pickle.load(f)
with open(dataset_folder_path + '/test_obj.pkl', 'rb') as f:
    test_obj = pickle.load(f)
with open(dataset_folder_path + '/meta.pkl', 'rb') as f:
    meta_dict = pickle.load(f)
with open(dataset_folder_path + '/class_to_idx.pkl', 'rb') as f:
    class_to_idx = pickle.load(f)
with open(dataset_folder_path + '/idx_to_class.pkl', 'rb') as f:
    idx_to_class = pickle.load(f)
with open(dataset_folder_path + '/open_class_to_idx.pkl', 'rb') as f:
    open_class_to_idx = pickle.load(f)
with open(dataset_folder_path + '/open_idx_to_class.pkl', 'rb') as f:
    open_idx_to_class = pickle.load(f)
with open(dataset_folder_path + '/open_test_obj.pkl', 'rb') as fo:
    open_test_obj = pickle.load(fo)
with open(dataset_folder_path + '/open_meta.pkl', 'rb') as fo:
    open_meta_dict = pickle.load(fo)
logger.info('Loading dataset {}'.format(args.data))

# Dataset and hyperparameters
if args.data == 'cifar10':
    im_dim = 3
    n_classes = 6

    # Classification-specific preprocessing.
    train_trans = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode=args.rcrop_pad_mode),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_trans = transforms.Compose([
        transforms.ToTensor(),
    ])

elif args.data == 'mnist':
    im_dim = 1
    n_classes = 6
    train_trans = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
    test_trans = train_trans

elif args.data == 'svhn':
    im_dim = 3
    n_classes = 6
    train_trans = transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.RandomCrop(32, padding=4, padding_mode=args.rcrop_pad_mode),
                transforms.ToTensor(),
            ])
    test_trans = transforms.Compose([
                transforms.ToTensor(),
            ])
train_dataset = CIFARDataset(train_obj, meta_dict, class_to_idx, train_trans)
test_dataset = CIFARDataset(test_obj, meta_dict, class_to_idx, test_trans)
open_test_dataset = CIFARDataset(open_test_obj, open_meta_dict, open_class_to_idx, test_trans)
train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args.val_batchsize, shuffle=False, num_workers=4)
open_test_loader = DataLoader(open_test_dataset, batch_size=args.val_batchsize, shuffle=False, num_workers=4)


logger.info('Dataset loaded.')
logger.info('Creating model.')

input_size = (args.batchsize, im_dim + args.padding, args.imagesize, args.imagesize)

dataset_size = len(train_loader.dataset)

init_layer = layers.LogitTransform(0.05)

# Model
model = ResidualFlow(
    [args.batchsize, 128, 2, 2],
    n_blocks=list(map(int, args.nblocks.split('-'))),
    intermediate_dim=im_dim,
    factor_out=args.factor_out,
    quadratic=args.quadratic,
    init_layer=init_layer,
    actnorm=args.actnorm,
    fc_actnorm=args.fc_actnorm,
    batchnorm=args.batchnorm,
    dropout=args.dropout,
    fc=args.fc,
    coeff=args.coeff,
    vnorms=args.vnorms,
    n_lipschitz_iters=args.n_lipschitz_iters,
    sn_atol=args.sn_tol,
    sn_rtol=args.sn_tol,
    n_power_series=args.n_power_series,
    n_dist=args.n_dist,
    n_samples=args.n_samples,
    kernels=args.kernels,
    activation_fn=args.act,
    fc_end=args.fc_end,
    fc_idim=args.fc_idim,
    n_exact_terms=args.n_exact_terms,
    preact=args.preact,
    neumann_grad=args.neumann_grad,
    grad_in_forward=args.mem_eff,
    first_resblock=args.first_resblock,
    learn_p=args.learn_p,
    classification='hybrid',
    classification_hdim=args.cdim,
    n_classes=n_classes,
    block_type=args.block,
)

model.to(device)
ema = utils.ExponentialMovingAverage(model)


logger.info(model)
logger.info('EMA: {}'.format(ema))


# Optimization
def tensor_in(t, a):
    for a_ in a:
        if t is a_:
            return True
    return False

scheduler = None

#from itertools import chain

params = list(model.encoder.parameters()) + list(model.transforms.parameters())
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=args.wd)
params_classifer = list(model.classification_heads.parameters())
# optimizer_classifier = torch.optim.SGD(params_classifer, lr=1e-1, momentum=0.9, weight_decay=args.wd)
optimizer_classifier = torch.optim.Adam(model.parameters(), lr=1e-1, betas=(0.9, 0.99), weight_decay=args.wd)
if args.scheduler:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2, last_epoch=args.begin_epoch - 1)

best_test_bpd = math.inf
if (args.resume is not None):
    logger.info('Resuming model from {}'.format(args.resume))
    with torch.no_grad():
        x = torch.rand(1, *input_size[1:]).to(device)
        model(x)
    checkpt = torch.load(args.resume)
    sd = {k: v for k, v in checkpt['state_dict'].items() if 'last_n_samples' not in k}
    state = model.state_dict()
    state.update(sd)
    model.load_state_dict(state, strict=True)
    ema.set(checkpt['ema'])
    if 'optimizer_state_dict' in checkpt:
        optimizer.load_state_dict(checkpt['optimizer_state_dict'])
        # Manually move optimizer state to GPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    del checkpt
    del state

logger.info(optimizer)

fixed_z = standard_normal_sample([min(32, args.batchsize),
                                  (im_dim + args.padding) * args.imagesize * args.imagesize]).to(device)

criterion = torch.nn.CrossEntropyLoss()


def compute_loss(x, model, beta=1.0):

    logpz, delta_logp = torch.zeros(1).to(x), torch.zeros(1).to(x)

    nvals = 256

    x, logpu = add_padding(x, nvals)

    (z, delta_logp), logits_tensor = model(x, 0, classify=True)

    # log p(z)
    logpz = standard_normal_logprob(z).view(z.size(0), -1).sum(1, keepdim=True)

    # log p(x)
    logpx = logpz - beta * delta_logp - np.log(nvals) * (
        2 * 2 * (128 + args.padding)
    ) - logpu
    bits_per_dim = -torch.mean(logpx) / (2 * 2 * 128) / np.log(2)

    logpz = torch.mean(logpz).detach()
    delta_logp = torch.mean(-delta_logp).detach()

    return bits_per_dim, logits_tensor, logpz, delta_logp, logpx


def estimator_moments(model, baseline=0):
    avg_first_moment = 0.
    avg_second_moment = 0.
    for m in model.modules():
        if isinstance(m, layers.iResBlock):
            avg_first_moment += m.last_firmom.item()
            avg_second_moment += m.last_secmom.item()
    return avg_first_moment, avg_second_moment


def compute_p_grads(model):
    scales = 0.
    nlayers = 0
    for m in model.modules():
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            scales = scales + m.compute_one_iter()
            nlayers += 1
    scales.mul(1 / nlayers).backward()
    for m in model.modules():
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            if m.domain.grad is not None and torch.isnan(m.domain.grad):
                m.domain.grad = None


batch_time = utils.RunningAverageMeter(0.97)
bpd_meter = utils.RunningAverageMeter(0.97)
logpz_meter = utils.RunningAverageMeter(0.97)
deltalogp_meter = utils.RunningAverageMeter(0.97)
firmom_meter = utils.RunningAverageMeter(0.97)
secmom_meter = utils.RunningAverageMeter(0.97)
gnorm_meter = utils.RunningAverageMeter(0.97)
ce_meter = utils.RunningAverageMeter(0.97)


def train(epoch, model):
    model.train()
    end = time.time()
    total = 0
    correct = 0
    for i, (x, y) in enumerate(train_loader):

        global_itr = epoch * len(train_loader) + i
        update_lr(optimizer, global_itr)
        x = x.to(device)

        beta = min(1, global_itr / args.annealing_iters) if args.annealing_iters > 0 else 1.
        bpd, logits, logpz, neg_delta_logp, _ = compute_loss(x, model, beta=beta)

        firmom, secmom = estimator_moments(model)

        bpd_meter.update(bpd.item())
        logpz_meter.update(logpz.item())
        deltalogp_meter.update(neg_delta_logp.item())
        firmom_meter.update(firmom)
        secmom_meter.update(secmom)

        y = y.to(device)
        crossent = criterion(logits, y)
        ce_meter.update(crossent.item())

        # Compute accuracy.
        _, predicted = logits.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

        if global_itr % args.update_freq == args.update_freq - 1:


            if args.learn_p: compute_p_grads(model)
            if i % 2 == 0:
                if args.update_freq > 1:
                    with torch.no_grad():
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad = p.grad / args.update_freq

                grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.)

                loss = bpd
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                update_lipschitz(model)
            else:
                loss = crossent
                loss.backward()
                optimizer_classifier.step()
                optimizer_classifier.zero_grad()

            ema.apply()

            gnorm_meter.update(grad_norm)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            s = (
                'Epoch: [{0}][{1}/{2}] | Time {batch_time.val:.3f} | '
                'GradNorm {gnorm_meter.avg:.2f}'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, gnorm_meter=gnorm_meter
                )
            )

            s += (
                ' | Bits/dim {bpd_meter.val:.4f}({bpd_meter.avg:.4f}) | '
                'Logpz {logpz_meter.avg:.0f} | '
                '-DeltaLogp {deltalogp_meter.avg:.0f} | '
                'EstMoment ({firmom_meter.avg:.0f},{secmom_meter.avg:.0f})'.format(
                    bpd_meter=bpd_meter, logpz_meter=logpz_meter, deltalogp_meter=deltalogp_meter,
                    firmom_meter=firmom_meter, secmom_meter=secmom_meter
                )
            )

            s += ' | CE {ce_meter.avg:.4f} | Acc {0:.4f}'.format(100 * correct / total, ce_meter=ce_meter)

            logger.info(s)
        del x
        torch.cuda.empty_cache()
        gc.collect()

def validate(epoch, model, ema=None):
    """
    Evaluates the cross entropy between p_data and p_model.
    """
    bpd_meter = utils.AverageMeter()
    ce_meter = utils.AverageMeter()

    if ema is not None:
        ema.swap()

    update_lipschitz(model)

    model.eval()

    correct = 0
    total = 0

    start = time.time()
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(test_loader)):
            x = x.to(device)
            bpd, logits, _, _, logpx = compute_loss(x, model)
            bpd_meter.update(bpd.item(), x.size(0))

            y = y.to(device)
            loss = criterion(logits, y)
            ce_meter.update(loss.item(), x.size(0))
            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    val_time = time.time() - start

    if ema is not None:
        ema.swap()
    s = 'Epoch: [{0}]\tTime {1:.2f} | Test bits/dim {bpd_meter.avg:.4f}'.format(epoch, val_time, bpd_meter=bpd_meter)

    s += ' | CE {:.4f} | Acc {:.2f}'.format(ce_meter.avg, 100 * correct / total)
    logger.info(s)
    return bpd_meter.avg

def open_test(epoch, model, ema=None):
    """
    Evaluates the cross entropy between p_data and p_model.
    """
    if ema is not None:
        ema.swap()
    update_lipschitz(model)
    model.eval()
    correct = 0
    total = 0
    known_logpx = []
    unknown_logpx = []
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(test_loader)):
            x = x.to(device)
            bpd, logits, _, _, logpx = compute_loss(x, model)

            y = y.to(device)
            probs, predicted = logits.max(1)
            known_logpx.extend(logpx.tolist())
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
        acc = correct / total
        for i, (x, y) in enumerate(tqdm(open_test_loader)):
            x = x.to(device)
            bpd, logits, _, _, logpx = compute_loss(x, model)

            y = y.to(device)
            probs, predicted = logits.max(1)
            unknown_logpx.extend(logpx.tolist())
    if ema is not None:
        ema.swap()
    labels = [1]*len(known_logpx) + [0]*len(unknown_logpx)
    preds = known_logpx + unknown_logpx
    auc = roc_auc_score(labels, preds)
    print('Closed ACC:', correct/total, 'AUC:', auc)
    return bpd_meter.avg

def get_lipschitz_constants(model):
    lipschitz_constants = []
    for m in model.modules():
        if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
            lipschitz_constants.append(m.scale)
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            lipschitz_constants.append(m.scale)
        if isinstance(m, base_layers.LopConv2d) or isinstance(m, base_layers.LopLinear):
            lipschitz_constants.append(m.scale)
    return lipschitz_constants

def update_lipschitz(model):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
                m.compute_weight(update=True)
            if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
                m.compute_weight(update=True)


def get_ords(model):
    ords = []
    for m in model.modules():
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            domain, codomain = m.compute_domain_codomain()
            if torch.is_tensor(domain):
                domain = domain.item()
            if torch.is_tensor(codomain):
                codomain = codomain.item()
            ords.append(domain)
            ords.append(codomain)
    return ords

def pretty_repr(a):
    return '[[' + ','.join(list(map(lambda i: f'{i:.2f}', a))) + ']]'

def main():
    global best_test_bpd

    last_checkpoints = []
    lipschitz_constants = []
    ords = []

    if args.resume:
        #validate(args.begin_epoch - 1, model, ema)
        open_test(args.begin_epoch - 1, model, ema)
    for epoch in range(args.begin_epoch, args.nepochs):

        logger.info('Current LR {}'.format(optimizer.param_groups[0]['lr']))

        train(epoch, model)
        lipschitz_constants.append(get_lipschitz_constants(model))
        ords.append(get_ords(model))
        logger.info('Lipsh: {}'.format(pretty_repr(lipschitz_constants[-1])))
        logger.info('Order: {}'.format(pretty_repr(ords[-1])))

        if args.ema_val:
            test_bpd = validate(epoch, model, ema)
        else:
            test_bpd = validate(epoch, model)

        if args.scheduler and scheduler is not None:
            scheduler.step()

        if test_bpd < best_test_bpd:
            best_test_bpd = test_bpd
            utils.save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
                'ema': ema,
                'test_bpd': test_bpd,
            }, os.path.join(args.save, 'models'), epoch, last_checkpoints, num_checkpoints=5)

        torch.save({
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args,
            'ema': ema,
            'test_bpd': test_bpd,
        }, os.path.join(args.save, 'models', 'most_recent.pth'))


if __name__ == '__main__':
    main()