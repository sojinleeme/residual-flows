import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os
import time
import datetime
import math
import numpy as np

import torch

import lib.optimizers as optim
import lib.layers.base as base_layers
import lib.layers as layers
import lib.toy_data as toy_data
import lib.utils as utils
from lib.visualize_flow import visualize_transform

from lib.kde import GaussianKernel, KernelDensityEstimator

ACTIVATION_FNS = {
    'relu': torch.nn.ReLU,
    'tanh': torch.nn.Tanh,
    'elu': torch.nn.ELU,
    'selu': torch.nn.SELU,
    'fullsort': base_layers.FullSort,
    'maxmin': base_layers.MaxMin,
    'swish': base_layers.Swish,
    'lcube': base_layers.LipschitzCube,
}

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings'],
    type=str, default='8gaussians'
)
parser.add_argument('--arch', choices=['iresnet', 'realnvp'], default='iresnet')
parser.add_argument('--coeff', type=float, default=0.9)
parser.add_argument('--vnorms', type=str, default='222222')
parser.add_argument('--n-lipschitz-iters', type=int, default=5)
parser.add_argument('--atol', type=float, default=None)
parser.add_argument('--rtol', type=float, default=None)
parser.add_argument('--learn-p', type=eval, choices=[True, False], default=False)
parser.add_argument('--mixed', type=eval, choices=[True, False], default=True)

parser.add_argument('--dims', type=str, default='128-128-128-128')
parser.add_argument('--act', type=str, choices=ACTIVATION_FNS.keys(), default='swish')
parser.add_argument('--nblocks', type=int, default=100)
parser.add_argument('--brute-force', type=eval, choices=[True, False], default=False)
parser.add_argument('--actnorm', type=eval, choices=[True, False], default=False)
parser.add_argument('--batchnorm', type=eval, choices=[True, False], default=False)
parser.add_argument('--exact-trace', type=eval, choices=[True, False], default=False)
parser.add_argument('--n-power-series', type=int, default=None)
parser.add_argument('--n-samples', type=int, default=1)
parser.add_argument('--n-dist', choices=['geometric', 'poisson'], default='geometric')

parser.add_argument('--niters', type=int, default=50000)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--test_batch_size', type=int, default=10000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight-decay', type=float, default=1e-5)
parser.add_argument('--annealing-iters', type=int, default=0)

parser.add_argument('--save', type=str, default='experiments/')
parser.add_argument('--viz_freq', type=int, default=100)
parser.add_argument('--val_freq', type=int, default=100)
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=123)


parser.add_argument('--use_wandb', type=str, default='False')
parser.add_argument('--norm_hyp', type=float, default=0.0001)
parser.add_argument('--toy_exp_type', type=str, default='default') # base
parser.add_argument('--sampling_num', type=int, default=256) # base
parser.add_argument('--kde_bandwidth', type=float, default=0.4) # base

args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
args.device = device

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# intialize snapshots directory for saving models and results
args.model_signature = str(datetime.datetime.now())[0:19].replace(' ', '_').replace(':', '_').replace('-', '_')
args.name = args.arch + '_' + '_' + args.data + '/' + args.toy_exp_type + '_' + str(args.nblocks) + 'blocks' + '_'+ str(args.norm_hyp)

lr_schedule = f'_lr{str(args.lr)[2:]}'
args.snap_dir = os.path.join(args.save, args.name)
args.snap_dir += f'_seed{args.seed}' + lr_schedule + '_' + f"_bs{args.batch_size}"
args.dirs = f'{args.snap_dir}/{args.model_signature}/'
utils.makedirs(args.dirs)

logger = utils.get_logger(logpath=os.path.join(args.dirs, 'logs'), filepath=os.path.abspath(__file__))
logger.info(args)

mean = torch.zeros(2).to(args.device) # mean=0
cov = torch.eye(2).to(args.device) # covariance=1
priorMVG_Z = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=cov) # MVG ~ N(0,I)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def standard_normal_sample(size):
    return torch.randn(size)


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def compute_loss(args, model, batch_size=None, beta=1.):
    if batch_size is None: batch_size = args.batch_size

    # load data
    x = toy_data.inf_train_gen(args, args.data, batch_size=batch_size)
    x = torch.from_numpy(x).type(torch.float32).to(device)
    zero = torch.zeros(x.shape[0], 1).to(x)

    # transform to z
    z, delta_logp = model(x, zero) # delta_logp = logpx - logdetgrad

    # compute log p(z)
    logpz = standard_normal_logprob(z).sum(1, keepdim=True)

    logpx = logpz - beta * delta_logp  
    loss = -torch.mean(logpx)
    return loss, torch.mean(logpz), torch.mean(-delta_logp)


def parse_vnorms():
    ps = []
    for p in args.vnorms:
        if p == 'f':
            ps.append(float('inf'))
        else:
            ps.append(float(p))
    return ps[:-1], ps[1:]


def compute_p_grads(model):
    scales = 0.
    nlayers = 0
    for m in model.modules():
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            scales = scales + m.compute_one_iter()
            nlayers += 1
    scales.mul(1 / nlayers).mul(0.01).backward()
    for m in model.modules():
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            if m.domain.grad is not None and torch.isnan(m.domain.grad):
                m.domain.grad = None


def build_nnet(dims, activation_fn=torch.nn.ReLU):
    nnet = []
    domains, codomains = parse_vnorms()
    if args.learn_p:
        if args.mixed:
            domains = [torch.nn.Parameter(torch.tensor(0.)) for _ in domains]
        else:
            domains = [torch.nn.Parameter(torch.tensor(0.))] * len(domains)
        codomains = domains[1:] + [domains[0]]
    for i, (in_dim, out_dim, domain, codomain) in enumerate(zip(dims[:-1], dims[1:], domains, codomains)):
        nnet.append(activation_fn())
        nnet.append(
            base_layers.get_linear(
                in_dim,
                out_dim,
                coeff=args.coeff,
                n_iterations=args.n_lipschitz_iters,
                atol=args.atol,
                rtol=args.rtol,
                domain=domain,
                codomain=codomain,
                zero_init=(out_dim == 2),
            )
        )
    return torch.nn.Sequential(*nnet)


def update_lipschitz(model, n_iterations):
    for m in model.modules():
        if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
            m.compute_weight(update=True, n_iterations=n_iterations)
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            m.compute_weight(update=True, n_iterations=n_iterations)


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


if __name__ == '__main__':

    activation_fn = ACTIVATION_FNS[args.act]

    if args.arch == 'iresnet':
        dims = [2] + list(map(int, args.dims.split('-'))) + [2]
        blocks = []
        if args.actnorm: blocks.append(layers.ActNorm1d(2))
        for _ in range(args.nblocks):
            blocks.append(
                layers.iResBlock(
                    build_nnet(dims, activation_fn),
                    n_dist=args.n_dist,
                    n_power_series=args.n_power_series,
                    exact_trace=args.exact_trace,
                    brute_force=args.brute_force,
                    n_samples=args.n_samples,
                    neumann_grad=False,
                    grad_in_forward=False,
                )
            )
            if args.actnorm: blocks.append(layers.ActNorm1d(2))
            if args.batchnorm: blocks.append(layers.MovingBatchNorm1d(2))
        model = layers.SequentialFlow(blocks).to(device)
    else:
        print("Not Implemented yet.")
    # elif args.arch == 'realnvp':
    #     blocks = []
    #     for _ in range(args.nblocks):
    #         blocks.append(layers.CouplingBlock(2, swap=False))
    #         blocks.append(layers.CouplingBlock(2, swap=True))
    #         if args.actnorm: blocks.append(layers.ActNorm1d(2))
    #         if args.batchnorm: blocks.append(layers.MovingBatchNorm1d(2))
    #     model = layers.SequentialFlow(blocks).to(device)

    logger.info(model)
    logger.info("Number of trainable parameters: {}".format(count_parameters(model)))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    time_meter = utils.RunningAverageMeter(0.93)
    loss_meter = utils.RunningAverageMeter(0.93)
    logpz_meter = utils.RunningAverageMeter(0.93)
    delta_logp_meter = utils.RunningAverageMeter(0.93)

    new_loss_meter = utils.RunningAverageMeter(0.93)
    regularizer_meter = utils.RunningAverageMeter(0.93)


    end = time.time()
    best_loss = float('inf')
    model.train()

    # Set kernel density estimator  tat_data: [batchsize, 2]
    tgt_data = toy_data.inf_train_gen(args, args.data, batch_size=args.batch_size) # kde로 측정할 우리가 정답아는 분포
    tgt_data = torch.from_numpy(tgt_data).type(torch.float32).to(args.device) # kde로 측정할 우리가 정답아는 분포
    kernel_estimator = KernelDensityEstimator(tgt_data, bandwidth=args.kde_bandwidth).to(args.device)
    
    # Set KLD loss
    kl_loss_mean = torch.nn.KLDivLoss(reduction='batchmean').to(args.device)
    kl_loss_mean_log = torch.nn.KLDivLoss(reduction='batchmean', log_target=True).to(args.device)
    kl_loss = torch.nn.KLDivLoss(reduction="none").to(args.device)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        beta = min(1, itr / args.annealing_iters) if args.annealing_iters > 0 else 1.

        ''' 1. Sampling z from MVG and keep log_p_z '''
        z = priorMVG_Z.sample((args.batch_size,)).to(args.device) # [batchsize, 2]
        log_p_z = priorMVG_Z.log_prob(z) # [batchsize]
        zero = torch.zeros(z.shape[0], 1).to(z)

        ''' 2. Calculate f(x; theta) '''
        sampled_x, delta_log_p_x = model.inverse(z, zero)
        log_p_x = log_p_z + delta_log_p_x # f(x) = log p(z) - log |det J| = log p(x)

        ''' 3. Calculate g(x; KDE) '''
        kde_q = kernel_estimator(sampled_x) # Estimate denstiy of sampled data in KDE
    # # load data
    # x = toy_data.inf_train_gen(args, args.data, batch_size=batch_size)
    # x = torch.from_numpy(x).type(torch.float32).to(device)
    # zero = torch.zeros(x.shape[0], 1).to(x)

    # # transform to z
    # z, delta_logp = model(x, zero) # delta_logp = logpx - logdetgrad

    # # compute log p(z)
    # logpz = standard_normal_logprob(z).sum(1, keepdim=True)

    # logpx = logpz - beta * delta_logp  
    # loss = -torch.mean(logpx)
    # return loss, torch.mean(logpz), torch.mean(-delta_logp)
        losses = {}
        ''' 4. Calculate Objective 1 = -log p(x): losses['nll'] '''
        losses['nll'], losses['logpz'], losses['delta_logp'] = compute_loss(args, model, beta=beta)

        ''' 5. Calculate Objective 2 = KL_divergence '''
        # KL(P||Q) = kl_loss(Q.log, P)
        # KL(Q||P) = kl_loss(P.log, Q)
        # f(x) = log_p_x -> log density
        # g(x) = kde_q -> density
        
        losses['kl_mean_gf'] = kl_loss_mean(log_p_x, kde_q)
        losses['kl_mean_g_minusf'] = kl_loss_mean(-1 * log_p_x, kde_q)       
        
        losses['kl_gf_norm'] = torch.norm(kl_loss(log_p_x, kde_q))
        losses['kl_g_minusf_norm'] = torch.norm(kl_loss(-1 * log_p_x, kde_q))

        losses['qp_ratio'] = torch.norm((kde_q - log_p_x.exp())-1)
        losses['pq_ratio'] = torch.norm((log_p_x.exp() - kde_q)-1)

        M = 0.5 * (log_p_x.exp() + kde_q) # Calculate as the form of density
        KL_PM = kl_loss_mean_log(M.log(), log_p_x)
        KL_QM = kl_loss_mean_log(M.log(), kde_q.log())
        losses['jsd'] = 0.5 * (KL_PM + KL_QM)


        ''' 6. Calculate norm between f and g '''
        density_x = torch.exp(log_p_x)
        norm_density = torch.norm(density_x - kde_q) # l2 norm

        if args.toy_exp_type == 'default':
            losses['new_objective'] = losses['nll']
            regulaizer = 0
        elif args.toy_exp_type == 'KL_gf':
            losses['new_objective'] = losses['nll'] + args.norm_hyp * losses['kl_mean_gf']
            regulaizer = args.norm_hyp * losses['kl_mean_gf']
        elif args.toy_exp_type == 'KL_g-f':
            losses['new_objective'] = losses['nll'] + args.norm_hyp * losses['kl_mean_g_minusf']
            regulaizer = args.norm_hyp * losses['kl_mean_g_minusf']
        elif args.toy_exp_type == 'KLeleNorm_gf':
            losses['new_objective'] = losses['nll'] + args.norm_hyp * losses['kl_gf_norm']
            regulaizer = args.norm_hyp * losses['kl_gf_norm']
        elif args.toy_exp_type == 'KLeleNorm_g-f':
            losses['new_objective'] = losses['nll'] + args.norm_hyp * losses['kl_g_minusf_norm']
            regulaizer = args.norm_hyp * losses['kl_g_minusf_norm']
        elif args.toy_exp_type == 'qp_ratio':
            losses['new_objective'] = losses['nll'] + args.norm_hyp * losses['qp_ratio']
            regulaizer = args.norm_hyp * losses['qp_ratio']
        elif args.toy_exp_type == 'pq_ratio':
            losses['new_objective'] = losses['nll'] + args.norm_hyp * losses['pq_ratio']
            regulaizer = args.norm_hyp * losses['pq_ratio']
        elif args.toy_exp_type == 'jsd':   
            losses['new_objective'] = losses['nll'] + args.norm_hyp * losses['jsd']
            regulaizer = args.norm_hyp * losses['jsd']
        else:
            print("Not Implemented")
    

        loss_meter.update(losses['nll'].item())
        logpz_meter.update(losses['logpz'].item())
        delta_logp_meter.update(losses['delta_logp'].item())
        new_loss_meter.update(losses['new_objective'].item())
        regularizer_meter.update(regulaizer)
        # loss.backward()
        
        losses['new_objective'].backward()

        if args.learn_p and itr > args.annealing_iters: compute_p_grads(model)
        optimizer.step()
        update_lipschitz(model, args.n_lipschitz_iters)

        time_meter.update(time.time() - end)

        logger.info(
            'Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f})'
            ' | Logp(z) {:.6f}({:.6f}) | DeltaLogp {:.6f}({:.6f})'.format(
                itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg, logpz_meter.val, logpz_meter.avg,
                delta_logp_meter.val, delta_logp_meter.avg
            )
        )

        if itr % args.val_freq == 0 or itr == args.niters:
            update_lipschitz(model, 200)
            with torch.no_grad():
                model.eval()
                test_loss, test_logpz, test_delta_logp = compute_loss(args, model, batch_size=args.test_batch_size)
                log_message = (
                    '[TEST] Iter {:04d} | Test Loss {:.6f} '
                    '| Test Logp(z) {:.6f} | Test DeltaLogp {:.6f}'.format(
                        itr, test_loss.item(), test_logpz.item(), test_delta_logp.item()
                    )
                )
                logger.info(log_message)

                logger.info('Ords: {}'.format(pretty_repr(get_ords(model))))

                if test_loss.item() < best_loss:
                    best_loss = test_loss.item()
                    utils.makedirs(args.save)
                    torch.save({
                        'args': args,
                        'state_dict': model.state_dict(),
                    }, os.path.join(args.save, 'checkpt.pth'))
                model.train()

        if itr == 1 or itr % args.viz_freq == 0:
            with torch.no_grad():
                model.eval()
                p_samples = toy_data.inf_train_gen(args, args.data, batch_size=20000)

                sample_fn, density_fn = model.inverse, model.forward

                plt.figure(figsize=(9, 3))
                visualize_transform(
                    p_samples, torch.randn, standard_normal_logprob, transform=sample_fn, inverse_transform=density_fn,
                    samples=True, npts=400, device=device
                )
                fig_filename = os.path.join(args.dirs, 'figs', '{:04d}.jpg'.format(itr))
                utils.makedirs(os.path.dirname(fig_filename))
                plt.savefig(fig_filename)
                plt.close()
                model.train()

        end = time.time()

    logger.info('Training has finished.')
