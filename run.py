import subprocess
import argparse
parser=argparse.ArgumentParser(description="run")
parser.add_argument("--n", type=int)
args = parser.parse_args()
n = args.n


DATASET = "8gaussians"

'''64 Server'''
# if args.n == 1:
#     for toy_exp_type in ['default']:
#         subprocess.call(f"python train_toy.py --use_wandb 'True' --norm_hyp 0 --toy_exp_type {toy_exp_type} --data {DATASET} --gpu {n}", shell=True)

# if args.n == 0:
#     for toy_exp_type in ['KL_gf']:
#         for norm_hyp in [1,10,100]:
#             subprocess.call(f"python train_toy.py --use_wandb 'True' --norm_hyp {norm_hyp} --toy_exp_type {toy_exp_type} --data {DATASET} --gpu {n}", shell=True)

# if args.n == 1:
#     for toy_exp_type in ['KL_g-f']:
#         for norm_hyp in [1,10,100]:
#             subprocess.call(f"python train_toy.py --use_wandb 'True' --norm_hyp {norm_hyp} --toy_exp_type {toy_exp_type} --data {DATASET} --gpu {n}", shell=True)

# if args.n == 2:
#     for toy_exp_type in ['KLeleNorm_gf']:
#         for norm_hyp in [1,10,100]:
#             subprocess.call(f"python train_toy.py --use_wandb 'True' --norm_hyp {norm_hyp} --toy_exp_type {toy_exp_type} --data {DATASET} --gpu {n}", shell=True)

# if args.n == 3:
#     for toy_exp_type in ['KLeleNorm_g-f']:
#         for norm_hyp in [1,10,100]:
#             subprocess.call(f"python train_toy.py --use_wandb 'True' --norm_hyp {norm_hyp} --toy_exp_type {toy_exp_type} --data {DATASET} --gpu {n}", shell=True)

'''AI Hub'''
# if args.n == 0:
#     for toy_exp_type in ['qp_ratio']:
#         for norm_hyp in [1,10,100]:
#             subprocess.call(f"python train_toy.py --use_wandb 'True' --norm_hyp {norm_hyp} --toy_exp_type {toy_exp_type} --data {DATASET} --gpu {n}", shell=True)

# if args.n == 1:
#     for toy_exp_type in ['pq_ratio']:
#         for norm_hyp in [1,10,100]:
#             subprocess.call(f"python train_toy.py --use_wandb 'True' --norm_hyp {norm_hyp} --toy_exp_type {toy_exp_type} --data {DATASET} --gpu {n}", shell=True)

'''???'''
# if args.n == 0:
#     for toy_exp_type in ['jsd']:
#         for norm_hyp in [1,10,100]:
#             subprocess.call(f"python train_toy.py --use_wandb 'True' --norm_hyp {norm_hyp} --toy_exp_type {toy_exp_type} --data {DATASET} --gpu {n}", shell=True)
