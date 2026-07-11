import os
from datetime import datetime
import random
import argparse
import torch
import numpy as np
import torch.backends.cudnn as cudnn

from pnpflow.utils import load_cfg_from_cfg_file, merge_cfg_from_list
from pnpflow.degradations import *
from pnpflow.dataloaders import DataLoaders
from pnpflow.train_flow_matching import FLOW_MATCHING
from pnpflow.train_denoiser import GRADIENT_STEP_DENOISER
from pnpflow.compute_metric import ComputeMetric
from pnpflow.methods.pnp_flow import PNP_FLOW
from pnpflow.methods.d_flow import D_FLOW
from pnpflow.methods.ot_ode import OT_ODE
from pnpflow.methods.flow_priors import FLOW_PRIORS
from pnpflow.methods.pnp_gs import PROX_PNP
from pnpflow.methods.pnp_diff import PNP_DIFF
from pnpflow.utils import gaussian_blur, define_model, load_model
import warnings
warnings.filterwarnings("ignore", module="matplotlib\\..*")


def make_haadf_known_mask(kind, dim, device='cpu', radius=16, line_width=8, random_missing_fraction=0.15, seed=0):
    """Return known_mask with shape 1 x 1 x dim x dim for synthetic HAADF tests.

    known_mask = 1 means observed/known pixel; 0 means missing pixel.
    """
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    known = torch.ones(1, 1, dim, dim, device=device)

    if kind == 'blob':
        yy, xx = torch.meshgrid(torch.arange(dim, device=device), torch.arange(dim, device=device), indexing='ij')
        cy, cx = dim // 2, dim // 2
        missing = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        known[:, :, missing] = 0
    elif kind == 'line':
        c = dim // 2
        half = max(1, line_width // 2)
        known[:, :, max(0, c-half):min(dim, c+half), :] = 0
    elif kind == 'random':
        rand = torch.rand((1, 1, dim, dim), generator=g, device=device)
        known = (rand > random_missing_fraction).float()
    else:
        raise ValueError(f'Unknown HAADF mask kind: {kind}')

    return known.float()

torch.cuda.empty_cache()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Main')
    cfg = load_cfg_from_cfg_file('./' + 'config/main_config.yaml')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    parser.add_argument(
        '--no_cap',
        action='store_true',
        help='Train over the full DataLoader instead of stopping after 21 batches per epoch.',
    )
    args = parser.parse_args()
    cfg.no_cap = args.no_cap
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)

    dataset_config = cfg.root + \
        'config/dataset_config/{}.yaml'.format(
            cfg.dataset)
    cfg.update(load_cfg_from_cfg_file(dataset_config))

    method_config_file = cfg.root + \
        'config/method_config/{}.yaml'.format(
            cfg.method)
    cfg.update(load_cfg_from_cfg_file(method_config_file))

    if args.opts is not None:
        # override config with command line input
        cfg = merge_cfg_from_list(cfg, args.opts)

    # for all keys in the method config file, create a dictionary {key: value} in the cfg object cfg.dict_cfg_method
    method_cfg = load_cfg_from_cfg_file(method_config_file)
    cfg.dict_cfg_method = {}
    for key in method_cfg.keys():
        cfg.dict_cfg_method[key] = cfg[key]
    return cfg


def resolve_model_checkpoint(args):
    base_dir = os.path.join(args.root, 'model', args.dataset, args.model)
    model_run = getattr(args, 'model_run', None)
    if model_run in [None, 'None', '']:
        latest_file = os.path.join(base_dir, 'latest_run.txt')
        if os.path.exists(latest_file):
            with open(latest_file, 'r') as file:
                model_run = file.read().strip()

    filename = 'gradient_step_denoiser_final.pt' if args.model == 'gradient_step' else 'model_final.pt'
    candidates = []
    if model_run not in [None, 'None', '']:
        candidates.append(os.path.join(base_dir, model_run, filename))
    candidates.append(os.path.join(base_dir, filename))

    for checkpoint_path in candidates:
        if os.path.exists(checkpoint_path):
            print(f'Loading checkpoint: {checkpoint_path}')
            return checkpoint_path

    raise FileNotFoundError(
        f'No checkpoint found for dataset={args.dataset}, model={args.model}. '
        f'Checked: {candidates}')


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    (model, state) = define_model(args)

    if args.train:
        args.batch_size = args.batch_size_train
        print('Training...')
        data_loaders = DataLoaders(
            args.dataset, args.batch_size_train, args.batch_size_train).load_data()
        if args.model == "ot":
            generative_method = FLOW_MATCHING(model, device, args)
        elif args.model == "gradient_step":
            generative_method = GRADIENT_STEP_DENOISER(model, device, args)
        else:
            raise ValueError(
                "Model not implemented yet: you can choose between 'ot' and 'gradient_step'")
        generative_method.train(data_loaders)
        print('Training done!')

    if args.eval:

        if args.model == "ot" or args.model == "gradient_step":
            model_path = resolve_model_checkpoint(args)
            load_model(args.model, model, state, download=False,
                       checkpoint_path=model_path, dataset=None,  device=device)
            model.eval()

        elif args.model == "rectified":
            model_path = args.root + 'model/{}/{}/model_final.pth'.format(
                args.dataset, args.model)
            load_model(args.model, model, state, download=False,
                       checkpoint_path=model_path, dataset=None, device=device)
            model.eval()

        elif args.model == "diffusion":
            model.eval()

        if args.model == "gradient_step":
            generative_method = GRADIENT_STEP_DENOISER(model, device, args)
        else:
            generative_method = FLOW_MATCHING(model, device, args)

        if args.compute_metrics:
            print('Computing metrics...')
            data_loaders = DataLoaders(args.dataset, 5000, 5000).load_data()
            metric = ComputeMetric(
                data_loaders, generative_method, device, args)
            metric.compute_metrics(5000)
            print('Computing metrics done!')

        if args.problem == "denoising":
            if args.noise_type == 'laplace':
                sigma_noise = 0.3
            elif args.noise_type == 'gaussian':
                sigma_noise = 0.2
            degradation = Denoising()

        elif args.problem == "inpainting":
            if args.noise_type == 'laplace':
                sigma_noise = 0.3
            elif args.noise_type == 'gaussian':
                sigma_noise = 0.05
            if args.dim_image == 128:
                half_size_mask = 20
            elif args.dim_image == 256:
                half_size_mask = 40
            degradation = BoxInpainting(half_size_mask)

        elif args.problem == "paintbrush_inpainting":
            if args.noise_type == 'laplace':
                sigma_noise = 0.3
            elif args.noise_type == 'gaussian':
                sigma_noise = 0.05
            degradation = PaintbrushInpainting()

        elif args.problem == "random_inpainting":
            if args.noise_type == 'laplace':
                sigma_noise = 0.3
            elif args.noise_type == 'gaussian':
                sigma_noise = 0.01
            p = 0.7
            degradation = RandomInpainting(p)

        elif args.problem in ["haadf_blob_inpainting", "haadf_line_inpainting", "haadf_random_inpainting"]:
            # Fixed synthetic masks for controlled HAADF inpainting benchmarks.
            # These are preferable to randomly regenerated masks because H and H_adj
            # must use the same operator.
            if args.noise_type == 'laplace':
                sigma_noise = 0.03
            elif args.noise_type == 'gaussian':
                sigma_noise = 0.01
            if args.problem == "haadf_blob_inpainting":
                known_mask = make_haadf_known_mask('blob', args.dim_image, device=device, radius=max(8, args.dim_image // 8), seed=args.seed or 0)
            elif args.problem == "haadf_line_inpainting":
                known_mask = make_haadf_known_mask('line', args.dim_image, device=device, line_width=max(4, args.dim_image // 16), seed=args.seed or 0)
            else:
                known_mask = make_haadf_known_mask('random', args.dim_image, device=device, random_missing_fraction=0.15, seed=args.seed or 0)
            degradation = FixedMaskInpainting(known_mask)

        elif args.problem == "superresolution":
            if args.dim_image == 128:
                print('Superresolution with scale factor 2')
                sf = 2
            elif args.dim_image == 256:
                print('Superresolution with scale factor 4')
                sf = 4
            if args.noise_type == 'laplace':
                sigma_noise = 0.3

            elif args.noise_type == 'gaussian':
                sigma_noise = 0.05
            degradation = Superresolution(sf, args.dim_image)

        elif args.problem == "gaussian_deblurring_FFT":
            if args.dim_image == 128:
                sigma_blur = 1.0
            elif args.dim_image == 256:
                sigma_blur = 3.0

            if args.noise_type == 'laplace':
                sigma_noise = 0.3
            elif args.noise_type == 'gaussian':
                sigma_noise = 0.05
            kernel_size = 61
            degradation = GaussianDeblurring(
                sigma_blur, kernel_size, "fft", args.num_channels, args.dim_image, device)

        print('Solving the {} inverse problem with the method {}...'.format(
            args.problem, args.method))
        print('sigma_noise', sigma_noise)
        data_loaders = DataLoaders(
            args.dataset, args.batch_size_ip, args.batch_size_ip).load_data()
        eval_run_name = getattr(args, 'eval_run_name', None)
        if eval_run_name in [None, 'None', '']:
            eval_run_name = (
                datetime.now().strftime('%Y%m%d_%H%M%S')
                + f'_{args.dataset}_{args.model}_{args.problem}_{args.method}_{args.eval_split}'
            )
        args.eval_run_name = eval_run_name

        results_root = 'results_laplace' if args.noise_type == 'laplace' else 'results'
        eval_base_path = os.path.join(
            args.root, results_root, args.dataset, args.model, args.problem, args.method, args.eval_split)
        os.makedirs(eval_base_path, exist_ok=True)
        args.save_path = os.path.join(eval_base_path, eval_run_name)
        os.makedirs(args.save_path, exist_ok=True)
        with open(os.path.join(eval_base_path, 'latest_eval_run.txt'), 'w') as file:
            file.write(eval_run_name + '\n')

        if args.method == 'pnp_flow':
            method = PNP_FLOW(model, device, args)
        elif args.method == 'd_flow':
            method = D_FLOW(model, device, args)
        elif args.method == 'ot_ode':
            method = OT_ODE(model, device, args)
        elif args.method == 'flow_priors':
            method = FLOW_PRIORS(model, device, args)
        elif args.method == 'pnp_gs':
            method = PROX_PNP(generative_method, device, args)
        elif args.method == 'pnp_diff':
            method = PNP_DIFF(model, device, args)
        else:
            raise ValueError("The method your entered does not exist")

        method.run_method(data_loaders, degradation, sigma_noise)


if __name__ == "__main__":
    main()
