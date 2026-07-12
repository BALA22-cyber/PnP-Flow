# Code adapted from
#
# Tong, A., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Fatras, K., ... & Bengio, Y. (2023).
# Improving and generalizing flow-based generative models with minibatch optimal transport.
# arXiv preprint arXiv:2302.00482.
# (https://github.com/atong01/conditional-flow-matching)
#
# Chemseddine, J., Hagemann, P., Wald, C., & Steidl, G. (2024).
# Conditional Wasserstein Distances with Applications in Bayesian OT Flow Matching.
# arXiv preprint arXiv:2403.18705.
# (https://github.com/JChemseddine/Conditional_Wasserstein_Distances/blob/main/utils/utils_FID.py)

import torch
import torch.distributed as dist
import os
from datetime import datetime
import skimage.io as io
import numpy as np
import torch
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import ot
from torchdiffeq import odeint_adjoint as odeint
import pnpflow.fid_score as fs
from pnpflow.dataloaders import DataLoaders
import pnpflow.utils as utils
try:
    from fld.metrics.FID import FID
    from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor
except ModuleNotFoundError:
    FID = None
    InceptionFeatureExtractor = None
from pnpflow.dataloaders import CelebADataset, AFHQDataset
from torchvision import transforms
from torchvision.utils import save_image


img_dir_celeba = './data/celeba/img_align_celeba/'
partition_csv_celeba = './data/celeba/list_eval_partition.csv'
img_dir_afhq = './data/afhq_cat/test/cat/'


def _is_dist_ready():
    return dist.is_available() and dist.is_initialized()


def _is_main_process(args):
    return getattr(args, 'rank', 0) == 0


def _barrier_if_needed():
    if _is_dist_ready():
        dist.barrier()


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


class FLOW_MATCHING(object):

    def __init__(self, model, device, args):
        self.d = args.dim_image
        self.num_channels = args.num_channels
        self.device = device
        self.args = args
        self.lr = args.lr
        self.model = model.to(device)
        self.coupling = self.args.model

    def train_FM_model(self, train_loader, opt, num_epoch, start_epoch=0):

        ft_extractor = None
        test_feat = None
        if (_is_main_process(self.args)
                and self.args.dataset in ["celeba", "afhq_cat"]
                and InceptionFeatureExtractor is not None):
            ft_extractor = InceptionFeatureExtractor(save_path="features")
        elif _is_main_process(self.args) and self.args.dataset in ["celeba", "afhq_cat"]:
            print(
                "Skipping FID feature extraction because optional package 'fld' "
                "is not installed. Training will continue."
            )

        if ft_extractor is not None and self.args.dataset == "celeba":
            test_feat = ft_extractor.get_features(CelebADataset(
                img_dir_celeba, partition_csv_celeba, partition=2, transform=transforms.Compose([transforms.CenterCrop(178), transforms.Resize([self.args.dim_image, self.args.dim_image]),])), name=f"celeba{self.args.dim_image}_test")
        elif ft_extractor is not None and self.args.dataset == "afhq_cat":
            test_feat = AFHQDataset(
                img_dir_afhq, batchsize=self.args.batch_size_test, transform = transforms.Compose([transforms.Resize((256, 256)),
                transforms.ToTensor()]))
        elif _is_main_process(self.args) and self.args.dataset not in ["celeba", "afhq_cat"]:
            print(f"Skipping FID feature extraction for custom dataset {self.args.dataset}.")

        num_batches = len(train_loader) if hasattr(train_loader, '__len__') else None
        num_samples = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else None
        if _is_main_process(self.args):
            print(
                f"Training loader: samples={num_samples}, batches={num_batches}, "
                f"batch_size={self.args.batch_size_train}, no_cap={getattr(self.args, 'no_cap', False)}",
                flush=True,
            )

        tq = tqdm(range(start_epoch, num_epoch), desc='loss', disable=not _is_main_process(self.args))
        for ep in tq:
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(ep)
            for iteration, (x, labels) in enumerate(train_loader):
                if x.size(0) == 0:
                    continue
                if not getattr(self.args, 'no_cap', False) and iteration > 20:
                    break
                if _is_main_process(self.args):
                    print(f'Epoch: {ep}, iter: {iteration}', flush=True)
                x = x.to(self.device)
                z = torch.randn(
                    x.shape[0],
                    self.num_channels,
                    self.d,
                    self.d,
                    device=self.device,
                    requires_grad=True)

                t1 = torch.rand(x.shape[0], 1, 1, 1, device=self.device)

                # compute coupling
                if self.coupling == "ot":
                    x0 = z.clone()
                    x1 = x.clone()
                    a, b = np.ones(len(x0)) / len(x0), np.ones(len(x0)) / len(x0)

                    M = ot.dist(x0.view(len(x0), -1).cpu().data.numpy(),
                            x1.view(len(x1), -1).cpu().data.numpy())
                    plan = ot.emd(a, b, M)
                    p = plan.flatten()
                    p = p / p.sum()
                    choices = np.random.choice(
                        plan.shape[0] * plan.shape[1], p=p, size=len(x0), replace=True)
                    i, j = np.divmod(choices, plan.shape[1])
                    x0 = x0[i]
                    x1 = x1[j]
                else:
                    x0 = z
                    x1 = x
                xt = t1 * x1 + (1 - t1) * x0
                loss = torch.sum(
                    (self.model(xt, t1.squeeze()) - (x1 - x0))**2) / x.shape[0]
                opt.zero_grad()
                loss.backward()
                opt.step()

                # save loss in txt file
                if _is_main_process(self.args):
                    with open(self.save_path + 'loss_training.txt', 'a') as file:
                        file.write(
                            f'Epoch: {ep}, iter: {iteration}, Loss: {loss.item()}\n')

            # save samples, plot them, and compute FID on small dataset
            if _is_main_process(self.args):
                self.sample_plot(x, ep)
                if ep % 5 == 0:
                    # save model
                    torch.save(_unwrap_model(self.model).state_dict(),
                               self.model_path + 'model_{}.pt'.format(ep))

                    # FID is only meaningful/configured for the original RGB natural-image
                    # datasets. For custom HAADF microscopy patches, skip it.
                    if ft_extractor is not None and test_feat is not None:
                        print("Computing FID 5K")
                        num_gen = 5_000
                        fid_value = self.compute_fid(
                            num_gen, test_feat, ft_extractor, batch_size=124,
                            integration_method="euler", integration_steps=10)

                        with open(self.save_path + f'FID_{(num_gen // 1000)}k.txt', 'a') as file:
                            file.write(f'Epoch: {ep}, FID: {fid_value}\n')
                    else:
                        with open(self.save_path + 'FID_skipped.txt', 'a') as file:
                            file.write(f'Epoch: {ep}, FID skipped for dataset {self.args.dataset}\n')
                self._save_training_state(opt, ep)
            _barrier_if_needed()

    def _save_training_state(self, optimizer, completed_epoch):
        """Write the state required to continue after a completed epoch."""
        if not _is_main_process(self.args):
            return

        checkpoint = {
            'format_version': 1,
            'completed_epoch': completed_epoch,
            'model_state_dict': _unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'dataset': self.args.dataset,
            'model': self.args.model,
        }
        checkpoint_path = self.model_path + 'training_state_latest.pt'
        temporary_path = checkpoint_path + '.tmp'
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, checkpoint_path)

    def apply_flow_matching(self, NO_samples):
        self.model.eval()
        with torch.no_grad():
            model_class = cnf(_unwrap_model(self.model))
            latent = torch.randn(
                NO_samples,
                self.num_channels,
                self.d,
                self.d,
                device=self.device,
                requires_grad=False)
            z_t = odeint(model_class, latent,
                         torch.tensor([0.0, 1.0]).to(self.device),
                         atol=1e-5,
                         rtol=1e-5,
                         method='dopri5',
                         )
            x = z_t[-1].detach()
        self.model.train()
        return x

    def sample_plot(self, x, ep=None):
        try:
            os.makedirs(self.save_path + 'results_samplings/')
        except BaseException:
            pass

        reco = utils.postprocess(self.apply_flow_matching(16), self.args)
        gt = utils.postprocess(x[:16], self.args)
        utils.save_samples(reco.detach().cpu(), gt.detach().cpu(), self.save_path + 'results_samplings/' +
                           'samplings_ep_{}.pdf'.format(ep), self.args)

        # check the plots by saving training samples
        if ep == 0:
            utils.save_samples(gt.detach().cpu(), gt.detach().cpu(), self.save_path + 'results_samplings/' +
                               'train_samples_ep_{}.pdf'.format(ep), self.args)

    def generate_samples(self, integration_method="dopri5", tol=1e-5,
                         n_samples=1028, batch_size=None, num_channels=3,
                         integration_steps=100, tmax=1):
        """
        Return a tensor of size (TODO).
        """

        if batch_size is None:
            batch_size = n_samples

        images_list = []
        batches = [batch_size] * (n_samples // batch_size)
        if n_samples % batch_size:
            batches += [n_samples % batch_size]

        with torch.no_grad():
            for k, batch in enumerate(tqdm(batches)):
                time_points = torch.linspace(
                    0, tmax, int(tmax * integration_steps), device=self.device)

                x0 = torch.randn(batch, num_channels, self.d,
                                 self.d, device=self.device)
                model_class = cnf(_unwrap_model(self.model))
                traj = odeint(model_class, x0, time_points, rtol=tol, atol=tol,
                    method=integration_method)
                images_list.append(traj[-1, :])

        images = torch.cat(images_list, dim=0)
        return images

    def compute_fid(self, num_images_fid, train_feat, ft_extractor, batch_size=512, integration_method="dopri5", integration_steps=100,  epoch='final'):
        gen_images = self.generate_samples(integration_method=integration_method, tol=1e-4,
                                           n_samples=num_images_fid, batch_size=batch_size, integration_steps=integration_steps)
        rescaled_imgs = (gen_images * 127.5 + 128).clip(0, 255).to(torch.uint8)
        gen_feat = ft_extractor.get_tensor_features(
            rescaled_imgs)

        fid_val = FID().compute_metric(
            train_feat, None, gen_feat)

        # save the 16 first generated images in a grid
        os.makedirs(f"training_images/{self.args.dataset}", exist_ok=True)
        images = gen_images[:16]
        save_image(images, f"training_images/{self.args.dataset}/gen_images_epoch{epoch}.png")
        return fid_val

    def train(self, data_loaders):

        run_name = getattr(self.args, 'run_name', None)
        if run_name in [None, 'None', '']:
            run_name = datetime.now().strftime('%Y%m%d_%H%M%S') + f'_{self.args.dataset}_{self.args.model}'
        self.args.run_name = run_name

        self.save_path = os.path.join(
            self.args.root, 'results', self.args.dataset, self.args.model, run_name) + os.sep
        self.model_path = os.path.join(
            self.args.root, 'model', self.args.dataset, self.args.model, run_name) + os.sep
        if _is_main_process(self.args):
            os.makedirs(self.save_path, exist_ok=True)
            os.makedirs(self.model_path, exist_ok=True)
        _barrier_if_needed()

        # load model
        train_loader = data_loaders['train']

        resume_checkpoint = getattr(self.args, 'resume_checkpoint', None)
        if resume_checkpoint:
            resume_checkpoint = os.path.abspath(resume_checkpoint)
            if not os.path.isfile(resume_checkpoint):
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")

        # Create a new metadata file for fresh runs and append resume details otherwise.
        if _is_main_process(self.args):
            mode = 'a' if resume_checkpoint else 'w'
            with open(self.save_path + 'model_info.txt', mode) as file:
                if resume_checkpoint:
                    file.write(f'Resumed from: {resume_checkpoint}\n')
                    file.write(f'Resume target epochs: {self.args.num_epoch}\n')
                else:
                    file.write(f'PARAMETERS\n')
                    file.write(
                        f'Number of parameters: {sum(p.numel() for p in _unwrap_model(self.model).parameters())}\n')
                    file.write(f'Number of epochs: {self.args.num_epoch}\n')
                    file.write(f'Batch size: {self.args.batch_size_train}\n')
                    file.write(f'Learning rate: {self.lr}\n')
                    file.write(f'Run name: {run_name}\n')
                    file.write(f'Model path: {self.model_path}\n')
                    file.write(f'Results path: {self.save_path}\n')

        # Start training, optionally restoring an epoch-complete training state.
        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        start_epoch = 0
        if resume_checkpoint:
            _barrier_if_needed()
            checkpoint = torch.load(resume_checkpoint, map_location=self.device)
            required_keys = {'completed_epoch', 'model_state_dict', 'optimizer_state_dict'}
            missing_keys = required_keys.difference(checkpoint)
            if missing_keys:
                raise ValueError(
                    f"{resume_checkpoint} is not a resumable training-state checkpoint; "
                    f"missing: {sorted(missing_keys)}"
                )
            if checkpoint.get('dataset') not in {None, self.args.dataset}:
                raise ValueError("Resume checkpoint dataset does not match the requested dataset.")
            if checkpoint.get('model') not in {None, self.args.model}:
                raise ValueError("Resume checkpoint model does not match the requested model.")
            _unwrap_model(self.model).load_state_dict(checkpoint['model_state_dict'])
            opt.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = int(checkpoint['completed_epoch']) + 1
            if start_epoch >= self.args.num_epoch:
                raise ValueError(
                    f"Resume checkpoint completed epoch {start_epoch - 1}, which already reaches "
                    f"the requested total of {self.args.num_epoch} epochs."
                )
            if _is_main_process(self.args):
                print(
                    f"Resuming run {run_name} from epoch {start_epoch} "
                    f"to epoch {self.args.num_epoch - 1}.",
                    flush=True,
                )
        self.train_FM_model(
            train_loader, opt, num_epoch=self.args.num_epoch, start_epoch=start_epoch)

        # save final model
        if _is_main_process(self.args):
            torch.save(_unwrap_model(self.model).state_dict(), self.model_path + 'model_final.pt')
            for base_dir in [
                os.path.join(self.args.root, 'results', self.args.dataset, self.args.model),
                os.path.join(self.args.root, 'model', self.args.dataset, self.args.model),
            ]:
                os.makedirs(base_dir, exist_ok=True)
                with open(os.path.join(base_dir, 'latest_run.txt'), 'w') as file:
                    file.write(run_name + '\n')
        _barrier_if_needed()



class cnf(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x):
        with torch.no_grad():
            # z = self.model(x, t.squeeze())
            z = self.model(x, t.repeat(x.shape[0]))
        return z

