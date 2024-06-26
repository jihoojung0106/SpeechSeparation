from math import ceil
import warnings
from models.networks import Resnet18
from models.lipreading_model import Lipreading
import matplotlib.pyplot as plt
import torchvision
import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import wandb
import time
import os
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from sgmse import sampling
from sgmse.sdes import SDERegistry
from sgmse.backbones import BackboneRegistry
from sgmse.util.inference import evaluate_model
from sgmse.util.graphics import visualize_example, visualize_one
from sgmse.util.other import pad_spec, si_sdr_torch,load_json
VIS_EPOCHS = 5 
from itertools import chain


torch.autograd.set_detect_anomaly(True)

class MyScoreModel(pl.LightningModule):
    def __init__(self,
        backbone: str = "blade", sde: str = "ouvesde",
        lr: float = 1e-4, ema_decay: float = 0.999,
        t_eps: float = 3e-2, transform: str = 'none', nolog: bool = False,
        num_eval_files: int = 50, loss_type: str = 'mse', data_module_cls = None, **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: The underlying backbone DNN that serves as a score-based model.
                Must have an output dimensionality equal to the input dimensionality.
            sde: The SDE to use for the diffusion.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            reduce_mean: If `True`, average the loss across data dimensions.
                Otherwise sum the loss across data dimensions.
        """
        super().__init__()
        # Initialize Backbone DNN
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        kwargs.update(input_channels=4)
        self.dnn = dnn_cls(**kwargs)
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.dnn.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files
        self.facialnetmodel=self.build_facial()
        self.net_lipreading = self.build_lipreadingnet()
        for param in self.facialnetmodel.parameters():
            param.requires_grad = False
        for param in self.net_lipreading.parameters():
            param.requires_grad = False
        self.save_hyperparameters(ignore=['nolog'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        self._reduce_op = lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
        self.nolog = nolog 

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (3e-2 by default)")
        parser.add_argument("--num_eval_files", type=int, default=10, help="Number of files for speech enhancement performance evaluation during training.")
        parser.add_argument("--loss_type", type=str, default="mse", choices=("mse", "mae", "gaussian_entropy", "kristina", "sisdr", "time_mse"), help="The type of loss function to use.")
        parser.add_argument("--spatial_channels", type=int, default=1)
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.dnn.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.dnn.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        print("로드합니다.")
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        print(f"{self.global_step} 저장합니다.")
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.dnn.parameters())        # store current params in EMA
                self.ema.copy_to(self.dnn.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.dnn.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, err, err_time=None, err_mag=None):
        if self.loss_type == 'mse':
            losses = torch.square(err.abs())
            loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

        elif self.loss_type == 'mae':
            losses = err.abs()
            loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

        return loss

    def _weighted_mean(self, x, w):
        return torch.mean(x * w)

    def _raw_dnn_output(self, x, t, y,visual_embedding):
        dnn_input = torch.cat([x, y], dim=1) #b,2*d,f,t
        return self.dnn(dnn_input, t,visual_embedding)

    def forward(self, x, t, y, visual_embedding,**kwargs):
        score = -self._raw_dnn_output(x, t, y,visual_embedding)
        std = self.sde._std(t, y=y)
        if std.ndim < y.ndim:
            std = std.view(*std.size(), *((1,)*(y.ndim - std.ndim)))
        return score

    def _step(self, batch, batch_idx): #x가 clean, y가 noisy
        mouthroi_A1 = batch["mouthroi_A1"] #(2,1,64,112,112)
        frame_A = batch["frame_A"] #(2,3,224,224)
        audio_spec_A1 = batch["audio_spec_A1"]#(1,1,257,256)
        audio_spec_mix1 = batch["audio_spec_mix1"] #(1,1,257,256)
        mouthroi_A1_embed = self.net_lipreading(Variable(mouthroi_A1, requires_grad=False), 64) #(2,512,1,64)
        identity_feature_A = self.facialnetmodel(Variable(frame_A, requires_grad=False))
        identity_feature_A = F.normalize(identity_feature_A, p=2, dim=1) #(2,128,1,1)
        identity_feature_A = identity_feature_A.repeat(1, 1, 1, mouthroi_A1_embed.shape[-1]) #(2,128,1,64)
        visual_feature_A1 = torch.cat((identity_feature_A, mouthroi_A1_embed), dim=1) #(2,640,1,64)
        visual_embedding=visual_feature_A1.permute(0, 3, 1, 2).squeeze(3) #(2,64=순서,640=embed_d)
        x,y=audio_spec_A1,audio_spec_mix1
        
        
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        mean, std = self.sde.marginal_prob(x, t, y)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5 ---> problem: this cannot work for FreqOUVE, because is standard, and tries to match a score with a sigma which is not standard
        if std.ndim < y.ndim:
            std = std.view(*std.size(), *((1,)*(y.ndim - std.ndim)))
        sigmas = std
        perturbed_data = mean + sigmas * z
        score = self(perturbed_data, t, y, visual_embedding)
        err = score * sigmas + z
        loss = self._loss(err)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=self.data_module.batch_size)
        return loss

    def validation_step(self, batch, batch_idx, discriminative=False, sr=16000):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size)

        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq_est, si_sdr_est, estoi_est, spec, audio = evaluate_model(self, self.num_eval_files, spec=not self.current_epoch%VIS_EPOCHS, audio=not self.current_epoch%VIS_EPOCHS, discriminative=discriminative)
            print(f"PESQ at epoch {self.current_epoch} : {pesq_est:.2f}")
            print(f"SISDR at epoch {self.current_epoch} : {si_sdr_est:.1f}")
            print(f"ESTOI at epoch {self.current_epoch} : {estoi_est:.2f}")
            print('__________________________________________________________________')
            
            self.log('ValidationPESQ', pesq_est, on_step=False, on_epoch=True)
            self.log('ValidationSISDR', si_sdr_est, on_step=False, on_epoch=True)
            self.log('ValidationESTOI', estoi_est, on_step=False, on_epoch=True)

            if audio is not None:
                y_list, x_hat_list, x_list = audio
                for idx, (y, x_hat, x) in enumerate(zip(y_list, x_hat_list, x_list)):
                    if self.current_epoch == 0:
                        self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Mix/{idx}", (y / torch.max(torch.abs(y))).unsqueeze(0), sample_rate=sr)
                        self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Clean/{idx}", (x / torch.max(x)).unsqueeze(0), sample_rate=sr)
                    self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Estimate/{idx}", (x_hat / torch.max(torch.abs(x_hat))).unsqueeze(0), sample_rate=sr)

            if spec is not None:
                figures = []
                y_stft_list, x_hat_stft_list, x_stft_list = spec
                for idx, (y_stft, x_hat_stft, x_stft) in enumerate(zip(y_stft_list, x_hat_stft_list, x_stft_list)):
                    figures.append(
                        visualize_example(
                        torch.abs(y_stft), 
                        torch.abs(x_hat_stft), 
                        torch.abs(x_stft), return_fig=True))
                self.logger.experiment.add_figure(f"Epoch={self.current_epoch}/Spec", figures)

        return loss

    def to(self, *args, **kwargs):
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, visual_embedding,N=None, minibatch=None, scale_factor=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y, visual_embedding=visual_embedding,**kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=1, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_ode_sampler(sde, self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def enhance(self, y, sampler_type="pc", predictor="reverse_diffusion",
        corrector="ald", N=50, corrector_steps=1, snr=0.5, timeit=False,
        scale_factor = None, return_stft=False,
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        start = time.time()
        T_orig = y.size(1)
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        if sampler_type == "pc":
            sampler = self.get_pc_sampler(predictor, corrector, Y, N=N,
                corrector_steps=corrector_steps, snr=snr, intermediate=False,
                scale_factor=scale_factor,
                **kwargs)
        elif sampler_type == "ode":
            sampler = self.get_ode_sampler(Y, N=N, **kwargs)
        else:
            print("{} is not a valid sampler type!".format(sampler_type))
        sample, nfe = sampler()

        if return_stft:
            return sample.squeeze(), Y.squeeze(), T_orig, norm_factor

        x_hat = self.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu()
        end = time.time()
        if timeit:
            sr = 16000
            rtf = (end-start)/(len(x_hat)/sr)
            return x_hat, nfe, rtf
        else:
            return x_hat
        
    def build_facial(self, pool_type='maxpool', fc_out=128, with_fc=True, weights="checkpoints/facial_best.pth"):
        pretrained = False
        original_resnet = torchvision.models.resnet18(pretrained)
        net = Resnet18(original_resnet, pool_type=pool_type, with_fc=with_fc, fc_in=512, fc_out=fc_out)

        if weights is not None:
            print('Loading weights for facial attributes analysis stream')
            pretrained_state = torch.load(weights)
            model_state = net.state_dict()
            pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
            model_state.update(pretrained_state)
            net.load_state_dict(model_state)
        return net
    
    def build_lipreadingnet(self, config_path="configs/lrw_snv1x_tcn2x.json", weights='checkpoints/lipreading_best.pth', extract_feats=True):
        if os.path.exists(config_path):
            args_loaded = load_json(config_path)
            print('Lipreading configuration file loaded.')
            tcn_options = { 'num_layers': args_loaded['tcn_num_layers'],
                            'kernel_size': args_loaded['tcn_kernel_size'],
                            'dropout': args_loaded['tcn_dropout'],
                            'dwpw': args_loaded['tcn_dwpw'],
                            'width_mult': args_loaded['tcn_width_mult']}                 
        net = Lipreading(tcn_options=tcn_options,
                        backbone_type=args_loaded['backbone_type'],
                        relu_type=args_loaded['relu_type'],
                        width_mult=args_loaded['width_mult'],
                        extract_feats=extract_feats)

        if len(weights) > 0:
            print('Loading weights for lipreading stream')
            net.load_state_dict(torch.load(weights))
        return net


class DiscriminativeModel(MyScoreModel):

    def forward(self, y,visual_embedding):
        if self.dnn.FORCE_STFT_OUT:
            y = self._istft(self._backward_transform(y.clone().squeeze(1)))
        t = torch.ones(y.shape[0], device=y.device)
        x_hat = self.dnn(y, t, visual_embedding)
        return x_hat

    def _loss(self, x, xhat):
        if self.dnn.FORCE_STFT_OUT:
            x = self._istft(self._backward_transform(x.clone().squeeze(1)))

        if self.loss_type == 'mse':
            losses = torch.square((x - xhat).abs())
            loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

        elif self.loss_type == 'mae':
            losses = (x - xhat).abs()
            loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

        elif self.loss_type == "sisdr":
            loss = - torch.mean(torch.stack([si_sdr_torch(x[i], xhat[i]) for i in range(x.size(0))]))
        return loss

    def _step(self, batch, batch_idx):
        X, Y = batch
        Xhat = self(Y)
        loss = self._loss(X, Xhat)
        return loss

    def enhance(self, y, **ignored_kwargs):
        with torch.no_grad():
            norm_factor = y.abs().max().item()
            T_orig = y.size(1)

            if self.data_module.return_time:
                Y = torch.unsqueeze((y/norm_factor).cuda(), 0) #1,D=1,T
            else:
                Y = torch.unsqueeze(self._forward_transform(self._stft((y/norm_factor).cuda())), 0) #1,D,F,T
                Y = pad_spec(Y)
            X_hat = self(Y)
            if self.dnn.FORCE_STFT_OUT:
                X_hat = self._forward_transform(self._stft(X_hat)).unsqueeze(1)

            if self.data_module.return_time:
                x_hat = X_hat.squeeze()
            else:
                x_hat = self.to_audio(X_hat.squeeze(), T_orig)

            return (x_hat * norm_factor).squeeze()
                    
    def validation_step(self, batch, batch_idx):
        return super().validation_step(batch, batch_idx, discriminative=True)
    

















class StochasticRegenerationModel(pl.LightningModule):
    def __init__(self,
        backbone_denoiser: str, backbone_score: str, sde: str,
        lr: float = 1e-4, ema_decay: float = 0.999,
        t_eps: float = 3e-2, nolog: bool = False, num_eval_files: int = 50,
        loss_type_denoiser: str = "none", loss_type_score: str = 'mse', data_module_cls = None, 
        mode = "regen-joint-training", condition = "both",debug=False,
        **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: The underlying backbone DNN that serves as a score-based model.
                Must have an output dimensionality equal to the input dimensionality.
            sde: The SDE to use for the diffusion.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            reduce_mean: If `True`, average the loss across data dimensions.
                Otherwise sum the loss across data dimensions.
        """
        super().__init__()
        # Initialize Backbone DNN
        kwargs_denoiser = kwargs
        kwargs_denoiser.update(input_channels=2)
        kwargs_denoiser.update(discriminative=True)
        self.denoiser_net = BackboneRegistry.get_by_name(backbone_denoiser)(**kwargs) if backbone_denoiser != "none" else None

        kwargs.update(input_channels=(6 if condition == "both" else 4))
        kwargs_denoiser.update(discriminative=False)
        self.score_net = BackboneRegistry.get_by_name(backbone_score)(**kwargs) if backbone_score != "none" else None

        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        self.t_eps = t_eps

        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        if debug:
            self.facialnetmodel=self.build_facial(weights=None)
            self.net_lipreading = self.build_lipreadingnet(weights=None)
        else:
            self.facialnetmodel=self.build_facial()
            self.net_lipreading = self.build_lipreadingnet()
        for param in self.facialnetmodel.parameters():
            param.requires_grad = False
        for param in self.net_lipreading.parameters():
            param.requires_grad = False
        # Only include parameters of denoiser_net and score_net for EMA
        # self.trainable_params = chain(self.denoiser_net.parameters(), self.score_net.parameters())

        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False

        self.loss_type_denoiser = loss_type_denoiser
        self.loss_type_score = loss_type_score
        if "weighting_denoiser_to_score" in kwargs.keys():
            self.weighting_denoiser_to_score = kwargs["weighting_denoiser_to_score"]
        else:
            self.weighting_denoiser_to_score = .5
        self.condition = condition
        self.mode = mode
        self.configure_losses()

        self.num_eval_files = num_eval_files
        self.save_hyperparameters(ignore=['nolog'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        self._reduce_op = lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
        self.nolog = nolog

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (3e-2 by default)")
        parser.add_argument("--num_eval_files", type=int, default=10, help="Number of files for speech enhancement performance evaluation during training.")
        parser.add_argument("--loss_type_denoiser", type=str, default="mse", choices=("none", "mse", "mae", "sisdr", "mse_cplx+mag", "mse_time+mag"), help="The type of loss function to use.")
        parser.add_argument("--loss_type_score", type=str, default="mse", choices=("none", "mse", "mae"), help="The type of loss function to use.")
        parser.add_argument("--weighting_denoiser_to_score", type=float, default=0.9, help="a, as in L = a * L_denoiser + (1-a) * .")
        parser.add_argument("--condition", default="both", choices=["noisy", "post_denoiser", "both"])
        parser.add_argument("--spatial_channels", type=int, default=1)
        return parser

    def configure_losses(self):
        # Score Loss
        if self.loss_type_score == "mse":
            self.loss_fn_score = lambda err: self._reduce_op(torch.square(torch.abs(err)))
        elif self.loss_type_score == "mae":
            self.loss_fn_score = lambda err: self._reduce_op(torch.abs(err))
        elif self.loss_type_score == "none":
            raise NotImplementedError
            self.loss_fn_score = None
        else:
            raise NotImplementedError
        
        # Denoiser Loss
        if self.loss_type_denoiser == "mse":
            self.loss_fn_denoiser = lambda x, y: self._reduce_op(torch.square(torch.abs(x - y)))
        elif self.loss_type_denoiser == "mae":
            self.loss_fn_denoiser = lambda x, y: self._reduce_op(torch.abs(x - y))
        elif self.loss_type_denoiser == "none":
            self.loss_fn_denoiser = None
        else:
            raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    def load_denoiser_model(self, checkpoint):
        self.denoiser_net = DiscriminativeModel.load_from_checkpoint(checkpoint).dnn
        if self.mode == "regen-freeze-denoiser":
            for param in self.denoiser_net.parameters():
                param.requires_grad = False

    def load_score_model(self, checkpoint):
        self.score_net = MyScoreModel.load_from_checkpoint(checkpoint).dnn

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, err, y_denoised, x):
        loss_score = self.loss_fn_score(err) if self.loss_type_score != "none" else None
        loss_denoiser = self.loss_fn_denoiser(y_denoised, x) if self.loss_type_denoiser != "none" else None
        if loss_score is not None:
            
            if loss_denoiser is not None: #여기 비율 조정
                self.weighting_denoiser_to_score=1.0
                loss = self.weighting_denoiser_to_score * loss_denoiser + (1 - self.weighting_denoiser_to_score) * loss_score
            else:
                loss = loss_score
        else:
            loss = loss_denoiser
        return loss, loss_score, loss_denoiser

    def _weighted_mean(self, x, w):
        return torch.mean(x * w)
    #이게 call된대.
    def forward_score(self, x, t, visual_embedding,score_conditioning, sde_input, **kwargs):
        dnn_input = torch.cat([x] + score_conditioning, dim=1) #b,n_input*d,f,t
        score = -self.score_net(x=dnn_input, time_cond=t,visual_embedding=visual_embedding) #NCSNpp
        std = self.sde._std(t, y=sde_input)
        if std.ndim < sde_input.ndim:
            std = std.view(*std.size(), *((1,)*(sde_input.ndim - std.ndim)))
        return score

    def forward_denoiser(self, y, visual_embedding,**kwargs):
        x_hat = self.denoiser_net(x=y,visual_embedding=visual_embedding)
        return x_hat

    def _step(self, batch, batch_idx):
        mouthroi_A1 = batch["mouthroi_A1"] #(2,1,64,112,112)
        frame_A = batch["frame_A"] #(2,3,224,224)
        audio_spec_A1 = batch["audio_spec_A1"]#(1,1,257,256)
        audio_spec_mix1 = batch["audio_spec_mix1"] #(1,1,257,256)
        with torch.set_grad_enabled(False):
            mouthroi_A1_embed = self.net_lipreading(Variable(mouthroi_A1, requires_grad=False), 64) #(2,512,1,64)
            identity_feature_A = self.facialnetmodel(Variable(frame_A, requires_grad=False))
        identity_feature_A = F.normalize(identity_feature_A, p=2, dim=1) #(2,128,1,1)
        identity_feature_A = identity_feature_A.repeat(1, 1, 1, mouthroi_A1_embed.shape[-1]) #(2,128,1,64)
        visual_feature_A1 = torch.cat((identity_feature_A, mouthroi_A1_embed), dim=1) #(2,640,1,64)
        visual_embedding=visual_feature_A1.permute(0, 3, 1, 2).squeeze(3) #(2,64=순서,640=embed_d)
        x,y=audio_spec_A1,audio_spec_mix1

        # Denoising step
        with torch.set_grad_enabled(self.mode != "regen-freeze-denoiser"):
            y_denoised = self.forward_denoiser(y,visual_embedding)
        

        # Score step

        sde_target = x
        sde_input = y_denoised
        # Forward process
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        mean, std = self.sde.marginal_prob(sde_target, t, sde_input)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        if std.ndim < y.ndim:
            std = std.view(*std.size(), *((1,)*(y.ndim - std.ndim)))
        sigmas = std
        perturbed_data = mean + sigmas * z

        # Score estimation
        if self.condition == "noisy":
            score_conditioning = [y]
        elif self.condition == "post_denoiser":
            score_conditioning = [y_denoised]
        elif self.condition == "both":
            score_conditioning = [y, y_denoised]
        else:
            raise NotImplementedError(f"Don't know the conditioning you have wished for: {self.condition}")

        score = self.forward_score(perturbed_data, t, visual_embedding,score_conditioning, sde_input)
        err = score * sigmas + z

        loss, loss_score, loss_denoiser = self._loss(err, y_denoised, x)

        return loss, loss_score, loss_denoiser

    def training_step(self, batch, batch_idx):
        loss, loss_score, loss_denoiser = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=self.data_module.batch_size)
        self.log('train_loss_score', loss_score, on_step=True, on_epoch=True, batch_size=self.data_module.batch_size)
        if loss_denoiser is not None:
            self.log('train_loss_denoiser', loss_denoiser, on_step=True, on_epoch=True, batch_size=self.data_module.batch_size)
        return loss

    # def validation_step(self, batch, batch_idx, discriminative=False, sr=16000):
        loss, loss_score, loss_denoiser = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size)
        self.log('valid_loss_score', loss_score, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size)
        if loss_denoiser is not None:
            self.log('valid_loss_denoiser', loss_denoiser, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size)

        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq_est, si_sdr_est, estoi_est, spec, audio = evaluate_model(self, self.num_eval_files, spec=not self.current_epoch%VIS_EPOCHS, audio=not self.current_epoch%VIS_EPOCHS, discriminative=discriminative)
            print(f"PESQ at epoch {self.current_epoch} : {pesq_est:.2f}")
            print(f"SISDR at epoch {self.current_epoch} : {si_sdr_est:.1f}")
            print(f"ESTOI at epoch {self.current_epoch} : {estoi_est:.2f}")
            print('__________________________________________________________________')
            
            self.log('ValidationPESQ', pesq_est, on_step=False, on_epoch=True)
            self.log('ValidationSISDR', si_sdr_est, on_step=False, on_epoch=True)
            self.log('ValidationESTOI', estoi_est, on_step=False, on_epoch=True)

            if audio is not None:
                y_list, x_hat_list, x_list = audio
                for idx, (y, x_hat, x) in enumerate(zip(y_list, x_hat_list, x_list)):
                    if self.current_epoch == 0:
                        self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Mix/{idx}", (y / torch.max(torch.abs(y))).unsqueeze(0), sample_rate=sr)
                        self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Clean/{idx}", (x / torch.max(x)).unsqueeze(0), sample_rate=sr)
                    self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Estimate/{idx}", (x_hat / torch.max(torch.abs(x_hat))).unsqueeze(0), sample_rate=sr)

            if spec is not None:
                figures = []
                y_stft_list, x_hat_stft_list, x_stft_list = spec
                for idx, (y_stft, x_hat_stft, x_stft) in enumerate(zip(y_stft_list, x_hat_stft_list, x_stft_list)):
                    figures.append(
                        visualize_example(
                        torch.abs(y_stft), 
                        torch.abs(x_hat_stft), 
                        torch.abs(x_stft), return_fig=True))
                self.logger.experiment.add_figure(f"Epoch={self.current_epoch}/Spec", figures)

        return loss

    def to(self, *args, **kwargs):
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, visual_embedding,N=None, minibatch=None, scale_factor=None, conditioning=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self.forward_score, y=y, visual_embedding=visual_embedding,conditioning=conditioning, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self.forward_score, y=y_mini, conditioning=conditioning, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=1, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_ode_sampler(sde, self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        back=self._backward_transform(spec)
        return back,self._istft(back,length)
        
    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def separate(self, Y, visual_embedding, sampler_type="pc", predictor="reverse_diffusion",
        corrector="langevin", N=500, corrector_steps=1, snr=0.5, timeit=False,
        scale_factor = None, return_stft=False, denoiser_only=False,
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        
        with torch.no_grad():

            if self.denoiser_net is not None:
                Y_denoised = self.forward_denoiser(Y,visual_embedding=visual_embedding)
            else:
                Y_denoised = None

            if self.score_net is not None and not denoiser_only:
                # Conditioning
                if self.condition == "noisy":
                    score_conditioning = [Y]
                elif self.condition == "post_denoiser":
                    score_conditioning = [Y_denoised]
                elif self.condition == "both":
                    score_conditioning = [Y, Y_denoised]
                else:
                    raise NotImplementedError(f"Don't know the conditioning you have wished for: {self.condition}")

                # Reverse process
                if sampler_type == "pc":
                    sampler = self.get_pc_sampler(predictor, corrector, Y_denoised,visual_embedding, N=N,
                        corrector_steps=corrector_steps, snr=snr, intermediate=False,
                        scale_factor=scale_factor, conditioning=score_conditioning,
                        **kwargs)
                elif sampler_type == "ode":
                    sampler = self.get_ode_sampler(Y_denoised, N=N, 
                        conditioning=score_conditioning, 
                        **kwargs)
                else:
                    print("{} is not a valid sampler type!".format(sampler_type))
                sample, nfe = sampler() #x_result,ns
            else:
                sample = Y_denoised
        return Y_denoised,sample
        
    def enhance(self, y, sampler_type="pc", predictor="reverse_diffusion",
        corrector="none", N=30, corrector_steps=1, snr=0.5, timeit=False,
        scale_factor = None, return_stft=False, denoiser_only=False,
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        start = time.time()
        T_orig = y.size(1)
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        with torch.no_grad():

            if self.denoiser_net is not None:
                Y_denoised = self.forward_denoiser(Y)
            else:
                Y_denoised = None

            if self.score_net is not None and not denoiser_only:
                # Conditioning
                if self.condition == "noisy":
                    score_conditioning = [Y]
                elif self.condition == "post_denoiser":
                    score_conditioning = [Y_denoised]
                elif self.condition == "both":
                    score_conditioning = [Y, Y_denoised]
                else:
                    raise NotImplementedError(f"Don't know the conditioning you have wished for: {self.condition}")

                # Reverse process
                if sampler_type == "pc":
                    sampler = self.get_pc_sampler(predictor, corrector, Y_denoised, N=N,
                        corrector_steps=corrector_steps, snr=snr, intermediate=False,
                        scale_factor=scale_factor, conditioning=score_conditioning,
                        **kwargs)
                elif sampler_type == "ode":
                    sampler = self.get_ode_sampler(Y_denoised, N=N, 
                        conditioning=score_conditioning, 
                        **kwargs)
                else:
                    print("{} is not a valid sampler type!".format(sampler_type))
                sample, nfe = sampler()
            else:
                sample = Y_denoised

            if return_stft:
                return sample.squeeze(), Y.squeeze(), T_orig, norm_factor

        x_hat = self.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu()
        end = time.time()
        if timeit:
            sr = 16000
            rtf = (end-start)/(len(x_hat)/sr)
            return x_hat, nfe, rtf
        else:
            return x_hat

    def build_facial(self, pool_type='maxpool', fc_out=128, with_fc=True, weights="checkpoints/facial_best.pth"):
        pretrained = False
        original_resnet = torchvision.models.resnet18(pretrained)
        net = Resnet18(original_resnet, pool_type=pool_type, with_fc=with_fc, fc_in=512, fc_out=fc_out)

        if weights is not None:
            print('Loading weights for facial attributes analysis stream')
            pretrained_state = torch.load(weights)
            model_state = net.state_dict()
            pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
            model_state.update(pretrained_state)
            net.load_state_dict(model_state)
        return net
    
    def build_lipreadingnet(self, config_path="configs/lrw_snv1x_tcn2x.json", weights='checkpoints/lipreading_best.pth', extract_feats=True):
        if os.path.exists(config_path):
            args_loaded = load_json(config_path)
            print('Lipreading configuration file loaded.')
            tcn_options = { 'num_layers': args_loaded['tcn_num_layers'],
                            'kernel_size': args_loaded['tcn_kernel_size'],
                            'dropout': args_loaded['tcn_dropout'],
                            'dwpw': args_loaded['tcn_dwpw'],
                            'width_mult': args_loaded['tcn_width_mult']}                 
        net = Lipreading(tcn_options=tcn_options,
                        backbone_type=args_loaded['backbone_type'],
                        relu_type=args_loaded['relu_type'],
                        width_mult=args_loaded['width_mult'],
                        extract_feats=extract_feats)

        if weights is not None:
            print('Loading weights for lipreading stream')
            net.load_state_dict(torch.load(weights))
        return net
