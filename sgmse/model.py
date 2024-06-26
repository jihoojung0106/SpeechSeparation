import time
from math import ceil
import warnings
import torchvision
import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
from talk_model.talkNetModel import talkNetModel
from sgmse import sampling
from sgmse.sdes import SDERegistry
from sgmse.backbones import BackboneRegistry
from sgmse.util.inference import evaluate_model
from sgmse.util.other import pad_spec,load_json
from talk_model.networks import Resnet18_
from torch.autograd import Variable
import os
import torch.nn.functional as F
from models.lipreading_model import Lipreading


class MyScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (3e-2 by default)")
        parser.add_argument("--num_eval_files", type=int, default=20, help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="mse", choices=("mse", "mae"), help="The type of loss function to use.")
        return parser

    def __init__(
        self, backbone, sde, lr=1e-4, ema_decay=0.999, t_eps=3e-2,
        num_eval_files=20, loss_type='mse', data_module_cls=None, **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()
        # Initialize Backbone DNN
        dnn_cls = BackboneRegistry.get_by_name(backbone) #NCSNpp, U-Net 아키텍처
        self.dnn = dnn_cls(**kwargs) #NCSNpp를 초기화했다.
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde) #OUVESDE
        self.sde = sde_cls(**kwargs) #OUVESDE 초기화
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0) #specdatamodule
        # self.talknetmodel=talkNetModel()
        self.facialnetmodel=self.build_facial()
        self.net_lipreading = self.build_lipreadingnet()
        for param in self.facialnetmodel.parameters():
            param.requires_grad = False
        for param in self.net_lipreading.parameters():
            param.requires_grad = False
        self.ema = ExponentialMovingAverage(self.dnn.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files
        
        
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
        
    #학습 모드와 평가 모드를 전환합니다.
    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.dnn.parameters())        # store current params in EMA
                self.ema.copy_to(self.dnn.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None: #이거 안 함
                    self.ema.restore(self.dnn.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, err): #err : (2,1,256,256)
        if self.loss_type == 'mse':
            losses = torch.square(err.abs()) #(2,1,256,256)
        elif self.loss_type == 'mae':
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1)) #배치당 loss의 평균
        return loss #한 개의 텐서

    def _step(self, batch, batch_idx):
        # batch[]
        mouthroi_A1 = batch["mouthroi_A1"] #(2,1,64,112,112)
        frame_A = batch["frame_A"] #(2,3,224,224)
        audio_spec_A1 = batch["audio_spec_A1"]#(1,1,257,256)
        audio_spec_mix1 = batch["audio_spec_mix1"] #(1,1,257,256)
        # mouthroi_A1_embed=self.talknetmodel(mouthroi_A1) #(1,64,128)
        mouthroi_A1_embed = self.net_lipreading(Variable(mouthroi_A1, requires_grad=False), 64) #(2,512,1,64)
        identity_feature_A = self.facialnetmodel(Variable(frame_A, requires_grad=False))
        identity_feature_A = F.normalize(identity_feature_A, p=2, dim=1) #(2,128,1,1)
        identity_feature_A = identity_feature_A.repeat(1, 1, 1, mouthroi_A1_embed.shape[-1]) #(2,128,1,64)
        visual_feature_A1 = torch.cat((identity_feature_A, mouthroi_A1_embed), dim=1) #(2,640,1,64)
        visual_embedding=visual_feature_A1.permute(0, 3, 1, 2).squeeze(3)
        # x, y = batch #x는 clean, y가 noisy
        x,y=audio_spec_A1,audio_spec_mix1 #(1,1,257,256)
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        mean, std = self.sde.marginal_prob(x, t, y)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        sigmas = std[:, None, None, None]
        perturbed_data = mean + sigmas * z #14번 식 perturbed_data=x_t
        score = self(perturbed_data, t, y, visual_embedding) #self.forward 불러버리기~
        err = score * sigmas + z #15번식, 얘네 맨날 논문이랑 다르게 score에 std 곱해버리잖아.
        loss = self._loss(err)
        return loss

    def training_step(self, batch, batch_idx): #256개의 주파수 빈과 256개의 시간 프레임을 가진 복소수 텐서
        loss = self._step(batch, batch_idx) #batch는 [(2,1,256,256),(2,1,256,256)]
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        # print(f"train_loss : {loss}")
        # current_step = self.trainer.global_step  # 현재 global_step 값을 가져옵니다.
        # print(f"Global step: {current_step}")
        return loss

    # def validation_step(self, batch, batch_idx):
    #     loss = self._step(batch, batch_idx)
    #     self.log('valid_loss', loss, on_step=False, on_epoch=True)

    #     # Evaluate speech enhancement performance
    #     if batch_idx == 0 and self.num_eval_files != 0:
    #         pesq, si_sdr, estoi = evaluate_model(self, self.num_eval_files)
    #         self.log('pesq', pesq, on_step=False, on_epoch=True)
    #         self.log('si_sdr', si_sdr, on_step=False, on_epoch=True)
    #         self.log('estoi', estoi, on_step=False, on_epoch=True)

    #     return loss

    def forward(self, x, t, y,visual_embedding):
        # Concatenate y as an extra channel
        dnn_input = torch.cat([x, y], dim=1)
        
        # the minus is most likely unimportant here - taken from Song's repo
        score = -self.dnn(dnn_input, t, visual_embedding)
        return score

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, visual_embedding, N=None, minibatch=None, **kwargs): #predictor_name=reverse_diffusion, corrector_name=ald
        N = self.sde.N if N is None else N #N=30
        sde = self.sde.copy() #OUVESDE 
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.my_get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y,visual_embedding=visual_embedding, **kwargs)
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

    def get_ode_sampler(self, y, visual_embedding,N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y,visual_embedding=visual_embedding, **kwargs)
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
                return sample, ns
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
        return back,self._istft(back)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def enhance(self, y, sampler_type="pc", predictor="reverse_diffusion",
        corrector="ald", N=30, corrector_steps=1, snr=0.5, timeit=False,
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        sr=16000
        start = time.time()
        T_orig = y.size(1) 
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        if sampler_type == "pc":
            sampler = self.get_pc_sampler(predictor, corrector, Y.cuda(), N=N, 
                corrector_steps=corrector_steps, snr=snr, intermediate=False,
                **kwargs)
        elif sampler_type == "ode":
            sampler = self.get_ode_sampler(Y.cuda(), N=N, **kwargs)
        else:
            print("{} is not a valid sampler type!".format(sampler_type))
        sample, nfe = sampler()
        x_hat = self.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()
        end = time.time()
        if timeit:
            rtf = (end-start)/(len(x_hat)/sr)
            return x_hat, nfe, rtf
        else:
            return x_hat
        
    def build_facial(self, pool_type='maxpool', fc_out=128, with_fc=True, weights="checkpoints/facial_best.pth"):
        pretrained = False
        original_resnet = torchvision.models.resnet18(pretrained)
        net = Resnet18_(original_resnet, pool_type=pool_type, with_fc=with_fc, fc_in=512, fc_out=fc_out)

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

