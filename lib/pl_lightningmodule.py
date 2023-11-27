from matplotlib import pyplot as plt
from typing import Any
import pytorch_lightning as pl
from pathlib import Path
import torch
from torch import nn
import time
from lib.torch_layers import Modular, MSE_img, L1_img
from numpy import ceil
from skimage import io
import numpy as np

class LitCNN(pl.LightningModule):
    '''
    Class defining the training, testing etc logic around the models.
    Takes an arbitrary nn.Module defining the architecture, for specific models we can subclass LitCNN and pass the corresponding model.
    '''
    def __init__(self, 
        model : nn.Module,
        PL_scale : float = 100, 
        lr : float = 1e-4, 
        optimizer : str = "Adam",
        optimizer_params : dict = {},
        scheduler : str | None = 'ReduceLROnPlateau',
        scheduler_params : dict[str, int | float | bool | str] = dict(
            threshold=1e-4, 
            patience=4, 
            verbose=True,
            threshold_mode='abs'
        ),
        scheduler_interval : str = "epoch",
        scheduler_frequency : int = 1,
        loss_fn : str = 'MSELoss',
        channels_last : bool = True,
        model_weights : str | None = None,
        previous_model  :   str | None = None,
        train_previous_model    :   bool = False,
        **kwargs_tracking
        ):
        '''
        Inputs:
            model               -   the torch model to be used
            PL_scale            -  based on the dataset, only to convert grayscale loss/error to dB
            lr                  -  base learning rate
            optimizer           -  optimizer name
            optimizer_params    -  parameters for initialization of optimizer
            scheduler           -  scheduler name 
            scheduler_params    -  parameters for initialization of scheduler
            scheduler_interval, scheduler_frequency - define how PL handles the scheduler (change e.g. for annealing)
            loss_fn             -  function to calculating loss, subclass of nn.Module, given as string for CLI and logging to work properly
            channels_last       -  use channels_last tensor layout
            model_weights       -  path to checkpoint to load only weights but nothing else saved in the checkpoint (e.g. optimizer states)
            previous_model      -  log directory of an existing model that shall be used before the current one (curriculum training as in RadioUNet)
            train_previous_model-  whether to train the previous_model as well or not (freeze it)
            **kwargs_tracking   -  not used, only to track other parameters (e.g. from the dataset) with tensorboard   
        '''
        super().__init__()
        #########
        self.example_input_array = torch.zeros((32, 5, 256, 256))
        #########
        if previous_model is not None:
            print(f'LitCNN previous: {previous_model}')
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        if model_weights is not None:
            weights = torch.load(Path(model_weights))['state_dict']
            ### remove the "model." in the beginning
            self.model.load_state_dict({k[6:] : v for k, v in weights.items()})
        self.model_weights = model_weights
        self.PL_scale = PL_scale
        self.lr = lr 
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency
        self.loss_fn = getattr(torch.nn, loss_fn)()
        self.channels_last = channels_last

        if previous_model is None:
            self.previous = previous_model
        else:
            import yaml
            sub_dir = list(Path(previous_model).iterdir())
            assert len(sub_dir)==1, f'run_dir of {previous_model} could not be determined uniquely, found sub_dirs {sub_dir}'
            sub_dir = sub_dir[0]
            ckpt1 = list((sub_dir / 'checkpoints').glob('*loss*.ckpt'))
            assert len(ckpt1) == 1, f'checkpoint for model1 ({previous_model}) not (uniquely) determined, found {ckpt1}'
            ckpt1 = ckpt1[0]
            config = sub_dir / 'config.yaml'
            assert config.is_file(), f'config for model1 ({previous_model})not existing/not a file'
            with open(config, 'r') as f:
                params = yaml.safe_load(f)
            model1_class = LitModelDict[params['model']['class_path'].split('.')[-1]]
            self.previous = model1_class.load_from_checkpoint(ckpt1)
            print(f'loaded {model1_class} from {ckpt1} using {config}')
            if not train_previous_model:
                self.previous.freeze()

        ### print unused params to the log/terminal to allow recognizing errors, in particular arguments not understood due to typos, and to pass e.g. params from the dataset to Tensorboard
        if kwargs_tracking:
            print(f"\nunused params in LitCNN: {kwargs_tracking}\n")

        self.train_losses = []
        self.val_losses = []
        self.metrics = {
            ### may be used to generate more metrics during testing
            'mse'       :   MSE_img(reduction='none'),
            # 'l1'        :   L1_img(reduction='none'),
        }

        self.test_losses = {
            k   :  [] for k in self.metrics.keys()
        }
        ### for testing, also store id and magnitude (squared) of each sample for later analysis, also number of positive pixels
        self.test_magnitudes = {
            k   :  [] for k in self.metrics.keys()
        }
        self.pos_pixels = []
        self.test_ids = []
    
    def __repr__(self):
        if hasattr(self, "name"):
            return self.name
        else:
            return "LitCNN"

    def forward(self, x : Any) -> Any:
        if self.previous is not None:
            x = torch.cat([x, self.previous(x)], dim=-3)
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        
        inputs, target = batch
        output = self(inputs)
        loss = self.loss_fn(output, target)

        self.log("loss_train_it", loss.detach().cpu(), batch_size=target.shape[0], on_step=True, prog_bar=True)
        self.log("loss_train_avg", loss.detach().cpu(), batch_size=target.shape[0], on_epoch=True, on_step=False)

        self.train_losses.append(loss.detach().cpu())

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        inputs, target = batch
        output = self(inputs)
        loss = self.loss_fn(output, target).detach().cpu()

        self.log("loss_val_avg", loss, batch_size=target.shape[0], on_epoch=True, prog_bar=True)
        self.val_losses.append(loss)

        if batch_idx==0:
            ### save examples to TB log
            fig_list = []
            for m in range(min(inputs.shape[0], 20)):
                fig_list.append(show_samples(batch, m, output))
            self.logger.experiment.add_figure(f"val_samples", fig_list, global_step=self.current_epoch)
            plt.close('all')

        return loss
        
    def test_step(self, batch, batch_idx) -> None:
        if len(batch)==3:
            inputs, target, map_id = batch
            masks = None
        elif len(batch)==4:
            inputs, target, masks, map_id = batch
            ### we don't know masks beforehand, add them to metrics in first batch
            if batch_idx==0:
                for m in masks.keys():
                    for me in self.metrics.keys():
                        self.test_losses[f'{me}_mask_{m}'] = []
                        self.test_magnitudes[f'{me}_mask_{m}'] = []

        output = self(inputs)

        for k, v in self.metrics.items():
            self.test_losses[k].append(v(output, target).cpu())
            self.test_magnitudes[k].append(v(target, torch.zeros_like(target)).cpu())
            if masks is not None:
                for km, m in masks.items():
                    self.test_losses[f'{k}_mask_{km}'].append(v(output * m,target * m).cpu())
                    self.test_magnitudes[f'{k}_mask_{km}'].append(v(target * m, torch.zeros_like(target)).cpu())

        self.pos_pixels.append(torch.squeeze(torch.sum(target > 0, dim=(-1,-2))))
        self.test_ids.append(['_'.join([map_id[i][j] for i in range(len(map_id))]) for j in range(len(map_id[0]))])

    def predict_step(self, batch, batch_idx): 
        inputs, target, map_id = batch
        output = self(inputs)
        ### save prediction and loss for each sample
        for b in range(inputs.shape[0]):
            map_id_str = ''
            for i in map_id:
                map_id_str += f'{i[b]}_'
            io.imsave(self.predict_dir / f'{map_id_str[:-1]}.png', np.clip((255 * torch.squeeze(output[b,:]).cpu().numpy()), 0, 255).astype(np.uint8), check_contrast=False)
      
    def on_train_epoch_start(self) -> None:
        self.train_losses = []

    def on_validation_epoch_start(self) -> None:
        self.val_losses = []

    def on_test_epoch_start(self) -> None:
        self.test_losses = {
            k   :   [] for k in self.test_losses.keys()
        }
        self.test_magnitudes = {
            k   :  [] for k in self.test_magnitudes.keys()
        }
        self.pos_pixels = []
        self.test_ids = []
    
    def on_train_epoch_end(self) -> None:
        '''Add average losses of this epoch to the log file and check whether val loss improved.'''
        ### in case we continue training from a checkpoint, this hook gets triggered before training actually starts again
        if len(self.train_losses)==0:
            return
        avg_loss_train = float(torch.mean(torch.stack(self.train_losses)))
        avg_loss_val = float(torch.mean(torch.stack(self.val_losses))) if len(self.val_losses) > 0 else 1e5
        

        ### we are using "loss_val_avg" to track the avg loss over all epochs
        ### hp_metric is instead used to track the best avg loss so far
        if avg_loss_val < self.best_val[0]:
            self.best_val = (avg_loss_val, self.current_epoch)
        self.log('hp_metric', self.best_val[0])

        if self.trainer.log_dir is not None:
            with open(Path(self.trainer.log_dir) / "log_file.txt", "a", encoding='utf-8') as f:
                f.write(f'Epoch\t{self.current_epoch}/{self.trainer.max_epochs}\t{time.strftime("%d.%m.%y-%H:%M:%S")}\t\ttrain: {avg_loss_train:.7f}\t\tval: {avg_loss_val:.7f} (best: {self.best_val[0]:.7f} in ep. {self.best_val[1]})\t\tlr: {self.trainer.optimizers[0].param_groups[0]["lr"]:.7f}\n')

    def on_test_epoch_end(self) -> None:
        '''
        Add average losses of this epoch to the log file and TB.
        '''
        torch.set_printoptions(precision=6)
        save_dir = Path(self.trainer.log_dir)

        import pandas as pd
        columns = ['id']
        for k in self.test_losses.keys():
            columns.append(k)
            columns.append(f'{k}_magnitude')
            # columns.append(f'{k}_normalized')
        columns.append('pos_pixels')
        df = pd.DataFrame(columns=columns)

        ### loop over all batches, store results per sample in df, collect losses and normalized losses for averaging
        n_samples = 0
        losses_acc = {
            k   :   0 for k in self.test_losses.keys()
        }
        losses_norm_acc = {
            k   :   0 for k in self.test_losses.keys()
        }
        for i in range(len(self.test_ids)):
            ids = self.test_ids[i]
            pos_pixels = self.pos_pixels[i]
            losses = {
                k : self.test_losses[k][i] for k in self.test_losses.keys()
            }
            magnitudes = {
                k : self.test_magnitudes[k][i] for k in self.test_losses.keys()
            }
            for j in range(len(ids)):
                row = [ids[j]]
                for k in self.test_losses.keys():

                    row.append(losses[k][j].item())
                    row.append(magnitudes[k][j].item())
                    # row.append(losses[k][j].item() / (magnitudes[k][j].item() + 1e-4))
                    losses_acc[k] += losses[k][j].item()
                    # losses_norm_acc[k] += losses[k][j].item() / (magnitudes[k][j].item() + 1e-4)
                row.append(pos_pixels[j].item())
                df = pd.concat([df, pd.DataFrame([row], columns=columns)])

            n_samples += len(ids)
        df.to_csv(save_dir / "errors_per_sample.csv", index=False)

        with open(save_dir / "log_file_test.txt", "a", encoding='utf-8') as f:
            f.write(f'Test \t{time.strftime("%d.%m.%y-%H:%M:%S")}\nAverage losses/metrics: ({n_samples} samples)\n\n')
            ### calculate averages, save to test_log
            for k, v in losses_acc.items():
                message = f'{k}:\t{v / n_samples}\n'
                print(message)
                f.write(message)
            
    def configure_optimizers(self):
        if hasattr(torch.optim, self.optimizer):
            opt_class = getattr(torch.optim, self.optimizer)
            opt = opt_class(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, **self.optimizer_params)
            if self.scheduler is None:
                return {'optimizer' : opt}
            elif hasattr(torch.optim.lr_scheduler, self.scheduler):
                sched_class = getattr(torch.optim.lr_scheduler, self.scheduler)
                sched = sched_class(opt, **self.scheduler_params)
                return {
                "optimizer" : opt,
                "lr_scheduler" : {
                        "scheduler" : sched,
                        "interval" : self.scheduler_interval,
                        "frequency": self.scheduler_frequency,
                        "monitor": "loss_val_avg",
                        "strict": True,
                        "name": "lr"
                    }
                }
            else:
                raise ValueError(f"scheduler {self.scheduler} not found")        
        else:
            raise ValueError(f"optimizer {self.optimizer} not found")

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: torch.optim.Optimizer) -> None:
        optimizer.zero_grad(set_to_none=True)
    
    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> tuple:
        ### can we do a nicer check here?
        if self.channels_last:
            for inp in batch:
                if isinstance(inp, torch.Tensor) and inp.dim() > 3:
                    inp = inp.to(memory_format=torch.channels_last)
        
        return batch

    def on_test_start(self) -> None:
        if self.channels_last:
            self = self.to(memory_format=torch.channels_last)
    
    def on_validation_start(self) -> None:
        if self.channels_last:
            self = self.to(memory_format=torch.channels_last)

    def on_predict_epoch_start(self) -> None:
        self.predict_dir = Path(self.trainer.logger.log_dir).parent / 'predictions'
        self.predict_dir.mkdir(exist_ok=True)
        self.start_time = time.time()

    def on_predict_epoch_end(self) -> None:
        time_diff = time.time() - self.start_time
        days = int(time_diff // (24 * 60 * 60))
        left = time_diff % (24 * 60 * 60)
        hours = int(left // (60 * 60))
        left = left % (60 * 60)
        mins = int(left // 60)
        secs = int(left % 60)
        message = f"prediction ended at {time.strftime('%y%m%d-%H:%M:%S')} after {days} days, {hours}:{mins}:{secs}"
        if self.trainer.log_dir is not None: ##could be None in fast_dev_run
            with open(Path(self.trainer.log_dir) / "log_file_predict.txt", "a", encoding='utf-8') as f:
                f.write(message)
        print(message)

    def on_fit_start(self):
        ### store best val loss and corresponding epoch
        self.best_val = (1e5, -1)

        ### store all the losses for a single train, validation epoch to calculate average afterwards
        ### reset them here in case we load an already trained model
        self.train_losses = []
        self.val_losses = []
        
        self.start_time = time.time()

        if self.channels_last:
            self = self.to(memory_format=torch.channels_last)

        message = f"\n\ntraining started at {time.strftime('%y%m%d-%H:%M:%S')}\n\n"
        if self.model_weights is not None:
            message = f'\nLoaded weights from {self.model_weights}\n' + message
        if self.trainer.log_dir is not None: ##could be None in fast_dev_run
            with open(Path(self.trainer.log_dir) / "log_file.txt", "a", encoding='utf-8') as f:
                f.write(message)
        print(message)
        print(f'Optimizers: {self.optimizers()}\tSchedulers: {self.lr_schedulers()}')

    def on_fit_end(self):
        time_diff = time.time() - self.start_time
        days = int(time_diff // (24 * 60 * 60))
        left = time_diff % (24 * 60 * 60)
        hours = int(left // (60 * 60))
        left = left % (60 * 60)
        mins = int(left // 60)
        secs = int(left % 60)
        message = f"training ended at {time.strftime('%y%m%d-%H:%M:%S')} after {days} days, {hours}:{mins}:{secs}\nbest loss:\t{self.best_val[0]} in epoch {self.best_val[1]}"
        if self.trainer.log_dir is not None: ##could be None in fast_dev_run
            with open(Path(self.trainer.log_dir) / "log_file.txt", "a", encoding='utf-8') as f:
                f.write(message)
        print(message)

class LitUNet(LitCNN):
    'Basic UNet with several options for adjustments, e.g. residual connections (ResNet-like), Dropout, different activation, down and up sampling options (see torch_layers)'
    def __init__(self, 
        in_ch : int,
        out_ch : int = 1,
        inBlock : tuple | list = ('nConvBlocks', {}),
        encoderBlock : tuple | list = ('nConvBlocks', {}),
        skipBlock : tuple | list = ('Identity', {}),
        bottleneck : tuple | list | None = None,
        decoderBlock : tuple | list = ('nConvBlocks', {}),
        outBlock : tuple | list = ('nConvBlocks', {'res' : False}),
        num_layers : int = 2,
        channel : int | list = 32,
        depth : int | None = 5,
        activation : str = 'ReLU',
        kernel_size : int = 3,
        batchnorm : bool = True,
        bn_eps : float = 1e-05,
        dropout : tuple | float = 0,
        img_size : int = 256,
        res : bool = True,
        params_down : dict[str,int|str] = {},
        params_up : dict[str,int|str] = {},
        ### other params to be passed to LitCNN super class
        previous_model : str = None,
        **kwargs,
        ):

        self.name = 'LitUNet'
        self.save_hyperparameters()

        super().__init__(
            model = Modular(
                in_ch=in_ch + 1 * (previous_model is not None), 
                out_ch=out_ch, 
                inBlock=inBlock, 
                encoderBlock=encoderBlock, 
                skipBlock=skipBlock, 
                bottleneck=bottleneck, 
                decoderBlock=decoderBlock, 
                outBlock=outBlock, 
                num_layers=num_layers, 
                channel=channel, 
                depth=depth, 
                activation=activation, 
                kernel_size=kernel_size, 
                batchnorm=batchnorm, 
                bn_eps=bn_eps, 
                dropout=dropout, 
                img_size=img_size, 
                params_down=params_down, 
                params_up=params_up, 
                res=res),
            previous_model=previous_model,
            **kwargs
        )
    
class LitUNet_ViT(LitUNet):
    '''UNet with ViT from TransUNet in the bottleneck, not well tested'''
    def __init__(self, 
        in_ch : int = 3, 
        hidden_size : int = 768,
        grid_size : int = 16, 
        num_layers : int = 12,
        nhead : int = 12,
        dropout : float = 0.1, 
        attn_bias : bool = False,
        dim_feedforward : int | None = None,
        previous_model : str = None,
        **kwargs
    ):
        super().__init__(
            bottleneck=('ViT', dict(
                hidden_size=hidden_size,
                grid_size=grid_size,
                num_layers=num_layers,
                nhead=nhead,
                dropout=dropout,
                attn_bias=attn_bias,
                dim_feedforward=dim_feedforward
            )), 
            in_ch=in_ch + 1 * (previous_model is not None), 
            previous_model=previous_model,
            **kwargs)
        self.name = 'LitUNet_ViT'

class LitUNetDCN(LitCNN):
    '''UNet with deformable convolutions'''
    def __init__(self, 
                 in_ch,
                 lr = 1e-4,
                 params_down={},
                 params_up={},
                 dropout=0,
                 channel=32,
                 depth=5,
                 img_size=256,
                 batchnorm=True,
                 res=True,
                 batchnorm_dcn=True,
                 deactivate_last_res=False,
                 previous_model : str = None,
                 skip_first=True,
                 **kwargs):
        self.name = 'LitUNetDCN'
        outBlock = [('nConvBlocks', {})] if not deactivate_last_res else [('nConvBlocks', {'res' : False})]
        super().__init__(
            model = Modular(
                in_ch=in_ch + 1 * (previous_model is not None), 
                out_ch=1,
                channel=channel,
                depth=depth,
                res=res,
                batchnorm=batchnorm,
                inBlock=[('nConvBlocks', {})],
                outBlock=outBlock,
                decoderBlock=[('nConvBlocks', {}), ('DCNBlock', dict(batchnorm=batchnorm_dcn))],
                bottleneck=[('nConvBlocks', {}), ('DCNBlock', dict(batchnorm=batchnorm_dcn))],
                encoderBlock=[('convBlock', {}), ('DCNBlock', dict(batchnorm=batchnorm_dcn))],
                skipBlock=('Identity', {}),
                params_down=params_down,
                params_up=params_up,
                dropout=dropout,
                img_size=img_size,
                skip_first=skip_first
            ),
            previous_model=previous_model,
            lr=lr,
            **kwargs)

class LitUNet_DCN_old2(LitUNetDCN):
    '''Legacy alias for LitUNetDCN (for loading old checkpoints etc)'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
'''
models from other authors
'''
class LitPMNet(LitCNN):
    def __init__(self,
        in_ch : int = 3,
        n_blocks : list = [3, 3, 27, 3],
        atrous_rates : list = [6, 12, 18],
        multi_grids : list | None = [1, 2, 4],
        output_stride : int = 8,
        ceil_mode : bool = True, 
        output_padding : tuple[int, int] = (1, 1),
        bn_eps : float = 1e-05,
        dcn = False,
        previous_model : str = None,
        **kwargs
        ):
        from .PMNet.PMNet import PMNet
        ### workaround 
        assert multi_grids is None or len(multi_grids)==n_blocks[-1]
        model = PMNet(in_ch=in_ch + 1 * (previous_model is not None), n_blocks=n_blocks, atrous_rates=atrous_rates, multi_grids=multi_grids, output_stride=output_stride, ceil_mode=ceil_mode, output_padding=output_padding, bn_eps=bn_eps, dcn=dcn)
        self.name = "LitPMNet"
        self.save_hyperparameters()
        super().__init__(model=model, previous_model=previous_model, **kwargs)
    
class LitRadioUNet(LitCNN):
    def __init__(self,
        in_ch : int = 3,
        previous_model : str = None,
        **kwargs
        ):
        from .RadioUNet.RadioUNet import RadioWNet
        model = RadioWNet(inputs=in_ch + 1 * (previous_model is not None))
        self.name = "LitRadioUNet"
        self.save_hyperparameters()
        super().__init__(model=model, previous_model=previous_model, **kwargs)

class LitRadioUNet2(LitCNN):
    def __init__(self,
        in_ch : int = 3,
        previous_model : str = None,
        **kwargs
        ):
        from .RadioUNet.RadioUNet import RadioWNet2
        model = RadioWNet2(inputs=in_ch + 1 * (previous_model is not None))
        self.name = "LitRadioUNet2"
        self.save_hyperparameters()
        super().__init__(model=model, previous_model=previous_model, **kwargs)

### register all our LitModels in this dict to link strings to class names
### this is needed if we want to train more than one model in curricuum (as in RadioUNet)
LitModelDict = {
    'LitUNet'   :   LitUNet,
    'LitUNet_ViT'   :   LitUNet_ViT,
    'LitPMNet'      :   LitPMNet,
    'LitUNet_DCN_old2'  :   LitUNet_DCN_old2,
    'LitRadioUNet'  :   LitRadioUNet,
    'LitRadioUNet2'  :   LitRadioUNet2,
}


# for colorbars of the same size as the image
def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

def show_samples(
        batch : tuple, 
        batch_id : int = 0, 
        output : None | torch.Tensor = None, 
        filename : None | str | Path = None
    ):
    inputs, target = batch[0][batch_id,:].detach().cpu().to(torch.float32), torch.squeeze(batch[1][batch_id,:]).detach().cpu().to(torch.float32)
    n_inp = inputs.shape[0]
    cols = 5
    rows = int(ceil((n_inp + 1 + (output is not None)) / 5))
    fig = plt.figure(figsize=(4*cols, 3*rows))
    fig.add_subplot(rows, cols, 1)
    plt.imshow(target, vmin=0, vmax=1)
    plt.colorbar()
    plt.title('target')
    b = 2
    if output is not None:
        output = torch.squeeze(output[batch_id,:].detach().cpu().to(torch.float32))
        fig.add_subplot(rows, cols, 2)
        plt.imshow(output)
        plt.colorbar()
        plt.title('output')
        b += 1
    for i in range(n_inp):
        fig.add_subplot(rows, cols, i + b)
        ### inputs are either in the range [0,1] or [-1,1] or [-1,0]
        if torch.all(inputs[i,:] >=0):
            vmin = 0
        else:
            vmin = -1
        if torch.all(inputs[i,:] <=0):
            vmax = 0
        else:
            vmax = 1
        plt.imshow(inputs[i,:], vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(f'input {i}')
    if filename is not None:
        plt.savefig(filename)
    return fig