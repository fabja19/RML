'''Our custom callbacks.'''
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torchinfo import summary
from pathlib import Path 

class SummaryCustom(Callback):
    '''
    This callback generates a model summary based on torchinfo, prints it to the shell and saves it to the log dir of the trainer if possible.
    The summary generated by torchinfo is much more detailed than the one generated by PL.
    Also add hparams of the dataset so we can get (almost) all information about the simulation from the log file.
    '''
    def __init__(self, depth=5, batch_size=32):
        self.depth = depth
        self.batch_size = batch_size

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        try:
            input_size = (self.batch_size, pl_module.hparams['in_ch'], 256, 256) 
        except ValueError as e:
            print(e)
            return
        device = torch.device('cuda') if torch.cuda.is_available() and isinstance(trainer.accelerator, pl.accelerators.cuda.CUDAAccelerator) else torch.device('cpu')

        sum = summary(pl_module, input_size=input_size, depth=6, device=device, verbose=0)
        strPrint = (
            f'\nSaving to {trainer.log_dir}'
            f'\n{pl_module}\n'
            f'\n{pl_module.hparams_initial}\n'
            f'\n{sum}'
            f'\n(for input_size={input_size}, with AMP significantly less memory)'
        )
        
        try:
            strPrint += f'\nData:\t{trainer.datamodule}\n\n{trainer.datamodule.hparams}'
        except Exception as e:
            print(e)

        print(strPrint)

        ### in fast_dev_run, the trainer has no log_dir
        if hasattr(trainer, 'log_dir') and trainer.log_dir is not None:
            with open(Path(trainer.log_dir) / "log_file.txt", "a", encoding='utf-8') as f:
                print(strPrint, file=f)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        print(
            f'\nSaving to {trainer.log_dir}\n'
            f'loaded {pl_module}'
            )
            
class AdjustThreads(Callback):
    '''Adjust threads used by PyTorch (for use on cluster)'''
    def __init__(self, t : None | int | str = 'auto', verb : bool = False) -> None:
        '''
        Set number of threads to a fixed value if t is int or derive automatically from os.sched_affinity if 'auto' or skip if None
        '''
        from os import sched_getaffinity
        if t is None or t=='None':
            from torch import get_num_threads, get_num_interop_threads
            if verb:
                print(f'\n\nnot adjusting threads (using {get_num_threads()}, {get_num_interop_threads} out of {len(sched_getaffinity(0))}\n\n')
            return
        from torch import set_num_threads, set_num_interop_threads
        if t == 'auto':
            t = len(sched_getaffinity(0))
        else:
            t = t
        set_num_threads(t)
        set_num_interop_threads(t)
        if verb:
            print(f'\n\nadjusted threads to {t}\n\n')

class Precision(Callback):
    '''EXPLAIN'''
    def __init__(self, matmul_precision : str = 'medium', conv_tf32 : bool = True) -> None:
        '''Allows to set matmul precision and TF32 for convs from the CLI, should not have an effect with AMP'''
        torch.set_float32_matmul_precision(precision=matmul_precision)
        torch.backends.cudnn.allow_tf32 = conv_tf32