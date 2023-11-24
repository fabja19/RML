from pytorch_lightning.cli import LightningCLI, SaveConfigCallback, ArgsType
import lib.pl_datamodule # noqa: F401
import lib.pl_lightningmodule # noqa: F401
from lib import pl_callbacks
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
import datetime 
import warnings
warnings.filterwarnings('ignore', message='.*shuffling enabled', )
warnings.filterwarnings('ignore', message='.*You have overridden', )
warnings.filterwarnings('ignore', message='.*number of training batches')

class CLICustom(LightningCLI):
    ''''
    LightningCLI for training any model (potentially with any dataset).
    By default, results will be saved to ./logs/DATETIME/. TensorBoard logs are in a subdirectory showing the model name and inputs, checkpoints in a separate subdirectory.
    '''
    def add_arguments_to_parser(self, parser) -> None:

        ### define callbacks we usually use with defaults, can still be changed from CLI
        parser.add_lightning_class_args(EarlyStopping, 'cb_early_stopping')
        parser.add_lightning_class_args(ModelCheckpoint, 'cb_ckpt_best')
        parser.add_lightning_class_args(ModelCheckpoint, 'cb_ckpt_last')
        parser.add_lightning_class_args(pl_callbacks.SummaryCustom, 'cb_summary')
        parser.add_lightning_class_args(pl_callbacks.AdjustThreads, 'cb_threads')
        parser.add_lightning_class_args(pl_callbacks.Precision, 'cb_precision')
        parser.add_lightning_class_args(TQDMProgressBar, 'cb_progbar')

        ### set default arguments for organization of logs, checkpoints, callbacks
        parser.set_defaults(
            {
                'model' :   'LitUNetDCN',
                'data'  :    'LitRM_directional',
                'trainer.logger'   :   {
                    'class_path'    :   'pytorch_lightning.loggers.TensorBoardLogger',
                    'init_args' :   {
                        'version'   :   '',
                        'sub_dir'   :   ''
                    }
                },
                'trainer.default_root_dir'  :   f'./logs/{datetime.datetime.now().strftime("%y%m%d-%H%M%S")}',
                'cb_early_stopping.monitor' :   "loss_val_avg",
                'cb_early_stopping.min_delta' :   1e-4,
                'cb_early_stopping.patience'    :   10,
                'cb_early_stopping.check_on_train_epoch_end'    :   False,
                'cb_early_stopping.verbose'    :   True,
                'cb_early_stopping.check_finite'    :   False,
                'cb_ckpt_best.filename'    : "ep_{epoch}_loss_{loss_val_avg:.5f}",
                'cb_ckpt_best.save_top_k'  :   1, 
                'cb_ckpt_best.monitor' : "loss_val_avg",
                'cb_ckpt_best.save_on_train_epoch_end' :   False,
                'cb_ckpt_last.filename'    : "ep_{epoch}_step_{step}",
                'cb_ckpt_last.save_top_k'  :   1, 
                'cb_ckpt_last.train_time_interval' :   datetime.timedelta(minutes=30),
                'cb_progbar.refresh_rate'   :   10,    

            }
        )
        ### link some values between data and model
        parser.link_arguments('data.PL_scale', 'model.init_args.PL_scale', apply_on='instantiate')
        parser.link_arguments('data.in_ch', 'model.init_args.in_ch', apply_on='instantiate')
        ### link the used inputs and the model name to the directory for the logger, checkpoints and so on
        parser.link_arguments('trainer.default_root_dir', 'trainer.logger.init_args.save_dir')
        parser.link_arguments(('model.name', 'data.name'), 'trainer.logger.init_args.name', compute_fn=lambda w, x: f"{w}_{x}", apply_on='instantiate')
          
def main(args: ArgsType = None):
    CLICustom(
        model_class = lib.pl_lightningmodule.LitCNN,
        datamodule_class = lib.pl_datamodule.LitRM,
        subclass_mode_model = True,
        subclass_mode_data = True,
        seed_everything_default = 123,
        trainer_defaults = {
            'max_epochs'    :   120,
            'devices'   :   [0],
            'default_root_dir'  :   '.',
            'precision' :   '16-mixed',
            'num_sanity_val_steps'  :   '0',
            'enable_model_summary'  :   False,
            'deterministic' :   False,
            'benchmark' :   True,
        },
        args = args
    )

if __name__ == '__main__':
    main()