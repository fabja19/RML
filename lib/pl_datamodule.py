import pytorch_lightning as pl
from pathlib import Path
import json
import numpy as np
import torch
from matplotlib import pyplot as plt
from . import torch_datasets


class LitRM(pl.LightningDataModule):
    '''
    Abstract DataModule, to be subclassed for specific datasets. This is to specify some common parameters and methods and avoid copying.
    '''
    def __init__(self,
        dataset_class   : torch_datasets.RM,
        dataset_path    : str | Path,
        id_file         : str,
        augmentation    : bool = True, 
        ###### params for dataloader
        batch_size      : int = 32,
        shuffle         : bool = True,
        num_workers     : int = 8,
        pin_memory      : bool = True,
        **dataset_params
    ):
        '''
        '''
        super().__init__()

        self.dataset_class = dataset_class

        self.loader_params = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.dataset_params = {
            'dataset_path'  :   Path(dataset_path),
            'augmentation' : augmentation,
            **dataset_params
            }
        
        with open(Path(dataset_path) / id_file, 'r') as f:
            id_data = json.load(f)

        self.train_ids = id_data['train']
        self.val_ids = id_data['val']
        self.test_ids = id_data['test']

        self.in_ch = dataset_class.get_in_ch(**self.dataset_params)
        self.PL_scale = dataset_class.get_PL_scale(**self.dataset_params)

        # to be overridden by subclasses after calling super init
        if not hasattr(self, 'name'):
            self.name = 'LitRM' 

        print(f"prepared dataset {dataset_class}\t{len(self.train_ids)} train / {len(self.val_ids)} val / {len(self.test_ids)} test ids")

    def setup(self, stage: str=None) -> None:
        ''''
        DESCRIPTION
        '''
        if stage=="fit" or stage is None:
            self.train_set = self.dataset_class(ids=self.train_ids, **self.dataset_params)
            self.val_set = self.dataset_class(ids=self.val_ids, **self.dataset_params)
        elif stage=="validate":
            self.val_set = self.dataset_class(ids=self.val_ids, **{**self.dataset_params, 'augmentation' : False})
        elif stage=="test":
            self.test_set = self.dataset_class(ids=self.test_ids, **{**self.dataset_params, 'augmentation' : False, "get_ids" : True, "test" : True})
        elif stage=="predict":
            self.predict_set = self.dataset_class(ids=self.test_ids,**{**self.dataset_params, 'augmentation' : False, "get_ids" : True})
        else:
            raise NotImplementedError(f"stage={stage}") 
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, **self.loader_params)

    def val_dataloader(self):
        ### shuffle the validation set so we see different outputs in TB every epoch
        return torch.utils.data.DataLoader(self.val_set, **{**self.loader_params, "shuffle" : True})

    def test_dataloader(self, shuffle=False):
        return torch.utils.data.DataLoader(self.test_set, **{**self.loader_params, "shuffle" : shuffle})
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_set, **{**self.loader_params, "shuffle" : False})
    
    def __repr__(self):
        return self.name
        
class LitRM_directional(LitRM):
    '''
    Directional radio map dataset in lightning, on top of torch_dataset_directional.RMD and subclassing LitRM.
    All of the important logic is implemented in the torch class and LitRM, but this class allows us to easily track the used parameters, set defaults and have a common base with other datasets.
    '''
    def __init__(self,
        dataset_path : str | Path = Path('./dataset'),
        id_file : str = 'splits_0.1_0.1_seeds_1_2_ant_all.json',
        ### inputs
        tx_one_hot      : bool = True,
        ndsms           : bool = True,
        ndsm_all        : bool = False,
        dist2d          : bool = False,
        dist2d_log      : bool = False,
        coords_euclidian : bool = False,
        coords_cylindrical : bool = False,
        coords_spherical : bool = False,
        coords_ga       : bool = False,
        los_floor       : bool = False,
        los_top         : bool = False,
        los_z_min       : bool = False,
        los_z_min_rel   : bool = False,
        los_theta_max   : bool = False,
        ant_gain_floor  : bool = True,
        ant_gain_top    : bool = True,
        ant_gain_slices : bool = False,
        ant_gain_los_slices : bool = False,
        ant_gain_floor_dist : bool = False,
        ant_gain_top_dist   : bool = False,
        ant_gain_slices_dist : bool = False,
        ant_gain_los_slices_dist : bool = False,
        img             : bool = False,
        img_rgb         : bool = False,
        elevation       : bool = False,
        azimuth         : bool = False,
        ### values for scaling and cutting height values/path loss
        z_step      : int = 4,
        z_max       : int = 32, 
        thresh      : float | None = 0.2,
        ### params for dataloader, augmentation
        **kwargs
    ):
        '''
        Inputs that can be requested by passing True:
            tx_one_hot              -   tensor with Tx height as value in Tx position, 0 elsewhere
            ndsms                   -   height maps buildings and vegetation
            ndsm_all                -   height map buildings and vegetation together (unclassified)
            dist2d                  -   2D distance of each pixel to Tx
            dist2d_log              -   log10(dist2d)
            coords_euclidian        -   euclidian coordinate system (Tx perspective)
            coords_cylindrical      -   cylindrical coordinate system (Tx perspective)
            coords_spherical        -   spherical coordinate system (Tx perspective)
            coords_ga               -   grid anchor (see https://ieeexplore.ieee.org/document/9753644)
            los_floor               -   binary LoS information for floor/ground in each pixel
            los_top                 -   binary LoS information for building top in each pixel
            los_z_min               -   minimum z-value visible from Tx in each pixel
            los_z_min_rel           -   minimum z-value visible from Tx in each pixel, minus Tx height
            los_theta_max           -   maximum theta-value visible from Tx in each pixel (spherical coordinates)
            ant_gain_floor          -   antenna gain projected onto the ground
            ant_gain_top            -   antenna gain projected onto the building top
            ant_gain_slices         -   antenna gain projected onto planes parallel to the ground according to z_step, z_max
            ant_gain_los_slices     -   antenna gain projected onto planes parallel to the ground according to z_step, z_max, additionally 0 if no LoS
            ant_gain_X_dist         -   gain in dB - 2*log_10(dist2d) , corresponding to free space path loss
            img                     -   aerial image (RGBI)
            img_rgb                 -   aerial image (RGB)
            elevation               -   tilt/elevation angle (spherical coordinates)
            azimuth                 -   azimuth angle (spherical/cylindrical coordinates)
            
            z_step, z_max           -   generate slices (gain) at heights 0, z_step, 2*z_step, ..., (z_step-1) * z_max
            thresh                  -   cut off lower part of the dB range from radio map
        '''
        self.save_hyperparameters()

        super().__init__(
            dataset_class=torch_datasets.RM_directional,
            dataset_path=dataset_path,
            id_file=id_file,
            tx_one_hot = tx_one_hot,
            ndsms = ndsms,
            ndsm_all = ndsm_all,
            dist2d = dist2d, 
            dist2d_log = dist2d_log,
            coords_euclidian = coords_euclidian,
            coords_cylindrical = coords_cylindrical,
            coords_spherical = coords_spherical,
            coords_ga = coords_ga,
            los_floor = los_floor,
            los_top = los_top,
            los_z_min = los_z_min,
            los_z_min_rel = los_z_min_rel,
            los_theta_max = los_theta_max,
            ant_gain_floor = ant_gain_floor,
            ant_gain_top = ant_gain_top,
            ant_gain_slices = ant_gain_slices,
            ant_gain_los_slices = ant_gain_los_slices,
            ant_gain_floor_dist = ant_gain_floor_dist,
            ant_gain_top_dist = ant_gain_top_dist,
            ant_gain_slices_dist = ant_gain_slices_dist,
            ant_gain_los_slices_dist = ant_gain_los_slices_dist, 
            img = img,
            img_rgb = img_rgb,
            azimuth=azimuth,
            elevation=elevation,
            z_step = z_step,
            z_max = z_max,
            thresh=thresh,
            **kwargs
        )
        self.name = 'LitRM_directional'

    def __repr__(self) -> str:
        return self.name

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



'''
Test
'''
if __name__ == "__main__":
    print("testing LitRMD")
    litrm = LitRM_directional()
    litrm.prepare_data()
    print("prepared data")
    litrm.setup("fit")
    print("setup data stage fit")
    litrm.val_dataloader()
    t = litrm.train_dataloader()
    print("generated train, val loaders")
    litrm.setup("test")
    print("setup stage test")
    litrm.test_dataloader()
    print("generated test loader")
    
    inputs, target = next(iter(t))
    print(f'first batch in train loader:\ninputs: {type(inputs), inputs.shape, inputs.dtype}\ntarget: {type(target), target.shape, target.dtype}\nsaving to {Path("./test_dataset.png")}')
    try:
        from .pl_lightningmodule import show_samples
        show_samples(inputs, target=target, path_save=Path("./test_dataset.png"))
    except ImportError as e:
        print(e, '.pl_lightningmodule not found?')