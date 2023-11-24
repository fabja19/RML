import torch
from pathlib import Path
import numpy as np
import json
from skimage import io
from operator import itemgetter
from . import utils_coords
from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()
import torchvision.transforms.v2 as T

class RM(torch.utils.data.Dataset):
    '''
    Generic dataset class for Radio Maps. Defines some logic we use for all different datasets and some methods that have to be defined by subclasses.
    '''
    def __init__(self,
        ids : list,
        dataset_path : str | Path,
        augmentation : bool,
        get_ids : bool = False
    ) -> None:
        '''
        Inputs:
            ids             -   list of ids, ids will be passed to __getsample__ of subclass
            dataset_path    -   where to find the dataset directory
            augmentation    -   whether to use random flips and rotations
            get_ids         -   whether to get sample id with each sample (used in inference)
        '''
        self.ids = ids
        self.dataset_path = Path(dataset_path)
        assert self.dataset_path.is_dir(), f"Please check the path to the dataset ({dataset_path}), not a directory."

        self.t1 = T.ToImageTensor()
        if augmentation:
            self.transforms = T.Compose([
                ### this combination of transforms allows us to obtain exactly the 8 ways of flipping and rotating that keep the image shape
                # T.ToImageTensor(), 
                T.ConvertImageDtype(torch.float32),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomApply([T.RandomRotation((90, 90))], p=0.5),
            ])
        else:
            self.transforms = T.Compose([
                # T.ToImageTensor(), 
                T.ConvertImageDtype(torch.float32),
            ])
        self.get_ids = get_ids

    def get_in_ch(*args, **kwargs) -> int:
        '''
        Dummy method.
        '''
        raise NotImplementedError('subclasses of RM have to implement get_in_ch')

    def get_PL_scale(*args, **kwargs):
        '''
        Dummy method.
        '''
        raise NotImplementedError('subclasses or RMhave to implement get_PL_scale')
        
    def __len__(self) -> int:
        '''
        Total number of samples.
        '''
        return len(self.ids)
    
    def __getsample__(self, idx : int) -> tuple | list:
        '''
        Dummy method to be implemented by subclasses. This does the job usually __getitem__ does.
        '''
        raise NotImplementedError('subclasses or RMhave to implement __getsample__')
    
    def __getitem__(self, idx : int) -> tuple | list:
        '''
        Calls method __getsample__ that all subclasses have to implement. Applies augmentation and, if requested, adds ids for inference.
        '''
        curr_id = self.ids[idx]
        sample = list(self.__getsample__(curr_id))
        for i in range(len(sample)):
            ### ToImageTensor doesn't work on dicts
            if isinstance(sample[i], dict):
                for k, v in sample[i].items():
                    sample[i][k] = self.t1(v)
            sample[i] = self.t1(sample[i])
        sample = self.transforms(sample)
        
        if self.get_ids:
            return *sample, curr_id
        else:
            return sample
            
class RM_directional(RM):
    '''
    PyTorch dataset containing radio maps, corresponding gis layers (nDSMs of buildings and vegetation) and transmitter positions and parameters (gain, direction).
    Additional information based on this data like coordinate systems, line-of-sight-maps can also be generated.    
    '''
    def __init__(self,
        dataset_path : str | Path,
        ### inputs
        tx_one_hot : bool,
        ndsms : bool,
        ndsm_all : bool,
        dist2d : bool,
        dist2d_log : bool,
        coords_euclidian : bool,
        coords_cylindrical : bool,
        coords_spherical : bool,
        coords_ga : bool,
        los_floor : bool,
        los_top : bool,
        los_z_min : bool,
        los_z_min_rel : bool,
        los_theta_max : bool,
        ant_gain_floor : bool,
        ant_gain_top : bool,
        ant_gain_slices : bool,
        ant_gain_los_slices : bool,
        ant_gain_floor_dist : bool,
        ant_gain_top_dist : bool,
        ant_gain_slices_dist : bool,
        ant_gain_los_slices_dist : bool,
        img : bool,
        img_rgb : bool,
        elevation : bool,
        azimuth :   bool,
        ### values for scaling and cutting height values
        z_step : int,
        z_max : int,
        thresh : float,
        ### activate to pass masks when testing (allows to calculate loss in LoS/non-LoS areas)
        test    :   bool = False,
        ### kwargs for super class
        **kwargs
        ) -> None:
        ''' Arguments:
        dataset_path    -   string or path; Path to dataset main folder.        

        inputs          -   Several inputs can be turned on by setting cooresponding bool to True, all of these are one or several 256 x 256 tensors according to the map
            tx_one_hot          -   1 in tx position, 0 else        
            ndsms               -   ndsms of buildings and vegetation separately
            ndsm_all            -   one ndsm not separated by classes (mainly used to see how ndsm + image without class labels performs, as this information might be easier to obtain)
            dist2d              -   distance in x-y plance from Tx to each point
            dist2d_log          -   log_10(dist2d) , corresponding to the effect of the distance in free space path loss in dB (linear transform should be learned by the network)
            coordinates_X       -   coordinates of the floor, building top and vegetation top for a coordinates system with center in the Tx position
                                    e.g. coordinates_cylindrical produces five channels: phi, r, and for each of floor/building/vegetation the z_value
                                    GA is grid anchor as proposed in RadioTrans paper (https://ieeexplore.ieee.org/document/9753644)
            los_flor/top        -   1/0 for los/non-los, both top of buildings and floor
            los_z_min           -   minimal z-value for visibility (from the floor, as in nDSMs, GA)
            los_z_min_rel       -   minimal z-value for visibility, relative to tx height (euclidian/cylindrical coordinates)
            los_theta_max       -   maximal theta value for visibility, relative to tx position (spherical coordinates)
            ant_gain_floor/top  -   antenna gain in dB projected onto the floor of the map/ building top
            ant_gain_slices     -   antenna gain projected onto horizontal planes, according to z_step, z_max (see below)
            ant_gain_los_slices -   like ant_gain_slices, but additionally contains LoS information, gain in voxels with no LoS is set to 0
            ant_gain_X_dist     -   gain in dB - 2*log_10(dist2d) , corresponding to free space path loss
            img                 -   aerial image (RGBI)
            img_rgb             -   aerial image, only RGB channels
            elevation           -   tilt/elevation angle (spherical coordinates)
            azimuth             -   azimuth angle (spherical/cylindrical coordinates)
        z_step          -   int; This is only used for ant_gain_slices and determines to how many meters in z-direction each slice corresponds. The maximum height considered in the 
                            dataset is 32m, so with e.g. z_step=4 we produce 8 slices for the heights 0-4, 4-8,...,28-32
        z_max           -   int; Maximum height value relevant for this dataset. This is used to cut off gis layers above this value and rescale height values linearly from [0, z_max] to [0, 1].
        thresh          -   float; To cut off lower part of the radio map
        test            -   bool; pass True to receive masks for LoS, areas directly illuminated by Tx for additional metrics
        The output of the get_item method is a tuple of two tensors, first is all inputs stacked along the channel dimension, second one is the target radio map.
        All inputs and the target are normalized and in the case of heights cut to values in the interval [-1,1].
        '''
        
        self.antenna_gains = {}
        for ant_file in (dataset_path / 'antenna_patterns').glob('pattern_*.npy'):
            try:
                ant_id = int(ant_file.stem.split('_')[-1])
            except ValueError as e:
                # print(e)
                continue
            self.antenna_gains[ant_id] = np.load(ant_file)

        self.tx_one_hot = tx_one_hot
        self.ndsms = ndsms
        self.ndsm_all = ndsm_all
        self.dist2d = dist2d
        self.dist2d_log = dist2d_log
        self.coords_euclidian = coords_euclidian
        self.coords_cylindrical = coords_cylindrical
        self.coords_spherical = coords_spherical
        self.coords_ga = coords_ga
        self.los_floor = los_floor
        self.los_top = los_top
        self.los_z_min = los_z_min
        self.los_z_min_rel = los_z_min_rel
        self.los_theta_max = los_theta_max
        self.ant_gain_floor = ant_gain_floor
        self.ant_gain_top = ant_gain_top
        self.ant_gain_slices = ant_gain_slices
        self.ant_gain_los_slices = ant_gain_los_slices
        self.ant_gain_floor_dist = ant_gain_floor_dist
        self.ant_gain_top_dist = ant_gain_top_dist
        self.ant_gain_slices_dist = ant_gain_slices_dist
        self.ant_gain_los_slices_dist = ant_gain_los_slices_dist
        self.img = img
        self.img_rgb = img_rgb
        self.azimuth = azimuth
        self.elevation = elevation

        self.z_step = z_step
        self.z_max = z_max
        self.thresh = thresh ## explain. may be 0

        self.test = test
        
        super().__init__(dataset_path=dataset_path, **kwargs)

    def get_in_ch(
            tx_one_hot=False, 
            ndsms=False, 
            ndsm_all=False, 
            dist2d=False, 
            dist2d_log=False, 
            coords_euclidian=False, 
            coords_cylindrical=False, 
            coords_spherical=False, 
            coords_ga=False, 
            los_floor=False, 
            los_top=False, 
            los_z_min=False, 
            los_z_min_rel=False, 
            los_theta_max=False, 
            ant_gain_floor=False, 
            ant_gain_top=False, 
            ant_gain_slices=False, 
            ant_gain_los_slices=False, 
            ant_gain_floor_dist=False, 
            ant_gain_top_dist=False, 
            ant_gain_slices_dist=False, 
            ant_gain_los_slices_dist=False, 
            img=False, 
            img_rgb=False,
            z_max=32, 
            z_step=4, 
            #######
            azimuth=False,
            elevation=False,
            #######
            ### catch all arguments not needed for calculating channel
            *args, **kwargs) -> int:
        '''
        Calculates the number of in channels for the network for the given data configuration.
        The way linking of arguments in the CLI works, we have to make this avaailable in the __init__ of the DataModule, therefore we cannot work with the self attributes here.
        *args, **kwargs don't do anything, these are just defined so we can throw in the whole configuration of the dataset at once.

        Output  :   int; number of channels of input tensor 
        '''
        return tx_one_hot + dist2d +  dist2d_log + los_floor + los_top + los_z_min + los_z_min_rel + los_theta_max + ant_gain_floor + ant_gain_top + ant_gain_floor_dist + ant_gain_top_dist + azimuth \
            + 2 * ndsms + ndsm_all \
            + 3 * (elevation + img_rgb) \
            + 4 * img \
            + 5 * (coords_euclidian + coords_cylindrical + coords_ga) \
            + 7 * coords_spherical \
            + int(z_max // z_step) * (ant_gain_slices + ant_gain_los_slices + ant_gain_slices_dist + ant_gain_los_slices_dist) 

    def get_PL_scale(dataset_path : str | Path = 'dataset', **kwargs):
        '''
        Reads PL scale from file containing information about PL threshold and max PL.
        '''
        print(dataset_path)
        with open(Path(dataset_path) / 'max_power.json', 'r') as f:
            power_data = json.load(f)
        return float(power_data['pg_max']) - float(power_data['pg_trnc'])

    def __getsample__(self, curr_id : tuple):
        '''
        The function loads all data for the current idx and generates requested inputs.
        Output: 
        (inputs, target)    -   tuple of two tensors; first is all inputs stacked along the channel dimension, second one is the target radio map
        if test is passed in init, additionally the tuple contains a dict of names and masks
        '''
        if self.test:
            los_floor = None
            ant_gain_floor = None

        with open(self.dataset_path / "tx_antennas" / "{}_{}_{}_txparams.json".format(*curr_id[:3]), 'r') as f:
            tx_antenna_params = json.load(f)[curr_id[3]]
        tx_coords, tx_phi, tx_theta, tx_antenna = itemgetter('tx_coords', 'phi', 'theta', 'antenna')(tx_antenna_params)

        tx_antenna_pattern = self.antenna_gains[tx_antenna]

        nbuild = torch.minimum(torch.tensor(io.imread(self.dataset_path / "gis" / "nbuildings_{}_{}_{}.png".format(*curr_id[:3])), dtype=torch.float32), torch.tensor(self.z_max))
        nveg = torch.minimum(torch.tensor(io.imread(self.dataset_path / "gis" / "nveg_{}_{}_{}.png".format(*curr_id[:3])), dtype=torch.float32), torch.tensor(self.z_max))

        target = torch.tensor(io.imread(self.dataset_path / "path_gain" / "pg_{}_{}_{}_{}.png".format(*curr_id)), dtype=torch.float32) / 255
        target = torch.maximum((torch.reshape(target, (1, *target.shape))- self.thresh) / (1 - self.thresh), torch.tensor([0]))

        inputs = []

        ### always generate spherical coordinates, these are needed for generating gain and LoS input as well
        dist_3d_build, dist_3d_veg, dist_3d_floor, theta_build, theta_veg, theta_floor, phi = utils_coords.spherical_coords(nbuild=nbuild, nveg=nveg, tx_coords=tx_coords, phi_base=tx_phi, theta_base=tx_theta)

        if self.los_floor or self.los_top or self.los_theta_max or self.los_z_min or self.los_z_min_rel or self.ant_gain_los_slices or self.test:
            los_theta_max = torch.minimum(torch.tensor(np.load(self.dataset_path / "los" / "theta_max_{}_{}_{}_{}.npy".format(*curr_id)), dtype=torch.float32), theta_floor)

        ### generate all requested input tensors, normalize and add them to inputs list
        if self.tx_one_hot:
            tx_one_hot_tens = torch.zeros_like(nbuild)
            tx_one_hot_tens[tx_coords[0], tx_coords[1]] = tx_coords[2] / self.z_max
            inputs.append(tx_one_hot_tens)
        if self.ndsms:
            inputs.append(nbuild / self.z_max)
            inputs.append(nveg / self.z_max)
        if self.ndsm_all:
            inputs.append(torch.where(nbuild > 0, nbuild, nveg) / self.z_max)
        if self.dist2d:
            inputs.append(utils_coords.dist_2d(tx_coords=tx_coords) / (np.sqrt(2)*255))
        if self.dist2d_log:
            dist2d = utils_coords.dist_2d(tx_coords=tx_coords)
            ### we change the value at the Tx position, as log(0) doesn't make sense
            ### in this position, there will always be a building anyways and hence 0 path loss, and the usual path loss formula with log(dist) doesn't hold here
            dist2d[dist2d==0] = 1
            inputs.append(torch.log10(dist2d) / np.log10(np.sqrt(2)*255))
        if self.coords_euclidian:
            xc, yc, zc_build, zc_veg, zc_floor = utils_coords.euclidian_coords(nbuild=nbuild, nveg=nveg, tx_coords=tx_coords)
            inputs.extend([xc / 255, yc / 255, zc_build / self.z_max, zc_veg / self.z_max, zc_floor / self.z_max])
        if self.coords_cylindrical:
            dist2d, phi_cyl, zc_build, zc_veg, zc_floor = utils_coords.cylindrical_coords(nbuild=nbuild, nveg=nveg, tx_coords=tx_coords, phi_base=tx_phi)
            inputs.extend([dist2d / (np.sqrt(2)*255), phi_cyl / torch.pi, zc_build / self.z_max, zc_veg / self.z_max, zc_floor / self.z_max])
        if self.coords_spherical:
            max_dist_3d = np.sqrt(255**2 + 255**2 + self.z_max**2)
            inputs.extend([dist_3d_build / max_dist_3d, dist_3d_veg / max_dist_3d, dist_3d_floor / max_dist_3d, theta_build / torch.pi, theta_veg / torch.pi, theta_floor / torch.pi, phi / torch.pi])
        if self.coords_ga:
            xt, yt, zt, xs, ys = utils_coords.GA_coords(tx_coords=tx_coords)
            inputs.extend([xt / 256, yt / 256, zt / self.z_max, xs / 256, ys / 256])
        ### for binary LoS, add a small constant to los_theta_max, otherwise we have small holes/distortions in the LoS maps due to rounding errors
        if self.los_floor:
            los_floor = (los_theta_max + 1e-4 >= theta_floor).to(dtype=torch.float32)
            inputs.append(los_floor)
        if self.los_top:
            inputs.append((los_theta_max + 1e-4 >= theta_build).to(dtype=torch.float32))
        if self.los_z_min:
            los_z_min = utils_coords.get_heights(theta=los_theta_max, dist_2d=utils_coords.dist_2d(tx_coords=tx_coords), tx_z=tx_coords[2], theta_base=tx_theta, z_max=self.z_max)
            inputs.append(los_z_min / self.z_max)
        if self.los_z_min_rel:
            los_z_min_rel = utils_coords.get_heights(theta=los_theta_max, dist_2d=utils_coords.dist_2d(tx_coords=tx_coords), tx_z=tx_coords[2], theta_base=tx_theta, z_max=self.z_max) - tx_coords[2]
            inputs.append(los_z_min_rel / self.z_max)
        if self.los_theta_max:
            inputs.append(los_theta_max / torch.pi)
        if self.ant_gain_floor:
            ant_gain_floor = utils_coords.project_gain(phi=phi, theta=theta_floor, gain_array=tx_antenna_pattern, normalize=True)
            inputs.append(ant_gain_floor)
        if self.ant_gain_top:
            inputs.append(utils_coords.project_gain(phi=phi, theta=theta_build, gain_array=tx_antenna_pattern, normalize=True))
        if self.ant_gain_slices:
            theta3d, _ = utils_coords.spherical_slices(tx_coords=tx_coords, theta_base=tx_theta, z_max=self.z_max, z_step=self.z_step)
            inputs.extend([utils_coords.project_gain(phi=phi, theta=theta3d[k,:], gain_array=tx_antenna_pattern, normalize=True) for k in range(theta3d.shape[0])])
        if self.ant_gain_los_slices:
            theta3d, _ = utils_coords.spherical_slices(tx_coords=tx_coords, theta_base=tx_theta, z_max=self.z_max, z_step=self.z_step)
            inputs.extend([utils_coords.project_gain(phi=phi, theta=theta3d[k,:], gain_array=tx_antenna_pattern, theta_max=los_theta_max, normalize=True) for k in range(theta3d.shape[0])])
        if self.ant_gain_floor_dist:
            inputs.append(utils_coords.project_gain(phi=phi, theta=theta_floor, gain_array=tx_antenna_pattern, normalize=True, dist=utils_coords.dist_3d(tx_coords)))
        if self.ant_gain_top_dist:
            inputs.append(utils_coords.project_gain(phi=phi, theta=theta_build, gain_array=tx_antenna_pattern, normalize=True, dist=utils_coords.dist_3d(tx_coords)))
        if self.ant_gain_slices_dist:
            theta3d, dist3d = utils_coords.spherical_slices(tx_coords=tx_coords, theta_base=tx_theta, z_max=self.z_max, z_step=self.z_step)
            inputs.extend([utils_coords.project_gain(phi=phi, theta=theta3d[k,:], gain_array=tx_antenna_pattern, normalize=True, dist=dist3d[k,:], z_max=self.z_max) for k in range(theta3d.shape[0])])
        if self.ant_gain_los_slices_dist:
            theta3d, dist3d = utils_coords.spherical_slices(tx_coords=tx_coords, theta_base=tx_theta, z_max=self.z_max, z_step=self.z_step)
            inputs.extend([utils_coords.project_gain(phi=phi, theta=theta3d[k,:], gain_array=tx_antenna_pattern, theta_max=los_theta_max, normalize=True, dist=dist3d[k,:], z_max=self.z_max) for k in range(theta3d.shape[0])])
        if self.azimuth:
            inputs.append(phi)
        if self.elevation:
            inputs.extend([theta_build, theta_floor, theta_veg])
        if self.img:
            img_arr = io.imread(self.dataset_path / "img" /  "img_{}_{}_{}.tif".format(*curr_id[:3]))
            for i in range(img_arr.shape[-1]):
                inputs.append(torch.tensor(img_arr[:,:,i], dtype=torch.float32) / 255)
        if self.img_rgb:
            img_arr = io.imread(self.dataset_path / "img" /  "img_{}_{}_{}.tif".format(*curr_id[:3]))
            for i in range(3):
                inputs.append(torch.tensor(img_arr[:,:,i], dtype=torch.float32) / 255)
        
        inputs = torch.stack(inputs, dim=0)

        if self.test:
            if ant_gain_floor is None:
                ant_gain_floor = utils_coords.project_gain(phi=phi, theta=theta_floor, gain_array=tx_antenna_pattern, normalize=True)
            if los_floor is None:
                los_floor = (los_theta_max + 1e-4 >= theta_floor).to(dtype=torch.float32)
            return inputs, target, {'gain_los_floor' : 1.0*((los_floor * ant_gain_floor) == 0), 'gain_floor' : 1.0*(ant_gain_floor == 0)}
        else:
            return inputs, target
        