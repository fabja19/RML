'''
This file contains several functions to produce tensors containing coordinates and encode antenna gain as inputs to the CNN models.
'''
import torch
import warnings
import numpy as np

xy_range = torch.arange(256, dtype=torch.float32)
xc_base = torch.repeat_interleave(xy_range.reshape((-1, 1)), repeats=256, dim=1)
yc_base = torch.repeat_interleave(xy_range.reshape((1, -1)), repeats=256, dim=0)

def GA_coords(tx_coords : tuple | list):
    '''
    out: xt, yt, zt (coordinates of Tx, constant tensors), xs, ys (coordinates of spatial positions)
    
    grid anchor from RadioTrans paper (https://ieeexplore.ieee.org/document/9753644)
    '''
    xt = tx_coords[0] * torch.ones((256, 256))
    yt = tx_coords[1] * torch.ones((256, 256))
    zt = tx_coords[2] * torch.ones((256, 256))

    return xt, yt, zt, xc_base, yc_base

def euclidian_coords(nbuild : torch.Tensor, nveg : torch.Tensor, tx_coords : tuple | list):
    '''
    out: cartesian coordinates of top of building/veg/floor in each 2D location, cut off at tx-height, relative to tx position
        
    xc, yc, zc_build, zc_veg, zc_floor
    '''
    xt, yt, zt = tx_coords
    xc = xc_base - xt
    yc = yc_base - yt

    zc_build = torch.minimum(torch.zeros_like(nbuild), nbuild - zt)
    zc_veg = torch.minimum(torch.zeros_like(nbuild), nveg - zt)
    zc_floor = -zt * torch.ones_like(nbuild)

    return xc, yc, zc_build, zc_veg, zc_floor

def cylindrical_coords(nbuild : torch.Tensor, nveg : torch.Tensor, tx_coords : tuple | list, phi_base : float):
    '''
    out: cylindrical coordinates of top of building/veg/floor in each 2D location, cut off at tx-height, relative to tx position and orientation
        
    dist2d, phi, zc_build, zc_veg, zc_floor
    '''
    xc, yc, zc_build, zc_veg, zc_floor = euclidian_coords(nbuild, nveg, tx_coords)

    phi = torch.arctan2(yc, xc)
    ### rotate
    phi = phi - phi_base
    ### correct
    phi = torch.where(phi < -1 * torch.pi, 2 * torch.pi + phi, phi)
    phi = torch.where(phi > torch.pi, phi - 2 * torch.pi, phi)

    r = torch.sqrt(xc**2 + yc**2)

    return r, phi, zc_build, zc_veg, zc_floor

def spherical_coords(nbuild : torch.Tensor, nveg : torch.Tensor, tx_coords : tuple | list, phi_base : float, theta_base : float):
    '''out: spherical coordinates of top of building/veg/floor in each 2D location, cut off at tx-height, relative to tx position and orientation
        
    dist_3d_build, dist_3d_veg, dist_3d_floor, theta_build, theta_veg, theta_floor, phi'''
    assert theta_base >= 0 and theta_base < torch.pi, f"check phi_base={phi_base}, theta_base={theta_base}"
    ### REMOVE LATER

    if phi_base < -torch.pi or phi_base > torch.pi:
        # print(f'correcting phi={phi_base} to {(phi_base+torch.pi)%(2*torch.pi) -torch.pi}')
        phi_base = (phi_base + torch.pi)%(2 * torch.pi) - torch.pi

    xc, yc, zc_build, zc_veg, zc_floor = euclidian_coords(nbuild, nveg, tx_coords)
    
    dist_3d_build = torch.sqrt(xc**2 + yc**2 + zc_build**2)
    dist_3d_veg = torch.sqrt(xc**2 + yc**2 + zc_veg**2)
    dist_3d_floor = torch.sqrt(xc**2 + yc**2 + zc_floor**2)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theta_build = torch.where(dist_3d_build==0, torch.pi,  torch.arccos(zc_build / dist_3d_build))
        theta_veg = torch.where(dist_3d_veg==0, torch.pi,  torch.arccos(zc_veg / dist_3d_veg))
        theta_floor = torch.where(dist_3d_floor==0, torch.pi,  torch.arccos(zc_floor / dist_3d_floor))

    phi = torch.arctan2(yc, xc)
    ### rotate
    phi = phi - phi_base
    theta_build = theta_build - theta_base + torch.pi/2
    theta_veg = theta_veg - theta_base + torch.pi/2
    theta_floor = theta_floor - theta_base + torch.pi/2
    ### correct
    phi = torch.where(phi < -1 * torch.pi, 2 * torch.pi + phi, phi)
    phi = torch.where(phi > torch.pi, phi - 2 * torch.pi, phi) 

    return dist_3d_build, dist_3d_veg, dist_3d_floor, theta_build, theta_veg, theta_floor, phi

def spherical_slices(tx_coords : tuple | list, theta_base : float, z_max : int = 32, z_step : int = 2):
    ''''
    generates thetas, distances of spherical coordinates of the 3D positions (not buildings/veg) on the 2D grid and at heights 0, 0+z_step, 0+2*z_step,...z_max-z_step, 
    for generating gain in slices

    out: theta, dist 3D tensors
    '''
    xt, yt, zt = tx_coords
    xc = (torch.unsqueeze(xc_base, dim=0) - xt).repeat(z_max//z_step, 1, 1)
    yc = (torch.unsqueeze(yc_base, dim=0) - yt).repeat(z_max//z_step, 1, 1)
    zc = (torch.arange(start=0, end=z_max, step=z_step, dtype=torch.float32).reshape((-1,1,1)) - zt).repeat(1, 256, 256)

    r = torch.sqrt(xc**2 + yc**2 + zc**2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theta = torch.where(r==0, torch.pi, torch.arccos(torch.maximum(zc / r, torch.tensor(-1))))

    ### rotate
    theta = theta - theta_base + torch.pi/2

    return theta, r

def dist_2d(tx_coords : tuple | list):
    '''
    out: tensor containing 2D distance of each spatial location to Tx location 9only in x-y plane)
    '''
    xt, yt = tx_coords[0], tx_coords[1]
    xc = xc_base - xt
    yc = yc_base - yt
    dist2d = torch.sqrt(xc**2 + yc**2)
    return  dist2d

def dist_3d(tx_coords : tuple | list):
    '''
    out: tensor containing 3D distance of Rx at 1.5m height (Rx in our dataset) to Tx
    '''
    xt, yt, zt = tx_coords[0], tx_coords[1], tx_coords[2]
    xc = xc_base - xt
    yc = yc_base - yt
    dist3d = torch.sqrt(xc**2 + yc**2 + (zt - 1.5)**2)
    return  dist3d

def get_heights(theta : torch.Tensor, dist_2d : torch.Tensor, tx_z : float, theta_base : float, z_max : int = 32):
    '''
    takes theta, dist_2d and calculates back to height value in each pixel location
    max_height specifies value for locations with theta<=-3 (assigned to pixels which are behind Tx by our LoS-algorithm)
    
    out: height values corresponding to theta in each spatial lcoation
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        heights = torch.minimum(tx_z - dist_2d / torch.tan(3/ 2 *torch.pi - theta - theta_base), torch.tensor(z_max, dtype=torch.float32))
    ### set values 
    heights[(theta<=-3)] = z_max
    heights[dist_2d==0] = tx_z

    return heights

def project_gain(phi : torch.Tensor, theta : torch.Tensor, gain_array : np.ndarray, theta_max : torch.Tensor | None = None, normalize : bool =False, dist : torch.Tensor | None = None, z_max : int = 0):
    '''     "draw" antenna gain on the map according to antenna pattern, angles
            antenna gain is expected to be in the form theta x phi, angles in int steps 0,...,179 x 0,...,359
            theta_max can be given optionally, to include visibility (0 gain if LoS to the pixel is obstructed)
            slices at different heights can be taken by varying theta
            if dist 2d is given, we subtract -2log_10(dist2d), according to free psace path loss
            values are between -250 (no gain) and 0 without normalization, or shifted and scaled to [0,1]
    '''
    assert gain_array[0,0]==-250.0, f'gain_array[0,0]={gain_array[0,0]}'
    ### set everything behind Tx to -250dB
    behind = (theta <= 0) | (theta>=torch.pi) | (phi >= torch.pi/2) | (phi <= -torch.pi/2)

    ### if theta_max is given, also set all pixels in shadows to (0,0), i.e. -250 dB
    if theta_max is not None:
        shadow = (theta_max + 1e-4 < theta)
    else:
        shadow = torch.full(theta.shape, False, dtype=torch.bool)

    phi_deg = torch.where(behind | shadow, 0, torch.floor(((phi / (2*torch.pi) * 360)))).type(torch.int)%360
    theta_deg = torch.where(behind | shadow, 0, torch.floor(((theta / (2*torch.pi) * 360)))).type(torch.int)%360

    assert ((phi_deg >= 0) & (phi_deg < 360) & (theta_deg >= 0) & (theta_deg < 180)).all(), f'torch.amin(phi_deg)={torch.amin(phi_deg)}, torch.amax(phi_deg)={torch.amax(phi_deg)}, torch.amin(theta_deg)={torch.amin(theta_deg)}, torch.amax(theta_deg)={torch.amax(theta_deg)}'

    ### torch doesn't offer a ravel_multi_index function unfortunately
    gain_proj = torch.tensor(np.take(gain_array[:180,:360], np.ravel_multi_index((theta_deg.numpy(), phi_deg.numpy()), (180, 360))), dtype=torch.float32)
    if dist is not None:
        ### correct in Tx position to avoid nans (no Rx here anyways)
        ### keep minimal power -250dB
        dist[dist==0] = 1
        gain_proj = torch.max(gain_proj - 20 * torch.log10(dist), -250 * torch.ones_like(gain_proj))
        if normalize:
            gain_proj = (gain_proj + 250) / 250
            # gain_proj = (gain_proj + 250 + 20 * np.log10(np.sqrt(2 * 255**2 + z_max**2))) / (250 + 20 * np.log10(np.sqrt(2 * 255**2 + z_max**2)))
    elif normalize:
        gain_proj = (gain_proj + 250) / 250

    return gain_proj