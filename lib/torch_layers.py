import torch
from torch import nn
import copy
import numpy as np
from .dcn import DeformableConv2d

'''Standard building blocks for CNN'''

def get_padding(mode="same", kernel_size=3, dilation=1):
    '''returns value for padding and potentially output_padding to maintain input size or increase/reduce it by a factor of 2'''
    # same: applying normal conv, retaining dimensions, assuming stride==1
    if mode=="same":
        if dilation%2==1 and kernel_size%2==0:
            raise Exception(f"invalid padding parameters: mode {mode}, kernel_size {kernel_size}, dilation {dilation}")
        return int(0.5 * dilation * (kernel_size -1))
    # down sampling with stride 2:
    elif mode=="down":
        return int(dilation * (kernel_size - 1) //2)
    # up sampling with stride 2 and convTranspose
    elif mode=="up":
        output_padding = 0 if dilation * (kernel_size - 1) % 2 ==1 else 1
        padding = int((dilation * (kernel_size - 1) + output_padding - 1) // 2)
        return padding, output_padding
    else:
        raise ValueError(f"get_padding got mode={mode}")

class nConvBlocks(nn.Module):
    '''repeated convBlocks, see below'''
    def __init__(self, in_ch, num_layers=2, out_ch=None, mid_factor=None, kernel_size=3, stride=1, dilation=1, dropout=0, batchnorm=True, bn_eps : float = 1e-05, activation='ReLU', res=False, **kwargs) -> None:
        super().__init__()
            
        if out_ch is None:
            out_ch = in_ch
        if mid_factor is None:
            mid_ch = out_ch
        else:
            mid_ch = mid_factor * in_ch

        # for __repr__ function
        self.repr = f"nConvBlocks, in_ch={in_ch}, num_layers={num_layers}, mid_ch={mid_ch}, out_ch={out_ch}, kernel_size={kernel_size}, stride={stride}, dilation={dilation}, res={res}"
        
        mod_list = nn.ModuleList()
        for i in range(num_layers):
            in_ch_here = in_ch if i==0 else mid_ch
            out_ch_here = out_ch if i==num_layers-1 else mid_ch
            mod_list.append(convBlock(in_ch=in_ch_here, out_ch=out_ch_here, kernel_size=kernel_size, stride=stride, dilation=dilation, dropout=dropout, batchnorm=batchnorm,
                                      bn_eps=bn_eps, activation=activation))
        
        self.mod_list = nn.Sequential(*mod_list)
        ### if we use residual skip connection and in_ch doesn't match out_ch, we need to adjust
        self.ch_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False) if res and in_ch != out_ch else nn.Identity()
        
        ### store some of the parameters needed in the forward pass
        self.res=res

    def forward(self, x):
        y = self.mod_list(x)
        if self.res:
            residual = self.ch_proj(x)
            return residual + y
        else:
            return y

    def __repr__(self):
        return self.repr
    
class convBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch=None, kernel_size=3, stride=1, dilation=1, dropout=0, batchnorm=True, bn_eps : float = 1e-05, activation='ReLU', **kwargs) -> None:
        super().__init__()
        
        padd = get_padding(mode="same", kernel_size=kernel_size, dilation=dilation)
        
        act = getattr(nn, activation)
        
        if out_ch is None:
            out_ch = in_ch

        mod_list = nn.ModuleList([nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padd, bias=not batchnorm, dilation=dilation)])
        if batchnorm:    
            mod_list.append(nn.BatchNorm2d(out_ch, eps=bn_eps))
        mod_list.append(act())
        if dropout > 0:
            mod_list.append(nn.Dropout2d(dropout))

        # for __repr__ function
        self.repr = f"convBlock, in_ch={in_ch}, out_ch={out_ch}, kernel_size={kernel_size}, stride={stride}, dilation={dilation}"

        super().__init__(*mod_list)

    def __repr__(self):
        return self.repr

class resNeXtBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None, mid_ch=None, groups=32, mode="same", kernel_size=3, dilation=1, dropout=0, batchnorm=False, \
        bn_eps : float = 1e-05, activation='ReLU', down_params=None, up_params=None, **kwargs):
        super().__init__()
     
        # standard
        if mid_ch is None:
            # choose mid_ch about in_ch/2 but divisible by 32
            mid_ch = int((in_ch // (2*groups)) * groups)
            if mid_ch == 0:
                mid_ch = groups
        if out_ch is None:
            out_ch = in_ch

        # repr
        self.repr = f"resNeXtBlock, in_ch={in_ch}, mid_ch={mid_ch}, out_ch={out_ch}"

        act = getattr(nn, activation)
        
        if mode=="same":
            stride = 1
            padd = get_padding(mode=mode, kernel_size=kernel_size, dilation=dilation)
            if in_ch != out_ch:
                self.transform_skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
            else:
                self.transform_skip = nn.Identity()
        elif mode=="up":
            if up_params is not None:
                up_p = copy.deepcopy(up_params)
            else:
                up_p = dict()
            stride = 2
            padd, output_padding = get_padding(mode=mode, kernel_size=kernel_size, dilation=dilation)
            up_p["in_ch"] = in_ch
            up_p["out_ch"] = out_ch
            self.transform_skip = up(**up_p)
        elif mode=="down":
            if down_params is not None:
                down_p = copy.deepcopy(down_params)
            else:
                down_p = dict()
            stride = 2
            padd = get_padding(mode=mode, kernel_size=kernel_size, dilation=dilation)
            down_p["in_ch"] = in_ch
            down_p["out_ch"] = out_ch
            self.transform_skip = down(**down_p)
        else:
            raise NotImplementedError(f"mode {mode} in resNeXtBlock")

        self.mod_list = nn.Sequential(nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1, padding=0))

        if mode in ["same", "down"]:
            self.mod_list.append(nn.Conv2d(mid_ch, mid_ch, kernel_size=kernel_size, stride=stride, groups=groups, padding=padd, dilation=dilation, bias=not batchnorm))
        else:
            self.mod_list.append(nn.ConvTranspose2d(mid_ch, mid_ch, kernel_size=kernel_size, stride=stride, groups=groups, padding=padd, output_padding=output_padding, bias=not batchnorm))
        
        if batchnorm:
            self.mod_list.append(nn.BatchNorm2d(mid_ch, eps=bn_eps))
        self.mod_list.append(nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0))
        self.mod_list.append(act())
        if dropout > 0:
            self.mod_list.append(nn.Dropout2d(dropout))
        
    def forward(self, x):
        y = self.mod_list(x)

        residual = self.transform_skip(x)
        z = y + residual
        return z

    def __repr__(self):
        return self.repr

class dilationBlock(nn.Module):
    ##### inspired by https://ieeexplore.ieee.org/document/9653079, puts blocks of the same type (e.g. conv, ResNeXt) in parallel with different dilation values
    def __init__(self, in_ch, out_ch=None, dilations=(1,2,3,4), block="resNeXtBlock", **block_params) -> None:
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        # repr
        self.repr = f"dilationBlock, in_ch={in_ch}, out_ch={out_ch}"

        ### copy the parameters for the block to not change them in place
        block_params_copy = copy.deepcopy(block_params)
        block_params_copy["in_ch"], block_params_copy["out_ch"] = in_ch, out_ch
        self.mod_list = nn.ModuleList()
        
        try:
            layer = globals()[block]
        except:
            layer = getattr(nn, block)

        for dilation in dilations:
            self.mod_list.append(layer(**block_params_copy, dilation=dilation))
        ### 1x1 conv to go back to out_ch
        self.channel_conv = nn.Conv2d(len(dilations)*out_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = []
        for layer in self.mod_list:
            out.append(layer(x))
        ### concat all outputs along channel dimension and bring number of channels to out_ch
        return self.channel_conv(torch.cat(out, dim=1))
    
    def __repr__(self):
        return self.repr
        
def down(in_ch, sampling_down='max', kernel_size_down=2, out_ch=None, dilation_down=1, **kwargs):  
    if out_ch is None:
        out_ch = in_ch
    padd = get_padding(mode="down", kernel_size=kernel_size_down, dilation=dilation_down)

    if sampling_down=='max':
        layer = nn.MaxPool2d(kernel_size_down, padding=padd, dilation=dilation_down)
    elif sampling_down=='conv':
        # strided conv
        layer = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size_down, stride=2, padding=padd, dilation=dilation_down),
                                nn.ReLU())
    elif sampling_down=="seg":
        layer = nn.MaxPool2d(kernel_size_down, padding=padd, dilation=dilation_down, return_indices=True)
        
    elif sampling_down=='avg':
        layer = nn.AvgPool2d(kernel_size_down, padding=padd)
    else:
        raise ValueError(f'sampling={sampling_down}')
    return layer

def up(in_ch, sampling='conv', kernel_size_up=2, out_ch=None, dilation_up=1, **kwargs):
    if out_ch is None:
        out_ch = in_ch

    if sampling in ['bilinear', 'nearest']:
        layer = nn.Upsample(scale_factor=kernel_size_up, mode=sampling)
    elif sampling=='conv':
        padd, output_padd = get_padding(mode="up", kernel_size=kernel_size_up, dilation=dilation_up)
        layer = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size_up, stride=kernel_size_up, padding=padd, output_padding=output_padd, dilation=dilation_up),
                                   nn.ReLU())
    elif sampling=="seg":
        layer = nn.MaxUnpool2d(kernel_size=kernel_size_up)
    else:
        raise NotImplementedError("wrong upsampling ", sampling)
    return layer
        
class DCNBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch=None, kernel_size=3, stride=1, dropout=0, batchnorm=True, bn_eps : float = 1e-05, activation='ReLU', **kwargs) -> None:
        super().__init__()

        padd = get_padding(mode="same", kernel_size=kernel_size, dilation=1)
        
        act = getattr(nn, activation)
        
        if out_ch is None:
            out_ch = in_ch

        mod_list = nn.Sequential(DeformableConv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padd, bias=not batchnorm))
        if batchnorm:    
            mod_list.append(nn.BatchNorm2d(out_ch, eps=bn_eps))
        mod_list.append(act())
        if dropout > 0:
            mod_list.append(nn.Dropout2d(dropout))

        # for __repr__ function
        self.repr = f"DCNBlock, in_ch={in_ch}, out_ch={out_ch}, kernel_size={kernel_size}, stride={stride}"

        super().__init__(mod_list)

    def __repr__(self):
        return self.repr

class nDCNBlocks(nn.Module):    
    ''''''
    def __init__(self, in_ch, num_layers=2, out_ch=None, mid_factor=None, kernel_size=3, stride=1, dropout=0, batchnorm=True, bn_eps : float = 1e-05, activation='ReLU', res=False, **kwargs) -> None:
        super().__init__()
            
        if out_ch is None:
            out_ch = in_ch
        if mid_factor is None:
            mid_ch = out_ch
        else:
            mid_ch = mid_factor * in_ch

        # for __repr__ function
        self.repr = f"nDCNBlocks, in_ch={in_ch}, num_layers={num_layers}, mid_ch={mid_ch}, out_ch={out_ch}, kernel_size={kernel_size}, stride={stride}, res={res}"
        
        mod_list = nn.Sequential()
        for i in range(num_layers):
            in_ch_here = in_ch if i==0 else mid_ch
            out_ch_here = out_ch if i==num_layers-1 else mid_ch
            mod_list.append(DCNBlock(in_ch=in_ch_here, out_ch=out_ch_here, kernel_size=kernel_size, stride=stride, dropout=dropout, batchnorm=batchnorm,
                                     bn_eps=bn_eps, activation=activation))
            
        self.mod_list = nn.Sequential(*mod_list)        
        ### if we use residual skip connection and in_ch doesn't match out_ch, we need to adjust
        self.ch_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False) if res and in_ch != out_ch else nn.Identity()
        
        ### store some of the parameters needed in the forward pass
        self.res=res


    def forward(self, x):
        y = self.mod_list(x)
        if self.res:
            residual = self.ch_proj(x)
            return residual + y
        else:
            return y

    def __repr__(self):
        return self.repr

'''Building blocks for UNet-like CNN (encoder, decoder, skip-connections)'''

class Modular(nn.Module):
    '''
    general CNN with encoder-(bottleneck)-decoder plus skip connections structure
        encoder handles skip connections and operations on them, decoder connects them
        by default, each encoder block doubles the number of channels and each decoder block halves them, the bottleneck uses 2*channels[0] as mid_ch if this is possible. 
        img_size has to be given for ViT
        blocks are defined as lists of tuples, tuples consist of a string referring to a Module defined by us or in torch and the parameters given to the Module
        blocks in the encoder and decoder are repeated according to number of channels
        skip connections are addded from the inBlock and each of the repeated encoder blocks to the decoder blocks and the outBlock, where their output gets concatenated

        some common params used in several architectures that will be tried to be passed to all layers where they make sense (/inBlock/outBlock):
        activation, last_activation, kernel_size, kernel_size_in, kernel_size_out, batchnorm, dropout
        will be override by parameters explicitely given in the definitions of the blocks

    '''
    def __init__(self, 
        in_ch : int,
        out_ch : int ,
        inBlock : list | tuple,
        encoderBlock : list | tuple,
        skipBlock : list | tuple,
        bottleneck : list | tuple | None,
        decoderBlock : list | tuple,
        outBlock : list | tuple,
        channel : int | list[int],
        depth : int | None ,
        dropout : tuple | float | None,
        img_size : int,
        params_down : dict,
        params_up : dict,
        skip_first=False,
        ### other params like stride...
        **kwargs
        ):

        super().__init__()
        img_size_orig = img_size
        dropout = (dropout, dropout) if isinstance(dropout, (float, int)) else dropout
        if isinstance(channel, int):
            channels = [channel * 2**i for i in range(depth + 1)]
            ### adjust channels in encoder in case they are lower than in_ch to the next power of 2
            channel_min = min([c for c in channels if c >= in_ch])
            channels_enc = [max(c, channel_min) for c in channels]
        else:
            ### depth ignored in this case
            channels = channel
            channels_enc = channels
        channels_dec = [c1 + c2 for c1, c2 in zip(channels_enc, channels)]
        # print(f'MOdular got channel={channel}\ncalculated:\tchannels={channels}, channels_enc={channels_enc}, channels_dec={channels_dec}')
        # channels_dec.append(channels_enc[-1])
        self.inBlock, f = Block(inBlock, in_ch=in_ch, out_ch=channels_enc[0], img_size=img_size, **kwargs)
        img_size *= f
        self.encoder= Encoder(encoderBlock, skipBlock=skipBlock, channels=channels_enc, img_size=img_size, dropout=dropout[0], params_down=params_down, skip_first=skip_first, **kwargs)
        img_size *= self.encoder.factor
        self.bottleneck, f = Block(bottleneck, in_ch=channels_enc[-1], out_ch=channels_enc[-1], img_size=img_size,dropout=dropout[0], **kwargs)
        img_size *= f
        self.decoder = Decoder(decoderBlock, channels_in=channels_dec, channels_out=channels, img_size=img_size, dropout=dropout[1], params_up=params_up, skip_first=skip_first, **kwargs)
        img_size *= self.decoder.factor
        self.outBlock, f = Block(outBlock, in_ch=channels_dec[0], out_ch=out_ch, img_size=img_size, **kwargs)
        img_size *= f

        assert img_size==img_size_orig, f"img_size={img_size_orig} given but {img_size} after all layers"
        
    def forward(self, x) -> torch.Tensor:
        x = self.inBlock(x)
        x, skips = self.encoder(x)
        if self.bottleneck is not None:
            x = self.bottleneck(x)
        x = self.decoder(x, skips)
        x = self.outBlock(x)
        return x
    
    def print_layers(self):
        return (f'inBlock:\t{self.inBlock}'
                f'encoder:\t{self.encoder}'
                f'bottleneck:\t{self.bottleneck}'
                f'decoder:\t{self.decoder}'
                f'outBlock:\t{self.outBlock}'
        )

class Encoder(nn.Module):
    '''
    encoder consists of repeated downsampling plus arbitrary block (e.g. nConvBlocks) given as encoderBlock 

    additionally, a specific block can be given for the values passed on the skip connections (default identity)
    '''
    def __init__(
            self, 
            encoderBlock : list[tuple] | tuple,
            channels : list[int],
            skipBlock : tuple | list,
            img_size : int,
            params_down : dict,
            skip_first=True,
            **kwargs
            ) -> nn.Module:
        
        super().__init__()
        
        self.factor = 1

        self.encBlocks = torch.nn.ModuleList()
        self.downBlocks = torch.nn.ModuleList()
        self.skipBlocks = torch.nn.ModuleList()
        self.skip_first = skip_first

        for i in range(len(channels) - 1):
            ### skip
            b, _ = Block(skipBlock, in_ch=channels[i], out_ch=channels[i], img_size=img_size, **kwargs)
            self.skipBlocks.append(b)

            ### some arbitrary block BEFORE downsampling
            b, f = Block(encoderBlock, in_ch=channels[i], out_ch=channels[i+1], img_size=img_size, **kwargs)
            self.factor *= f
            img_size *= f
            self.encBlocks.append(b)

            ### down sampling, this one doesn't receive the general **kwargs (to avoid setting kernel_size and so on only meant for other layers)
            b, f = Block(('down', params_down), in_ch=channels[i+1], out_ch=channels[i+1], img_size=img_size)
            self.factor *= f
            img_size *= f
            self.downBlocks.append(b)

        b, _ = Block(skipBlock, in_ch=channels[-1], out_ch=channels[-1], img_size=img_size, **kwargs)
        self.skipBlocks.append(b)
        
    def forward(
        self, 
        x : torch.Tensor
        ) -> tuple[torch.Tensor, list]:
        skips = []
        # print(f'encoder in: {x.shape}')
        for i in range(len(self.encBlocks)):
            if self.skip_first or i==0:
                skips.append(self.skipBlocks[i](x))
            x = self.encBlocks[i](x)
            if not self.skip_first:
                skips.append(self.skipBlocks[i](x))
            x = self.downBlocks[i](x)
            # print(f'after encoder {i}: {x.shape} and skip {skips[-1].shape}')
        if self.skip_first:
            skips.append(self.skipBlocks[-1](x))
        # print(f'skips: {[s.shape for s in skips]}')
        return x, skips
    
    def get_features(
        self, 
        x : torch.Tensor
        ) -> tuple[torch.Tensor, list, list]:
        if not self.skip_first:
            raise NotImplementedError()
        skips = [self.skipBlocks[0](x)]
        features = []
        for i in range(len(self.encBlocks)):
            x = self.encBlocks[i](x)
            features.append(x.detach().clone())
            skips.append(self.skipBlocks[i+1](x))
        return x, skips, features

class Decoder(nn.Module):
    def __init__(
            self, 
            decoderBlock : tuple | list,
            channels_in : list[int],
            channels_out : list[int],
            img_size : int,
            params_up : dict,
            skip_first = True,
            **kwargs
            ) -> nn.Module:
        super().__init__()

        self.factor = 1

        self.decBlocks = torch.nn.ModuleList()
        self.upBlocks = torch.nn.ModuleList()
        self.skip_first = skip_first

        for i in range(len(channels_in) - 1):
            b, f = Block(decoderBlock, in_ch=channels_in[i+1], out_ch=channels_out[i], img_size=img_size, **kwargs)
            self.factor *= f
            img_size *= f
            self.decBlocks.append(b)

            b, f = Block(('up', params_up), in_ch=channels_out[i] if skip_first else channels_out[i+1], out_ch=channels_out[i] if skip_first else channels_out[i+1], img_size=img_size, **kwargs)
            self.factor *= f
            img_size *= f
            self.upBlocks.append(b)

    def forward(
        self, 
        x : torch.Tensor,
        skips : list,
        ) -> torch.Tensor:
        # print(f'decoder: len(skips)={len(skips)}, x.shape={x.shape}')
        for i in range(len(self.decBlocks) - 1, -1, -1):
            if self.skip_first:
                x = torch.cat([x, skips[i+1]], dim=1)
                x = self.decBlocks[i](x)
                x = self.upBlocks[i](x)
            else:
                x = self.upBlocks[i](x)
                # print(f'after up: {x.shape}')
                x = torch.cat([x, skips[i+1]], dim=1)
                x = self.decBlocks[i](x)
        #     print(f'after decoder i={i}, x: {x.shape}')
        # print(f'last step decoder cat {x.shape}, {skips[0].shape}')
        x = torch.cat([x, skips[0]], dim=1)
        return x
    
    def get_features(
        self, 
        x : torch.Tensor,
        skips : list,
        ) -> torch.Tensor:
        features = []
        for i in range(len(self.decBlocks) - 1, -1, -1):
            x = torch.cat([x, skips[i+1]], dim=1)
            x = self.decBlocks[i](x)
            features.append(x.detach().clone())
        x = torch.cat([x, skips[0]], dim=1)
        return x, features
    
def Block(
    mods_args : list | tuple | None, 
    in_ch : int | None = None,
    out_ch : int | None = None,
    **kwargs
    ) -> tuple[nn.Module, int]:
    '''
    construct a block from a tuple of nn.Module (or function returning nn.Module), arguments, and optionally number of repetitions of the block 
    returns the current image size as well (for ViT)
    alternatively, give a list of such tuples, then Block will call itself iteratively and construct a sequential module
    in_ch is only applied to the first block in the list, afterwards we always use out_ch, this has to be specified for layers requiring in_ch    
    **kwargs can be used to overwrite values in arguments, this is useful for for specific layers (e.g. inBlock, bottleneck) that should otherwise use the basic arguments given inside the tuple
    '''
    if mods_args is None:
        return None, 1
    elif isinstance(mods_args, list):
        b, factor = Block(mods_args=mods_args[0], in_ch=in_ch, out_ch=out_ch, **kwargs)
        modList = nn.Sequential(b)
        for i in range(1, len(mods_args)):
            b, f = Block(mods_args=mods_args[i], in_ch=out_ch, out_ch=out_ch, **kwargs)
            factor *= f
            if b is not None:
                modList.append(b)
        return modList, factor
    else:
        modName, arguments = mods_args
        ### check special case for nConvBlocks, Transformer layers and so on
        if hasattr(arguments, 'num_layers') and arguments['num_layers']==0:
            return None, 1
        ### calculate how the spatial size changes for ViT
        if 'down' in modName or 'Pool' in modName or 'down' in arguments.values():
            if 'kernel_size' in arguments:
                factor = 1 / arguments['kernel_size']
            else:
                factor = 1 / 2
            ### remove the standard kernel size, stride, dilation from the parameters, these should only be applied to other layers and explicitely given for up/down scaling layers if wanted
            for arg in ['kernel_size', 'stride', 'dilation']:
                if arg in kwargs.keys():
                    del kwargs[arg]
        elif 'up' in modName or 'Unpool' in modName or 'up' in arguments.values():
            if 'kernel_size' in arguments:
                factor = arguments['kernel_size']
            else:
                factor = 2
            for arg in ['kernel_size', 'stride', 'dilation']:
                if arg in kwargs.keys():
                    del kwargs[arg]
        else:
            factor = 1
        module = get_layer_dict()[modName](in_ch=in_ch, out_ch=out_ch, **{**kwargs, **arguments})

        return module, factor

#### metrics
    
class MSE_img(nn.Module):
    '''MSE with "proper" reducton for images (always average over spatial dimensions, 'reduction' argument for batches and channels)'''
    def __init__(self, reduction : str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, prediction : torch.Tensor, target  : torch.Tensor):
        loss = torch.mean((prediction - target)**2, dim=(-1,-2))
        if self.reduction =='mean':
            return torch.mean(loss)
        elif self.reduction == 'none':
            ### remove potentially useless channel dimension 
            return torch.squeeze(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            raise NotImplementedError(f'reduction={self.reduction}')

class L1_img(nn.Module):
    '''L1 with "proper" reducton for images (always average over spatial dimensions, 'reduction' argument for batches and channels)'''
    def __init__(self, reduction : str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, prediction : torch.Tensor, target  : torch.Tensor):
        loss = torch.mean(torch.abs(prediction - target), dim=(-1,-2))
        if self.reduction =='mean':
            return torch.mean(loss)
        elif self.reduction == 'none':
            ### remove potentially useless channel dimension 
            return torch.squeeze(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            raise NotImplementedError(f'reduction={self.reduction}')

'''
Mostly copied from TransUNet, with minor changes
https://arxiv.org/abs/2102.04306
https://github.com/Beckschen/TransUNet

 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

'''

class Embedder(nn.Module):
    def __init__(self, in_ch, img_size,  hidden_size, grid_size=16, positional_embedding="zero", dropout=0):
        ### adapted from TransUNet
        ### img_size: real size at the current stage, update while creating the blocks accounting for down sampling before the transformer block
        ### only for square images so far
        super().__init__()
        ### calculate shapes
        assert img_size%grid_size==0, f"Embedder: img_size {img_size} must be divisible by grid_size {grid_size}"
        if img_size < grid_size:
            print(f"\n\nViT embedder got grid_size={grid_size}, but img_size={img_size}. Reducing grid_size to img_size.")
            grid_size = img_size
        patch_size = int(img_size // grid_size)
        n_patches = grid_size**2

        self.repr = f"embedder, in_ch={in_ch}, img_size={img_size}, hidden_size={hidden_size}, grid_size={grid_size}"
        ### define layers
        ### TBD: change this for lower layers, e.g. when kernel_size > 3, use >=two convs instead
        self.patch_embeddings = nn.Conv2d(  in_ch, 
                                            out_channels=hidden_size, 
                                            kernel_size=patch_size, 
                                            stride=patch_size)

        if positional_embedding=="zero":
            ### more TBD
            ### can we model spatial relations here by a graph and have a small graph CNN learn position embeddings? 
            self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))
        elif positional_embedding is None:
            self.position_embeddings = torch.zeros(1, n_patches, hidden_size)
        # elif positional_embedding=="coords":
            ### can we use coordinates here in any way?
        else:
            raise NotImplementedError("other position embeddings TBD")
        self.do = nn.Dropout(dropout)
        # print(f"embedder: {img_size}=img_size, hidden_size={hidden_size}, grid_size={grid_size}, calculated patch_size={patch_size}, n_patches={n_patches}")

    def forward(self, x):
        ### as in TransUNet
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.do(embeddings)
        return embeddings   
    
    def __repr__(self):
        return self.repr

class Mlp(nn.Module):
    def __init__(self, hidden_size, dim_feedforward=None, dropout=0):
        super(Mlp, self).__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * hidden_size
        self.fc1 = nn.Linear(hidden_size, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, hidden_size)
        self.act_fn = nn.functional.gelu
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

        self.repr = f"Mlp, hidden_size={hidden_size}, dim_feedforward={dim_feedforward}"

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

    def __repr__(self):
        return self.repr

class selfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, nhead=12, dropout=0, bias=True, vis = False):
        super().__init__()
        assert hidden_size % nhead == 0, f'choose hidden_size={hidden_size} divisible by nhead={nhead}'
        self.vis = False
        
        self.num_attention_heads = nhead
        self.attention_head_size = int(hidden_size / nhead)
        self.all_head_size = nhead * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=bias)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

        self.repr = f"selfAttentionBlock, hidden_size={hidden_size}, nhead={nhead}, attention_head_size={self.attention_head_size}"

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

    def __repr__(self):
        return self.repr

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, nhead=12, dropout=0, attn_bias=True, dim_feedforward=None, layer_norm_eps=1e-6, vis=False):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size, dim_feedforward, dropout)
        self.attn = selfAttentionBlock(hidden_size, nhead, dropout=dropout, bias=attn_bias, vis=vis)

        self.repr = f"TransformerBlock, hidden_size={hidden_size}, nhead={nhead}"

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def __repr__(self):
        return self.repr

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers=12, nhead=12, dropout=0, attn_bias=False, dim_feedforward=None, layer_norm_eps=1e-6, vis=False):
        super().__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        for _ in range(num_layers):
            layer = TransformerBlock(hidden_size, nhead, dropout, attn_bias, dim_feedforward, layer_norm_eps, vis)
            self.layer.append(copy.deepcopy(layer))
        self.repr = f"TransformerEncoder, hidden_size={hidden_size}, num_layers={num_layers}, nhead={nhead}"

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

    def __repr__(self):
        return self.repr

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_size, out_ch, out_size, grid_size=16, up="conv", bn=True) -> None:
        super().__init__()
        ### channels go from hidden_size to out_ch
        ### resolution goes from grid_size to out_size
        ##########
        # BN eps???

        ##########
        self.out = nn.Sequential(
            nn.Conv2d(hidden_size, out_ch, kernel_size=3, padding=1),
        )
        if out_size > grid_size:
            if up in ["nearest", "bilinear"]:
                self.out.append(nn.Upsample(size=out_size, mode=up))
            elif up=="conv":
                assert out_size%grid_size==0, f"in TransformerDecoder grid_size={grid_size} and out_size={out_size} don't work with mode conv"
                factor = int(out_size//grid_size)
                self.out.append(nn.ConvTranspose2d(out_ch, out_ch, kernel_size=factor, stride=factor))
        self.out.append(nn.ReLU())
        if bn: 
            self.out.append(nn.BatchNorm2d(out_ch))
        
        self.repr = f"TransformerDecoder, hidden_size={hidden_size}, out_ch={out_ch}, out_size={out_size}, grid_size={grid_size}"
        
    def forward(self, hidden_states):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)

        return self.out(x)

    def __repr__(self):
        return self.repr

class ViT(nn.Module):
    def __init__(self, in_ch, img_size,  hidden_size=768, out_size=None, out_ch=None, grid_size=16, num_layers=12, nhead=12, dropout=0.1, attn_bias=False,\
        dim_feedforward=None, layer_norm_eps=1e-6, positional_embedding="zero", vis=False, up_out="conv", res=False, **kwargs) -> None:
        super().__init__()
        ### add assertion regarding hidden_size,...

        if grid_size is None or grid_size > img_size:
            print(f"got grid_size={grid_size}>{img_size}=img_size, adjusting grid_size")
            grid_size = int(img_size)

        if out_size is None:
            out_size = img_size
        if res and img_size!=out_size:
            raise NotImplementedError(f"ViT with img_size={img_size} != out_size={out_size} and res={res}")
        if out_ch is None:
            out_ch = in_ch

        img_size = int(img_size)
        
        self.embedder = Embedder(in_ch, img_size,  hidden_size, grid_size, positional_embedding, dropout)
        self.encoder = TransformerEncoder(hidden_size, num_layers=num_layers, nhead=nhead, dropout=dropout, attn_bias=attn_bias, dim_feedforward=dim_feedforward, layer_norm_eps=layer_norm_eps, vis=vis)
        self.out = TransformerDecoder(hidden_size, out_ch, out_size, grid_size, up=up_out)

        self.res = res

        self.repr = f"ViT, in_ch={in_ch}, out_ch={out_ch}, img_size={img_size}, hidden_size={hidden_size}, out_size={out_size}, grid_size={grid_size}, num_layers={num_layers}, nhead={nhead}"

    def forward(self, x):
        residual = x
        #print(f"ViT got input {x.shape}")
        x = self.embedder(x)
        #print(f"from embedder {x.shape}")
        x, attn_weights = self.encoder(x)
        #print(f"from transformer encoder {x.shape}")
        x = self.out(x)

        if self.res:
            x = x + residual
        return x
    
    def __repr__(self):
        return self.repr

### used to assign layers in Block
def get_layer_dict():
    return  {
    'get_padding'   :   get_padding,
    'nConvBlocks'   :   nConvBlocks,
    'convBlock'     :   convBlock,
    'resNeXtBlock'  :   resNeXtBlock,
    'dilationBlock' :   dilationBlock,
    'down'          :   down,
    'up'            :   up,
    'DCNBlock'      :   DCNBlock,
    'nDCNBlocks'    :   nDCNBlocks,
    'Encoder'       :   Encoder,
    'Decoder'       :   Decoder,
    #### torch
    'Identity'      :   nn.Identity,
    #### experimental
    'ViT'           :   ViT,
}
