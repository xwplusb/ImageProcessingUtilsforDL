import torch

def add_pos_info(img_as_tensor:torch.tensor, dim=1):
    '''
    add position information to img
    param:
        dim=1  add a new channel of [0, 1, 2, ..., h*w] to image
        dim=2  add [[0,0], [0,1], ..., [h, w]] two channel to image
    '''
    *fron_size, c, h, w = img_as_tensor.shape
    if dim == 1:
        pos_info = torch.arange(h*w, device=img_as_tensor.device)
        pos_info = pos_info.repeat(fron_size).reshape(*fron_size, 1, h,w)
        img_as_tensor = torch.cat([img_as_tensor, pos_info], dim=-3)
    if dim == 2:
        x = torch.arange(h)
        y = torch.arange(w)
        pos_h, pos_w = torch.meshgrid(x,y)
        pos_info = torch.stack([pos_h, pos_w])
        pos_info = pos_info.repeat(*fron_size,1,1).reshape(*fron_size, 2, h, w)
        pos_info = pos_info.to(device=img_as_tensor.device)
        img_as_tensor = torch.cat([img_as_tensor, pos_info], dim=-3)
    return img_as_tensor
