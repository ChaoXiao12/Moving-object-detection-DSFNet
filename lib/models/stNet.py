from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from thop import profile
from lib.models.DSFNet_with_Static import DSFNet_with_Static
from lib.models.DSFNet import DSFNet
from lib.models.DSFNet_with_Dynamic import DSFNet_with_Dynamic

def model_lib(model_chose):
    model_factory = {
                     'DSFNet_with_Static': DSFNet_with_Static,
                     'DSFNet': DSFNet,
                     'DSFNet_with_Dynamic': DSFNet_with_Dynamic,
                     }
    return model_factory[model_chose]

def get_det_net(heads, model_name):
    model_name = model_lib(model_name)
    model = model_name(heads)
    return model


def load_model(model, model_path, optimizer=None, resume=False,
               lr=None, lr_step=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    heads = {'hm': 1, 'wh': 2, 'reg': 2}
    model_nameAll = ['DSFNet_with_Static',  'DSFNet_with_Dynamic', 'DSFNet_with_dynamic_3D_full', 'DSFNet']
    device = 'cuda:0'

    out = {}

    for model_name in model_nameAll:
        net = get_det_net(heads, model_name).to(device)
        input = torch.rand(1,3,5,512,512).to(device)
        flops, params = profile(net, inputs=(input,))
        out[model_name] = [flops, params]

    for k,v in out.items():
        print('---------------------------------------------')
        print(k + '   Number of flops: %.2fG' % (v[0] / 1e9))
        print(k + '   Number of params: %.2fM' % (v[1] / 1e6))