import os
import os.path as osp

pretrained_models = {'resnet18': ['resnet', 'resnet18-5c106cde.pth']}

def get_pretrained_path(model_name='resnet18'):
    model_name = model_name.lower()
    pretrained_dir = os.path.split(__file__)[0]
    model_dir = pretrained_models.get(model_name, None)
    if model_dir is None:
        raise ValueError('model_name {} not in '.format(model_name), pretrained_models)
    pretrained_path = osp.join(pretrained_dir, *model_dir)
    return pretrained_path