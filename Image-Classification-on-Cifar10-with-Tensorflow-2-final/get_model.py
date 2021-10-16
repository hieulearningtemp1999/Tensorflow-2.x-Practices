from models import vgg, resnet

MODEL_TYPE = [
    'vgg11',
    'vgg13',
    'vgg16',
    'vgg19',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152'
]


def get_model(model_type):
    if model_type not in MODEL_TYPE:
        raise ValueError("Model type should be in : ", MODEL_TYPE)

    if model_type == 'vgg11':
        return vgg.VGG11()

    if model_type == 'vgg13':
        return vgg.VGG13()

    if model_type == 'vgg16':
        return vgg.VGG16()

    if model_type == 'vgg19':
        return vgg.VGG19()

    if model_type == 'resnet18':
        return resnet.ResNet18()

    if model_type == 'resnet34':
        return resnet.ResNet34()

    if model_type == 'resnet50':
        return resnet.ResNet50()

    if model_type == 'resnet101':
        return resnet.ResNet101()

    if model_type == 'resnet152':
        return resnet.ResNet152()


