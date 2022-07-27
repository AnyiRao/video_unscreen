from . import deeplab


def get_deeplab_model(num_classes=2):
    model_map = {
        'deeplabv3_resnet50': deeplab.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': deeplab.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': deeplab.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': deeplab.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': deeplab.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': deeplab.deeplabv3plus_mobilenet
    }
    model = model_map['deeplabv3plus_resnet50'](num_classes=num_classes)
    return model
