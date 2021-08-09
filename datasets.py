from torchvision import transforms
from torchvision.datasets import ImageNet


def build_val_dataset(dataset='imagenet', data_path=''):
    if dataset == 'imagenet':
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_data = ImageNet(root='{}/imagenet/'.format(data_path), transform=val_transforms, split='val')
    else:
        raise ValueError('{:} is not concluded'.format(dataset))
    return val_data
