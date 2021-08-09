import argparse, torch, cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

from model import build_separated_model


def attack(opts):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).to(opts.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).to(opts.device)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    extractor, pooling, classifier = build_separated_model(model_name=opts.model_name)
    extractor, pooling, classifier = extractor.to(opts.device).eval(), pooling.to(opts.device).eval(), classifier.to(
        opts.device).eval()
    weights = classifier._modules['0'].weight.data
    x = transform(Image.open(opts.image_file)).unsqueeze(dim=0).to(opts.device)
    with torch.no_grad():
        f = extractor(x)
        p = classifier(pooling(f).view(x.shape[0], -1))
        y = torch.argmax(p, dim=1)
        weight = weights[y, :].unsqueeze(dim=-1).unsqueeze(dim=-1)
        cam = (f * weight).sum(dim=1, keepdim=True).clamp_min(0).view(x.shape[0], -1)
        cam -= cam.min(dim=-1, keepdim=True)[0]
        cam /= (cam.max(dim=-1, keepdim=True)[0] + 1e-7)

    clamp_max = (torch.ones_like(x) - mean) / std
    clamp_min = (torch.zeros_like(x) - mean) / std
    n = torch.rand_like(x) * (clamp_max - clamp_min) + clamp_min
    mask = torch.zeros_like(x)
    mask[:, :, -opts.e:, :], mask[:, :, :opts.e, :], mask[:, :, :, -opts.e:], mask[:, :, :, :opts.e] = 1, 1, 1, 1
    for _ in range(opts.iters):
        n = Variable(n.data, requires_grad=True)
        z = x * (1 - mask) + torch.max(torch.min(n, clamp_max), clamp_min) * mask
        u = extractor(z)
        v = classifier(pooling(u).view(x.shape[0], -1))
        t = torch.argmax(v, dim=1) == y
        adv_cam = (u * weight).sum(dim=1, keepdims=True).clamp_min(0).view(x.shape[0], -1)
        adv_cam -= adv_cam.min(dim=-1, keepdim=True)[0]
        adv_cam /= (adv_cam.max(dim=-1, keepdim=True)[0] + 1e-7)
        loss_c = F.cross_entropy(v[t, :], y[t]) if t.sum() != 0 else 0
        loss_s = (1 - F.cosine_similarity(adv_cam, cam, dim=-1).mean(dim=0))
        loss = loss_c + opts.lamb * loss_s
        extractor.zero_grad(), classifier.zero_grad(), pooling.zero_grad()
        loss.backward()
        n = n.data + n.grad.data.sign_() * (clamp_max - clamp_min) / opts.iters

    z = x * (1 - mask) + torch.max(torch.min(n, clamp_max), clamp_min) * mask


    with torch.no_grad():
        f = extractor(x)
        cam = (f * weight).sum(dim=1, keepdim=True).clamp_min(0)
    sal = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=True)
    sal = sal.view(sal.shape[0], sal.shape[1], -1)
    sal -= torch.min(sal, dim=-1, keepdim=True)[0]
    sal /= (torch.max(sal, dim=-1, keepdim=True)[0] + 1e-7)
    sal = sal.view(sal.shape[0], sal.shape[1], 224, 224)
    sal = sal.detach().cpu().numpy()
    sal = np.float32(cv2.applyColorMap(np.uint8(255 * (1 - sal[0, 0, :, :])), cv2.COLORMAP_JET)) / 255

    plt.subplot(221), plt.axis('off')
    x = (x * std + mean).detach().cpu().numpy().transpose(0, 2, 3, 1)
    plt.imshow(x[0, :, :, :])
    plt.subplot(222), plt.axis('off')
    plt.imshow(sal)

    with torch.no_grad():
        f = extractor(z)
        cam = (f * weight).sum(dim=1, keepdim=True).clamp_min(0)
    sal = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=True)
    sal = sal.view(sal.shape[0], sal.shape[1], -1)
    sal -= torch.min(sal, dim=-1, keepdim=True)[0]
    sal /= (torch.max(sal, dim=-1, keepdim=True)[0] + 1e-7)
    sal = sal.view(sal.shape[0], sal.shape[1], 224, 224)
    sal = sal.detach().cpu().numpy()
    sal = np.float32(cv2.applyColorMap(np.uint8(255 * (1 - sal[0, 0, :, :])), cv2.COLORMAP_JET)) / 255
    plt.subplot(223), plt.axis('off')
    z = (z * std + mean).detach().cpu().numpy().transpose(0, 2, 3, 1)
    plt.imshow(z[0, :, :, :])
    plt.subplot(224), plt.axis('off')
    plt.imshow(sal)

    plt.tight_layout()
    plt.savefig('result.jpg')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--model_name', type=str, default='resnet101')
    parser.add_argument('--image_file', type=str, default='./test.jpg')

    parser.add_argument('--e', type=int, default=1, choices=[1, 2, 4, 8])
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--lamb', type=float, default=100.0)

    opts = parser.parse_args()
    attack(opts)
