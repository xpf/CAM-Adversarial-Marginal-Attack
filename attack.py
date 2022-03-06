import argparse, torch, cv2
from torchvision import transforms
from model import build_separated_model
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


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
    extractor = extractor.to(opts.device).eval()
    pooling = pooling.to(opts.device).eval()
    classifier = classifier.to(opts.device).eval()

    weights = classifier._modules['0'].weight.data

    x = transform(Image.open(opts.image)).unsqueeze(dim=0).to(opts.device)

    with torch.no_grad():
        f = extractor(x)
        p = classifier(pooling(f).view(x.shape[0], -1))
        y = torch.argmax(p, dim=1)
        weight = weights[y, :].unsqueeze(dim=-1).unsqueeze(dim=-1)
        cam_x = (f * weight).sum(dim=1, keepdim=True).clamp_min(0).view(x.shape[0], -1)
        cam_x = cam_x - cam_x.min(dim=-1, keepdim=True)[0]
        cam_x = cam_x / (cam_x.max(dim=-1, keepdim=True)[0] + 1e-12)

    clamp_max = (torch.ones_like(x) - mean) / std
    clamp_min = (torch.zeros_like(x) - mean) / std
    n = torch.rand_like(x) * (clamp_max - clamp_min) + clamp_min
    mask = torch.zeros_like(x)
    mask[:, :, -opts.e:, :], mask[:, :, :opts.e, :], mask[:, :, :, -opts.e:], mask[:, :, :, :opts.e] = 1, 1, 1, 1
    for _ in range(opts.n_iter):
        n = Variable(n.data, requires_grad=True)
        z = x * (1 - mask) + torch.max(torch.min(n, clamp_max), clamp_min) * mask
        u = extractor(z)
        v = classifier(pooling(u).view(x.shape[0], -1))
        t = torch.argmax(v, dim=1) == y
        cam_z = (u * weight).sum(dim=1, keepdims=True).clamp_min(0).view(z.shape[0], -1)
        cam_z = cam_z - cam_z.min(dim=-1, keepdim=True)[0]
        cam_z = cam_z / (cam_z.max(dim=-1, keepdim=True)[0] + 1e-12)
        loss_c = F.cross_entropy(v[t, :], y[t]) if t.sum() != 0 else 0
        loss_s = (1 - F.cosine_similarity(cam_z, cam_x, dim=-1).mean(dim=0))
        loss = loss_c + opts.lamb * loss_s
        extractor.zero_grad(), classifier.zero_grad(), pooling.zero_grad()
        loss.backward()
        n = n.data + n.grad.data.sign_() * (clamp_max - clamp_min) / opts.n_iter
    z = x * (1 - mask) + torch.max(torch.min(n, clamp_max), clamp_min) * mask
    with torch.no_grad():
        u = extractor(z)
        v = classifier(pooling(u).view(x.shape[0], -1))
        t = torch.argmax(v, dim=1)
        cam_z = (u * weight).sum(dim=1, keepdims=True).clamp_min(0).view(z.shape[0], -1)
        cam_z = cam_z - cam_z.min(dim=-1, keepdim=True)[0]
        cam_z = cam_z / (cam_z.max(dim=-1, keepdim=True)[0] + 1e-12)

    sal_x = F.interpolate(cam_x.view(x.shape[0], 1, f.shape[2], f.shape[3]), size=(224, 224),
                          mode='bilinear', align_corners=True)
    sal_x = sal_x.detach().cpu().numpy()
    sal_x = np.float32(cv2.applyColorMap(np.uint8(255 * (1 - sal_x[0, 0, :, :])), cv2.COLORMAP_JET)) / 255
    x = (x * std + mean).detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, :]

    sal_z = F.interpolate(cam_z.view(z.shape[0], 1, u.shape[2], u.shape[3]), size=(224, 224),
                          mode='bilinear', align_corners=True)
    sal_z = sal_z.detach().cpu().numpy()
    sal_z = np.float32(cv2.applyColorMap(np.uint8(255 * (1 - sal_z[0, 0, :, :])), cv2.COLORMAP_JET)) / 255
    z = (z * std + mean).detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, :]

    plt.figure(figsize=(6.4, 6.4))
    plt.subplot(221), plt.axis('off')
    plt.imshow(x)
    plt.text(5, 15, 'pred: {}'.format(y[0].item()), c='w')
    plt.subplot(222), plt.axis('off')
    plt.imshow(sal_x)
    plt.subplot(223), plt.axis('off')
    plt.imshow(z)
    plt.text(5, 15, 'pred: {}'.format(t[0].item()), c='w')
    plt.subplot(224), plt.axis('off')
    plt.imshow(sal_z)
    plt.tight_layout()
    plt.savefig(opts.image.replace('.jpg', '_ama_{}.jpg'.format(opts.e)))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default='resnet101')
    parser.add_argument('--image', type=str, default='./figures/test.jpg')
    parser.add_argument('--e', type=int, default=2)
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--lamb', type=float, default=100)
    opts = parser.parse_args()
    attack(opts)
