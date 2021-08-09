import os, argparse, torch, tqdm

from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable

from model import build_separated_model
from datasets import build_val_dataset

def attack(opts):
    print('model: {}, save: {}, name: {}'.format(opts.model_name, opts.dataset, opts.save_path, name))
    save_path = os.path.join(opts.save_path, '{}_{}'.format(opts.model_name, opts.dataset))
    if not os.path.isdir(save_path): os.mkdir(save_path)
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).to(opts.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).to(opts.device)
    val_data = build_val_dataset(dataset=opts.dataset, data_path=opts.data_path)
    val_loader = DataLoader(dataset=val_data, batch_size=opts.batch_size, shuffle=False, num_workers=2)
    extractor, pooling, classifier = build_separated_model(model_name=opts.model_name)
    weight = classifier._modules['0'].weight.data.unsqueeze(dim=-1).unsqueeze(dim=-1)

    run_tqdm = tqdm.tqdm(val_loader)
    for x, y in run_tqdm:
        x, y = x.to(opts.device), y.to(opts.device)
        with torch.no_grad():
            f = extractor(x)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--model_name', type=str, default='resnet101')
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--data_path', type=str, default='./../../../data2/xpf/datasets')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--e', type=int, default=8, choices=[1, 2, 4, 8])
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--lamb', type=float, default=100.0)

    parser.add_argument('--ths', type=list, default=[0.3, 0.4, 0.5, 0.6, 0.7])
    opts = parser.parse_args()


    attack(opts)