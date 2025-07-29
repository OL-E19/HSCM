import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import numpy as np
import os
import glob
import lpips
from PIL import Image
from pytorch_fid import fid_score
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


class Eval_Dataset(Dataset):
    def __init__(self, image_flist, gen_flist, input_size=256):
        super(Dataset, self).__init__()

        self.img_data = self.load_flist(image_flist)
        self.gen_data = self.load_flist(gen_flist)

        self.input_size = input_size

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index):

        img = Image.open(self.img_data[index]).resize((self.input_size, self.input_size))
        if not img.mode == "RGB":
            img = img.convert("RGB")
        img = np.array(img).astype(np.uint8)
        img = (img / 255.0).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)

        gen = Image.open(self.gen_data[index]).resize((self.input_size, self.input_size))
        if not gen.mode == "RGB":
            gen = gen.convert("RGB")
        gen = np.array(gen).astype(np.uint8)
        gen = (gen / 255.0).astype(np.float32)
        gen = np.transpose(gen, (2, 0, 1))
        gen = torch.from_numpy(gen)

        return img.to('cuda'), gen.to('cuda')

    def load_name(self, index):
        name = self.img_data[index]
        return os.path.basename(name)

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: images file path, images directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=str, encoding='utf-8')
                except Exception as e:
                    print(e)
                    return [flist]

        return []


def metric(gt, pre):
    pre = pre.clamp_(0, 1) * 255.0
    pre = pre.permute(0, 2, 3, 1)
    pre = pre.detach().cpu().numpy().astype(np.uint8)[0]

    gt = gt.clamp_(0, 1) * 255.0
    gt = gt.permute(0, 2, 3, 1)
    gt = gt.cpu().detach().numpy().astype(np.uint8)[0]

    psnr = min(100., compare_psnr(gt, pre))
    ssim = compare_ssim(gt, pre, multichannel=True, data_range=255, channel_axis=2)

    return psnr, ssim


def cal_fid(img_path_, gen_path_):
    fid_value = fid_score.calculate_fid_given_paths([img_path_, gen_path_], device='cuda', batch_size=50, dims=2048)

    return fid_value


def main(img_flist_, gen_flist_):
    eval_dataset = Eval_Dataset(img_flist_, gen_flist_)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=1)

    loss_fn_vgg = lpips.LPIPS(net='vgg').to('cuda')
    transf = torchvision.transforms.Compose(
        [torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    psnr_list = []
    ssim_list = []
    l1_list = []
    lpips_list = []

    index = 0
    for items in eval_loader:
        img, gen = items
        index += 1

        # psnr, ssim
        psnr, ssim = metric(img, gen)

        # l1
        l1 = torch.nn.functional.l1_loss(gen, img, reduction='mean').item()

        # lpips
        pl = loss_fn_vgg(transf(gen).to('cuda'), transf(img).to('cuda')).item()



        psnr_list.append(psnr)
        ssim_list.append(ssim)
        l1_list.append(l1)
        lpips_list.append(pl)

        print("img{}, psnr:{}, ssim:{}, ls:{}, lpips:{}".format(index, psnr, ssim, l1, pl))

    print("====================================AVG====================================")
    print("num:{}, psnr:{}, ssim:{}, ls:{}, lpips:{}".format(index,
                                                             np.average(psnr_list),
                                                             np.average(ssim_list),
                                                             np.average(l1_list),
                                                             np.average(lpips_list)))
    print("====================================FID====================================")
    fid = cal_fid(img_flist_, gen_flist_)
    print(f"fid:{fid}")
    print("====================================END====================================")


def make_flist(out_put_, img_path_):
    ext = {'.jpg', '.png', '.txt'}

    images = []
    for root, dirs, files in os.walk(img_path_):
        print('loading ' + root)
        for file in files:
            if os.path.splitext(file)[1] in ext:
                images.append(os.path.join(root, file))

    images = sorted(images)
    np.savetxt(out_put_, images, fmt='%s')


if __name__ == "__main__":
   
    img_flist = ""		# image path
    gen_flist = ""		# inpainting path

    main(img_flist, gen_flist)


