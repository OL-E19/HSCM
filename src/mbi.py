import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .dataset  import INP_Dataset
from utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR
from cv2 import circle
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import lpips
import torchvision
import time
import torch.optim as optim
from .networks import Generator, Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss


class MBI(nn.Module):
    def __init__(self, config):
        super(MBI, self).__init__()
        self.config = config

        self.model_name = 'HSCM'

        self.generator = Generator().to(config.DEVICE)
        self.discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge').to(config.DEVICE)

        self.iteration = 0
        self.gen_weights_path = os.path.join(config.PATH, self.model_name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, self.model_name + '_dis.pth')

        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss().to(config.DEVICE)
        self.style_loss = StyleLoss().to(config.DEVICE)
        self.adversarial_loss = AdversarialLoss(type=config.GAN_LOSS).to(config.DEVICE)

        self.gen_optimizer = optim.Adam(
            params=self.generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=self.discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.gen_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.gen_optimizer, last_epoch=-1,
                                                                  milestones=[20000, 40000, 60000, 80000, 120000],
                                                                  gamma=self.config.LR_Decay)
        self.dis_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.dis_optimizer, last_epoch=-1,
                                                                  milestones=[20000, 40000, 60000, 80000, 120000],
                                                                  gamma=self.config.LR_Decay)

        self.scaler = torch.cuda.amp.GradScaler()

        self.mask_type = config.MASK
        self.input_size = config.INPUT_SIZE

        self.transf = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)
        self.psnr = PSNR(255.0).to(config.DEVICE)

        # train mode
        if self.config.MODE == 1:
            self.image_flist = config.TRAIN_INPAINT_IMAGE_FLIST
            self.train_dataset = INP_Dataset(config.DEVICE, self.image_flist, self.mask_type, self.input_size)

        # test mode
        if self.config.MODE == 2:
            self.image_flist = config.TEST_INPAINT_IMAGE_FLIST
            self.mask_flist = config.TEST_MASK_FLIST
            self.test_dataset = INP_Dataset(config.DEVICE, self.image_flist, self.mask_type, self.mask_flist, self.input_size)

        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        self.log_file = os.path.join(config.PATH, 'log_' + self.model_name + '.dat')

    def process(self, images, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs

        outputs_img = self.forward(images, masks)

        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs_img.detach()

        dis_real, _ = self.discriminator(dis_input_real)
        dis_fake, _ = self.discriminator(dis_input_fake)

        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = outputs_img
        gen_fake, _ = self.discriminator(gen_input_fake)
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        gen_l1_loss = self.l1_loss(outputs_img, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_loss += gen_l1_loss

        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs_img, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss

        # generator style loss
        gen_style_loss = self.style_loss(outputs_img * masks, images * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        #############################

        # create logs
        logs = [
            ("gLoss", gen_loss.item()),
            ("dLoss", dis_loss.item())
        ]

        return outputs_img, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss

    def forward(self, images, masks):
        images_masked = (images * (1 - masks).float()) + masks

        outputs_img = self.generator(images_masked, masks)

        return outputs_img

    # def backward(self, gen_loss=None, dis_loss=None):
    #     gen_loss.backward()
    #     dis_loss.backward(retain_graph=True)
    #
    #     self.dis_optimizer.step()
    #     self.gen_optimizer.step()

    def backward(self, gen_loss=None, dis_loss=None):

        self.scaler.scale(dis_loss).backward(retain_graph=True)

        self.scaler.scale(gen_loss).backward()

        self.scaler.step(self.dis_optimizer)

        self.scaler.step(self.gen_optimizer)
        self.scaler.update()

        print(self.gen_scheduler.get_lr())

    def backward_joint(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading generator...')

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'], strict=False)
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading discriminator...')

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving...\n')
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def coarse_train(self):

        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        max_iteration = self.config.MAX_ITERS
        total = len(self.train_dataset)
        self.generator.train()
        self.discriminator.train()

        while keep_training:

            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                images, masks = self.cuda(*items)

                outputs_img, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss = self.process(
                    images, masks)
                outputs_merged = (outputs_img * masks) + (images * (1 - masks))

                # images = ((images + 1) / 2).clamp(0, 1)
                # outputs_merged = ((outputs_merged + 1) / 2).clamp(0, 1)

                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()

                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))

                self.backward(gen_loss, dis_loss)
                iteration = self.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                           ("epoch", epoch),
                           ("iter", iteration),
                       ] + logs

                progbar.add(len(images),
                            values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                ###################### visialization
                if iteration % 40 == 0:
                    create_dir(self.results_path)
                    inputs = (images * (1 - masks))
                    images_joint = stitch_images(
                        self.postprocess(images),
                        self.postprocess(inputs),
                        self.postprocess(outputs_img),
                        self.postprocess(outputs_merged),
                        img_per_row=1
                    )

                    path_masked = os.path.join(self.results_path, self.model_name, 'masked')
                    path_result = os.path.join(self.results_path, self.model_name, 'result')
                    path_joint = os.path.join(self.results_path, self.model_name, 'joint')
                    name = self.train_dataset.load_name(epoch - 1)[:-4] + '.png'

                    create_dir(path_masked)
                    create_dir(path_result)
                    create_dir(path_joint)

                    masked_images = self.postprocess(images * (1 - masks) + masks)[0]
                    images_result = self.postprocess(outputs_merged)[0]

                    print(os.path.join(path_joint, name[:-4] + '.png'))

                    images_joint.save(os.path.join(path_joint, name[:-4] + '.png'))
                    imsave(masked_images, os.path.join(path_masked, name))
                    imsave(images_result, os.path.join(path_result, name))

                    print(name + ' complete!')

                ##############

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()
        print('\nEnd training....')

    def test(self):

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        self.generator.eval()
        self.discriminator.eval()

        create_dir(self.results_path)

        psnr_list = []
        ssim_list = []
        l1_list = []
        lpips_list = []

        print('here')
        index = 0
        for items in test_loader:
            images, masks = self.cuda(*items)
            index += 1

            inputs = (images * (1 - masks))
            with torch.no_grad():
                tsince = int(round(time.time() * 1000))
                # print(images.shape)
                # print(masks.shape)
                outputs_img = self.forward(images, masks)
                ttime_elapsed = int(round(time.time() * 1000)) - tsince
                print('test time elaspsed {}ms'.format(ttime_elapsed))
            outputs_merged = (outputs_img * masks) + (images * (1 - masks))

            # print(outputs_merged.shape)

            psnr, ssim = self.metric(images, outputs_merged)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            if torch.cuda.is_available():
                pl = self.loss_fn_vgg(self.transf(outputs_merged[0].cpu()).cuda(),
                                      self.transf(images[0].cpu()).cuda()).item()
                lpips_list.append(pl)
            else:
                pl = self.loss_fn_vgg(self.transf(outputs_merged[0].cpu()), self.transf(images[0].cpu())).item()
                lpips_list.append(pl)

            l1_loss = torch.nn.functional.l1_loss(outputs_merged, images, reduction='mean').item()
            l1_list.append(l1_loss)

            print("psnr:{}/{}  ssim:{}/{} l1:{}/{}  lpips:{}/{}  {}".format(psnr, np.average(psnr_list),
                                                                            ssim, np.average(ssim_list),
                                                                            l1_loss, np.average(l1_list),
                                                                            pl, np.average(lpips_list),
                                                                            len(ssim_list)))

            images_joint = stitch_images(
                self.postprocess(images),
                self.postprocess(inputs),
                self.postprocess(outputs_img),
                self.postprocess(outputs_merged),
                img_per_row=1
            )

            path_masked = os.path.join(self.results_path, self.model_name, 'masked4060')
            path_result = os.path.join(self.results_path, self.model_name, 'result4060')
            path_joint = os.path.join(self.results_path, self.model_name, 'joint4060')

            name = self.test_dataset.load_name(index - 1)[:-4] + '.png'

            create_dir(path_masked)
            create_dir(path_result)
            create_dir(path_joint)

            masked_images = self.postprocess(images * (1 - masks) + masks)[0]
            images_result = self.postprocess(outputs_merged)[0]

            print(os.path.join(path_joint, name[:-4] + '.png'))

            images_joint.save(os.path.join(path_joint, name[:-4] + '.png'))
            imsave(masked_images, os.path.join(path_masked, name))
            imsave(images_result, os.path.join(path_result, name))

            print(name + ' complete!')

        print('\nEnd Testing')

        print('edge_psnr_ave:{} edge_ssim_ave:{} l1_ave:{} lpips:{}'.format(np.average(psnr_list),
                                                                            np.average(ssim_list),
                                                                            np.average(l1_list),
                                                                            np.average(lpips_list)))

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            print('load the generator:')
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))
            print('finish load')

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def metric(self, gt, pre):
        pre = pre.clamp_(0, 1) * 255.0
        pre = pre.permute(0, 2, 3, 1)
        pre = pre.detach().cpu().numpy().astype(np.uint8)[0]

        gt = gt.clamp_(0, 1) * 255.0
        gt = gt.permute(0, 2, 3, 1)
        gt = gt.cpu().detach().numpy().astype(np.uint8)[0]

        psnr = min(100, compare_psnr(gt, pre))
        ssim = compare_ssim(gt, pre, multichannel=True, data_range=255, channel_axis=2)

        return psnr, ssim

    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def tensor_to_img(self, image):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().detach().permute(0, 2, 3, 1).float().numpy()
        image_result = self.numpy_to_pil(image)[0]
        return image_result
