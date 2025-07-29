import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
# from torchinfo import summary
from mamba_ssm import Mamba


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class CrossGatedConvFFN(nn.Module):
    def __init__(self, dim, ffn_ratio=2, groups=2, bias=True, LayerNorm_type='WithBias'):
        super().__init__()
        hidden_dim = int(dim * ffn_ratio)
        self.groups = groups

        self.project_in = nn.Conv2d(dim, hidden_dim, 1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_dim, hidden_dim, 3, 1, 1,
            groups=hidden_dim // groups, bias=bias
        )
        self.norm = LayerNorm(hidden_dim, LayerNorm_type)

        self.channel_mixer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 16, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 16, hidden_dim, 1),
            nn.Sigmoid()
        )

        self.gate = nn.Conv2d(hidden_dim * 2, hidden_dim, 1, bias=bias)

        self.project_out = nn.Conv2d(hidden_dim, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)

        local_feat = self.dwconv(x)
        local_feat = self.norm(local_feat)

        global_feat = self.channel_mixer(x) * x

        fused_feat = torch.cat([local_feat, global_feat], dim=1)
        gate_weight = torch.sigmoid(self.gate(fused_feat))
        x = local_feat * gate_weight + global_feat * (1 - gate_weight)

        return self.project_out(x)


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.mamba1 = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.mamba2 = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

    def forward(self, x):

        # local
        B1, C1, H1, W1 = x.shape
        w = 2
        L1 = H1 * W1
        Hg, Wg = math.ceil(H1 / w), math.ceil(W1 / w)

        x1h = x.view(B1, C1, Hg, w, Wg, w).permute(0, 1, 2, 4, 3, 5).reshape(B1, C1, -1)
        x1v = x.view(B1, C1, Hg, w, Wg, w).permute(0, 1, 4, 2, 5, 3).reshape(B1, C1, -1)

        y1h = self.mamba1(x1h.transpose(-1, -2)).transpose(-1, -2)
        y1h = y1h.view(B1, C1, Hg, Wg, w, w).permute(0, 1, 2, 4, 3, 5).reshape(B1, C1, L1)
        y1v = self.mamba2(x1v.transpose(-1, -2)).transpose(-1, -2)
        y1v = y1v.view(B1, C1, Wg, Hg, w, w).permute(0, 1, 3, 5, 2, 4).reshape(B1, C1, L1)
        y1hv = y1h + y1v

        out = y1hv.reshape(B1, C1, H1, W1)

        return out


class MambaLayer2(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.mamba3 = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)


    def forward(self, x):

        # global
        x2 = self.avg_pool(x)
        B2, C2, H2, W2 = x2.shape
        L2 = H2 * W2

        x2h = torch.flip(x2.reshape(B2, -1, L2), dims=[-1]).float()
        x2v = torch.flip(x2.transpose(-1, -2).reshape(B2, -1, L2), dims=[-1]).float()
        x2hv = torch.cat((x2h, x2v), dim=-1)

        y2 = self.mamba3(x2hv.transpose(-1, -2)).transpose(-1, -2)
        y2v, y2h = torch.flip(y2, dims=[-1]).chunk(2, dim=-1)
        y2hv = y2h.reshape(B2, C2, H2, W2) + y2v.reshape(B2, C2, W2, H2).transpose(-1, -2)
        y2hv = self.upsample(y2hv)

        out = y2hv

        return out


class MambaBlock(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(MambaBlock, self).__init__()

        self.attn = MambaLayer(dim)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn2 = MambaLayer2(dim)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = CrossGatedConvFFN(dim)
        self.norm3 = LayerNorm(dim, LayerNorm_type)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.attn2(self.norm2(x))
        x = x + self.ffn(self.norm3(x))

        return x


class GatedEmb(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(GatedEmb, self).__init__()

        self.gproj1 = nn.Conv2d(in_c, embed_dim * 2, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        # x = self.proj(x)
        x = self.gproj1(x)
        x1, x2 = x.chunk(2, dim=1)

        x = F.gelu(x1) * x2

        return x


class Generator(nn.Module):
    def __init__(self,
                 inp_channels=4,
                 out_channels=3,
                 dim=48,
                 num_blocks=[2, 3, 3, 11],
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 ):
        super(Generator, self).__init__()

        self.patch_embed = GatedEmb(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            MambaBlock(dim=dim, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            MambaBlock(dim=int(dim * 2 ** 1), LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            MambaBlock(dim=int(dim * 2 ** 2), LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))

        self.latent = nn.Sequential(*[
            MambaBlock(dim=int(dim * 2 ** 3), LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)

        self.decoder_level3 = nn.Sequential(*[
            MambaBlock(dim=int(dim * 2 ** 2), LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(*[
            MambaBlock(dim=int(dim * 2 ** 1), LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential(*[
            MambaBlock(dim=int(dim * 2 ** 1), LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.output = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        )

    def forward(self, inp_img, mask):
        inp_enc_level1 = self.patch_embed(torch.cat((inp_img, mask), dim=1))

        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)

        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)

        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.output(out_dec_level1)

        out_dec_level1 = (torch.tanh(out_dec_level1) + 1) / 2
        return out_dec_level1


def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r**2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class Downsample(nn.Module):
    def __init__(self,n_feat):
        super(Downsample, self).__init__()
        self.body = DWT()
        self.out = nn.Conv2d(n_feat*4,n_feat*2,kernel_size=1,stride=1,padding=0,bias=False)

    def forward(self,x):
        # x1,x2 = x.chunk(2,dim=1)
        x = self.body(x)
        return self.out(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = IWT()
        self.out = nn.Conv2d(n_feat//4,n_feat//2,kernel_size=1,stride=1,padding=0,bias=False)

    def forward(self, x):
        x = self.body(x)
        return self.out(x)


if __name__ == '__main__':
    net = Generator().to('cuda')


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of parameters: {count_parameters(net)}")

