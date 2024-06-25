"""
https://github.com/tchambon/IADB/blob/main/iadb.py

Iterative alpha de-blending

In this version we replace the u net from the source code with smaller blocks

https://arxiv.org/abs/2303.03667
<<Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks>>

https://github.com/liaomingg/FasterNet/blob/master/pconv.py

"""
import torch
import torchvision
from torchvision import transforms
from diffusers import UNet2DModel
from torch.optim import Adam
from py.utils import *
from PIL import Image

class SmolActivation(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        """
            Tanh that is scaled down by a Gaussian
        """
        return self.tanh(x) * torch.exp(-x**2)


class SmolBlock(torch.nn.Module):

    def __init__(self, out_channels, in_channels, stride, kernel_size=3, padding=1, dilation=1) -> None:
        super().__init__()

        interim_features = in_channels

        self.conv1          = torch.nn.Conv2d(in_channels=in_channels, out_channels=interim_features,
                                              kernel_size=kernel_size, stride=stride, padding=padding, dilation=stride, groups=1)
        self.dwconv         = torch.nn.Conv2d(in_channels=interim_features + in_channels,
                                              out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=out_channels)
        # self.spline_degree  = 4
        # self.spline_k       = torch.nn.Parameter(torch.randn(self.spline_degree, 1, out_channels, 1, 1), requires_grad=True)
        self.activation     = torch.nn.GELU()
        self.activation2    = SmolActivation()

    def forward(self, i):
        x = self.conv1(i)
        x = self.activation(x)
        x = self.dwconv(torch.cat([i, x], dim=1))
        # apply spline
        x = self.activation(x)
        # x = x * self.spline_k[0] / 1.0 + self.activation2(x) * self.spline_k[1] / 1.0
        return x

class SmolUnet(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.num_features_0 = 4
        self.num_output_channels = 3
        self.num_features_1 = 16
        self.feature_extraction_conv    = torch.nn.Conv2d(in_channels=self.num_features_0,
                                                          out_channels=self.num_features_1, kernel_size=3, stride=1, padding=1) 
        
        self.smol_block_down_1          = SmolBlock(out_channels=32,    in_channels=self.num_features_1,    kernel_size=3, stride=1, dilation=16)
        self.smol_block_down_2          = SmolBlock(out_channels=64,    in_channels=32,                     kernel_size=3, stride=1, dilation=4)
        self.smol_block_down_3          = SmolBlock(out_channels=128,   in_channels=64,                     kernel_size=3, stride=1, dilation=2)
        
        self.smol_bottleneck            = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.smol_bottleneck_dw         = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        
        self.smol_block_up_1            = SmolBlock(out_channels=64,    in_channels=128,    kernel_size=3, stride=1)
        self.smol_block_up_2            = SmolBlock(out_channels=32,    in_channels=64,     kernel_size=3, stride=1)
        self.smol_block_up_3            = SmolBlock(out_channels=16,    in_channels=32,     kernel_size=3, stride=1)

        self.feature_reconstruction_conv = torch.nn.Conv2d(in_channels=self.num_features_1,
                                                            out_channels=self.num_output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, alpha):
        N, C, H, W = x.shape
        # print(f"x.shape={x.shape}")
        # print(f"alpha.shape={alpha.shape}")
        x = torch.cat([x, torch.sin(alpha * torch.pi).unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(N, 1, H, W)], dim=1)
        x0 = self.feature_extraction_conv(x)
        # print(f"x0.shape={x0.shape}")
        
        x1 = self.smol_block_down_1(x0)
        _x1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)(x1)
        # print(f"x1.shape={x1.shape}")

        x2 = self.smol_block_down_2(_x1)
        _x2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)(x2)
        # print(f"x2.shape={x2.shape}")
        
        x3 = self.smol_block_down_3(_x2)
        _x3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)(x3)
        # print(f"x3.shape={x3.shape}")

        _x4 = self.smol_bottleneck(_x3)
        _x41 = torch.nn.GELU()(_x4)
        _x42 = self.smol_bottleneck_dw(_x41)
        _x43 = torch.nn.GELU()(_x42)
        x4 = torch.nn.Upsample(scale_factor=2, mode='bilinear')(_x43)
        x5 = self.smol_block_up_1(x3 + x4)
        x6 = torch.nn.Upsample(scale_factor=2, mode='bilinear')(x5)
        x7 = self.smol_block_up_2(x2 + x6)
        x8 = torch.nn.Upsample(scale_factor=2, mode='bilinear')(x7)
        x9 = self.smol_block_up_3(x1 + x8)
        x10 = self.feature_reconstruction_conv(x0 + x9)
        x11 = torch.nn.Sigmoid()(x10)
        return x10


def get_model():
    return SmolUnet()

@torch.no_grad()
def sample_iadb(model, x0, nb_step):
    x_alpha = x0
    for t in range(nb_step):
        alpha_start = (t/nb_step)
        alpha_end   = ((t+1)/nb_step)
        d           = model(x_alpha, torch.tensor([alpha_start], device=x_alpha.device))
        x_alpha     = x_alpha + (alpha_end - alpha_start) * d

    return x_alpha

assert torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Load cats
# https://www.kaggle.com/datasets/spandan2/cats-faces-64x64-for-generative-models
size            = 64
batch_size      = 16
cat_tensors     = []
cur_batch       = []
MAX_NUM_CATS    = 1 << 10
cats_folder = get_or_create_tmp() / "cats"
for f in cats_folder.iterdir():
    if f.is_file() and f.suffix == ".jpg":
        # Load .jpg and convert to tensor
        img = Image.open(f).convert("RGB")
        img = img.resize((size, size))
        img = transforms.ToTensor()(img)
        # convert to NCHW
        img = img.unsqueeze(0)
        # upload to GPU
        img = img.to(device)
        N, C, H, W = img.shape
        assert C == 3, "Only RGB images are supported"
        assert H == size and W == size, "Only 64x64 images are supported"
        cur_batch.append(img)
        if len(cur_batch) == batch_size:
            cat_tensors.append(torch.cat(cur_batch, dim=0))
            cur_batch = []
    
model = get_model()
model = model.to(device)

# load checkpoint if exists

ckpt = get_or_create_tmp() / "iadb.ckpt"
try:
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt))
except Exception as e:
    print(f"Failed to load checkpoint {ckpt}: {e}")


optimizer = Adam(model.parameters(), lr=1e-4)
nb_iter = 0
print('Start training')
for current_epoch in range(100):
    for i, data in enumerate(cat_tensors):
        x1 = (data * 2.0) - 1.0
        x0 = torch.randn_like(x1)
        bs = x0.shape[0]

        alpha   = torch.rand(bs, device=device)
        x_alpha = alpha.view(-1, 1, 1, 1) * x1 + (1 - alpha).view(-1, 1, 1, 1) * x0
        
        d = model(x_alpha, alpha)
        loss = torch.sum((d - (x1-x0))**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch {current_epoch}, iter {i}, loss {loss.item()}')

        nb_iter += 1

        if nb_iter % 200 == 0:
            with torch.no_grad():
                print(f'Save export {nb_iter}')
                sample = (sample_iadb(model, x0, nb_step=128) * 0.5) + 0.5
                torchvision.utils.save_image(sample, get_or_create_tmp() / f'export_{str(nb_iter).zfill(8)}.png')
                torch.save(model.state_dict(), ckpt)