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

        interim_features = out_channels

        self.conv1          = torch.nn.Conv2d(in_channels=in_channels, out_channels=interim_features,
                                              kernel_size=kernel_size, stride=stride,
                                              padding=padding + (dilation - 1), dilation=dilation, groups=1, bias=True)
        self.dwconv         = torch.nn.Conv2d(in_channels=interim_features,
                                              out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                                              groups=out_channels, bias=True)
        # self.spline_degree  = 4
        # self.spline_k       = torch.nn.Parameter(torch.randn(self.spline_degree, 1, out_channels, 1, 1), requires_grad=True)
        self.activation     = torch.nn.GELU()
        self.activation2    = SmolActivation()

    def forward(self, i):
        x = self.conv1(i)
        x0 = self.activation(x)
        x1 = self.dwconv(x0)
        # apply spline
        # x = x1 + self.activation2(x)
        x = self.activation(x1)
        # x = x * self.spline_k[0] / 1.0 + self.activation2(x) * self.spline_k[1] / 1.0
        return x

class GaussianStrokes(torch.nn.Module):
    def __init__(self, output_resolution, num_gaussians, num_features, device) -> None:
        super().__init__()
        self.num_gaussians          = num_gaussians
        self.output_resolution      = output_resolution
        self.device                 = device
        self.gaussian_centers       = torch.nn.Linear(in_features=num_features, out_features=num_gaussians * 2).to(device)
        self.gaussian_invsigmas     = torch.nn.Linear(in_features=num_features, out_features=num_gaussians * 2).to(device)
        self.gaussian_correlations  = torch.nn.Linear(in_features=num_features, out_features=num_gaussians).to(device)
        self.gaussian_colors        = torch.nn.Linear(in_features=num_features, out_features=num_gaussians * 3).to(device)
        self.gaussian_magnitudes    = torch.nn.Linear(in_features=num_features, out_features=num_gaussians).to(device)

    def forward(self, x):
        N, C, H, W = x.shape
        # print(f"x.shape={x.shape}")
        x = x.view(N, C)
        magnitues   = self.gaussian_magnitudes(x).view(N, self.num_gaussians, 1)
        centers     = self.gaussian_centers(x).view(N, self.num_gaussians, 2)
        colors      = self.gaussian_colors(x).view(N, self.num_gaussians, 3)
        invsigmas   = self.gaussian_invsigmas(x).view(N, self.num_gaussians, 2)
        correlations= self.gaussian_correlations(x).view(N, self.num_gaussians, 1)
        
        activation2 = torch.nn.Tanh()
        activation  = torch.nn.Sigmoid()
        
        magnitude   = activation2(magnitues) * 1.0
        center      = activation(centers)
        color       = activation(colors)
        invsigmas   = activation(invsigmas) * 16.0 + 1.0
        correlation = activation(correlations)

        H, W = self.output_resolution, self.output_resolution
        output_tensor   = torch.zeros(N, 3, H, W, device=x.device)
        wacc            = torch.zeros(N, 1, H, W, device=x.device)
        X = torch.range(0, W-1, device=x.device).view(1, 1, 1, W).expand(N, 1, H, W)
        Y = torch.range(0, H-1, device=x.device).view(1, 1, H, 1).expand(N, 1, H, W)
        U = X / W
        V = Y / H

        # torchvision.utils.save_image(U, get_or_create_tmp() / f'test.png'); exit()

        for i in range(self.num_gaussians):
            center      = centers[:, i, :].view(N, 2, 1, 1).expand(N, 2, H, W)
            invsigma    = invsigmas[:, i, :].view(N, 2, 1, 1).expand(N, 2, H, W)
            correlation = correlations[:, i, :].view(N, 1, 1, 1).expand(N, 1, H, W)
            color       = colors[:, i, :].view(N, 3, 1, 1).expand(N, 3, H, W)
            magnitude   = magnitues[:, i, :].view(N, 1, 1, 1).expand(N, 1, H, W)

            # print(f"center.shape = {center.shape}")

            diff_x    = U - center[:, 0].unsqueeze(1)
            diff_y    = V - center[:, 1].unsqueeze(1)
            scaled_x  = diff_x * invsigma[:, 0].unsqueeze(1)
            scaled_y  = diff_y * invsigma[:, 1].unsqueeze(1)
            corr_xy   = correlation * scaled_x * scaled_y
            dist      = scaled_x * scaled_x + scaled_y * scaled_y - 2 * corr_xy
            w         = torch.exp(-dist / 1.0)
            # torchvision.utils.save_image(w, get_or_create_tmp() / f'test.png'); exit()
            output_tensor += (color * 2.0 - 1.0) * w
            wacc += w

        output_tensor /= wacc

        return output_tensor
        # return centers, invsigmas, correlations

class SmolUnet(torch.nn.Module):

    def __init__(self, device, input_resolution) -> None:
        super().__init__()

        self.num_features_0             = 4
        self.num_output_channels        = 3
        self.num_features_1             = 8
        self.feature_extraction_conv    = torch.nn.Conv2d(in_channels=self.num_features_0,
                                                          out_channels=self.num_features_1,
                                                          kernel_size=3, stride=1, padding=1).to(device) 
        
        self.down_modules   = []
        num_features        = self.num_features_1
        mip_size            = input_resolution
        while mip_size > 1:
            num_features    *= 2
            self.down_modules.append(SmolBlock(out_channels=num_features, in_channels=num_features // 2, kernel_size=3, stride=1, padding=1)
                                     .to(device))
            mip_size        = mip_size // 2
        
        self.final_features = num_features

        self.up_modules = []
        while mip_size != input_resolution:
            num_features = num_features // 2
            self.up_modules.append(SmolBlock(out_channels=num_features,
                                             in_channels=num_features * 2,
                                             kernel_size=3, stride=1, padding=1).to(device))
            mip_size *= 2

        self.feature_final_conv = torch.nn.Conv2d(in_channels=num_features,
                                                  out_channels=self.num_output_channels, kernel_size=3, stride=1, padding=1).to(device)

        self.gaussian_strokes = GaussianStrokes(
                output_resolution=input_resolution,
                num_gaussians=64,
                num_features=self.final_features,
                device=device)

    def forward(self, x, alpha):
        N, C, H, W = x.shape
        x = torch.cat([x, torch.sin(alpha * torch.pi).unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(N, 1, H, W)], dim=1)
        fc = self.feature_extraction_conv(x)
        x0 = fc
        downs = []
        for down_module in self.down_modules:
            x0 = down_module(x0)
            x0 = torch.nn.MaxPool2d(kernel_size=2, stride=2)(x0)
            downs.append(x0)
            # print(f"x0.shape = {x0.shape}")

        x1 = x0
        for i, up_module in enumerate(self.up_modules):
            d = downs.pop()
            if i != 0:
                x1 = up_module(x1 + d)
            else:
                x1 = up_module(x1)
            x1 = torch.nn.Upsample(scale_factor=2, mode='bilinear')(x1)

        x1 = self.feature_final_conv(x1 + fc)
        x1 = torch.tanh(x1)

        N, C, H, W = x0.shape
        assert (H == 1) and (W == 1), f"Expected H=1 and W=1 but got H={H} and W={W}"

        x0 = x1 + self.gaussian_strokes(x0)

        return x0

def get_model():
    return SmolUnet(64)

@torch.no_grad()
def sample_iadb(model, x0, nb_step):
    x_alpha = x0
    for t in range(nb_step):
        alpha_start = (t/nb_step)
        alpha_end   = ((t+1)/nb_step)
        d           = model(x_alpha, torch.tensor([alpha_start], device=x_alpha.device))
        # return d
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
    
model = SmolUnet(device=device, input_resolution=64)

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