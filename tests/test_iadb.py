"""
https://github.com/tchambon/IADB/blob/main/iadb.py

Iterative alpha de-blending

"""
import torch
import torchvision
from torchvision import transforms
from diffusers import UNet2DModel
from torch.optim import Adam
from py.utils import *
from PIL import Image

def get_model():
    block_out_channels=(128, 128, 256, 256, 512, 512)
    down_block_types=( 
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    )
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D"  
    )
    return UNet2DModel(block_out_channels=block_out_channels,out_channels=3, in_channels=3, up_block_types=up_block_types, down_block_types=down_block_types, add_attention=True)

@torch.no_grad()
def sample_iadb(model, x0, nb_step):
    x_alpha = x0
    for t in range(nb_step):
        alpha_start = (t/nb_step)
        alpha_end =((t+1)/nb_step)

        d = model(x_alpha, torch.tensor(alpha_start, device=x_alpha.device))['sample']
        x_alpha = x_alpha + (alpha_end-alpha_start)*d

    return x_alpha

assert torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Load cats
size            = 64
cat_tensors     = []
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
        cat_tensors.append(img)
    
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
        
        d = model(x_alpha, alpha)['sample']
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