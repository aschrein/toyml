import torch
import numpy as np

def test_torch():
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    c = a + b
    assert np.all(c.numpy() == np.array([5, 7, 9]))

def test_torch_cuda():
    a = torch.tensor([1, 2, 3]).cuda()
    b = torch.tensor([4, 5, 6]).cuda()
    c = a + b
    assert np.all(c.cpu().numpy() == np.array([5, 7, 9]))

if __name__ == "__main__":
    assert torch.cuda.is_available(), f"torch.cuda.is_available()={torch.cuda.is_available()}"
    
    # print device name
    print(f"torch.cuda.get_device_name(0)={torch.cuda.get_device_name(0)}")
    
    test_torch()
    test_torch_cuda()
    print("All tests passed.")