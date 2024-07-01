import wgpu
from wgpu.utils.compute import compute_with_buffers
import torch
import numpy as np

def make_conv2d_nhwc_fp32(num_features,
                num_channels,
                kernel_size,
                stride,
                padding,
                dst_width,
                dst_height,
                src_width,
                src_height,
                dilation=(1, 1),
                bias_enable=True,
                groups=1,
                dispatch_group_size=(8, 8)
                ):
    """
        Generate a Conv2D shader for NHWC layout.
    
        kernel_size : (int, int)
        stride      : (int, int)
        padding     : (int, int)
        dilation    : (int, int)
        groups      : int
        return      : str
    """
    assert len(kernel_size) == 2, f"Kernel size must be 2D."
    assert len(stride) == 2, f"Stride must be 2D."
    assert len(padding) == 2, f"Padding must be 2D."
    if kernel_size == (1, 1):
        assert dilation == (1, 1), f"Dilation must be 1 for 1x1 kernel."
        assert padding == (0, 0), f"Padding must be 0 for 1x1 kernel."
    elif kernel_size == (3, 3):
        assert dilation == (1, 1), f"Dilation must be 1 for 3x3 kernel."
        assert padding == (1, 1), f"Padding must be 1 for 3x3 kernel."
    else:
        assert False, f"Kernel size {kernel_size} not supported yet."

    assert dilation == (1, 1), f"Dilation not supported yet."
    assert groups == 1, f"Groups not supported yet."

    return f"""

    @group(0) @binding(0) var<storage, read_write>         g_rw_dst                 : array<f32>;
    @group(0) @binding(1) var<storage, read_write>         g_rw_src                 : array<f32>;
    @group(0) @binding(2) var<storage, read_write>         g_rw_weights             : array<f32>;
    {'@group(0) @binding(3) var<storage, read_write>         g_rw_biases : array<f32>;' if bias_enable else ''}

    @compute
    @workgroup_size({dispatch_group_size[0]}, {dispatch_group_size[1]}, 1)
    fn Main(
        @builtin(global_invocation_id) DispatchThreadID : vec3<u32>,
    ) {{


        let dst_coord : vec2<i32> = vec2<i32>(i32(DispatchThreadID.x), i32(DispatchThreadID.y));

        if (dst_coord.x >= {dst_width} || dst_coord.y >= {dst_height}) {{
            return;
        }}
        
        var features = array<f32, {num_features}>();

        for (var i : i32 = 0; i < {num_features}; i = i + 1) {{
            features[i] = f32(0.0);
        }}

        for (var _y : i32 = 0; _y < {kernel_size[1]}; _y = _y + 1) {{
            for (var _x : i32 = 0; _x < {kernel_size[0]}; _x = _x + 1) {{ 
                let src_coord : vec2<i32> = vec2<i32>(dst_coord.x * {stride[0]} + _x - {kernel_size[0] // 2}, dst_coord.y * {stride[1]} + _y - {kernel_size[1] // 2});
                if (src_coord.x < 0 || src_coord.x >= {src_width} || src_coord.y < 0 || src_coord.y >= {src_height}) {{
                    continue;
                }}
                for (var i : i32 = 0; i < {num_features}; i = i + 1) {{
                    for (var j : i32 = 0; j < {num_channels}; j = j + 1) {{
                        features[i] = features[i] +
                            g_rw_src[src_coord.y * {src_width} * {num_channels} + src_coord.x * {num_channels} + j]
                            * g_rw_weights[(_y * {kernel_size[0]} + _x) * {num_channels} * {num_features} + i * {num_channels} + j];
                    }}
                }}
            }}
        }}

        // Add bias
        
        """ + [f"""for (var i : i32 = 0; i < {num_features}; i = i + 1) {{
            features[i] = features[i] + g_rw_biases[i];
        }}""" if bias_enable else "" ][0] + f"""

        
        for (var i : i32 = 0; i < {num_features}; i = i + 1) {{
            g_rw_dst[dst_coord.y * {dst_width} * {num_features} + dst_coord.x * {num_features} + i] = features[i];

        }}
    }}
    """

width        = 512
height       = 512
src_channels = 3
src_shape    = [1, src_channels, width, height]

conv3x3_shader = make_conv2d_nhwc_fp32(
                num_features=64,
                num_channels=src_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                dst_width=width,
                dst_height=height,
                src_width=width,
                src_height=height,
                bias_enable=True,
                groups=1)

assert torch.cuda.is_available(), f"torch.cuda.is_available()={torch.cuda.is_available()}"
device      = torch.device("cuda")
torch_dtype = torch.float32
torch_to = {"device": device, "dtype": torch_dtype}
src_tensor  = torch.rand(*src_shape).to(**torch_to)
conv_weights = torch.rand(64, 3, 3, 3).to(**torch_to)
conv_biases = torch.rand(64).to(**torch_to)
dst_tensor  = torch.zeros(1, 64, height, width).to(**torch_to)

dst_tensor_ref = torch.nn.functional.conv2d(src_tensor,
                            weight=conv_weights,
                            bias=conv_biases,
                            stride=1,
                            padding=1)
# Get reference to numpy arrays and convert NCWH to NHWC
src_host            = src_tensor.cpu().numpy().transpose(0, 2, 3, 1)[:, :, :, :]
conv_weights_host   = conv_weights.cpu().numpy().transpose(2, 3, 0, 1)[:, :, :, :] # FCHW -> HWFC
conv_biases_host    = conv_biases.cpu().numpy()[:]
dst_host            = dst_tensor_ref.cpu().numpy().transpose(0, 2, 3, 1)[:,:,:, :]

# Initialize wgpu
device = wgpu.utils.get_default_device()
# adapters = wgpu.gpu.enumerate_adapters()
# for a in adapters: print(a.summary)
cshader     = device.create_shader_module(code=conv3x3_shader)
src_gpu     = device.create_buffer_with_data(data=src_host.flatten()[:].tobytes(),
                               usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
dst_gpu     = device.create_buffer(size=dst_host.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
weights_gpu = device.create_buffer_with_data(data=conv_weights_host.flatten()[:].tobytes(), usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
biases_gpu  = device.create_buffer_with_data(data=conv_biases_host.flatten()[:].tobytes(), usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)

binding_layouts = []
bindings = []
buffers = [dst_gpu, src_gpu, weights_gpu, biases_gpu]
for i in range(4):
    binding_layouts.append({
        "binding": i,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    })
    bindings.append({
        "binding": i,
        "resource": {"buffer": buffers[i],
                     "offset": 0,
                     "size": buffers[i].size # WHOLE SIZE
                     },
    })

bind_group_layout   = device.create_bind_group_layout(entries=binding_layouts)
pipeline_layout     = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
bind_group          = device.create_bind_group(layout=bind_group_layout, entries=bindings)
compute_pipeline    = device.create_compute_pipeline(
    layout  =   pipeline_layout,
    compute =   {
        "module": cshader,
        "entry_point": "Main"},
)
command_encoder     = device.create_command_encoder()
compute_pass        = command_encoder.begin_compute_pass()
compute_pass.set_pipeline(compute_pipeline)
compute_pass.set_bind_group(0, bind_group)
dispatch_resolution = [(width + 7) // 8, (height + 7) // 8, 1]
compute_pass.dispatch_workgroups(dispatch_resolution[0], dispatch_resolution[1], dispatch_resolution[2])
compute_pass.end()
device.queue.submit([command_encoder.finish()])
out         = device.queue.read_buffer(dst_gpu).cast("f")
result      = np.array(out.tolist()).reshape(1, height, width, 64)
assert np.allclose(result, dst_host, atol=1e-5), f"result={result}, dst_host={dst_host}"
print("All tests passed.")
# print(result)