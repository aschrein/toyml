import wgpu
from wgpu.utils.compute import compute_with_buffers

shader_source = """

@group(0) @binding(0) var<storage, read_write>         g_rw_dst: array<u32>;

@compute
@workgroup_size(64, 1, 1)

fn Main(
    @builtin(global_invocation_id) DispatchThreadID : vec3<u32>,
) {
    g_rw_dst[DispatchThreadID.x] = DispatchThreadID.x ^ u32(0xdeadbeef);
}
"""
N       = 1 << 20
data    = memoryview(bytearray(N * 4)).cast("i")
for i in range(N): data[i] = i
device = wgpu.utils.get_default_device()
# adapters = wgpu.gpu.enumerate_adapters()
# for a in adapters: print(a.summary)
cshader = device.create_shader_module(code=shader_source)
gpu_buffer = device.create_buffer(
    size=data.nbytes,
    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
)
binding_layouts = [
    {
        "binding": 0,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    },
]
bindings = [
    {
        "binding": 0,
        "resource": {"buffer": gpu_buffer, "offset": 0, "size": gpu_buffer.size},
    },
]
bind_group_layout   = device.create_bind_group_layout(entries=binding_layouts)
pipeline_layout     = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
bind_group          = device.create_bind_group(layout=bind_group_layout, entries=bindings)
compute_pipeline    = device.create_compute_pipeline(
    layout  =   pipeline_layout,
    compute =   {
        "module": cshader,
        "entry_point": "Main"},
)
command_encoder = device.create_command_encoder()
compute_pass = command_encoder.begin_compute_pass()
compute_pass.set_pipeline(compute_pipeline)
compute_pass.set_bind_group(0, bind_group)
dispatch_resolution = [(N + 63) // 64, 1, 1]
compute_pass.dispatch_workgroups(dispatch_resolution[0], dispatch_resolution[1], dispatch_resolution[2])
compute_pass.end()
device.queue.submit([command_encoder.finish()])
out = device.queue.read_buffer(gpu_buffer).cast("I")
result = out.tolist()
for i in range(N):
    # print(f"{i:2d}: {result[i]:08x}")
    assert result[i] == i ^ 0xdeadbeef
print("All tests passed.")
# print(result)