@group(0) @binding(0) var<storage, read_write> g_color : array<vec4f>;
@group(0) @binding(1) var g_sampler : sampler;
@group(0) @binding(2) var g_texture : texture_2d<f32>;
// Weights and biases storage buffer
@group(0) @binding(3) var<storage, read_write> g_rw_params : array<f32>;
// Means and variance storage buffer for adam optimizer exponential moving
// averages
// @group(0) @binding(4) var<storage, read_write> g_rw_adam_params: array<f32>;
@group(0) @binding(4) var<storage, read_write> g_rw_gradients : array<i32>;

const NUM_LAYERS : u32 = u32(6);
const NUM_NODES_PER_LAYER =
    array<u32, NUM_LAYERS>(u32(16), u32(16), u32(16), u32(16), u32(16), u32(3));

const MAX_NUM_NODES_PER_LAYER : u32     = u32(16);
const NUM_ACTIVATIONS_PER_NETWORK : u32 = NUM_LAYERS * MAX_NUM_NODES_PER_LAYER;

fn get_total_num_nodes()->u32 {
    var total_num_nodes : u32 = 0u;
    for (var i : u32 = 0u; i < NUM_LAYERS; i = i + 1u) {
        total_num_nodes = total_num_nodes + NUM_NODES_PER_LAYER[i];
    }
    return total_num_nodes;
}
/**
Layer params memory layout

--------------------
* where N is the number of nodes in the current layer and M
    is the number of nodes in the previous layer

weights[N][M] : f32
biases[N] : f32
adam_weights_mean[N][M] : f32
adam_weights_variance[N][M] : f32
adam_biases_mean[N] : f32
adam_biases_variance[N] : f32

*/

struct LayerConstants {
    num_nodes : u32,
    num_prev_nodes : u32,
    num_weights : u32,
    num_biases : u32,
    num_adam_params : u32,
    num_activations : u32,
    // offsets relative to the start of the layer
    weights_offset : u32,
    biases_offset : u32,
    adam_params_offset : u32,
};

fn get_layer_constants(layer_idx : u32)->LayerConstants {
    let num_prev_nodes : u32     = NUM_NODES_PER_LAYER[layer_idx - 1];
    let num_nodes : u32          = NUM_NODES_PER_LAYER[layer_idx];
    let num_weights : u32        = num_prev_nodes * num_nodes;
    let num_biases : u32         = num_nodes;
    let num_adam_params : u32    = 2 * (num_weights + num_biases);
    let num_activations : u32    = num_nodes;
    let weights_offset : u32     = 0u;
    let biases_offset : u32      = num_weights;
    let adam_params_offset : u32 = num_weights + num_biases;
    return LayerConstants(num_nodes, num_prev_nodes, num_weights, num_biases, num_adam_params,
                          num_activations, weights_offset, biases_offset,
                          adam_params_offset);
}
fn get_layer_activations_offset(layer_idx : u32)->u32 {
    var offset : u32 = 0u;
    for (var i : u32 = 0u; i < layer_idx; i = i + 1u) {
        let layer_constants = get_layer_constants(i);
        offset                = offset + layer_constants.num_nodes;
    }
    return offset;
}
fn get_layer_params_offset(layer_idx : u32)->u32 {
    // Get offset for a layer in the storage buffer
    // Layer 0 has no weights or biases
    // Layer 1 has weights and biases for input -> hidden layer etc.
    // layer_idx stores the mapping from previous layer to this layer
    var offset : u32 = 0u;
    for (var i : u32 = 1u; i < layer_idx; i = i + 1u) {
        let layer_constants = get_layer_constants(i);
        offset                = offset + layer_constants.num_weights +
                 layer_constants.num_biases + layer_constants.num_adam_params +
                 layer_constants.num_activations;
    }
    return offset;
}

struct CSInput {
    @builtin(global_invocation_id) global_id : vec3<u32>,
};

fn apply_gamma(color : vec3f, gamma : f32)->vec3f {
    return vec3f(pow(color.r, gamma), pow(color.g, gamma), pow(color.b, gamma));
}

const width : u32  = u32(512);
const height : u32 = u32(512u);

// low bias 32 bit random number generator
// https://github.com/skeeto/hash-prospector
fn lowbias32(_x : u32)->u32 {
    var x = _x;
    x = x ^ (x >> 16);
    x = x * 0x7feb352d;
    x = x ^ (x >> 15);
    x = x * 0x846ca68b;
    x = x ^ (x >> 16);
    return x;
}

// random number generator between 0 and 1 using 65535(0xffff) as the max value
fn random_uniform_unit_float(rnd_state : ptr<function, u32>)->f32 {
    let r : u32 = lowbias32(*rnd_state);
    *rnd_state  = r;
    return f32(r & u32(0xffff)) / f32(0xffff);
}

// clang-format off
@compute
@workgroup_size(8, 8, 1)
fn Initialize(input : CSInput) {
    // clang-format on

    let pos : vec2<u32> = input.global_id.xy;
    let idx : u32 = pos.x + pos.y * width;
    // Figure out layer index
    let layer_idx : u32 = idx / MAX_NUM_NODES_PER_LAYER;
    let node_idx : u32  = idx % MAX_NUM_NODES_PER_LAYER;

    if (layer_idx >= NUM_LAYERS) {
        return;
    }

    let layer_constants = get_layer_constants(layer_idx);
    
    if (node_idx >= layer_constants.num_nodes) {
        return;
    }
    
    let layer_params_offset = get_layer_params_offset(layer_idx);
    let normalization_const: f32 = 1.0 / sqrt(f32(layer_constants.num_prev_nodes));

    var rnd_state : u32 = idx;
    rnd_state = lowbias32(rnd_state);
    for (var i : u32 = 0u; i < layer_constants.num_prev_nodes; i = i + 1u) {
        let weight_offset : u32 =
                                layer_params_offset
                                + layer_constants.weights_offset
                                + node_idx * layer_constants.num_nodes
                                + i;
        g_rw_params[weight_offset] = (random_uniform_unit_float(&rnd_state) * 2.0 - 1.0)
                                    * normalization_const;
    }
    let biases_offset : u32 = layer_constants.biases_offset + node_idx;
    // Initialize biases to 0
    g_rw_params[biases_offset] = 0.0;

    // Initialize adam optimizer exponential moving averages
    // for weights
    for (var i : u32 = 0u; i < layer_constants.num_prev_nodes; i = i + 1u) {
        let adam_weights_mean_offset : u32 =
                                            layer_params_offset
                                            + layer_constants.adam_params_offset
                                            + node_idx * layer_constants.num_nodes
                                            + i;
        g_rw_params[adam_weights_mean_offset] = 0.0;
        let adam_weights_variance_offset : u32 = layer_params_offset
                                                + layer_constants.adam_params_offset
                                                + 2 * node_idx * layer_constants.num_nodes
                                                + i;
        g_rw_params[adam_weights_variance_offset * 2] = 0.0;
        g_rw_params[adam_weights_variance_offset * 1] = 0.0;
    }
    // for biases
    let adam_biases_mean_offset : u32 = layer_params_offset
                                        + layer_constants.adam_params_offset
                                        + 2 * layer_constants.num_weights
                                        + node_idx;
    g_rw_params[adam_biases_mean_offset] = 0.0;

}

fn leaky_relu(x : f32)->f32 {
    let alpha : f32 = 0.01;
    return max(alpha * x, x);
}

fn leaky_relu_derivative(x : f32)->f32 {
    let alpha : f32 = 0.01;
    if (x > 0.0) {
        return 1.0;
    } else {
        return alpha;
    }
}

fn Inference(layer_idx : u32,
             node_idx  : u32,
            activations : ptr<function, array<f32, NUM_ACTIVATIONS_PER_NETWORK>>) {

    let layer_constants     = get_layer_constants(layer_idx);
    let layer_params_offset = get_layer_params_offset(layer_idx);
    let weights_offset      = layer_params_offset + layer_constants.weights_offset;
    let biases_offset       = layer_params_offset + layer_constants.biases_offset;
    let activations_offset  = get_layer_activations_offset(layer_idx - 1);
    let activation_idx      = get_layer_activations_offset(layer_idx) + node_idx;
    var acc : f32            = 0.0;
    for (var i : u32 = 0u; i < layer_constants.num_prev_nodes; i = i + 1u) {
        let weight_idx : u32 = weights_offset
                                + node_idx * layer_constants.num_nodes
                                + i;
        let weight : f32     = g_rw_params[weight_idx];
        acc                  = acc + activations[activations_offset + i] * weight;
    }
    let bias_idx : u32 = biases_offset + node_idx;
    let bias : f32     = g_rw_params[bias_idx];
    activations[activation_idx + node_idx] = leaky_relu_derivative(acc + bias);
    // activations[activation_idx] = acc;
}

// clang-format off
@compute
@workgroup_size(8, 8, 1)
fn Main(input : CSInput) {
    // clang-format on
    let pos : vec2<u32> = input.global_id.xy;
    let uv : vec2<f32> =
                 vec2<f32>(f32(pos.x) / f32(width), f32(pos.y) / f32(height));
    // let color   : vec4f             = textureSample(g_texture, g_sampler,
    // uv);
    let color : vec4f = textureSampleLevel(g_texture, g_sampler, uv, 0);
    // g_color[pos.x + pos.y * width]  = vec4f(uv, 0.0, 1.0);
    // array of activations
    var activations = array<f32, NUM_ACTIVATIONS_PER_NETWORK>();
    for (var i : u32 = 0u; i < NUM_ACTIVATIONS_PER_NETWORK; i = i + 1u) {
        activations[i] = 0.0;
    }
    // Initialize input layer activations
    let input_layer_idx : u32 = u32(0);
    for (var i : u32 = 0u; i < NUM_NODES_PER_LAYER[input_layer_idx]; i = i + 1u) {
        let activation_idx : u32 = get_layer_activations_offset(input_layer_idx) + i;
        activations[activation_idx] = color[i];
    }
    // Frequency encoding of uv
    let num_frequncies : u32 = 3u; // 3 * 2 * 2 + 3 = 12 + 3 = 15
    for (var channel_idx : u32 = 0u; channel_idx < 2u; channel_idx = channel_idx + 1u) {
    for (var i : u32 = 0u; i < num_frequncies; i = i + 1u) {
        let activation_idx : u32 = 3
            + get_layer_activations_offset(input_layer_idx)
            + i * 2
            + channel_idx * num_frequncies * 2;
        let power_bias : u32 = 4;
        let sin_val : f32        = sin(uv.x * pow(2.0, f32(power_bias + i)) * 3.14159);
        let cos_val : f32        = cos(uv.y * pow(2.0, f32(power_bias + i)) * 3.14159);
        activations[activation_idx + 0] = sin_val;
        activations[activation_idx + 1] = cos_val;
    }
}

    const start_layer_idx : u32 = u32(1); // we start from the second layer
    for (var i : u32 = start_layer_idx; i < NUM_LAYERS; i = i + 1u) {
        for (var j : u32 = 0u; j < NUM_NODES_PER_LAYER[i]; j = j + 1u) {
            Inference(i, j, &activations);
        }
    }
    let final_activation_idx : u32 = get_layer_activations_offset(NUM_LAYERS - 1);
    let final_activation_r : f32     = activations[final_activation_idx + 0];
    let final_activation_g : f32     = activations[final_activation_idx + 1];
    let final_activation_b : f32     = activations[final_activation_idx + 2];

    // g_color[pos.x + pos.y * width] = vec4f(color.xyz, 1.0);
    g_color[pos.x + pos.y * width] = vec4f(final_activation_r, final_activation_g,
                                           final_activation_b, 1.0);
                                        // let t = 4;
    // g_color[pos.x + pos.y * width] = vec4f(
        // activations[t],
        // activations[t],
        // activations[t],
    // 1.0);
}

// clang-format off
@compute
@workgroup_size(8, 8, 1)
fn Backward(input : CSInput) {
    // clang-format on
}