from dataclasses import dataclass


@dataclass
class Wav2Vec2Config:
    _name_or_path = ''
    activation_dropout = 0.0
    adapter_kernel_size = 3
    adapter_stride = 2
    add_adapter = False
    apply_spec_augment = True
    architectures = ['Wav2Vec2ForCTC']
    attention_dropout = 0.0
    bos_token_id = 1
    classifier_proj_size = 256
    codevector_dim = 256
    contrastive_logits_temperature = 0.1
    conv_bias = False
    conv_dim = [
        512,
        512,
        512,
        512,
        512,
        512,
        512
    ]
    conv_kernel = [
        10,
        3,
        3,
        3,
        3,
        2,
        2
    ]
    conv_stride = [
        5,
        2,
        2,
        2,
        2,
        2,
        2
    ]
    ctc_loss_reduction = 'mean'
    ctc_zero_infinity = False
    diversity_loss_weight = 0.1
    do_stable_layer_norm = False
    eos_token_id = 2
    feat_extract_activation = 'gelu'
    feat_extract_norm = 'group'
    feat_proj_dropout = 0.0
    feat_quantizer_dropout = 0.0
    final_dropout = 0.0
    freeze_feat_extract_train = True
    hidden_act = 'gelu'
    hidden_dropout = 0.0
    hidden_size = 768
    initializer_range = 0.02
    intermediate_size = 3072
    layer_norm_eps = 1e-05
    layerdrop = 0.0
    mask_channel_length = 10
    mask_channel_min_space = 1
    mask_channel_other = 0.0
    mask_channel_prob = 0.0
    mask_channel_selection = 'static'
    mask_feature_length = 10
    mask_feature_min_masks = 0
    mask_feature_prob = 0.0
    mask_time_length = 10
    mask_time_min_masks = 2
    mask_time_min_space = 1
    mask_time_other = 0.0
    mask_time_prob = 0.05
    mask_time_selection = 'static'
    model_type = 'wav2vec2'
    no_mask_channel_overlap = False
    no_mask_time_overlap = False
    num_adapter_layers = 3
    num_attention_heads = 12
    num_codevector_groups = 2
    num_codevectors_per_group = 320
    num_conv_pos_embedding_groups = 16
    num_conv_pos_embeddings = 128
    num_feat_extract_layers = 7
    num_hidden_layers = 12
    num_negatives = 100
    output_hidden_size = 768
    pad_token_id = 42
    proj_codevector_dim = 256
    tdnn_dilation = [
        1,
        2,
        3,
        1,
        1
    ]
    tdnn_dim = [
        512,
        512,
        512,
        512,
        1500
    ]
    tdnn_kernel = [
        5,
        3,
        3,
        1,
        1
    ]
    torch_dtype = 'float32'
    transformers_version = '4.26.0.dev0'
    use_weighted_layer_sum = False
    vocab_size = 45
    xvector_output_dim = 512


config = Wav2Vec2Config()