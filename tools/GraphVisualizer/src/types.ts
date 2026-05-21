/** Mirrors NNTrainerLayerDef from Python */
export interface NNTrainerLayer {
    name: string;
    layer_type: string;
    properties: Record<string, unknown>;
    input_layers: string[];
    fx_node_name: string;
    hf_module_name: string;
    hf_module_type: string;
    has_weight: boolean;
    has_bias: boolean;
    weight_hf_key: string;
    bias_hf_key: string;
    transpose_weight: boolean;
    shared_from: string;
}

/** Torch.fx graph node */
export interface FxNode {
    name: string;
    op: 'placeholder' | 'call_module' | 'call_function' | 'call_method' | 'get_attr' | 'output';
    target: string;
    args: string[];
    output_shape: number[] | null;
    module_type: string | null;
    scope: string;
    meta: Record<string, unknown>;
}

/** Detected pattern structures */
export interface AttentionPattern {
    block_idx: number;
    q_proj: string;
    k_proj: string;
    v_proj: string;
    o_proj: string;
    attention_type: string;
    has_rope: boolean;
    num_heads: number;
    num_kv_heads: number;
    head_dim: number;
}

export interface FFNPattern {
    block_idx: number;
    ffn_type: string;
    gate_proj: string;
    up_proj: string;
    down_proj: string;
    intermediate_size: number;
}

export interface TransformerBlock {
    block_idx: number;
    pre_attn_norm: string;
    attention: AttentionPattern | null;
    attn_residual: string;
    pre_ffn_norm: string;
    ffn: FFNPattern | null;
    ffn_residual: string;
    norm_type: string;
}

export interface ModelStructure {
    model_type: string;
    arch_type: string;
    vocab_size: number;
    hidden_size: number;
    num_layers: number;
    num_heads: number;
    num_kv_heads: number;
    head_dim: number;
    intermediate_size: number;
    norm_eps: number;
    tie_word_embeddings: boolean;
    rope_theta: number;
    embedding: string;
    final_norm: string;
    lm_head: string;
    blocks: TransformerBlock[];
}

/** Weight mapping entry */
export interface WeightMapEntry {
    layer_name: string;
    layer_type: string;
    weight_key: string;
    transpose_weight: boolean;
}

/** Complete conversion result passed to webview */
export interface ConversionResult {
    // From TorchFXConverter
    nntrainerLayers: NNTrainerLayer[];
    fxGraph: FxNode[];
    modelStructure: ModelStructure | null;
    weightMap: WeightMapEntry[];

    // Mapping info for verification
    nodeMapping: NodeMapping[];

    // Conversion diagnostics
    unsupportedOps: string[];
    unknownLayers: string[];
    decomposedModules: string[];

    // Generated code (for cross-check)
    cppSource: string;
    iniConfig: string;

    // Source code (for local .py model conversion)
    torchSourceCode: string;
    torchSourcePath: string;
}

/** Mapping between fx node and nntrainer layer */
export interface NodeMapping {
    fxNodeName: string;
    nntrainerLayerName: string;
    hfModuleName: string;
    mappingType: 'direct' | 'decomposed' | 'skipped' | 'unmapped';
}

/** Per-layer profiling data */
export interface LayerProfile {
    name: string;
    layer_type: string;
    time_ms: number;
    memory_mb: number;
    flops: number;
    pct_of_total: number;
    /** Extended timing from NNTrainer profiler */
    avg_ms?: number;
    min_ms?: number;
    max_ms?: number;
    count?: number;
}

/** Model profiling result */
export interface ProfileData {
    model_name: string;
    total_time_ms: number;
    total_memory_mb: number;
    num_runs: number;
    seq_len: number;
    layers: LayerProfile[];
    bottlenecks: string[];
}
