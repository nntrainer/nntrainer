"""
Extract model architecture from HuggingFace config.json.

Maps HF architecture parameters to NNTrainer concepts:
- hidden_size → embedding dimension
- num_hidden_layers → number of transformer blocks
- num_attention_heads → attention heads
- intermediate_size → FFN hidden dimension
- hidden_act → activation function (gelu, silu, etc.)
- norm_type → normalization (layernorm, rmsnorm, etc.)
"""
import json
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class HFArchitecture:
    """Extracted HuggingFace architecture parameters"""
    model_name: str
    hf_id: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    hidden_act: str
    norm_type: str  # 'layernorm' or 'rmsnorm'
    layer_norm_eps: float
    use_cache: bool = True
    rope_scaling: Optional[Dict] = None
    num_heads_kv: Optional[int] = None  # For GQA/MQA
    sliding_window: Optional[int] = None
    attention_variant: str = "mha"  # 'mha', 'gqa', 'mqa'
    mlp_variant: str = "standard"  # 'standard', 'swiglu', 'geglu'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}


class HFArchitectureExtractor:
    """Extract architecture from HuggingFace model config"""

    # Map HF model type to our naming convention
    HF_TYPE_MAP = {
        'llama': 'llama',
        'mistral': 'mistral',
        'qwen': 'qwen',
        'qwen2': 'qwen2',
        'qwen3': 'qwen3',
        'gemma': 'gemma',
        'gemma2': 'gemma2',
        'phi': 'phi',
        'phi3': 'phi3',
    }

    def __init__(self, hf_model_id: str):
        """
        Args:
            hf_model_id: HuggingFace model ID (e.g., 'mistralai/Mistral-7B-v0.1')
        """
        self.hf_model_id = hf_model_id
        self.config = self._download_config()

    def _download_config(self) -> Dict:
        """Download config.json from HuggingFace hub"""
        try:
            from huggingface_hub import hf_hub_download
            import json

            config_file = hf_hub_download(
                repo_id=self.hf_model_id,
                filename="config.json",
                cache_dir="/tmp/hf_cache"
            )

            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to download config from {self.hf_model_id}: {e}")
            raise

    def extract(self) -> HFArchitecture:
        """Extract and normalize architecture from config"""

        model_type = self.config.get('model_type', '').lower()
        model_name = self._get_model_name(model_type)

        # Extract basic dimensions
        vocab_size = self.config.get('vocab_size', 32000)
        hidden_size = self.config.get('hidden_size', 768)
        num_hidden_layers = self.config.get('num_hidden_layers', 12)
        num_attention_heads = self.config.get('num_attention_heads', 12)
        intermediate_size = self.config.get('intermediate_size', hidden_size * 4)
        max_position_embeddings = self.config.get('max_position_embeddings', 2048)
        hidden_act = self.config.get('hidden_act', 'gelu')
        layer_norm_eps = self.config.get('layer_norm_eps', 1e-5)
        use_cache = self.config.get('use_cache', True)

        # Detect norm type
        norm_type = self._detect_norm_type()

        # Detect attention variant (GQA, MQA, standard MHA)
        attention_variant, num_heads_kv = self._detect_attention_variant()

        # Detect MLP variant
        mlp_variant = self._detect_mlp_variant()

        # Extract optional features
        rope_scaling = self.config.get('rope_scaling')
        sliding_window = self.config.get('sliding_window')

        return HFArchitecture(
            model_name=model_name,
            hf_id=self.hf_model_id,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            hidden_act=hidden_act,
            norm_type=norm_type,
            layer_norm_eps=layer_norm_eps,
            use_cache=use_cache,
            rope_scaling=rope_scaling,
            num_heads_kv=num_heads_kv,
            sliding_window=sliding_window,
            attention_variant=attention_variant,
            mlp_variant=mlp_variant,
        )

    def _get_model_name(self, model_type: str) -> str:
        """Normalize model type to our naming convention"""
        for hf_type, our_type in self.HF_TYPE_MAP.items():
            if hf_type in model_type:
                return our_type
        return model_type

    def _detect_norm_type(self) -> str:
        """Detect if model uses RMSNorm or LayerNorm"""
        # Check for explicit norm_type in config
        if 'norm_type' in self.config:
            norm = self.config['norm_type'].lower()
            if 'rms' in norm:
                return 'rmsnorm'

        # Check for hidden_act, some models use specific activations with RMSNorm
        # Qwen, Llama, Mistral typically use RMSNorm
        if any(x in self.config.get('model_type', '').lower()
               for x in ['qwen', 'llama', 'mistral']):
            return 'rmsnorm'

        return 'layernorm'

    def _detect_attention_variant(self) -> tuple:
        """Detect if model uses GQA (Grouped Query Attention) or MQA"""
        num_key_value_heads = self.config.get('num_key_value_heads')

        if num_key_value_heads is None:
            return 'mha', None

        num_attention_heads = self.config.get('num_attention_heads', 32)

        if num_key_value_heads == 1:
            return 'mqa', 1
        elif num_key_value_heads < num_attention_heads:
            return 'gqa', num_key_value_heads

        return 'mha', None

    def _detect_mlp_variant(self) -> str:
        """Detect if model uses SwiGLU, GEGLU, or standard FFN"""
        # Check intermediate_size vs hidden_size ratio
        hidden = self.config.get('hidden_size', 768)
        intermediate = self.config.get('intermediate_size', hidden * 4)

        # SwiGLU has 2/3 * (4 * hidden) = ~2.67 * hidden
        # Standard FFN has 4 * hidden
        ratio = intermediate / hidden

        if 2.6 < ratio < 2.8:
            return 'swiglu'
        elif 'swiglu' in str(self.config).lower():
            return 'swiglu'

        return 'standard'


def extract_hf_architecture(hf_model_id: str) -> HFArchitecture:
    """Convenience function to extract architecture"""
    extractor = HFArchitectureExtractor(hf_model_id)
    return extractor.extract()
