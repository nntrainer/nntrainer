#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise LFM2 native records with a synthetic mixed conv/attention MoE."""
import json
from pathlib import Path
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools'))
import convert_lfm2_htp as converter


def main():
    rng = np.random.default_rng(431)
    cfg = dict(model_type='lfm2_moe', architectures=['Lfm2MoeForCausalLM'],
        hidden_size=128, intermediate_size=256, moe_intermediate_size=128,
        num_hidden_layers=3, num_dense_layers=1, num_experts=32, num_experts_per_tok=4,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=128,
        conv_L_cache=3, conv_bias=False, tie_word_embeddings=False,
        layer_types=['conv', 'full_attention', 'conv'], norm_eps=1e-5)
    tensors = {}

    def matrix(name, n, k, ternary=False):
        q = rng.integers(-1 if ternary else -8, 2 if ternary else 8, (n,k))
        # Differing powers of two, and affine offset +1 for FCs.
        s = np.exp2(rng.integers(-10, -5, n)).astype(np.float32)
        tensors[name] = converter.bf16_round((q + (0 if ternary else 1)) * s[:, None])

    matrix('model.embed_tokens.weight',128,128)
    tensors['model.embedding_norm.weight'] = np.ones(128,np.float32)
    matrix('lm_head.weight',128,128)
    for l, kind in enumerate(cfg['layer_types']):
        p = f'model.layers.{l}.'
        tensors[p+'operator_norm.weight'] = np.ones(128,np.float32)
        tensors[p+'ffn_norm.weight'] = np.ones(128,np.float32)
        if kind == 'conv':
            matrix(p+'conv.in_proj.weight',384,128)
            matrix(p+'conv.out_proj.weight',128,128)
            # Distinct taps catch reversed time order and channel transposition.
            tensors[p+'conv.conv.weight'] = np.tile(np.array([.25,.5,1.],np.float32),(128,1)).reshape(128,1,3)
        else:
            for kind,n in [('q',128),('k',64),('v',64),('out',128)]:
                matrix(p+f'self_attn.{kind}_proj.weight',n,128)
            tensors[p+'self_attn.q_layernorm.weight'] = np.ones(64,np.float32)
            tensors[p+'self_attn.k_layernorm.weight'] = np.ones(64,np.float32)
        if l==0:
            for key,n,k in [('w1',256,128),('w3',256,128),('w2',128,256)]:
                matrix(p+f'feed_forward.{key}.weight',n,k)
        else:
            tensors[p+'feed_forward.gate.weight'] = converter.bf16_round(rng.standard_normal((32,128)).astype(np.float32))
            tensors[p+'feed_forward.expert_bias'] = rng.standard_normal(32).astype(np.float32)
            for e in range(32):
                for key in ('w1','w2','w3'):
                    matrix(p+f'feed_forward.experts.{e}.{key}.weight',128,128,True)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); source=root/'source';source.mkdir()
        (source/'config.json').write_text(json.dumps(cfg))
        (source/'generation_config.json').write_text('{}')
        (source/'tokenizer.json').write_text('{}')
        header={};payload=bytearray()
        for name,w in tensors.items():
            is_f32=name.endswith('expert_bias')
            data=w.astype('<f4').tobytes() if is_f32 else (w.view(np.uint32)>>16).astype('<u2').tobytes()
            start=len(payload);payload.extend(data)
            header[name]=dict(dtype='F32' if is_f32 else 'BF16',shape=list(w.shape),data_offsets=[start,len(payload)])
        encoded=json.dumps(header).encode();encoded+=b' '*(-len(encoded)%8)
        (source/'model.safetensors').write_bytes(struct.pack('<Q',len(encoded))+encoded+payload)
        output=root/'output'
        report=converter.convert(source,output,source,cache_experts=4)
        assert report['source_tensors_consumed']==len(tensors)
        assert report['expert_tensors_verified']==128
        assert report['max_expert_weight_error']==report['max_fc_bf16_roundtrip_error']==0
        raw=(output/report['model_file']).read_bytes()
        assert len(raw)==report['bytes']
        for item in report['tensors']:
            if item['name'].endswith('_conv:conv'):
                got=np.frombuffer(raw,'<f4',3*128,item['offset']).reshape(3,128)
                assert np.array_equal(got[:,0],[1.,.5,.25])
            if item['name'].endswith('_sparse_moe:gate'):
                got=np.frombuffer(raw,'<f4',128*32,item['offset']).reshape(128,32)
                assert np.array_equal(got,tensors[item['source'][0]].T)
        try: converter.convert(source,output,source)
        except FileExistsError: pass
        else: raise AssertionError('Existing pack overwritten')
        cp=converter.Checkpoint(source)
        try:
            cp.header['unused'] = cp.header['lm_head.weight']
            try:converter.weight_plan(cfg,cp)
            except ValueError:pass
            else:raise AssertionError('Unconsumed source accepted')
        finally:cp.close()
    print('LFM2 conversion: native TL2, affine WH, conv taps, router layout and rejection checks PASS')

if __name__=='__main__':main()
