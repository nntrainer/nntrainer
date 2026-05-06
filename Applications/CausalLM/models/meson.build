causallm_src += [
    meson.current_source_dir() / 'causal_lm.cpp',
    meson.current_source_dir() / 'transformer.cpp',
]

causallm_inc += include_directories('.')

if host_machine.system() == 'windows'
    subdir('qwen35')
else
    causallm_src += [
        meson.current_source_dir() / 'sentence_transformer.cpp',
    ]

    subdir('gpt_oss')
    subdir('gpt_oss_cached_slim')
    subdir('qwen2')
    subdir('qwen3')
    subdir('qwen35')
    subdir('qwen3_moe')
    subdir('qwen3_slim_moe')
    subdir('qwen3_cached_slim_moe')
    subdir('gemma3')
endif
