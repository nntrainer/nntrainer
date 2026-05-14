causallm_src += [
    meson.current_source_dir() / 'causal_lm.cpp',
    meson.current_source_dir() / 'transformer.cpp',
    meson.current_source_dir() / 'sentence_transformer.cpp',
]

causallm_inc += include_directories('.')

subdir('gpt_oss')
subdir('qwen2')
subdir('qwen3')
subdir('qwen3_moe')
subdir('qwen3_slim_moe')
subdir('gemma3')
if get_option('platform') != 'windows'
    subdir('bert')
    subdir('gpt_oss_cached_slim')
    subdir('qwen3_cached_slim_moe')
endif
