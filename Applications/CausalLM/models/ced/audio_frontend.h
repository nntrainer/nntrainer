// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   audio_frontend.h
 * @date   24 August 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Waveform to log-mel front-end matching torchaudio's MelSpectrogram
 * followed by AmplitudeToDB.
 */

#ifndef __CED_AUDIO_FRONTEND_H__
#define __CED_AUDIO_FRONTEND_H__

#include <cstdint>
#include <string>
#include <vector>

namespace causallm {
namespace audio {

/**
 * @brief Front-end geometry, mirroring torchaudio's MelSpectrogram arguments.
 */
struct FrontEndConfig {
  unsigned int sample_rate = 16000;
  unsigned int n_fft = 512;
  unsigned int win_size = 512;
  unsigned int hop_size = 160;
  bool center = true;
  float f_min = 0.0f;
  float f_max = 8000.0f;
  unsigned int n_mels = 64;
  float power = 2.0f;

  /**
   * How the mel energies are compressed. The two forms are not
   * interchangeable: they differ by the 10/ln(10) factor and in how the floor
   * is applied, so a model has to be fed the one it was trained with.
   *
   * DB:      10 log10(max(x, amin)), then a floor top_db below the window's
   *          maximum. This is torchaudio's AmplitudeToDB.
   * NATURAL: ln(max(x, log_eps)), no relative floor.
   */
  enum class LogMode { DB, NATURAL };
  LogMode log_mode = LogMode::DB;
  float top_db = 80.0f;  /**< DB mode: floor below the window maximum */
  float amin = 1e-10f;   /**< DB mode: absolute floor before the log */
  float log_eps = 1e-5f; /**< NATURAL mode: absolute floor before the log */
};

/**
 * @brief A decoded, mono, [-1, 1) normalized waveform.
 */
struct Waveform {
  std::vector<float> samples;
  unsigned int sample_rate = 0;
  unsigned int channels = 0;
};

/**
 * @brief Read a 16-bit PCM RIFF/WAVE file as mono float.
 *
 * The first channel is taken for multi-channel input and samples are scaled by
 * 1/32768, which is what the reference pipeline does (it reads int16 and
 * divides, rather than letting the decoder produce floats).
 *
 * @param path file to read
 * @param divisor value the int16 samples are divided by. 32768 maps int16 to
 * [-1, 1); some pipelines use 32767 (Java's Short.MAX_VALUE) instead, and the
 * 3e-5 relative difference is large enough to move this model's scores.
 * @return decoded waveform
 * @throw std::runtime_error on an unreadable or unsupported file
 */
Waveform readWav16(const std::string &path, float divisor = 32768.0f);

/**
 * @brief Number of frames the front-end produces for a given sample count.
 */
unsigned int frameCount(const FrontEndConfig &cfg, size_t n_samples);

/**
 * @brief Compute the log-mel spectrogram of one window.
 *
 * Equivalent to torchaudio MelSpectrogram(power=2, center=True,
 * pad_mode="reflect", norm=None, mel_scale="htk") followed by
 * AmplitudeToDB(stype="power", top_db=cfg.top_db). The Hann window and the mel
 * filterbank are derived from cfg, so no side files are needed.
 *
 * @param cfg front-end geometry
 * @param samples one window of audio
 * @param n_samples window length
 * @param[out] out row-major [n_mels, frames] dB values
 * @return frame count written
 */
unsigned int logMelSpectrogram(const FrontEndConfig &cfg, const float *samples,
                               size_t n_samples, std::vector<float> &out);

} // namespace audio
} // namespace causallm

#endif /* __CED_AUDIO_FRONTEND_H__ */
