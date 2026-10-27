// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   audio_frontend.cpp
 * @date   24 August 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Waveform to log-mel front-end matching torchaudio's MelSpectrogram
 * followed by AmplitudeToDB.
 */

#include "audio_frontend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace causallm {
namespace audio {

namespace {

/** Read a little-endian unsigned integer of `N` bytes. */
template <typename T> T readLE(std::istream &s) {
  unsigned char buf[sizeof(T)];
  s.read(reinterpret_cast<char *>(buf), sizeof(T));
  T v = 0;
  for (size_t i = 0; i < sizeof(T); ++i)
    v |= static_cast<T>(buf[i]) << (8 * i);
  return v;
}

/**
 * @brief torch.hann_window(n) -- periodic, so the divisor is n and not n-1.
 */
const std::vector<float> &hannWindow(unsigned int n) {
  static std::vector<float> cache;
  static unsigned int cached_n = 0;
  if (cached_n != n) {
    cache.resize(n);
    for (unsigned int i = 0; i < n; ++i)
      cache[i] = static_cast<float>(
        0.5 - 0.5 * std::cos(2.0 * M_PI * static_cast<double>(i) / n));
    cached_n = n;
  }
  return cache;
}

double hzToMel(double f) { return 2595.0 * std::log10(1.0 + f / 700.0); }
double melToHz(double m) { return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0); }

/**
 * @brief torchaudio melscale_fbanks(norm=None, mel_scale="htk").
 *
 * Returns a row-major [n_freqs, n_mels] triangular filterbank. Derived rather
 * than shipped: this reproduces the checkpoint's stored filterbank to 5e-6,
 * which is below the float32 noise of the spectrogram itself.
 */
std::vector<float> melFilterbank(unsigned int n_freqs, unsigned int n_mels,
                                 unsigned int sample_rate, double f_min,
                                 double f_max) {
  std::vector<double> all_freqs(n_freqs);
  const double nyquist = static_cast<double>(sample_rate / 2);
  for (unsigned int i = 0; i < n_freqs; ++i)
    all_freqs[i] = nyquist * i / (n_freqs - 1);

  const double m_min = hzToMel(f_min);
  const double m_max = hzToMel(f_max);
  std::vector<double> f_pts(n_mels + 2);
  for (unsigned int i = 0; i < n_mels + 2; ++i)
    f_pts[i] = melToHz(m_min + (m_max - m_min) * i / (n_mels + 1));

  std::vector<float> fb(static_cast<size_t>(n_freqs) * n_mels, 0.0f);
  for (unsigned int m = 0; m < n_mels; ++m) {
    const double d_lo = f_pts[m + 1] - f_pts[m];
    const double d_hi = f_pts[m + 2] - f_pts[m + 1];
    for (unsigned int k = 0; k < n_freqs; ++k) {
      const double down = (all_freqs[k] - f_pts[m]) / d_lo;
      const double up = (f_pts[m + 2] - all_freqs[k]) / d_hi;
      const double v = std::min(down, up);
      fb[static_cast<size_t>(k) * n_mels + m] =
        static_cast<float>(v > 0.0 ? v : 0.0);
    }
  }
  return fb;
}

/**
 * @brief Twiddle factors and the bit-reversal permutation for a fixed FFT size.
 *
 * Both depend only on n, so they are built once instead of calling cos/sin
 * inside every butterfly -- with 101 frames per window that transcendental cost
 * dominated the front end.
 */
struct FftPlan {
  size_t n = 0;
  std::vector<float> wr; /**< cos, indexed by stage offset + k */
  std::vector<float> wi; /**< sin, same indexing */
  std::vector<uint32_t> rev;
};

const FftPlan &fftPlan(size_t n) {
  static FftPlan plan;
  if (plan.n == n)
    return plan;

  plan.n = n;
  plan.rev.resize(n);
  const unsigned int bits = static_cast<unsigned int>(std::log2(n));
  for (size_t i = 0; i < n; ++i) {
    uint32_t r = 0;
    for (unsigned int b = 0; b < bits; ++b)
      if (i & (size_t(1) << b))
        r |= 1u << (bits - 1 - b);
    plan.rev[i] = r;
  }

  // Stage `len` needs len/2 twiddles; total is n-1 across all stages.
  plan.wr.resize(n);
  plan.wi.resize(n);
  size_t off = 0;
  for (size_t len = 2; len <= n; len <<= 1) {
    const double ang = -2.0 * M_PI / static_cast<double>(len);
    const size_t half = len >> 1;
    for (size_t k = 0; k < half; ++k) {
      plan.wr[off + k] = static_cast<float>(std::cos(ang * (double)k));
      plan.wi[off + k] = static_cast<float>(std::sin(ang * (double)k));
    }
    off += half;
  }
  return plan;
}

/**
 * @brief In-place iterative radix-2 FFT. `n` must be a power of two.
 */
void fftInPlace(std::vector<float> &re, std::vector<float> &im) {
  const size_t n = re.size();
  const FftPlan &plan = fftPlan(n);

  for (size_t i = 0; i < n; ++i) {
    const size_t j = plan.rev[i];
    if (i < j) {
      std::swap(re[i], re[j]);
      std::swap(im[i], im[j]);
    }
  }
  size_t off = 0;
  for (size_t len = 2; len <= n; len <<= 1) {
    const size_t half = len >> 1;
    for (size_t i = 0; i < n; i += len) {
      for (size_t k = 0; k < half; ++k) {
        const float wr = plan.wr[off + k];
        const float wi = plan.wi[off + k];
        const float ar = re[i + k], ai = im[i + k];
        const float br = re[i + k + half], bi = im[i + k + half];
        const float vr = br * wr - bi * wi;
        const float vi = br * wi + bi * wr;
        re[i + k] = ar + vr;
        im[i + k] = ai + vi;
        re[i + k + half] = ar - vr;
        im[i + k + half] = ai - vi;
      }
    }
    off += half;
  }
}

} // namespace

Waveform readWav16(const std::string &path, float divisor) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("Failed to open wav: " + path);

  char riff[4], wave[4];
  f.read(riff, 4);
  (void)readLE<uint32_t>(f); // total size, unused
  f.read(wave, 4);
  if (std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(wave, "WAVE", 4) != 0)
    throw std::runtime_error("Not a RIFF/WAVE file: " + path);

  Waveform wav;
  unsigned int bits = 0;
  bool have_fmt = false;

  // Walk the chunk list rather than assuming the canonical 44-byte header:
  // encoders routinely insert LIST/fact chunks before `data`.
  while (f && f.peek() != EOF) {
    char id[4];
    f.read(id, 4);
    if (f.gcount() != 4)
      break;
    const uint32_t size = readLE<uint32_t>(f);
    if (std::memcmp(id, "fmt ", 4) == 0) {
      const uint16_t format = readLE<uint16_t>(f);
      wav.channels = readLE<uint16_t>(f);
      wav.sample_rate = readLE<uint32_t>(f);
      (void)readLE<uint32_t>(f); // byte rate
      (void)readLE<uint16_t>(f); // block align
      bits = readLE<uint16_t>(f);
      if (format != 1)
        throw std::runtime_error(
          "Only uncompressed PCM wav is supported (format=" +
          std::to_string(format) + "): " + path);
      if (bits != 16)
        throw std::runtime_error("Only 16-bit PCM wav is supported (bits=" +
                                 std::to_string(bits) + "): " + path);
      f.seekg(static_cast<std::streamoff>(size) - 16, std::ios::cur);
      have_fmt = true;
    } else if (std::memcmp(id, "data", 4) == 0) {
      if (!have_fmt)
        throw std::runtime_error("wav data chunk precedes fmt: " + path);
      const size_t n_total = size / 2;
      std::vector<int16_t> pcm(n_total);
      f.read(reinterpret_cast<char *>(pcm.data()),
             static_cast<std::streamsize>(n_total * 2));
      const unsigned int ch = wav.channels ? wav.channels : 1;
      wav.samples.resize(n_total / ch);
      // Match the reference: take channel 0, then scale the int16.
      const float inv = 1.0f / divisor;
      for (size_t i = 0; i < wav.samples.size(); ++i)
        wav.samples[i] = static_cast<float>(pcm[i * ch]) * inv;
      return wav;
    } else {
      f.seekg(static_cast<std::streamoff>(size + (size & 1)), std::ios::cur);
    }
  }
  throw std::runtime_error("No data chunk found in wav: " + path);
}

unsigned int frameCount(const FrontEndConfig &cfg, size_t n_samples) {
  const size_t padded =
    cfg.center ? n_samples + cfg.n_fft : n_samples; // n_fft/2 on each side
  if (padded < cfg.n_fft)
    return 0;
  return static_cast<unsigned int>((padded - cfg.n_fft) / cfg.hop_size + 1);
}

unsigned int logMelSpectrogram(const FrontEndConfig &cfg, const float *samples,
                               size_t n_samples, std::vector<float> &out) {
  const unsigned int n_fft = cfg.n_fft;
  const unsigned int n_mels = cfg.n_mels;
  const unsigned int n_freqs = n_fft / 2 + 1;
  const unsigned int frames = frameCount(cfg, n_samples);
  if (frames == 0)
    throw std::runtime_error("audio window shorter than n_fft");

  // center=True with pad_mode="reflect": mirror n_fft/2 samples on each side,
  // excluding the edge sample itself.
  const size_t pad = cfg.center ? n_fft / 2 : 0;
  std::vector<float> x(n_samples + 2 * pad);
  std::copy(samples, samples + n_samples, x.begin() + pad);
  for (size_t i = 0; i < pad; ++i) {
    x[pad - 1 - i] = samples[std::min(i + 1, n_samples - 1)];
    x[pad + n_samples + i] =
      samples[n_samples >= i + 2 ? n_samples - 2 - i : 0];
  }

  const std::vector<float> &win = hannWindow(cfg.win_size);
  static std::vector<float> fb;
  static unsigned int fb_key = 0;
  const unsigned int key = n_freqs * 1000u + n_mels;
  if (fb_key != key) {
    fb = melFilterbank(n_freqs, n_mels, cfg.sample_rate, cfg.f_min, cfg.f_max);
    fb_key = key;
  }

  out.assign(static_cast<size_t>(n_mels) * frames, 0.0f);
  std::vector<float> re(n_fft), im(n_fft), power(n_freqs);

  for (unsigned int t = 0; t < frames; ++t) {
    const float *seg = x.data() + static_cast<size_t>(t) * cfg.hop_size;
    std::fill(im.begin(), im.end(), 0.0f);
    for (unsigned int i = 0; i < n_fft; ++i)
      re[i] = i < cfg.win_size ? seg[i] * win[i] : 0.0f;
    fftInPlace(re, im);
    for (unsigned int k = 0; k < n_freqs; ++k)
      power[k] = re[k] * re[k] + im[k] * im[k];

    for (unsigned int m = 0; m < n_mels; ++m) {
      float acc = 0.0f;
      for (unsigned int k = 0; k < n_freqs; ++k)
        acc += power[k] * fb[static_cast<size_t>(k) * n_mels + m];
      out[static_cast<size_t>(m) * frames + t] = acc;
    }
  }

  if (cfg.log_mode == FrontEndConfig::LogMode::NATURAL) {
    // ln(max(x, eps)). No relative floor: eps alone bounds the log.
    for (float &v : out)
      v = std::log(std::max(v, cfg.log_eps));
  } else {
    // AmplitudeToDB(stype="power"): 10 log10(max(x, amin)), then a floor
    // top_db below the maximum of this window.
    float max_db = -std::numeric_limits<float>::infinity();
    for (float &v : out) {
      v = 10.0f * std::log10(std::max(v, cfg.amin));
      max_db = std::max(max_db, v);
    }
    const float floor_db = max_db - cfg.top_db;
    for (float &v : out)
      v = std::max(v, floor_db);
  }

  return frames;
}

} // namespace audio
} // namespace causallm
