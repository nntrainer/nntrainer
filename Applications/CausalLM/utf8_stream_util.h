// SPDX-License-Identifier: Apache-2.0
/**
 * @file   utf8_stream_util.h
 * @brief  Helpers for holding incomplete multi-byte UTF-8 characters during
 *         streaming detokenization.
 *
 * A multi-byte UTF-8 character (e.g. a Korean syllable like '똻', 3 bytes) can
 * be generated as several consecutive byte-fallback tokens. Decoding only a
 * prefix of those tokens yields either incomplete raw bytes (native BPE) or a
 * trailing U+FFFD replacement character (HuggingFace/Rust tokenizer, which
 * decodes partial bytes via String::from_utf8_lossy). Streaming such a partial
 * result renders as broken glyphs.
 *
 * Usage: accumulate tokens into a pending buffer, Decode the whole buffer each
 * step, then call shouldHold() to decide whether to emit or keep buffering.
 * Used by CausalLM::registerOutputs() (CPU path) and the QNN generation loops
 * (Transformer-derived models that run their own decoding).
 */
#ifndef __CAUSALLM_UTF8_STREAM_UTIL_H__
#define __CAUSALLM_UTF8_STREAM_UTIL_H__

#include <cstddef>
#include <string>

namespace causallm {
namespace utf8stream {

// Upper bound on how many tokens we hold while a multi-byte character is being
// assembled. A UTF-8 code point is at most 4 bytes, so a valid character always
// resolves within this window; the cap keeps genuinely invalid model output (a
// stray byte that never completes) from stalling the stream forever.
constexpr size_t kMaxHeldByteTokens = 6;

// True if `s` ends with an incomplete UTF-8 byte sequence — a multi-byte lead
// byte whose continuation bytes have not all arrived yet (native BPE decode).
inline bool endsWithIncompleteUtf8(const std::string &s) {
  const size_t n = s.size();
  if (n == 0)
    return false;

  size_t lead = n - 1;
  size_t trailing_cont = 0;
  while ((static_cast<unsigned char>(s[lead]) & 0xc0) == 0x80) {
    ++trailing_cont;
    if (lead == 0 || trailing_cont >= 4)
      return false;
    --lead;
  }

  const unsigned char head = static_cast<unsigned char>(s[lead]);
  size_t expected;
  if ((head & 0x80) == 0x00)
    expected = 1;
  else if ((head & 0xe0) == 0xc0)
    expected = 2;
  else if ((head & 0xf0) == 0xe0)
    expected = 3;
  else if ((head & 0xf8) == 0xf0)
    expected = 4;
  else
    return false;

  return (n - lead) < expected;
}

// True if `s` ends with the UTF-8 replacement character U+FFFD (EF BF BD).
// The HuggingFace (Rust) tokenizer substitutes U+FFFD for byte-fallback bytes
// that do not yet form a complete code point, so a trailing U+FFFD signals
// "the character is still being assembled".
inline bool endsWithReplacementChar(const std::string &s) {
  return s.size() >= 3 && static_cast<unsigned char>(s[s.size() - 3]) == 0xef &&
         static_cast<unsigned char>(s[s.size() - 2]) == 0xbf &&
         static_cast<unsigned char>(s[s.size() - 1]) == 0xbd;
}

// Decide whether `decoded` (the text of all currently-buffered tokens) should
// be HELD rather than streamed now. `held_count` is how many tokens are
// currently buffered. Holds while the text is empty or ends mid-character,
// bounded by kMaxHeldByteTokens.
inline bool shouldHold(const std::string &decoded, size_t held_count) {
  if (decoded.empty())
    return true;
  return held_count <= kMaxHeldByteTokens &&
         (endsWithIncompleteUtf8(decoded) || endsWithReplacementChar(decoded));
}

} // namespace utf8stream
} // namespace causallm

#endif /* __CAUSALLM_UTF8_STREAM_UTIL_H__ */
