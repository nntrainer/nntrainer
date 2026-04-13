// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   test_api.cpp
 * @date   21 Jan 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @brief  Simple application to test CausalLM API
 * @bug    No known bugs except for NYI items
 *
 */

#include "../json.hpp"
#include "causal_lm_api.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr const char *COLOR_RESET = "\033[0m";
constexpr const char *COLOR_BOLD = "\033[1m";
constexpr const char *COLOR_CYAN = "\033[36m";
constexpr const char *COLOR_GREEN = "\033[32m";
constexpr const char *COLOR_YELLOW = "\033[33m";
constexpr const char *COLOR_BLUE = "\033[34m";
constexpr const char *COLOR_RED = "\033[31m";
constexpr const char *COLOR_MAGENTA = "\033[35m";
constexpr const char *COLOR_GRAY = "\033[90m";

void printLine(const std::string &s, int length = 80) {
  for (int i = 0; i < length; ++i)
    std::cout << s;
  std::cout << std::endl;
}

void printSection(const std::string &section) {
  std::cout << "\n"
            << COLOR_BOLD << COLOR_BLUE
            << "+-------------------------------------------------------------+"
            << COLOR_RESET << "\n";
  std::cout << COLOR_BOLD << COLOR_BLUE << "|  " << section
            << std::string(58 - section.length(), ' ') << "|" << COLOR_RESET
            << "\n";
  std::cout << COLOR_BOLD << COLOR_BLUE
            << "+-------------------------------------------------------------+"
            << COLOR_RESET << "\n\n";
}

void printSuccess(const std::string &msg) {
  std::cout << COLOR_GREEN << "✓ " << COLOR_BOLD << msg << COLOR_RESET
            << "\n\n";
}

void printError(const std::string &msg) {
  std::cerr << COLOR_RED << "✗ " << COLOR_BOLD << "Error: " << COLOR_RESET
            << msg << "\n";
}

void printWarning(const std::string &msg) {
  std::cout << COLOR_YELLOW << "⚠ " << msg << COLOR_RESET << "\n";
}

void printInfo(const std::string &label, const std::string &value) {
  std::cout << COLOR_CYAN << "  " << label << ":" << COLOR_RESET << " " << value
            << "\n";
}

void printLogo() {
  std::cout << "\n";
  std::cout << COLOR_BOLD << COLOR_MAGENTA;
  std::cout << "  ███╗   ██╗███╗   ██╗████████╗██████╗ \n";
  std::cout << "  ████╗  ██║████╗  ██║╚══██╔══╝██╔══██╗\n";
  std::cout << "  ██╔██╗ ██║██╔██╗ ██║   ██║   ██████╔╝\n";
  std::cout << "  ██║╚██╗██║██║╚██╗██║   ██║   ██╔══██╗\n";
  std::cout << "  ██║ ╚████║██║ ╚████║   ██║   ██║  ██║\n";
  std::cout << "  ╚═╝  ╚═══╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝\n";
  std::cout << COLOR_RESET;
  std::cout << COLOR_BOLD << COLOR_CYAN
            << "  ────────────────────────────────\n";
  std::cout << "      Causal Language Model API\n"
            << "  ────────────────────────────────\n";
  std::cout << COLOR_RESET << "\n";
}

void printUsage(const char *program_name) {
  std::cout << COLOR_YELLOW << "Usage:" << COLOR_RESET << "\n";
  std::cout << "  " << COLOR_BOLD << program_name << COLOR_RESET
            << " <model_name> [prompt] [use_chat_template] [quantization] "
               "[verbose] \n";
  std::cout << "  " << COLOR_BOLD << program_name << COLOR_RESET
            << " <model_name> --chat-file <path.json> [quantization] "
               "[verbose] \n\n";

  std::cout << COLOR_CYAN << "Arguments:" << COLOR_RESET << "\n";
  std::cout << "  model_name        " << COLOR_BOLD << "REQUIRED" << COLOR_RESET
            << "  - Model name (e.g., QWEN3-0.6B)\n";
  std::cout << "  prompt            " << COLOR_GREEN << "OPTIONAL"
            << COLOR_RESET
            << "  - Input prompt (default: 'Hello, how are you?')\n";
  std::cout << "  --chat-file       " << COLOR_GREEN << "OPTIONAL"
            << COLOR_RESET
            << "  - JSON file with chat messages [{role, content}, ...]\n";
  std::cout << "  use_chat_template " << COLOR_GREEN << "OPTIONAL"
            << COLOR_RESET << "  - 0/1 or true/false (default: 1)\n";
  std::cout << "  quantization      " << COLOR_GREEN << "OPTIONAL"
            << COLOR_RESET
            << "  - W4A32/W16A16/W8A16/W32A32/UNKNOWN (default: UNKNOWN)\n";
  std::cout << "  verbose           " << COLOR_GREEN << "OPTIONAL"
            << COLOR_RESET << "  - 0/1 or true/false (default: 0)\n\n";

  std::cout << COLOR_YELLOW << "Examples:" << COLOR_RESET << "\n";
  std::cout << "  " << COLOR_BOLD << program_name << COLOR_RESET
            << " QWEN3-0.6B \"Tell me a joke\" 1 W4A32\n";
  std::cout << "  " << COLOR_BOLD << program_name << COLOR_RESET
            << " QWEN3-0.6B --chat-file chat.json W32A32 1\n\n";

  std::cout << COLOR_YELLOW << "Chat file format (JSON):" << COLOR_RESET
            << "\n";
  std::cout << "  [\n";
  std::cout << "    {\"role\": \"system\",    \"content\": \"You are a helpful "
               "assistant.\"},\n";
  std::cout << "    {\"role\": \"user\",      \"content\": \"Hello!\"},\n";
  std::cout << "    {\"role\": \"assistant\", \"content\": \"Hi there!\"},\n";
  std::cout << "    {\"role\": \"user\",      \"content\": \"How are you?\"}\n";
  std::cout << "  ]\n\n";
}
} // namespace

int main(int argc, char *argv[]) {
  printLogo();

  if (argc < 2) {
    printSection("ERROR: Missing Required Arguments");
    printUsage(argv[0]);
    return 1;
  }

  const char *model_name = argv[1];
  const char *prompt = "Hello, how are you?";
  bool use_chat_template = true;
  std::string chat_file_path = "";
  std::string quant_str = "UNKNOWN";
  ModelQuantizationType quant_type = CAUSAL_LM_QUANTIZATION_UNKNOWN;
  bool verbose = true;

  // Parse --chat-file mode: <model> --chat-file <path> [quant] [verbose]
  if (argc >= 4 && std::string(argv[2]) == "--chat-file") {
    chat_file_path = argv[3];
    use_chat_template = true;
    if (argc >= 5) {
      quant_str = std::string(argv[4]);
      if (quant_str == "W4A32")
        quant_type = CAUSAL_LM_QUANTIZATION_W4A32;
      else if (quant_str == "W16A16")
        quant_type = CAUSAL_LM_QUANTIZATION_W16A16;
      else if (quant_str == "W8A16")
        quant_type = CAUSAL_LM_QUANTIZATION_W8A16;
      else if (quant_str == "W32A32")
        quant_type = CAUSAL_LM_QUANTIZATION_W32A32;
    }
    if (argc >= 6) {
      verbose = (std::string(argv[5]) == "1" || std::string(argv[5]) == "true");
    }
  } else {
    // Normal mode: <model> [prompt] [chat_template] [quant] [verbose]
    if (argc >= 3)
      prompt = argv[2];
    if (argc >= 4)
      use_chat_template =
        (std::string(argv[3]) == "1" || std::string(argv[3]) == "true");
    if (argc >= 5) {
      quant_str = std::string(argv[4]);
      if (quant_str == "W4A32")
        quant_type = CAUSAL_LM_QUANTIZATION_W4A32;
      else if (quant_str == "W16A16")
        quant_type = CAUSAL_LM_QUANTIZATION_W16A16;
      else if (quant_str == "W8A16")
        quant_type = CAUSAL_LM_QUANTIZATION_W8A16;
      else if (quant_str == "W32A32")
        quant_type = CAUSAL_LM_QUANTIZATION_W32A32;
    }
    if (argc >= 6)
      verbose = (std::string(argv[5]) == "1" || std::string(argv[5]) == "true");
  }

  printSection("Configuration");
  printInfo("Model Name", model_name);
  printInfo("Use Chat Template", use_chat_template ? "true" : "false");
  printInfo("Quantization", quant_str);
  printInfo("Verbose", verbose ? "true" : "false");
  if (!chat_file_path.empty()) {
    printInfo("Chat File", chat_file_path);
  }
  std::cout << "\n";

  printSection("Initialization");
  std::cout << COLOR_CYAN << "⏳ " << COLOR_RESET << "Configuring options...\n";
  Config config;
  config.use_chat_template = use_chat_template;
  config.debug_mode = true;
  config.verbose = verbose;
  ErrorCode err = setOptions(config);
  if (err != CAUSAL_LM_ERROR_NONE) {
    printError("Failed to set options");
    std::cerr << "  Error code: " << static_cast<int>(err) << "\n";
    return 1;
  }
  printSuccess("Options configured successfully");

  printSection("Model Loading");
  std::cout << COLOR_CYAN << "⏳ " << COLOR_RESET
            << "Loading model: " << COLOR_BOLD << model_name << COLOR_RESET
            << "\n";

  // Map string to ModelType
  ModelType model_type = CAUSAL_LM_MODEL_QWEN3_0_6B;
  std::string model_name_str(model_name);
  if (model_name_str == "QWEN3-0.6B") {
    model_type = CAUSAL_LM_MODEL_QWEN3_0_6B;
  } else {
    std::cout << COLOR_YELLOW << "⚠ Warning: Unknown model name '"
              << model_name_str << "'. Defaulting to QWEN3-0.6B." << COLOR_RESET
              << "\n";
  }

  err = loadModel(CAUSAL_LM_BACKEND_CPU, model_type, quant_type);

  if (err != CAUSAL_LM_ERROR_NONE) {
    printError("Failed to load model");
    std::cerr << "  Error code: " << static_cast<int>(err) << "\n";
    return 1;
  }
  printSuccess("Model loaded successfully");

  // ── --chat-file mode: load messages from JSON file ──
  using json = nlohmann::json;
  std::vector<CausalLMChatMessage> file_msgs;
  std::vector<std::string> role_strs, content_strs;

  if (!chat_file_path.empty()) {
    printSection("Test: Chat Template from File");
    std::ifstream chat_file(chat_file_path);
    if (!chat_file.is_open()) {
      printError("Cannot open chat file: " + chat_file_path);
      return 1;
    }

    json chat_json;
    try {
      chat_file >> chat_json;
    } catch (const json::parse_error &e) {
      printError("JSON parse error: " + std::string(e.what()));
      return 1;
    }

    // Support both formats:
    //   Array:  [{"role":"user","content":"Hi"}]
    //   Object: {"chat": [{"role":"user","content":"Hi"}]}
    json messages_json;
    if (chat_json.is_array()) {
      messages_json = chat_json;
    } else if (chat_json.is_object() && chat_json.contains("chat") &&
               chat_json["chat"].is_array()) {
      messages_json = chat_json["chat"];
    } else {
      printError("Chat file must contain a JSON array or {\"chat\": [...]}");
      return 1;
    }

    // Store strings to keep pointers valid
    for (const auto &msg : messages_json) {
      if (msg.contains("role") && msg.contains("content")) {
        role_strs.push_back(msg["role"].get<std::string>());
        content_strs.push_back(msg["content"].get<std::string>());
      }
    }
    for (size_t i = 0; i < role_strs.size(); ++i) {
      file_msgs.push_back({role_strs[i].c_str(), content_strs[i].c_str()});
    }

    std::cout << COLOR_CYAN << "📝 " << COLOR_RESET << "Messages from "
              << chat_file_path << ":\n";
    for (size_t i = 0; i < file_msgs.size(); ++i) {
      std::cout << COLOR_YELLOW << "  [" << file_msgs[i].role << "] "
                << COLOR_RESET << file_msgs[i].content << "\n";
    }
    std::cout << "\n";

    // Test applyChatTemplate with file messages
    const char *formattedText = nullptr;
    err = applyChatTemplate(file_msgs.data(), file_msgs.size(), true,
                            &formattedText);
    if (err == CAUSAL_LM_ERROR_NONE && formattedText) {
      std::cout << COLOR_CYAN << "💬 " << COLOR_RESET << "Formatted prompt:\n";
      std::cout << COLOR_BOLD << COLOR_YELLOW << formattedText << COLOR_RESET
                << "\n\n";
      printSuccess("applyChatTemplate works");
    } else {
      printError("applyChatTemplate failed");
      std::cerr << "  Error code: " << static_cast<int>(err) << "\n";
    }

    // Test runModelWithMessages with file messages
    printSection("Test: runModelWithMessages from File");
    std::cout << COLOR_CYAN << "⚡ " << COLOR_RESET
              << "Running inference with messages...\n\n";

    const char *msgOutput = nullptr;
    if (verbose) {
      std::cout << COLOR_CYAN << "💬 " << COLOR_RESET << "Streaming Output:\n";
      std::cout << COLOR_BOLD << COLOR_GRAY;
    }

    err = runModelWithMessages(file_msgs.data(), file_msgs.size(), true,
                               &msgOutput);

    if (verbose) {
      std::cout << COLOR_RESET << "\n\n";
    }

    if (err == CAUSAL_LM_ERROR_NONE && msgOutput) {
      std::cout << COLOR_CYAN << "💬 " << COLOR_RESET << "Output:\n";
      std::cout << COLOR_BOLD << COLOR_GREEN << "  " << msgOutput << COLOR_RESET
                << "\n\n";
      printSuccess("runModelWithMessages works");
    } else {
      printError("runModelWithMessages failed");
      std::cerr << "  Error code: " << static_cast<int>(err) << "\n";
    }

    // Skip to performance metrics
    goto print_metrics;
  }

  // ── Test 1: applyChatTemplate (no inference, format only) ──
  {
    printSection("Test: applyChatTemplate");
    std::cout << COLOR_CYAN << "📝 " << COLOR_RESET
              << "Testing chat template formatting (no inference)...\n\n";

    CausalLMChatMessage tmpl_msgs[] = {
      {"system", "You are a helpful AI assistant."}, {"user", prompt}};

    const char *formattedText = nullptr;
    err = applyChatTemplate(tmpl_msgs, 2, true, &formattedText);
    if (err == CAUSAL_LM_ERROR_NONE && formattedText) {
      std::cout << COLOR_CYAN << "💬 " << COLOR_RESET << "Formatted prompt:\n";
      std::cout << COLOR_BOLD << COLOR_YELLOW << formattedText << COLOR_RESET
                << "\n\n";
      printSuccess("applyChatTemplate works");
    } else {
      printError("applyChatTemplate failed");
      std::cerr << "  Error code: " << static_cast<int>(err) << "\n";
    }

    // ── Test 2: runModel (single prompt, existing API) ──
    printSection("Test: runModel (single prompt)");
    std::cout << COLOR_CYAN << "📝 " << COLOR_RESET << "Input Prompt:\n";
    std::cout << COLOR_BOLD << COLOR_YELLOW << "  " << prompt << COLOR_RESET
              << "\n\n";

    std::cout << COLOR_CYAN << "⚡ " << COLOR_RESET
              << "Running inference...\n\n";

    const char *outputText = nullptr;

    if (verbose) {
      std::cout << COLOR_CYAN << "💬 " << COLOR_RESET << "Streaming Output:\n";
      std::cout << COLOR_BOLD << COLOR_GRAY;
    }

    err = runModel(prompt, &outputText);

    if (verbose) {
      std::cout << COLOR_RESET << "\n\n";
    }

    if (err != CAUSAL_LM_ERROR_NONE) {
      printError("Failed to run model");
      std::cerr << "  Error code: " << static_cast<int>(err) << "\n";
      return 1;
    }

    if (outputText) {
      std::cout << COLOR_CYAN << "💬 " << COLOR_RESET << "Output:\n";
      std::cout << COLOR_BOLD << COLOR_GREEN << "  ";
      std::string out(outputText);
      size_t pos = 0;
      while (pos < out.length()) {
        size_t newlinePos = out.find('\n', pos);
        if (newlinePos == std::string::npos) {
          newlinePos = out.length();
        }
        std::string line = out.substr(pos, newlinePos - pos);
        std::cout << line;
        if (newlinePos < out.length()) {
          std::cout << "\n  ";
          pos = newlinePos + 1;
        } else {
          pos = out.length();
        }
      }
      std::cout << COLOR_RESET << "\n\n";
    } else {
      printWarning("No output generated");
    }

    // ── Test 3: runModelWithMessages (chat template with messages) ──
    printSection("Test: runModelWithMessages");
    CausalLMChatMessage chat_msgs[] = {
      {"system", "You are a helpful AI assistant."}, {"user", prompt}};

    std::cout << COLOR_CYAN << "📝 " << COLOR_RESET << "Messages:\n";
    for (size_t i = 0; i < 2; ++i) {
      std::cout << COLOR_YELLOW << "  [" << chat_msgs[i].role << "] "
                << COLOR_RESET << chat_msgs[i].content << "\n";
    }
    std::cout << "\n";

    std::cout << COLOR_CYAN << "⚡ " << COLOR_RESET
              << "Running inference with messages...\n\n";

    const char *msgOutput = nullptr;

    if (verbose) {
      std::cout << COLOR_CYAN << "💬 " << COLOR_RESET << "Streaming Output:\n";
      std::cout << COLOR_BOLD << COLOR_GRAY;
    }

    err = runModelWithMessages(chat_msgs, 2, true, &msgOutput);

    if (verbose) {
      std::cout << COLOR_RESET << "\n\n";
    }

    if (err == CAUSAL_LM_ERROR_NONE && msgOutput) {
      std::cout << COLOR_CYAN << "💬 " << COLOR_RESET << "Output:\n";
      std::cout << COLOR_BOLD << COLOR_GREEN << "  " << msgOutput << COLOR_RESET
                << "\n\n";
      printSuccess("runModelWithMessages works");
    } else {
      printError("runModelWithMessages failed");
      std::cerr << "  Error code: " << static_cast<int>(err) << "\n";
    }
  } // end of normal mode tests

print_metrics:
  printSection("Performance Metrics");
  PerformanceMetrics metrics;
  err = getPerformanceMetrics(&metrics);
  if (err != CAUSAL_LM_ERROR_NONE) {
    printWarning("Failed to get metrics");
    std::cout << "  Error code: " << static_cast<int>(err) << "\n";
  } else {
    double prefill_tps =
      metrics.prefill_duration_ms > 0
        ? (metrics.prefill_tokens / metrics.prefill_duration_ms * 1000.0)
        : 0.0;
    double gen_tps =
      metrics.generation_duration_ms > 0
        ? (metrics.generation_tokens / metrics.generation_duration_ms * 1000.0)
        : 0.0;

    std::cout << COLOR_CYAN << "  📊 " << COLOR_RESET << COLOR_BOLD
              << "Prefill Stage" << COLOR_RESET << "\n";
    std::cout << COLOR_CYAN << "    Tokens:" << COLOR_RESET << "       "
              << metrics.prefill_tokens << "\n";
    std::cout << COLOR_CYAN << "    Duration:" << COLOR_RESET << "     "
              << std::fixed << std::setprecision(2)
              << metrics.prefill_duration_ms << " ms\n";
    std::cout << COLOR_CYAN << "    Throughput:" << COLOR_RESET << "   "
              << COLOR_BOLD << COLOR_GREEN << std::fixed << std::setprecision(1)
              << prefill_tps << COLOR_RESET << " tokens/sec\n\n";

    std::cout << COLOR_CYAN << "  📊 " << COLOR_RESET << COLOR_BOLD
              << "Generation Stage" << COLOR_RESET << "\n";
    std::cout << COLOR_CYAN << "    Tokens:" << COLOR_RESET << "       "
              << metrics.generation_tokens << "\n";
    std::cout << COLOR_CYAN << "    Duration:" << COLOR_RESET << "     "
              << std::fixed << std::setprecision(2)
              << metrics.generation_duration_ms << " ms\n";
    std::cout << COLOR_CYAN << "    Throughput:" << COLOR_RESET << "   "
              << COLOR_BOLD << COLOR_GREEN << std::fixed << std::setprecision(1)
              << gen_tps << COLOR_RESET << " tokens/sec\n\n";

    std::cout << COLOR_CYAN << "  📊 " << COLOR_RESET << COLOR_BOLD
              << "Total Stats" << COLOR_RESET << "\n";
    std::cout << COLOR_CYAN << "    Init time:" << COLOR_RESET << "    "
              << std::fixed << std::setprecision(2)
              << metrics.initialization_duration_ms << " ms\n";
    std::cout << COLOR_CYAN << "    Duration :" << COLOR_RESET << "    "
              << std::fixed << std::setprecision(2) << metrics.total_duration_ms
              << " ms\n";
    std::cout << COLOR_CYAN << "    Peak Mem:" << COLOR_RESET << "     "
              << metrics.peak_memory_kb / 1024 << " MB\n\n";
  }

  printLine("═", 63);
  std::cout << COLOR_BOLD << COLOR_GREEN << "  ✓ Test completed successfully!"
            << COLOR_RESET << "\n";
  printLine("═", 63);
  std::cout << "\n";

  return 0;
}
