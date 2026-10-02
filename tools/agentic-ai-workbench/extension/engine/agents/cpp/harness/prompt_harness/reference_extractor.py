"""
Extract patterns from existing NNTrainer CausalLM reference implementations.

Given a working <model>_causallm.cpp file, extract:
- Method implementations (createAttention, createMLP, etc.)
- Architecture-specific constants (NORM_EPS, GQA_SIZE, etc.)
- Constraints and special patterns
- Comments documenting the architecture
"""
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ExtractedPattern:
    """A pattern extracted from reference implementation"""
    name: str                    # e.g., "createAttention"
    code: str                    # Full method body
    signature: str               # Method signature
    description: str             # Extracted comment/doc


class ReferenceExtractor:
    """Extract patterns from existing _causallm.cpp files"""

    def __init__(self, cpp_file_path: str):
        """
        Args:
            cpp_file_path: Path to <model>_causallm.cpp
        """
        self.cpp_file_path = cpp_file_path
        self.content = self._read_file()

    def _read_file(self) -> str:
        """Read the C++ file"""
        try:
            with open(self.cpp_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Failed to read {self.cpp_file_path}: {e}")

    def extract_all_patterns(self) -> Dict[str, ExtractedPattern]:
        """Extract all relevant patterns from the reference implementation"""
        patterns = {}

        # Extract key methods. Names match the real base-class signatures in
        # Applications/CausalLM/models/transformer.cpp (createMlp, not
        # createMLP; createTransformerDecoderBlock, not createTransformerBlock).
        for method_name in ['createAttention', 'createMlp', 'createTransformerDecoderBlock']:
            pattern = self.extract_method(method_name)
            if pattern:
                patterns[method_name] = pattern

        return patterns

    def extract_method(self, method_name: str) -> Optional[ExtractedPattern]:
        """
        Extract a specific method from the C++ file.

        Returns the method signature and body as a complete, executable function.
        """
        # Pattern to match: Tensor <ClassName>::methodName(args) { ... }
        # Handles nested braces correctly
        pattern = rf'(Tensor\s+\w+::{method_name}\s*\([^)]*\)\s*\{{)'

        match = re.search(pattern, self.content)
        if not match:
            return None

        start_pos = match.start()
        opening_brace_pos = match.end() - 1

        # Find matching closing brace
        closing_pos = self._find_matching_brace(opening_brace_pos)
        if closing_pos == -1:
            return None

        # Extract full method
        full_method = self.content[start_pos:closing_pos + 1]

        # Extract signature
        signature_match = re.match(r'(Tensor\s+\w+::\w+\s*\([^)]*\))', full_method)
        signature = signature_match.group(1) if signature_match else ""

        # Extract preceding comment
        preceding_comment = self._extract_preceding_comment(start_pos)

        return ExtractedPattern(
            name=method_name,
            code=full_method,
            signature=signature,
            description=preceding_comment
        )

    def extract_constants(self) -> Dict[str, str]:
        """Extract model-specific constants (NORM_EPS, GQA_SIZE, etc.)"""
        constants = {}

        # Pattern: static constexpr float NORM_EPS = 1e-5;
        # Pattern: static constexpr int GQA_SIZE = 8;
        const_pattern = r'static\s+constexpr\s+(\w+)\s+(\w+)\s*=\s*([^;]+);'

        for match in re.finditer(const_pattern, self.content):
            type_name = match.group(1)
            const_name = match.group(2)
            const_value = match.group(3).strip()
            constants[const_name] = const_value

        return constants

    def extract_architecture_notes(self) -> str:
        """
        Extract architecture-specific documentation.

        Looks for comments at the top of the class or file describing
        the architecture patterns (e.g., "Pre-attention RMSNorm", "SwiGLU MLP").
        """
        # Look for class comments
        class_doc_pattern = r'///.*\n(?:///.*\n)*'
        match = re.search(class_doc_pattern, self.content)

        if match:
            return match.group(0)
        return ""

    def extract_layer_usage(self) -> Dict[str, int]:
        """Count which NNTrainer layers are used in this model"""
        usage = {}

        # Find all createLayer calls
        layer_pattern = r'createLayer\s*\(\s*"([^"]+)"'

        for match in re.finditer(layer_pattern, self.content):
            layer_type = match.group(1)
            usage[layer_type] = usage.get(layer_type, 0) + 1

        return usage

    def _find_matching_brace(self, opening_pos: int) -> int:
        """
        Find the position of the closing brace that matches the opening brace at opening_pos.

        Handles nested braces and string literals.
        """
        depth = 1
        pos = opening_pos + 1
        in_string = False
        escape_next = False

        while pos < len(self.content) and depth > 0:
            char = self.content[pos]

            if escape_next:
                escape_next = False
                pos += 1
                continue

            if char == '\\':
                escape_next = True
                pos += 1
                continue

            if char == '"':
                in_string = not in_string
                pos += 1
                continue

            if not in_string:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1

            pos += 1

        return pos - 1 if depth == 0 else -1

    def _extract_preceding_comment(self, method_pos: int) -> str:
        """Extract comments that appear before the method definition"""
        # Look back up to 500 chars for comments
        search_start = max(0, method_pos - 500)
        preceding = self.content[search_start:method_pos]

        # Extract last comment block
        comment_pattern = r'(//.*\n)+$'
        match = re.search(comment_pattern, preceding)

        if match:
            return match.group(0).strip()
        return ""
