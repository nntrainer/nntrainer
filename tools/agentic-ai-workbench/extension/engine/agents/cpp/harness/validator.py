"""
Compilation validator for generated C++ code.
"""

import subprocess
import tempfile
import os
from dataclasses import dataclass
from typing import List


@dataclass
class ValidationResult:
    """Result of compilation validation."""
    success: bool
    errors: List[str]
    compile_log: str


class CompilationValidator:
    """
    Validates generated C++ code compiles correctly.
    
    Uses g++ with -fsyntax-only for fast syntax checking.
    """
    
    def __init__(self, causallm_root: str, nntrainer_root: str):
        """
        Initialize validator with paths to required directories.
        
        Args:
            causallm_root: Path to CausalLM project root
            nntrainer_root: Path to nntrainer repository root
        """
        self.nntrainer_root = nntrainer_root
        self.nntrainer_include = os.path.join(nntrainer_root, "include")
        self.causallm_layers = os.path.join(nntrainer_root, "Applications/CausalLM/layers")
        self.causallm_root = causallm_root
    
    def validate_syntax(self, header: str, source: str, architecture: str) -> ValidationResult:
        """
        Validate C++ syntax using g++ -fsyntax-only.
        
        Args:
            header: Header file content
            source: Source file content
            architecture: Architecture name (used for temp filenames)
        
        Returns:
            ValidationResult with success status and any errors
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write temporary files
            header_filename = f"{architecture.lower().replace('forcausallm', '')}_causallm.h"
            source_filename = f"{architecture.lower().replace('forcausallm', '')}_causallm.cpp"
            
            header_path = os.path.join(tmpdir, header_filename)
            source_path = os.path.join(tmpdir, source_filename)
            
            with open(header_path, 'w') as f:
                f.write(header)
            with open(source_path, 'w') as f:
                f.write(source)
            
            # Build include paths
            include_paths = [
                self.nntrainer_include,
            ]
            
            # Add causallm_layers include (where custom layers like reshaped_rms_norm, swiglu live)
            if os.path.isdir(self.causallm_layers):
                include_paths.append(self.causallm_layers)
            
            # Add causallm include if it exists
            causallm_include = os.path.join(self.causallm_root, "include")
            if os.path.isdir(causallm_include):
                include_paths.append(causallm_include)
            
            # Add causallm root include (where llm_util.hpp, kv_cache_manager.h live)
            if os.path.isdir(self.causallm_root):
                include_paths.append(self.causallm_root)
            
            # Build g++ command
            cmd = ["g++", "-std=c++17", "-fsyntax-only"]
            for inc_path in include_paths:
                if os.path.isdir(inc_path):
                    cmd.extend(["-I", inc_path])
            cmd.append(source_path)
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    return ValidationResult(
                        success=True,
                        errors=[],
                        compile_log="Syntax validation passed"
                    )
                else:
                    # Parse errors from stderr
                    errors = self._parse_compiler_errors(result.stderr)
                    return ValidationResult(
                        success=False,
                        errors=errors,
                        compile_log=result.stderr
                    )
                    
            except subprocess.TimeoutExpired:
                return ValidationResult(
                    success=False,
                    errors=["Compilation timed out after 60 seconds"],
                    compile_log=""
                )
            except Exception as e:
                return ValidationResult(
                    success=False,
                    errors=[str(e)],
                    compile_log=""
                )
    
    def _parse_compiler_errors(self, stderr: str) -> List[str]:
        """
        Parse compiler error messages from stderr.
        
        Returns a list of error messages.
        """
        errors = []
        for line in stderr.split('\n'):
            line = line.strip()
            if line and ('error:' in line.lower() or 'warning:' in line.lower()):
                errors.append(line)
        return errors if errors else [stderr.strip()] if stderr.strip() else ["Unknown compilation error"]
