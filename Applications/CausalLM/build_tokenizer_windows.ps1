# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2026 The NNTrainer Authors
#
# @file build_tokenizer_windows.ps1
# @brief Build the CausalLM tokenizers_c library for Windows.

param (
    [string]$BuildDir = "build",
    [string]$RustTarget = "",
    # Link the static CRT (/MT + crt-static) instead of the dynamic default.
    # Required when the consuming build uses -Db_vscrt=static_from_buildtype:
    # a single /MD object in this lib (the cc-crate C++ deps, e.g. esaxx)
    # otherwise fails every exe link with LNK2038 RuntimeLibrary mismatch.
    [switch]$StaticCrt
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$CrateDir = Join-Path $ScriptDir "tokenizers_c_win"
$CargoBin = Join-Path $env:USERPROFILE ".cargo\bin"

if (-Not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    $CargoExe = Join-Path $CargoBin "cargo.exe"
    if (Test-Path $CargoExe) {
        $env:PATH = "$CargoBin;$env:PATH"
    } else {
        Write-Error "cargo is required. Install Rust from https://rustup.rs/ and retry."
    }
}

if ([System.IO.Path]::IsPathRooted($BuildDir)) {
    $BuildRoot = $BuildDir
} else {
    $BuildRoot = Join-Path $RepoRoot $BuildDir
}

$TargetDir = Join-Path $BuildRoot "tokenizers_c_win\target"
$CargoArgs = @(
    "build",
    "--manifest-path", (Join-Path $CrateDir "Cargo.toml"),
    "--target-dir", $TargetDir,
    "--release",
    "--locked"
)

if (-Not [string]::IsNullOrWhiteSpace($RustTarget)) {
    if (Get-Command rustup -ErrorAction SilentlyContinue) {
        & rustup target add $RustTarget
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }

    $CargoArgs += @("--target", $RustTarget)
}

New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null

Write-Output "Building tokenizers_c for Windows"
Write-Output "  crate:  $CrateDir"
Write-Output "  target: $TargetDir"

$PreviousRustFlags = $env:RUSTFLAGS
$PreviousCFlags = $env:CFLAGS
$PreviousCxxFlags = $env:CXXFLAGS
# These are APPENDED after any caller-provided flags, so they always win --
# the CRT choice is owned by this switch, not by ambient env.
if ($StaticCrt) {
    $RustCrtFlag = "-C target-feature=+crt-static"
    $CCrtFlag = "/MT"
} else {
    $RustCrtFlag = "-C target-feature=-crt-static"
    $CCrtFlag = "/MD"
}

try {
    if ([string]::IsNullOrWhiteSpace($PreviousRustFlags)) {
        $env:RUSTFLAGS = $RustCrtFlag
    } else {
        $env:RUSTFLAGS = "$PreviousRustFlags $RustCrtFlag"
    }

    if ([string]::IsNullOrWhiteSpace($PreviousCFlags)) {
        $env:CFLAGS = $CCrtFlag
    } else {
        $env:CFLAGS = "$PreviousCFlags $CCrtFlag"
    }

    if ([string]::IsNullOrWhiteSpace($PreviousCxxFlags)) {
        $env:CXXFLAGS = $CCrtFlag
    } else {
        $env:CXXFLAGS = "$PreviousCxxFlags $CCrtFlag"
    }

    & cargo @CargoArgs
} finally {
    $env:RUSTFLAGS = $PreviousRustFlags
    $env:CFLAGS = $PreviousCFlags
    $env:CXXFLAGS = $PreviousCxxFlags
}

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ([string]::IsNullOrWhiteSpace($RustTarget)) {
    $LibraryPath = Join-Path $TargetDir "release\tokenizers_c.lib"
} else {
    $LibraryPath = Join-Path $TargetDir "$RustTarget\release\tokenizers_c.lib"
}

if (-Not (Test-Path $LibraryPath)) {
    Write-Error "tokenizers_c.lib was not produced at $LibraryPath"
}

Write-Output "Built $LibraryPath"
