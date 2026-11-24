# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file events.py
# @brief Backward-compatibility shim: re-exports bus from core.events.
# @author prachi-t17 <prachi.t@samsung.com>

"""Backward-compatibility shim: re-exports bus from core.events."""
from core.events import bus, EventBus
__all__ = ["bus", "EventBus"]
