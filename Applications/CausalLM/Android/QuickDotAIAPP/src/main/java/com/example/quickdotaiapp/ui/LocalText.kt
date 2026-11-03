// SPDX-License-Identifier: Apache-2.0
/**
 * @file  LocalText.kt
 * @brief Text a field edits in local snapshot state, so the field never
 *        recomposes with a value older than what was typed (a StateFlow
 *        collected into composition lags by a dispatch and can drop or
 *        reorder keystrokes). Every edit is still pushed to the controller;
 *        the local copy is re-read from it only when [revision] moves.
 */
package com.example.quickdotaiapp.ui

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember

@Composable
fun rememberLocalText(revision: Any, latest: () -> String): MutableState<String> {
    val text = remember { mutableStateOf(latest()) }
    LaunchedEffect(revision) { text.value = latest() }
    return text
}
