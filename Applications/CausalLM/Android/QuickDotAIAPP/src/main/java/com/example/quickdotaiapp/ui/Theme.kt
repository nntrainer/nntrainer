// SPDX-License-Identifier: Apache-2.0
/**
 * @file  Theme.kt
 * @brief The fixed sky-blue Material 3 schemes (no wallpaper dynamic
 *        color, so the brand colour is the same on every phone).
 */
package com.example.quickdotaiapp.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import com.example.quickdotaiapp.ThemeMode

/** The lightning bolt's own colour, identical in both schemes. */
val SkyBolt = Color(0xFF38BDF8)
val SkyStreak = Color(0xFF7DD3FC)

private val LightScheme = lightColorScheme(
    primary = Color(0xFF006492), onPrimary = Color(0xFFFFFFFF),
    primaryContainer = Color(0xFFCBE6FF), onPrimaryContainer = Color(0xFF001E2F),
    secondary = Color(0xFF50606E), onSecondary = Color(0xFFFFFFFF),
    secondaryContainer = Color(0xFFD3E4F5), onSecondaryContainer = Color(0xFF0B1D2A),
    tertiary = Color(0xFF65587B), onTertiary = Color(0xFFFFFFFF),
    tertiaryContainer = Color(0xFFE3DFFF), onTertiaryContainer = Color(0xFF1A1A49),
    error = Color(0xFFBA1A1A), onError = Color(0xFFFFFFFF),
    errorContainer = Color(0xFFFFDAD6), onErrorContainer = Color(0xFF410002),
    background = Color(0xFFF7F9FF), onBackground = Color(0xFF181C20),
    surface = Color(0xFFF7F9FF), onSurface = Color(0xFF181C20),
    surfaceVariant = Color(0xFFDEE3EA), onSurfaceVariant = Color(0xFF41484D),
    outline = Color(0xFF72787E), outlineVariant = Color(0xFFC2C7CE),
    surfaceContainerLowest = Color(0xFFFFFFFF), surfaceContainerLow = Color(0xFFF0F4FA),
    surfaceContainer = Color(0xFFEAEEF5), surfaceContainerHigh = Color(0xFFE4E9EF),
    surfaceContainerHighest = Color(0xFFDEE3EA),
)

private val DarkScheme = darkColorScheme(
    primary = Color(0xFF8DCDFF), onPrimary = Color(0xFF00344F),
    primaryContainer = Color(0xFF004B70), onPrimaryContainer = Color(0xFFCBE6FF),
    secondary = Color(0xFFB7C9D8), onSecondary = Color(0xFF22323E),
    secondaryContainer = Color(0xFF374955), onSecondaryContainer = Color(0xFFD3E4F5),
    tertiary = Color(0xFFCFBFE8), onTertiary = Color(0xFF362B4B),
    tertiaryContainer = Color(0xFF4D4162), onTertiaryContainer = Color(0xFFE3DFFF),
    error = Color(0xFFFFB4AB), onError = Color(0xFF690005),
    errorContainer = Color(0xFF93000A), onErrorContainer = Color(0xFFFFDAD6),
    background = Color(0xFF0F1419), onBackground = Color(0xFFDFE3E8),
    surface = Color(0xFF0F1419), onSurface = Color(0xFFDFE3E8),
    surfaceVariant = Color(0xFF41484D), onSurfaceVariant = Color(0xFFC2C7CE),
    outline = Color(0xFF8C9198), outlineVariant = Color(0xFF41484D),
    surfaceContainerLowest = Color(0xFF0A0F14), surfaceContainerLow = Color(0xFF171C21),
    surfaceContainer = Color(0xFF1C2025), surfaceContainerHigh = Color(0xFF262A30),
    surfaceContainerHighest = Color(0xFF31353B),
)

@Composable
fun QuickAiTheme(mode: ThemeMode, content: @Composable () -> Unit) {
    val dark = when (mode) {
        ThemeMode.SYSTEM -> isSystemInDarkTheme()
        ThemeMode.LIGHT -> false
        ThemeMode.DARK -> true
    }
    MaterialTheme(colorScheme = if (dark) DarkScheme else LightScheme, content = content)
}
