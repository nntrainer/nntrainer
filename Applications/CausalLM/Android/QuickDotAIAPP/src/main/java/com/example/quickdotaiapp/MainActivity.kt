// SPDX-License-Identifier: Apache-2.0
/**
 * @file  MainActivity.kt
 * @brief Compose host for the QuickDotAI sample. Everything else lives
 *        in AppViewModel / ChatController and the ui package.
 */
package com.example.quickdotaiapp

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import com.example.quickdotaiapp.ui.QuickAiApp

class MainActivity : ComponentActivity() {
    /** Public so the opt-in device test can read the state it drives through the UI. */
    val viewModel: AppViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        checkAllFilesAccess()
        setContent { QuickAiApp(viewModel) }
    }

    /** The native engine opens the model pack under /sdcard/Download; Android 11+ needs all-files access for that. */
    private fun checkAllFilesAccess() {
        if (Environment.isExternalStorageManager()) return
        startActivity(Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, Uri.parse("package:$packageName")))
    }
}
