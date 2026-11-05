// SPDX-License-Identifier: Apache-2.0
/**
 * @file  PowerFiles.kt
 * @brief Where a session's results go (spec §6): runs.csv appended per
 *        run and one JSON per session, under Download/QuickAI/power when
 *        the app has all-files access, else the app's own external dir.
 */
package com.example.quickdotaiapp.power

import android.content.Context
import android.os.Environment
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class PowerFiles(private val context: Context) {
    val dir: File
        get() = if (Environment.isExternalStorageManager())
            File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "QuickAI/power")
        else File(context.getExternalFilesDir(null), "power")

    /** Appends one CSV row per run and writes the session JSON; the JSON path, or null when nothing could be written. */
    fun write(log: PowerSessionLog): String? = runCatching {
        val d = dir.apply { mkdirs() }
        val csv = File(d, "runs.csv")
        val header = if (!csv.exists() || csv.length() == 0L) PowerLog.CSV_HEADER + "\n" else ""
        csv.appendText(header + log.runs.joinToString("") { PowerLog.csvRow(log, it) + "\n" })
        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        val json = File(d, "$stamp-${safe(log.meta.modelId)}-${safe(log.meta.backend)}.json")
        json.writeText(PowerLog.toJson(log))
        json.path
    }.getOrNull()

    private fun safe(s: String) = s.replace(Regex("[^A-Za-z0-9._-]"), "_")
}
