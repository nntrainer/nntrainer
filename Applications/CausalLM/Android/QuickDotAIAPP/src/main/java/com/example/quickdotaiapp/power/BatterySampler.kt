// SPDX-License-Identifier: Apache-2.0
/**
 * @file  BatterySampler.kt
 * @brief One PowerSample from the battery gauge (BatteryManager + the
 *        sticky battery intent), the thermal service (status, headroom)
 *        and the processor-block sensors under /sys/class/thermal. Every
 *        read fails soft: 0, false or null, never an exception.
 */
package com.example.quickdotaiapp.power

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.PowerManager
import java.io.File

class BatterySampler(private val context: Context) {
    private val battery = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
    private val power = context.getSystemService(Context.POWER_SERVICE) as PowerManager

    /** Block → the temp files of its zones, found once; the S26 names them nsphvx/nsphmx (NPU), cpu-N-N-N, gpuss-N. */
    private val zones: Map<String, List<File>> = enumerateZones()
    private var tick = 0
    private var status = 0
    private var headroom = Float.NaN
    private var npu: Float? = null
    private var cpu: Float? = null
    private var gpu: Float? = null

    /** Called every 250 ms by the runner; the thermal reads happen on every 4th call (1 Hz) and are carried in between. */
    fun sample(): PowerSample {
        val sticky = runCatching { context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED)) }.getOrNull()
        // An unsupported property reads as Int.MIN_VALUE, not as an exception.
        val microAmps = runCatching { battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW) }.getOrDefault(0)
            .let { if (it == Int.MIN_VALUE) 0 else it }
        val milliVolts = sticky?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0) ?: 0
        val tenths = sticky?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0
        val plugged = (sticky?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0) != 0
        if (tick++ % 4 == 0) {
            status = runCatching { power.currentThermalStatus }.getOrDefault(0)
            // getThermalHeadroom returns NaN when called much faster than once a second.
            headroom = runCatching { power.getThermalHeadroom(10) }.getOrDefault(Float.NaN)
            npu = maxTemp("npu"); cpu = maxTemp("cpu"); gpu = maxTemp("gpu")
        }
        return PowerSample(System.currentTimeMillis(), microAmps, milliVolts, tenths, plugged, status, headroom, npu, cpu, gpu)
    }

    private fun maxTemp(block: String): Float? =
        zones[block]?.mapNotNull { f -> runCatching { f.readText().trim().toInt() / 1000f }.getOrNull() }?.maxOrNull()

    private fun enumerateZones(): Map<String, List<File>> {
        val groups: Map<String, (String) -> Boolean> = mapOf(
            "npu" to { t -> t.startsWith("nsphvx") || t.startsWith("nsphmx") },
            "cpu" to { t -> t.startsWith("cpu-") && t.getOrNull(4)?.isDigit() == true },
            "gpu" to { t -> t.startsWith("gpuss") },
        )
        val found = groups.keys.associateWith { mutableListOf<File>() }
        val dirs = runCatching { File("/sys/class/thermal").listFiles { f -> f.name.startsWith("thermal_zone") } }.getOrNull()
            ?: return found
        for (dir in dirs) {
            val type = runCatching { File(dir, "type").readText().trim() }.getOrNull() ?: continue
            for ((block, matches) in groups) if (matches(type)) found.getValue(block) += File(dir, "temp")
        }
        return found
    }
}
