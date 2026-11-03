// SPDX-License-Identifier: Apache-2.0
/**
 * @file  AttachSheet.kt
 * @brief The attach menu: image (vision models only), .txt into the
 *        message, .json into Raw.
 */
package com.example.quickdotaiapp.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import com.example.quickdotaiapp.R

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AttachSheet(
    canAttachImages: Boolean,
    modelName: String,
    onImage: () -> Unit,
    onText: () -> Unit,
    onJson: () -> Unit,
    onDismiss: () -> Unit,
) {
    ModalBottomSheet(onDismissRequest = onDismiss, sheetState = rememberModalBottomSheetState()) {
        Column(Modifier.padding(bottom = 24.dp)) {
            Text("Attach", style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.padding(horizontal = 24.dp, vertical = 8.dp))
            ListItem(
                headlineContent = { Text("Image") },
                supportingContent = {
                    Text(if (canAttachImages) "Attached to the next message" else "Vision models only. $modelName is text only.")
                },
                leadingContent = { Icon(painterResource(R.drawable.ic_image), null) },
                modifier = Modifier
                    .alpha(if (canAttachImages) 1f else 0.5f)
                    .clickable(enabled = canAttachImages) { onImage(); onDismiss() }
                    .testTag(Tags.ATTACH_IMAGE),
            )
            ListItem(
                headlineContent = { Text("Text file (.txt)") },
                supportingContent = { Text("Its contents are inserted into the message, ready to edit and send.") },
                leadingContent = { Icon(painterResource(R.drawable.ic_text_file), null) },
                modifier = Modifier.clickable { onText(); onDismiss() }.testTag(Tags.ATTACH_TEXT),
            )
            ListItem(
                headlineContent = { Text("JSON request (.json)") },
                supportingContent = { Text("An OpenAI messages file opens in Raw; any other JSON is inserted as text.") },
                leadingContent = { Icon(painterResource(R.drawable.ic_braces), null) },
                modifier = Modifier.clickable { onJson(); onDismiss() }.testTag(Tags.ATTACH_JSON),
            )
            Spacer(Modifier.height(4.dp))
            Text(
                "Files up to 1 MB, picked with the system document picker",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(horizontal = 24.dp),
            )
        }
    }
}
