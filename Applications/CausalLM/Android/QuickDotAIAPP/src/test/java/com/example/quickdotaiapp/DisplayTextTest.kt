// SPDX-License-Identifier: Apache-2.0
package com.example.quickdotaiapp

import com.example.quickdotaiapp.ui.displayText
import org.junit.Assert.assertEquals
import org.junit.Test

class DisplayTextTest {
    private fun turn(text: String) = ChatTurn(1, TurnRole.ASSISTANT, text)

    @Test
    fun hidesOnlyTheTrailingEndOfTurnMarker() {
        assertEquals("Hello world", displayText(turn("Hello world<turn|>")))
        assertEquals("Hello world", displayText(turn("Hello world\n<turn|>")))
        assertEquals("a <turn|> b", displayText(turn("a <turn|> b")))
        assertEquals("no marker", displayText(turn("no marker")))
    }
}
