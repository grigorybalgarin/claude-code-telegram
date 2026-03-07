"""Telegram response delivery for agentic mode.

Consolidates the duplicated pattern of sending Claude responses back to
Telegram: delete progress message, send formatted text with retry,
send images (with optional caption), attach panel markup, show guard report.
"""

import asyncio
from typing import Any, List, Optional

import structlog
from telegram import Update

from ..utils.formatting import FormattedMessage
from .stream_handler import StreamHandler

logger = structlog.get_logger()


class ResponseSender:
    """Delivers formatted Claude responses to Telegram chats.

    Extracted from MessageOrchestrator to eliminate triple duplication
    of response sending logic across text, document, and media handlers.
    """

    def __init__(self, stream: StreamHandler):
        self.stream = stream

    async def deliver(
        self,
        update: Update,
        formatted_messages: List[FormattedMessage],
        *,
        images: Optional[List[Any]] = None,
        panel_markup: Optional[Any] = None,
        progress_msg: Optional[Any] = None,
        guard_report: Optional[Any] = None,
        change_guard: Optional[Any] = None,
    ) -> None:
        """Send formatted response messages, images, and panel to Telegram.

        Args:
            update: The original Telegram update (for reply_to).
            formatted_messages: Pre-formatted message parts from ResponseFormatter.
            images: Optional list of ImageAttachment objects.
            panel_markup: Optional InlineKeyboardMarkup for the last message.
            progress_msg: Progress message to delete before sending.
            guard_report: Optional ChangeGuardReport to append.
            change_guard: Optional ChangeGuard instance for formatting report.
        """
        if progress_msg:
            try:
                await progress_msg.delete()
            except Exception:
                logger.debug("Failed to delete progress message, ignoring")

        images = images or []
        reply_id = update.message.message_id

        # Try to combine text + images in one message when possible
        caption_sent = False
        if images and len(formatted_messages) == 1:
            msg = formatted_messages[0]
            if msg.text and len(msg.text) <= 1024:
                try:
                    caption_sent = await self.stream.send_images(
                        update,
                        images,
                        reply_to_message_id=reply_id,
                        caption=msg.text,
                        caption_parse_mode=msg.parse_mode,
                    )
                except Exception as img_err:
                    logger.warning("Image+caption send failed", error=str(img_err))

        # Send text messages (skip if caption was already embedded in photos)
        sent_any_text = False
        if not caption_sent:
            visible_messages = [
                m for m in formatted_messages if m.text and m.text.strip()
            ]
            for i, message in enumerate(formatted_messages):
                if not message.text or not message.text.strip():
                    continue
                msg_reply_id = reply_id if i == 0 else None
                sent_any_text = True
                is_last_visible = (
                    bool(visible_messages)
                    and message is visible_messages[-1]
                    and not (guard_report and change_guard)
                )
                sent = await self._send_with_retry(
                    update,
                    message.text,
                    message.parse_mode,
                    msg_reply_id,
                    reply_markup=panel_markup if is_last_visible else None,
                )
                if not sent:
                    try:
                        await update.message.reply_text(
                            message.text[:4000],
                            reply_to_message_id=msg_reply_id,
                            reply_markup=panel_markup if is_last_visible else None,
                        )
                    except Exception:
                        pass
                if i < len(formatted_messages) - 1:
                    await asyncio.sleep(0.5)

            # Send images separately if caption wasn't used
            if images:
                try:
                    await self.stream.send_images(
                        update,
                        images,
                        reply_to_message_id=reply_id,
                    )
                except Exception as img_err:
                    logger.warning("Image send failed", error=str(img_err))

        # Guard report (verification/rollback)
        if guard_report and change_guard:
            await update.message.reply_text(
                change_guard.format_report_html(guard_report),
                parse_mode="HTML",
                reply_to_message_id=reply_id,
                reply_markup=panel_markup,
            )
        elif caption_sent and not sent_any_text and panel_markup:
            await update.message.reply_text(
                "Actions:",
                reply_markup=panel_markup,
                reply_to_message_id=reply_id,
            )

    @staticmethod
    async def _send_with_retry(
        update: Update,
        text: str,
        parse_mode: Optional[str],
        reply_to_message_id: Optional[int],
        reply_markup: Optional[Any] = None,
        max_retries: int = 3,
    ) -> bool:
        """Send a message with exponential backoff retry.

        Tries HTML first, then plain text on parse error. Returns True on success.
        """
        for attempt in range(max_retries):
            try:
                pm = parse_mode if attempt == 0 else None
                await update.message.reply_text(
                    text,
                    parse_mode=pm,
                    reply_markup=reply_markup,
                    reply_to_message_id=reply_to_message_id,
                )
                return True
            except Exception as e:
                err_str = str(e).lower()
                if "parse" in err_str or "can't" in err_str:
                    if parse_mode:
                        parse_mode = None
                        continue
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))
                else:
                    logger.warning(
                        "Failed to send after retries",
                        error=str(e),
                        attempts=max_retries,
                    )
        return False
