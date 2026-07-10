"""Operational notifications for runs that cannot start."""

import smtplib
from email.header import Header
from email.mime.text import MIMEText
from email.utils import formataddr, parseaddr

from loguru import logger
from omegaconf import DictConfig


def send_gpu_unavailable_notification(config: DictConfig, reason: str) -> None:
    sender = config.email.sender
    receiver = config.email.receiver
    msg = MIMEText(
        "The daily paper pipeline was paused because no GPU had enough free memory.\n\n"
        f"Reason: {reason}\n\nRun it again when the GPU server is idle.",
        "plain",
        "utf-8",
    )
    name, address = parseaddr(f"Zotero arXiv Daily <{sender}>")
    msg["From"] = formataddr((Header(name, "utf-8").encode(), address))
    name, address = parseaddr(f"You <{receiver}>")
    msg["To"] = formataddr((Header(name, "utf-8").encode(), address))
    msg["Subject"] = Header(config.runtime.gpu.notification_subject, "utf-8").encode()
    try:
        server = smtplib.SMTP_SSL(config.email.smtp_server, config.email.smtp_port)
        server.login(sender, config.email.sender_password)
        server.sendmail(sender, [receiver], msg.as_string())
        server.quit()
    except Exception:
        logger.exception("Failed to send GPU-unavailable notification")
