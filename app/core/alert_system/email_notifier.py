"""
SMTP Email Notifier for HyperCore Alert System.

Provides email notification functionality for clinical alerts,
formatting AlertEvent data into readable email content.
"""
from __future__ import annotations

import smtplib
import ssl
import time
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr, formatdate
from typing import List, Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .models import AlertEvent

logger = logging.getLogger("hypercore.email_notifier")


@dataclass
class EmailResult:
    """Result of an email send operation."""
    success: bool
    recipients_sent: List[str]
    recipients_failed: List[str]
    error_message: Optional[str] = None
    retry_count: int = 0


class EmailNotifier:
    """
    SMTP Email Notifier for clinical alerts.

    Handles formatting AlertEvent into readable emails and
    sending via SMTP with retry logic and error handling.
    """

    def __init__(
        self,
        server: str,
        port: int,
        username: Optional[str] = None,
        password: Optional[str] = None,
        from_address: str = "hypercore-alerts@localhost",
        from_name: str = "HyperCore Alert System",
        use_tls: bool = True,
        use_ssl: bool = False,
        timeout_seconds: int = 30,
        retry_count: int = 3,
        retry_delay_seconds: float = 1.0,
    ):
        self.server = server
        self.port = port
        self.username = username
        self.password = password
        self.from_address = from_address
        self.from_name = from_name
        self.use_tls = use_tls
        self.use_ssl = use_ssl
        self.timeout_seconds = timeout_seconds
        self.retry_count = retry_count
        self.retry_delay_seconds = retry_delay_seconds

        logger.info(
            f"EmailNotifier initialized: server={server}:{port}, "
            f"tls={use_tls}, ssl={use_ssl}"
        )

    @classmethod
    def from_settings(cls) -> "EmailNotifier":
        """Create EmailNotifier from environment settings."""
        from .email_config import get_smtp_settings
        settings = get_smtp_settings()
        return cls(
            server=settings.server,
            port=settings.port,
            username=settings.username,
            password=settings.password,
            from_address=settings.from_address,
            from_name=settings.from_name,
            use_tls=settings.use_tls,
            use_ssl=settings.use_ssl,
            timeout_seconds=settings.timeout_seconds,
            retry_count=settings.retry_count,
            retry_delay_seconds=settings.retry_delay_seconds,
        )

    def format_alert_email(
        self,
        alert: "AlertEvent",
        recipient: str
    ) -> MIMEMultipart:
        """
        Format an AlertEvent into a MIME email message.

        Creates both plain text and HTML versions for maximum compatibility.
        """
        msg = MIMEMultipart("alternative")
        msg["Subject"] = self._build_subject(alert)
        msg["From"] = formataddr((self.from_name, self.from_address))
        msg["To"] = recipient
        msg["Date"] = formatdate(localtime=True)
        msg["X-Priority"] = self._get_priority_header(alert.severity.value)
        msg["X-HyperCore-Alert-ID"] = str(alert.event_id)
        msg["X-HyperCore-Patient-ID"] = str(alert.patient_id)

        # Plain text version
        text_content = self._build_plain_text(alert)
        msg.attach(MIMEText(text_content, "plain", "utf-8"))

        # HTML version
        html_content = self._build_html(alert)
        msg.attach(MIMEText(html_content, "html", "utf-8"))

        return msg

    def _build_subject(self, alert: "AlertEvent") -> str:
        """Build email subject line with severity indicator."""
        severity = alert.severity.value if hasattr(alert.severity, 'value') else str(alert.severity)
        severity_prefix = {
            "CRITICAL": "[CRITICAL]",
            "URGENT": "[URGENT]",
            "WARNING": "[WARNING]",
            "INFO": "[INFO]",
        }.get(severity.upper(), "[ALERT]")

        headline = alert.clinical_headline or f"{alert.risk_domain} alert"
        return f"{severity_prefix} {headline} - Patient {alert.patient_id}"

    def _get_priority_header(self, severity: str) -> str:
        """Map severity to email priority header."""
        return {
            "CRITICAL": "1",  # Highest
            "URGENT": "2",
            "WARNING": "3",
            "INFO": "4",
        }.get(severity.upper(), "3")

    def _build_plain_text(self, alert: "AlertEvent") -> str:
        """Build plain text email body."""
        biomarkers = alert.contributing_biomarkers or []
        biomarkers_str = ", ".join(biomarkers) if biomarkers else "None identified"
        routed = alert.routed_to or []
        routed_str = ", ".join(routed) if routed else "Not routed"
        severity = alert.severity.value if hasattr(alert.severity, 'value') else str(alert.severity)
        state = alert.state_current.value if hasattr(alert.state_current, 'value') else str(alert.state_current)
        tth = alert.time_to_harm_hours if alert.time_to_harm_hours else "Not calculated"

        return f"""
HYPERCORE CLINICAL ALERT
========================

Alert ID: {alert.event_id}
Patient ID: {alert.patient_id}
Severity: {severity}
Risk Domain: {alert.risk_domain}
Clinical State: {state}

CLINICAL HEADLINE
-----------------
{alert.clinical_headline or 'N/A'}

CLINICAL RATIONALE
------------------
{alert.clinical_rationale or 'N/A'}

SUGGESTED ACTION
----------------
{alert.suggested_action or 'N/A'}

TIME TO POTENTIAL HARM
----------------------
{tth} hours

CONTRIBUTING BIOMARKERS
-----------------------
{biomarkers_str}

ROUTING INFORMATION
-------------------
Routed to: {routed_str}

---
This is an automated alert from the HyperCore ML Service.
Do not reply to this email.
"""

    def _build_html(self, alert: "AlertEvent") -> str:
        """Build HTML email body with clinical styling."""
        severity = alert.severity.value if hasattr(alert.severity, 'value') else str(alert.severity)
        severity_color = {
            "CRITICAL": "#dc3545",  # Red
            "URGENT": "#fd7e14",    # Orange
            "WARNING": "#ffc107",   # Yellow
            "INFO": "#28a745",      # Green
        }.get(severity.upper(), "#6c757d")

        biomarkers = alert.contributing_biomarkers or []
        biomarkers_html = "".join(
            f"<li>{bm}</li>" for bm in biomarkers
        ) or "<li>None identified</li>"

        routed = alert.routed_to or []
        routed_html = ", ".join(routed) if routed else "Not routed"
        state = alert.state_current.value if hasattr(alert.state_current, 'value') else str(alert.state_current)
        tth = alert.time_to_harm_hours if alert.time_to_harm_hours else "N/A"

        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: {severity_color};
            color: white;
            padding: 15px 20px;
            border-radius: 8px 8px 0 0;
        }}
        .header h1 {{
            margin: 0;
            font-size: 18px;
        }}
        .content {{
            background: #f8f9fa;
            padding: 20px;
            border: 1px solid #dee2e6;
            border-top: none;
            border-radius: 0 0 8px 8px;
        }}
        .section {{
            margin-bottom: 20px;
        }}
        .section-title {{
            font-weight: 600;
            color: #495057;
            margin-bottom: 8px;
            border-bottom: 1px solid #dee2e6;
            padding-bottom: 5px;
        }}
        .meta {{
            font-size: 14px;
            color: #6c757d;
            margin-top: 10px;
        }}
        .severity-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 4px;
            background: {severity_color};
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 12px;
        }}
        .state-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            background: #6c757d;
            color: white;
            font-size: 12px;
            margin-left: 8px;
        }}
        .time-warning {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 10px 15px;
            border-radius: 4px;
            margin: 10px 0;
        }}
        .action-box {{
            background: #d4edda;
            border: 1px solid #28a745;
            padding: 15px;
            border-radius: 4px;
        }}
        ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .footer {{
            margin-top: 20px;
            padding-top: 15px;
            border-top: 1px solid #dee2e6;
            font-size: 12px;
            color: #6c757d;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>HyperCore Clinical Alert</h1>
    </div>
    <div class="content">
        <div class="meta">
            <strong>Alert ID:</strong> {alert.event_id}<br>
            <strong>Patient:</strong> {alert.patient_id}<br>
            <span class="severity-badge">{severity}</span>
            <span class="state-badge">{state}</span>
        </div>

        <div class="section">
            <div class="section-title">Clinical Headline</div>
            <p><strong>{alert.clinical_headline or 'N/A'}</strong></p>
        </div>

        <div class="section">
            <div class="section-title">Risk Domain</div>
            <p>{alert.risk_domain}</p>
        </div>

        <div class="section">
            <div class="section-title">Clinical Rationale</div>
            <p>{alert.clinical_rationale or 'N/A'}</p>
        </div>

        <div class="time-warning">
            <strong>Time to Potential Harm:</strong> {tth} hours
        </div>

        <div class="section">
            <div class="section-title">Suggested Action</div>
            <div class="action-box">
                {alert.suggested_action or 'N/A'}
            </div>
        </div>

        <div class="section">
            <div class="section-title">Contributing Biomarkers</div>
            <ul>
                {biomarkers_html}
            </ul>
        </div>

        <div class="section">
            <div class="section-title">Routing Information</div>
            <p><strong>Routed to:</strong> {routed_html}</p>
        </div>
    </div>
    <div class="footer">
        This is an automated alert from the HyperCore ML Service.<br>
        Do not reply to this email.
    </div>
</body>
</html>
"""

    def send_alert(
        self,
        alert: "AlertEvent",
        recipients: List[str]
    ) -> EmailResult:
        """
        Send an alert email to multiple recipients.

        Implements retry logic and graceful error handling.
        Does not raise exceptions - returns EmailResult with status.
        """
        if not recipients:
            logger.warning(f"No recipients for alert {alert.event_id}")
            return EmailResult(
                success=False,
                recipients_sent=[],
                recipients_failed=[],
                error_message="No recipients provided"
            )

        recipients_sent: List[str] = []
        recipients_failed: List[str] = []
        last_error: Optional[str] = None
        total_retries = 0

        for recipient in recipients:
            success = False
            retry = 0

            while retry <= self.retry_count and not success:
                try:
                    self._send_single_email(alert, recipient)
                    recipients_sent.append(recipient)
                    success = True
                    logger.info(
                        f"Alert {alert.event_id} sent to {recipient} "
                        f"(attempt {retry + 1})"
                    )
                except Exception as e:
                    retry += 1
                    total_retries += 1
                    last_error = str(e)
                    logger.warning(
                        f"Failed to send alert {alert.event_id} to {recipient}: "
                        f"{e} (attempt {retry}/{self.retry_count + 1})"
                    )
                    if retry <= self.retry_count:
                        time.sleep(self.retry_delay_seconds)

            if not success:
                recipients_failed.append(recipient)
                logger.error(
                    f"Exhausted retries for alert {alert.event_id} to {recipient}"
                )

        overall_success = len(recipients_sent) > 0

        return EmailResult(
            success=overall_success,
            recipients_sent=recipients_sent,
            recipients_failed=recipients_failed,
            error_message=last_error if recipients_failed else None,
            retry_count=total_retries
        )

    def _send_single_email(
        self,
        alert: "AlertEvent",
        recipient: str
    ) -> None:
        """
        Send email to a single recipient.

        Raises exception on failure for retry handling.
        """
        msg = self.format_alert_email(alert, recipient)

        # Create SSL context for secure connections
        context = ssl.create_default_context()

        if self.use_ssl:
            # Implicit SSL (port 465)
            with smtplib.SMTP_SSL(
                self.server,
                self.port,
                context=context,
                timeout=self.timeout_seconds
            ) as smtp:
                self._authenticate_and_send(smtp, msg, recipient)
        else:
            # STARTTLS or plain (ports 587, 25)
            with smtplib.SMTP(
                self.server,
                self.port,
                timeout=self.timeout_seconds
            ) as smtp:
                if self.use_tls:
                    smtp.starttls(context=context)
                self._authenticate_and_send(smtp, msg, recipient)

    def _authenticate_and_send(
        self,
        smtp: smtplib.SMTP,
        msg: MIMEMultipart,
        recipient: str
    ) -> None:
        """Authenticate if credentials provided and send message."""
        if self.username and self.password:
            smtp.login(self.username, self.password)
        smtp.sendmail(self.from_address, [recipient], msg.as_string())


# =============================================================================
# CALLBACK FACTORY
# =============================================================================

def create_email_callback() -> Callable[["AlertEvent", List[str]], bool]:
    """
    Create an email notification callback for the alert routing system.

    This function returns a callback that can be registered with
    router.register_notification_callback("email", callback).

    The callback:
    - Initializes EmailNotifier from environment settings
    - Maps targets to email addresses using recipient settings
    - Sends formatted alert emails
    - Returns True if at least one email was sent successfully
    """
    from .email_config import get_smtp_settings

    # Lazy initialization - notifier created on first call
    _notifier: Optional[EmailNotifier] = None

    def email_callback(alert: "AlertEvent", targets: List[str]) -> bool:
        """
        Email notification callback for alert routing.

        Args:
            alert: The AlertEvent to send
            targets: List of target identifiers (roles/emails)

        Returns:
            True if at least one email sent successfully, False otherwise
        """
        nonlocal _notifier

        settings = get_smtp_settings()

        # Check if email notifications are enabled
        if not settings.enabled:
            logger.info(f"Email notifications disabled, skipping alert {alert.event_id}")
            return True  # Return True to not block other channels

        # Check if SMTP is configured
        if not settings.is_configured():
            logger.warning(
                f"SMTP not configured (server={settings.server}), cannot send alert {alert.event_id}"
            )
            return False

        # Initialize notifier on first use
        if _notifier is None:
            _notifier = EmailNotifier.from_settings()

        # Map targets to email addresses
        recipients = settings.get_recipients_for_targets(targets)

        if not recipients:
            logger.warning(
                f"No email recipients resolved for alert {alert.event_id}, "
                f"targets={targets}"
            )
            return False

        # Send emails
        result = _notifier.send_alert(alert, recipients)

        # Log result
        if result.success:
            logger.info(
                f"Alert {alert.event_id} emailed to {len(result.recipients_sent)} "
                f"recipients (failed: {len(result.recipients_failed)})"
            )
        else:
            logger.error(
                f"Failed to email alert {alert.event_id}: {result.error_message}"
            )

        return result.success

    return email_callback
