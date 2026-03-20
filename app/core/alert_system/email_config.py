"""
Email Configuration for Alert System.

Uses pydantic-settings for environment variable loading.
"""
import os
import logging
from typing import Optional, List
from dataclasses import dataclass

logger = logging.getLogger("hypercore.email_config")


@dataclass
class SMTPSettings:
    """SMTP configuration loaded from environment variables."""

    # SMTP server settings
    server: str = "localhost"
    port: int = 587
    username: Optional[str] = None
    password: Optional[str] = None

    # Sender info
    from_address: str = "hypercore-alerts@localhost"
    from_name: str = "HyperCore Alert System"

    # Security
    use_tls: bool = True
    use_ssl: bool = False

    # Operational
    enabled: bool = True
    timeout_seconds: int = 30
    retry_count: int = 3
    retry_delay_seconds: float = 1.0

    # Recipient mapping
    default_recipients: str = ""

    @classmethod
    def from_env(cls) -> "SMTPSettings":
        """Load settings from environment variables."""
        def get_bool(key: str, default: bool) -> bool:
            val = os.environ.get(key, "").lower()
            if val in ("true", "1", "yes"):
                return True
            elif val in ("false", "0", "no"):
                return False
            return default

        return cls(
            server=os.environ.get("SMTP_SERVER", "localhost"),
            port=int(os.environ.get("SMTP_PORT", "587")),
            username=os.environ.get("SMTP_USERNAME") or os.environ.get("SMTP_USER"),
            password=os.environ.get("SMTP_PASSWORD") or os.environ.get("SMTP_PASS"),
            from_address=os.environ.get("SMTP_FROM_ADDRESS", "hypercore-alerts@localhost"),
            from_name=os.environ.get("SMTP_FROM_NAME", "HyperCore Alert System"),
            use_tls=get_bool("SMTP_USE_TLS", True),
            use_ssl=get_bool("SMTP_USE_SSL", False),
            enabled=get_bool("SMTP_ENABLED", True),
            timeout_seconds=int(os.environ.get("SMTP_TIMEOUT_SECONDS", "30")),
            retry_count=int(os.environ.get("SMTP_RETRY_COUNT", "3")),
            retry_delay_seconds=float(os.environ.get("SMTP_RETRY_DELAY_SECONDS", "1.0")),
            default_recipients=os.environ.get("ALERT_DEFAULT_RECIPIENTS", ""),
        )

    def is_configured(self) -> bool:
        """Check if SMTP is minimally configured for sending."""
        return bool(self.server and self.server != "localhost" and self.from_address)

    def get_recipients_for_targets(self, targets: List[str]) -> List[str]:
        """
        Map routing targets to email addresses.

        Targets can be:
        - Email addresses (contain @)
        - Role names (mapped via ALERT_<ROLE>_RECIPIENTS env vars)
        - Falls back to ALERT_DEFAULT_RECIPIENTS
        """
        recipients = []

        for target in targets:
            # If it looks like an email, use directly
            if "@" in target:
                recipients.append(target)
                continue

            # Try role-specific env var (e.g., ALERT_NEPHROLOGY_RECIPIENTS)
            env_key = f"ALERT_{target.upper().replace('-', '_')}_RECIPIENTS"
            role_recipients = os.environ.get(env_key, "")
            if role_recipients:
                recipients.extend([r.strip() for r in role_recipients.split(",") if r.strip()])

        # If no recipients found, use defaults
        if not recipients and self.default_recipients:
            recipients = [r.strip() for r in self.default_recipients.split(",") if r.strip()]

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for r in recipients:
            if r not in seen:
                seen.add(r)
                unique.append(r)

        return unique


# Singleton instance
_smtp_settings: Optional[SMTPSettings] = None


def get_smtp_settings() -> SMTPSettings:
    """Get singleton SMTP settings instance."""
    global _smtp_settings
    if _smtp_settings is None:
        _smtp_settings = SMTPSettings.from_env()
        logger.info(f"SMTP settings loaded: server={_smtp_settings.server}:{_smtp_settings.port}, enabled={_smtp_settings.enabled}")
    return _smtp_settings


def reset_smtp_settings() -> None:
    """Reset settings (useful for testing)."""
    global _smtp_settings
    _smtp_settings = None
