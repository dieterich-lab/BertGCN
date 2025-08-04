#!/usr/bin/env python3
"""
Notifications Module for BertGCN

Provides notification utilities for training alerts, model deployment, and system events.
"""

import logging
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NotificationConfig:
    """Configuration for notifications."""

    email_enabled: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_recipients: List[str] = None
    slack_enabled: bool = False
    slack_webhook_url: str = ""


class NotificationManager:
    """Manage various types of notifications."""

    def __init__(self, config: Optional[NotificationConfig] = None):
        """
        Initialize notification manager.

        Args:
            config: Notification configuration
        """
        self.config = config or NotificationConfig()

    def send_email(
        self, subject: str, body: str, recipients: Optional[List[str]] = None
    ) -> bool:
        """
        Send email notification.

        Args:
            subject: Email subject
            body: Email body (HTML or plain text)
            recipients: List of email recipients

        Returns:
            True if successful
        """
        if not self.config.email_enabled:
            logger.debug("Email notifications disabled")
            return False

        recipients = recipients or self.config.email_recipients
        if not recipients:
            logger.warning("No email recipients configured")
            return False

        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.config.email_username
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = subject

            msg.attach(MimeText(body, "html"))

            # Send email
            server = smtplib.SMTP(
                self.config.email_smtp_server, self.config.email_smtp_port
            )
            server.starttls()
            server.login(self.config.email_username, self.config.email_password)

            text = msg.as_string()
            server.sendmail(self.config.email_username, recipients, text)
            server.quit()

            logger.info(f"Email sent successfully to {recipients}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False

    def send_slack_message(self, message: str, channel: Optional[str] = None) -> bool:
        """
        Send Slack notification.

        Args:
            message: Message to send
            channel: Slack channel (optional)

        Returns:
            True if successful
        """
        if not self.config.slack_enabled:
            logger.debug("Slack notifications disabled")
            return False

        try:
            import requests

            payload = {
                "text": message,
                "username": "BertGCN Bot",
                "icon_emoji": ":robot_face:",
            }

            if channel:
                payload["channel"] = channel

            response = requests.post(
                self.config.slack_webhook_url, json=payload, timeout=10
            )

            if response.status_code == 200:
                logger.info("Slack message sent successfully")
                return True
            else:
                logger.error(f"Failed to send Slack message: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Failed to send Slack message: {str(e)}")
            return False


# Global notification manager
notification_manager = NotificationManager()


def send_training_alert(
    status: str,
    metrics: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Send training completion/failure alert.

    Args:
        status: Training status (success, failure, warning)
        metrics: Training metrics
        error: Error message if status is failure
        config: Training configuration
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if status == "success":
            subject = "🎉 BertGCN Training Completed Successfully"

            # Create HTML email body
            body = f"""
            <html>
            <body>
                <h2>🎉 Training Completed Successfully</h2>
                <p><strong>Timestamp:</strong> {timestamp}</p>
                
                <h3>📊 Final Metrics:</h3>
                <ul>
            """

            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, float):
                        body += f"<li><strong>{key}:</strong> {value:.4f}</li>"
                    else:
                        body += f"<li><strong>{key}:</strong> {value}</li>"

            body += """
                </ul>
                
                <h3>⚙️ Configuration:</h3>
                <ul>
            """

            if config:
                for key, value in config.items():
                    body += f"<li><strong>{key}:</strong> {value}</li>"

            body += """
                </ul>
                
                <p>The model is ready for validation and deployment.</p>
            </body>
            </html>
            """

            # Slack message
            slack_message = f"""
🎉 *BertGCN Training Completed Successfully*

⏰ *Time:* {timestamp}

📊 *Key Metrics:*
{chr(10).join([f"• {k}: {v:.4f}" if isinstance(v, float) else f"• {k}: {v}" for k, v in (metrics or {}).items()])}

✅ Model ready for deployment!
            """

        elif status == "failure":
            subject = "❌ BertGCN Training Failed"

            body = f"""
            <html>
            <body>
                <h2>❌ Training Failed</h2>
                <p><strong>Timestamp:</strong> {timestamp}</p>
                
                <h3>🚨 Error Details:</h3>
                <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px;">
{error or "Unknown error"}
                </pre>
                
                <h3>⚙️ Configuration:</h3>
                <ul>
            """

            if config:
                for key, value in config.items():
                    body += f"<li><strong>{key}:</strong> {value}</li>"

            body += """
                </ul>
                
                <p>Please check the logs for more details and retry the training.</p>
            </body>
            </html>
            """

            # Slack message
            slack_message = f"""
❌ *BertGCN Training Failed*

⏰ *Time:* {timestamp}

🚨 *Error:*
```
{error or "Unknown error"}
```

🔍 Please check logs for details.
            """

        else:  # warning or other status
            subject = f"⚠️ BertGCN Training Alert - {status.title()}"
            body = f"""
            <html>
            <body>
                <h2>⚠️ Training Alert</h2>
                <p><strong>Status:</strong> {status}</p>
                <p><strong>Timestamp:</strong> {timestamp}</p>
                
                <p>Please check the training logs for more information.</p>
            </body>
            </html>
            """

            slack_message = (
                f"⚠️ *BertGCN Training Alert*\n\n*Status:* {status}\n*Time:* {timestamp}"
            )

        # Send notifications
        notification_manager.send_email(subject, body)
        notification_manager.send_slack_message(slack_message)

        logger.info(f"Training alert sent: {status}")

    except Exception as e:
        logger.error(f"Failed to send training alert: {str(e)}")


def send_deployment_alert(
    model_name: str, version: str, stage: str, metrics: Optional[Dict[str, Any]] = None
) -> None:
    """
    Send model deployment alert.

    Args:
        model_name: Name of the deployed model
        version: Model version
        stage: Deployment stage (Staging, Production)
        metrics: Model metrics
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        subject = f"🚀 Model Deployed: {model_name} v{version} to {stage}"

        body = f"""
        <html>
        <body>
            <h2>🚀 Model Deployment Notification</h2>
            <p><strong>Model:</strong> {model_name}</p>
            <p><strong>Version:</strong> {version}</p>
            <p><strong>Stage:</strong> {stage}</p>
            <p><strong>Timestamp:</strong> {timestamp}</p>
            
            <h3>📊 Model Metrics:</h3>
            <ul>
        """

        if metrics:
            for key, value in metrics.items():
                if isinstance(value, float):
                    body += f"<li><strong>{key}:</strong> {value:.4f}</li>"
                else:
                    body += f"<li><strong>{key}:</strong> {value}</li>"

        body += """
            </ul>
            
            <p>The model is now available for inference.</p>
        </body>
        </html>
        """

        slack_message = f"""
🚀 *Model Deployed Successfully*

📦 *Model:* {model_name} v{version}
🎯 *Stage:* {stage}
⏰ *Time:* {timestamp}

📊 *Metrics:*
{chr(10).join([f"• {k}: {v:.4f}" if isinstance(v, float) else f"• {k}: {v}" for k, v in (metrics or {}).items()])}

✅ Ready for inference!
        """

        notification_manager.send_email(subject, body)
        notification_manager.send_slack_message(slack_message)

        logger.info(f"Deployment alert sent for {model_name} v{version}")

    except Exception as e:
        logger.error(f"Failed to send deployment alert: {str(e)}")


def send_performance_alert(
    alert_type: str, message: str, metrics: Optional[Dict[str, Any]] = None
) -> None:
    """
    Send performance-related alert.

    Args:
        alert_type: Type of alert (degradation, anomaly, etc.)
        message: Alert message
        metrics: Related metrics
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        subject = f"⚠️ BertGCN Performance Alert: {alert_type}"

        body = f"""
        <html>
        <body>
            <h2>⚠️ Performance Alert</h2>
            <p><strong>Alert Type:</strong> {alert_type}</p>
            <p><strong>Message:</strong> {message}</p>
            <p><strong>Timestamp:</strong> {timestamp}</p>
            
            <h3>📊 Related Metrics:</h3>
            <ul>
        """

        if metrics:
            for key, value in metrics.items():
                if isinstance(value, float):
                    body += f"<li><strong>{key}:</strong> {value:.4f}</li>"
                else:
                    body += f"<li><strong>{key}:</strong> {value}</li>"

        body += """
            </ul>
            
            <p>Please investigate and take appropriate action.</p>
        </body>
        </html>
        """

        slack_message = f"""
⚠️ *Performance Alert*

🔥 *Type:* {alert_type}
💬 *Message:* {message}
⏰ *Time:* {timestamp}

📊 *Metrics:*
{chr(10).join([f"• {k}: {v:.4f}" if isinstance(v, float) else f"• {k}: {v}" for k, v in (metrics or {}).items()])}

🔍 Investigation required!
        """

        notification_manager.send_email(subject, body)
        notification_manager.send_slack_message(slack_message)

        logger.warning(f"Performance alert sent: {alert_type} - {message}")

    except Exception as e:
        logger.error(f"Failed to send performance alert: {str(e)}")


def configure_notifications(
    email_enabled: bool = False,
    email_config: Optional[Dict[str, Any]] = None,
    slack_enabled: bool = False,
    slack_webhook_url: Optional[str] = None,
) -> None:
    """
    Configure global notification settings.

    Args:
        email_enabled: Enable email notifications
        email_config: Email configuration dictionary
        slack_enabled: Enable Slack notifications
        slack_webhook_url: Slack webhook URL
    """
    global notification_manager

    config = NotificationConfig()
    config.email_enabled = email_enabled
    config.slack_enabled = slack_enabled

    if email_config:
        config.email_smtp_server = email_config.get("smtp_server", "smtp.gmail.com")
        config.email_smtp_port = email_config.get("smtp_port", 587)
        config.email_username = email_config.get("username", "")
        config.email_password = email_config.get("password", "")
        config.email_recipients = email_config.get("recipients", [])

    if slack_webhook_url:
        config.slack_webhook_url = slack_webhook_url

    notification_manager = NotificationManager(config)
    logger.info("Notifications configured successfully")
