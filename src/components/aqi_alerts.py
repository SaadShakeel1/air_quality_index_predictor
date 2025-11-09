"""
AQI Alert System for hazardous air quality conditions
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import json
from pathlib import Path

from src.logger import logging
from src.exception import CustomException

class AQICategory(Enum):
    """AQI categories based on OpenWeather API"""
    GOOD = 1
    FAIR = 2
    MODERATE = 3
    POOR = 4
    VERY_POOR = 5

class AlertLevel(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    aqi_threshold: int
    duration_hours: int
    alert_level: AlertLevel
    message: str
    enabled: bool = True

@dataclass
class Alert:
    """Alert instance"""
    id: str
    rule_name: str
    alert_level: AlertLevel
    aqi_value: float
    timestamp: datetime
    message: str
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None

class AQIALertSystem:
    """AQI Alert System for monitoring and alerting on hazardous air quality"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the AQI Alert System
        
        Args:
            config_path: Path to configuration file
        """
        self.alerts: List[Alert] = []
        self.alert_rules: List[AlertRule] = []
        self.alert_history: List[Alert] = []
        self.config_path = config_path or "config/aqi_alerts_config.json"
        
        self._load_default_rules()
        self._load_config()
    
    def _load_default_rules(self):
        """Load default alert rules"""
        self.alert_rules = [
            AlertRule(
                name="Moderate AQI Alert",
                aqi_threshold=3,
                duration_hours=2,
                alert_level=AlertLevel.LOW,
                message="AQI has reached moderate levels (AQI ≥ 3). Consider limiting outdoor activities if you have respiratory conditions."
            ),
            AlertRule(
                name="Poor AQI Alert",
                aqi_threshold=4,
                duration_hours=1,
                alert_level=AlertLevel.MEDIUM,
                message="AQI has reached poor levels (AQI ≥ 4). Reduce outdoor activities, especially for sensitive groups."
            ),
            AlertRule(
                name="Very Poor AQI Alert",
                aqi_threshold=5,
                duration_hours=1,
                alert_level=AlertLevel.HIGH,
                message="AQI has reached very poor levels (AQI ≥ 5). Avoid outdoor activities. Sensitive groups should stay indoors."
            ),
            AlertRule(
                name="Hazardous AQI Spike",
                aqi_threshold=4.5,
                duration_hours=0.5,
                alert_level=AlertLevel.CRITICAL,
                message="Rapid AQI increase detected! Air quality has deteriorated significantly in a short time."
            )
        ]
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    
                # Update rules from config
                if 'alert_rules' in config:
                    self.alert_rules = []
                    for rule_config in config['alert_rules']:
                        rule = AlertRule(
                            name=rule_config['name'],
                            aqi_threshold=rule_config['aqi_threshold'],
                            duration_hours=rule_config['duration_hours'],
                            alert_level=AlertLevel(rule_config['alert_level']),
                            message=rule_config['message'],
                            enabled=rule_config.get('enabled', True)
                        )
                        self.alert_rules.append(rule)
                
                logging.info(f"Loaded AQI alert configuration from {self.config_path}")
            else:
                logging.info("Using default AQI alert configuration")
                
        except Exception as e:
            logging.error(f"Failed to load AQI alert configuration: {e}")
            # Continue with default rules
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            config = {
                'alert_rules': [
                    {
                        'name': rule.name,
                        'aqi_threshold': rule.aqi_threshold,
                        'duration_hours': rule.duration_hours,
                        'alert_level': rule.alert_level.value,
                        'message': rule.message,
                        'enabled': rule.enabled
                    }
                    for rule in self.alert_rules
                ]
            }
            
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logging.info(f"Saved AQI alert configuration to {self.config_path}")
            
        except Exception as e:
            logging.error(f"Failed to save AQI alert configuration: {e}")
            raise CustomException(e)
    
    def check_alerts(self, current_aqi: float, timestamp: datetime) -> List[Alert]:
        """
        Check if any alert rules are triggered
        
        Args:
            current_aqi: Current AQI value
            timestamp: Current timestamp
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        try:
            for rule in self.alert_rules:
                if not rule.enabled:
                    continue
                
                # Check if AQI threshold is exceeded
                if current_aqi >= rule.aqi_threshold:
                    # Check if this alert is already active
                    existing_alert = self._get_active_alert(rule.name)
                    
                    if not existing_alert:
                        # Create new alert
                        alert_id = f"{rule.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
                        alert = Alert(
                            id=alert_id,
                            rule_name=rule.name,
                            alert_level=rule.alert_level,
                            aqi_value=current_aqi,
                            timestamp=timestamp,
                            message=rule.message
                        )
                        
                        self.alerts.append(alert)
                        triggered_alerts.append(alert)
                        logging.warning(f"AQI Alert triggered: {rule.name} (AQI: {current_aqi:.1f})")
                    
                    # Reset the duration counter for this rule
                    self._reset_rule_duration(rule.name)
                else:
                    # Check if we should resolve any active alerts for this rule
                    self._check_resolve_alert(rule, current_aqi, timestamp)
            
            return triggered_alerts
            
        except Exception as e:
            logging.error(f"Failed to check alerts: {e}")
            raise CustomException(e)
    
    def _get_active_alert(self, rule_name: str) -> Optional[Alert]:
        """Get active alert for a rule"""
        for alert in self.alerts:
            if alert.rule_name == rule_name and not alert.resolved:
                return alert
        return None
    
    def _reset_rule_duration(self, rule_name: str):
        """Reset duration counter for a rule"""
        # Implementation would track when the rule was first triggered
        pass
    
    def _check_resolve_alert(self, rule: AlertRule, current_aqi: float, timestamp: datetime):
        """Check if an alert should be resolved"""
        active_alert = self._get_active_alert(rule.name)
        if active_alert and current_aqi < rule.aqi_threshold:
            # Resolve the alert
            active_alert.resolved = True
            active_alert.resolved_timestamp = timestamp
            self.alert_history.append(active_alert)
            self.alerts.remove(active_alert)
            logging.info(f"AQI Alert resolved: {rule.name} (AQI: {current_aqi:.1f})")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return self.alerts.copy()
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """
        Get alert history for the specified number of hours
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of alerts from the specified time period
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = []
        
        for alert in self.alert_history:
            if alert.timestamp >= cutoff_time:
                recent_alerts.append(alert)
        
        # Also include currently active alerts
        for alert in self.alerts:
            if alert.timestamp >= cutoff_time:
                recent_alerts.append(alert)
        
        return sorted(recent_alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_summary(self) -> Dict:
        """Get summary of current alert status"""
        active_alerts = self.get_active_alerts()
        recent_alerts = self.get_alert_history(24)
        
        summary = {
            'active_alerts_count': len(active_alerts),
            'active_alerts': [
                {
                    'rule_name': alert.rule_name,
                    'alert_level': alert.alert_level.value,
                    'aqi_value': alert.aqi_value,
                    'timestamp': alert.timestamp.isoformat(),
                    'message': alert.message
                }
                for alert in active_alerts
            ],
            'recent_alerts_count': len(recent_alerts),
            'last_alert': recent_alerts[0].timestamp.isoformat() if recent_alerts else None
        }
        
        return summary
    
    def send_email_alert(self, alert: Alert, recipients: List[str], smtp_config: Dict):
        """
        Send email alert notification
        
        Args:
            alert: Alert to send
            recipients: List of email recipients
            smtp_config: SMTP configuration
        """
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = smtp_config['sender']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"AQI Alert: {alert.rule_name}"
            
            # Create email body
            body = f"""
AQI Alert Notification

Alert: {alert.rule_name}
Level: {alert.alert_level.value.upper()}
Current AQI: {alert.aqi_value:.1f}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

{alert.message}

This is an automated alert from the AQI Prediction System.
"""
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_config['host'], smtp_config['port']) as server:
                server.starttls()
                server.login(smtp_config['username'], smtp_config['password'])
                server.send_message(msg)
            
            logging.info(f"Email alert sent to {recipients}")
            
        except Exception as e:
            logging.error(f"Failed to send email alert: {e}")
    
    def save_alert_log(self, log_path: str = "logs/aqi_alerts.json"):
        """Save alert history to file"""
        try:
            log_data = {
                'alerts': [
                    {
                        'id': alert.id,
                        'rule_name': alert.rule_name,
                        'alert_level': alert.alert_level.value,
                        'aqi_value': alert.aqi_value,
                        'timestamp': alert.timestamp.isoformat(),
                        'message': alert.message,
                        'resolved': alert.resolved,
                        'resolved_timestamp': alert.resolved_timestamp.isoformat() if alert.resolved_timestamp else None
                    }
                    for alert in self.alert_history + self.alerts
                ]
            }
            
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            logging.info(f"Alert log saved to {log_path}")
            
        except Exception as e:
            logging.error(f"Failed to save alert log: {e}")

# Convenience function for easy integration
def check_aqi_alerts(current_aqi: float, timestamp: Optional[datetime] = None) -> List[Alert]:
    """
    Check AQI alerts for a given AQI value
    
    Args:
        current_aqi: Current AQI value
        timestamp: Current timestamp (uses now() if None)
        
    Returns:
        List of triggered alerts
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    alert_system = AQIALertSystem()
    return alert_system.check_alerts(current_aqi, timestamp)