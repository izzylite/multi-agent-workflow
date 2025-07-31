"""
Verification Dashboard and Continuous Verification System

Provides a comprehensive dashboard for monitoring vendor tool verification results
and a continuous verification system for automated testing and alerting.
"""

import asyncio
import logging
import json
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import sqlite3
from contextlib import asynccontextmanager

from .vendor_verifier import VendorToolsVerifier, VerificationLevel, VerificationResult, VerificationReport
from .test_scenarios import TestScenarioManager, create_test_scenario_manager


class DashboardMetric(Enum):
    """Dashboard metrics for monitoring."""
    SUCCESS_RATE = "success_rate"
    AVERAGE_DURATION = "average_duration"
    ERROR_COUNT = "error_count"
    LAST_RUN = "last_run"
    TOTAL_TESTS = "total_tests"


@dataclass
class DashboardData:
    """Dashboard data structure."""
    vendor: str
    metric: DashboardMetric
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    condition: str  # "success_rate < 0.8", "error_count > 5", etc.
    severity: str  # "low", "medium", "high", "critical"
    enabled: bool = True
    notification_channels: List[str] = field(default_factory=list)


class VerificationDashboard:
    """Comprehensive dashboard for monitoring verification results."""

    def __init__(self, db_path: str = ".taskmaster/verification_dashboard.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.alert_rules: List[AlertRule] = []
        self._init_database()

    def _init_database(self):
        """Initialize the dashboard database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS verification_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vendor TEXT NOT NULL,
                    test_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration REAL,
                    details TEXT,
                    error_message TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    verification_level TEXT,
                    screenshots TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS dashboard_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vendor TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    value REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    triggered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    resolved_at DATETIME,
                    status TEXT DEFAULT 'active'
                )
            """)

    async def store_verification_result(self, result: VerificationResult, vendor: str, level: VerificationLevel):
        """Store a verification result in the dashboard database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO verification_results 
                (vendor, test_name, status, duration, details, error_message, verification_level, screenshots)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                vendor,
                result.test_name,
                result.status,
                result.duration,
                result.details,
                result.error_message,
                level.value,
                json.dumps(result.screenshots) if result.screenshots else None
            ))

    def get_vendor_metrics(self, vendor: str, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive metrics for a vendor."""
        with sqlite3.connect(self.db_path) as conn:
            # Get recent results
            cursor = conn.execute("""
                SELECT status, duration, timestamp
                FROM verification_results 
                WHERE vendor = ? AND timestamp >= datetime('now', '-{} days')
            """.format(days), (vendor,))
            
            results = cursor.fetchall()
            
            if not results:
                return {
                    "vendor": vendor,
                    "total_tests": 0,
                    "success_rate": 0.0,
                    "average_duration": 0.0,
                    "error_count": 0,
                    "last_run": None
                }

            total_tests = len(results)
            successful_tests = len([r for r in results if r[0] == "PASSED"])
            error_count = len([r for r in results if r[0] in ["ERROR", "FAILED"]])
            durations = [r[1] for r in results if r[1] is not None]
            last_run = max(r[2] for r in results) if results else None

            return {
                "vendor": vendor,
                "total_tests": total_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0.0,
                "average_duration": sum(durations) / len(durations) if durations else 0.0,
                "error_count": error_count,
                "last_run": last_run
            }

    def get_all_vendors_metrics(self, days: int = 7) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all vendors."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT DISTINCT vendor FROM verification_results
            """)
            vendors = [row[0] for row in cursor.fetchall()]
            
            return {
                vendor: self.get_vendor_metrics(vendor, days)
                for vendor in vendors
            }

    def get_recent_results(self, vendor: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent verification results."""
        with sqlite3.connect(self.db_path) as conn:
            if vendor:
                cursor = conn.execute("""
                    SELECT * FROM verification_results 
                    WHERE vendor = ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (vendor, limit))
            else:
                cursor = conn.execute("""
                    SELECT * FROM verification_results 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
            
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule to the dashboard."""
        self.alert_rules.append(rule)
        self.logger.info(f"Added alert rule: {rule.name}")

    def check_alerts(self, metrics: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check for triggered alerts based on current metrics."""
        triggered_alerts = []
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
                
            for vendor, vendor_metrics in metrics.items():
                try:
                    # Simple condition evaluation
                    if rule.condition == "success_rate < 0.8" and vendor_metrics["success_rate"] < 0.8:
                        triggered_alerts.append({
                            "rule_name": rule.name,
                            "severity": rule.severity,
                            "message": f"Low success rate for {vendor}: {vendor_metrics['success_rate']:.2%}",
                            "vendor": vendor,
                            "metric": "success_rate",
                            "value": vendor_metrics["success_rate"]
                        })
                    elif rule.condition == "error_count > 5" and vendor_metrics["error_count"] > 5:
                        triggered_alerts.append({
                            "rule_name": rule.name,
                            "severity": rule.severity,
                            "message": f"High error count for {vendor}: {vendor_metrics['error_count']} errors",
                            "vendor": vendor,
                            "metric": "error_count",
                            "value": vendor_metrics["error_count"]
                        })
                except Exception as e:
                    self.logger.error(f"Error evaluating alert rule {rule.name}: {e}")
        
        return triggered_alerts

    def store_alert(self, alert: Dict[str, Any]):
        """Store a triggered alert in the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO alerts (rule_name, severity, message)
                VALUES (?, ?, ?)
            """, (alert["rule_name"], alert["severity"], alert["message"]))


class ContinuousVerificationSystem:
    """Continuous verification system for automated testing and monitoring."""

    def __init__(self, verifier: VendorToolsVerifier, dashboard: VerificationDashboard):
        self.verifier = verifier
        self.dashboard = dashboard
        self.logger = logging.getLogger(__name__)
        self.scenario_manager = create_test_scenario_manager(verifier)
        self.running = False
        self.verification_interval = 3600  # 1 hour default
        self.vendors_to_monitor = ["tesco", "asda", "costco"]

    async def start_continuous_verification(self, interval_minutes: int = 60):
        """Start the continuous verification system."""
        self.verification_interval = interval_minutes * 60
        self.running = True
        self.logger.info(f"Starting continuous verification with {interval_minutes} minute intervals")
        
        # Add default alert rules
        self._setup_default_alerts()
        
        while self.running:
            try:
                await self._run_verification_cycle()
                await asyncio.sleep(self.verification_interval)
            except Exception as e:
                self.logger.error(f"Error in continuous verification cycle: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying

    async def stop_continuous_verification(self):
        """Stop the continuous verification system."""
        self.running = False
        self.logger.info("Stopping continuous verification")

    def _setup_default_alerts(self):
        """Setup default alert rules."""
        default_rules = [
            AlertRule(
                name="Low Success Rate",
                condition="success_rate < 0.8",
                severity="high",
                notification_channels=["log"]
            ),
            AlertRule(
                name="High Error Count",
                condition="error_count > 5",
                severity="medium",
                notification_channels=["log"]
            ),
            AlertRule(
                name="No Recent Activity",
                condition="last_run is None",
                severity="low",
                notification_channels=["log"]
            )
        ]
        
        for rule in default_rules:
            self.dashboard.add_alert_rule(rule)

    async def _run_verification_cycle(self):
        """Run a single verification cycle for all vendors."""
        self.logger.info("Starting verification cycle")
        
        all_results = []
        
        for vendor in self.vendors_to_monitor:
            try:
                self.logger.info(f"Running verification for {vendor}")
                results = await self.scenario_manager.run_vendor_scenarios(
                    vendor, VerificationLevel.BASIC
                )
                
                # Store results in dashboard
                for result in results:
                    await self.dashboard.store_verification_result(
                        result, vendor, VerificationLevel.BASIC
                    )
                
                all_results.extend(results)
                
            except Exception as e:
                self.logger.error(f"Error running verification for {vendor}: {e}")
        
        # Check for alerts
        metrics = self.dashboard.get_all_vendors_metrics(days=1)
        alerts = self.dashboard.check_alerts(metrics)
        
        for alert in alerts:
            self.dashboard.store_alert(alert)
            self.logger.warning(f"Alert triggered: {alert['message']}")
        
        self.logger.info(f"Verification cycle completed. {len(all_results)} tests run, {len(alerts)} alerts triggered")

    async def run_single_verification(self, vendor: str = None) -> List[VerificationResult]:
        """Run a single verification for specified vendor or all vendors."""
        if vendor:
            vendors = [vendor]
        else:
            vendors = self.vendors_to_monitor
        
        all_results = []
        
        for v in vendors:
            try:
                self.logger.info(f"Running single verification for {v}")
                results = await self.scenario_manager.run_vendor_scenarios(
                    v, VerificationLevel.BASIC
                )
                
                # Store results in dashboard
                for result in results:
                    await self.dashboard.store_verification_result(
                        result, v, VerificationLevel.BASIC
                    )
                
                all_results.extend(results)
                
            except Exception as e:
                self.logger.error(f"Error running verification for {v}: {e}")
        
        return all_results

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get a comprehensive dashboard summary."""
        metrics = self.dashboard.get_all_vendors_metrics(days=7)
        recent_results = self.dashboard.get_recent_results(limit=20)
        
        # Calculate overall statistics
        total_tests = sum(m["total_tests"] for m in metrics.values())
        overall_success_rate = sum(m["success_rate"] * m["total_tests"] for m in metrics.values()) / total_tests if total_tests > 0 else 0.0
        
        return {
            "overview": {
                "total_tests": total_tests,
                "overall_success_rate": overall_success_rate,
                "vendors_monitored": len(metrics),
                "last_verification": max(m["last_run"] for m in metrics.values()) if metrics else None
            },
            "vendor_metrics": metrics,
            "recent_results": recent_results,
            "system_status": {
                "continuous_verification_running": self.running,
                "verification_interval_minutes": self.verification_interval // 60
            }
        }


@asynccontextmanager
async def create_verification_dashboard(verifier: VendorToolsVerifier):
    """Context manager for creating and managing a verification dashboard."""
    dashboard = VerificationDashboard()
    continuous_system = ContinuousVerificationSystem(verifier, dashboard)
    
    try:
        yield dashboard, continuous_system
    finally:
        await continuous_system.stop_continuous_verification()


def create_dashboard_report(dashboard: VerificationDashboard, output_path: str = None) -> str:
    """Create a comprehensive dashboard report."""
    summary = dashboard.get_all_vendors_metrics(days=7)
    recent_results = dashboard.get_recent_results(limit=50)
    
    report_lines = [
        "# Vendor Verification Dashboard Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        f"Total vendors monitored: {len(summary)}",
        f"Total tests in last 7 days: {sum(m['total_tests'] for m in summary.values())}",
        "",
        "## Vendor Performance",
        ""
    ]
    
    for vendor, metrics in summary.items():
        report_lines.extend([
            f"### {vendor.title()}",
            f"- Success Rate: {metrics['success_rate']:.2%}",
            f"- Average Duration: {metrics['average_duration']:.2f}s",
            f"- Error Count: {metrics['error_count']}",
            f"- Last Run: {metrics['last_run'] or 'Never'}",
            ""
        ])
    
    report_lines.extend([
        "## Recent Results",
        ""
    ])
    
    for result in recent_results[:10]:  # Show last 10 results
        status_emoji = "✅" if result["status"] == "PASSED" else "❌"
        report_lines.append(f"{status_emoji} {result['vendor']} - {result['test_name']} ({result['status']})")
    
    report_content = "\n".join(report_lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_content)
    
    return report_content 