# Enhanced Health Monitor with Direct Reporting & Data Export

-----

## Add to `system_health_monitor.py`

Insert these methods into the `SystemHealthMonitor` class:

```python
class SystemHealthMonitor:
    def __init__(
        self, 
        output_dir: str = "validation_results/health",
        cfg: Optional[HealthConfig] = None,
        verbose: bool = False  # ← NEW: toggle for live readouts
    ):
        # ... existing init code ...
        self.verbose = verbose  # Control console output
        
        # NEW: Real-time monitoring flag
        self._live_monitoring = False
    
    # ========================================
    # NEW: Direct Reporting Methods
    # ========================================
    
    def enable_live_monitoring(self) -> None:
        """
        Enable live console readouts during session.
        Prints vital signs after each turn.
        """
        self._live_monitoring = True
        print("[Health Monitor] Live monitoring ENABLED")
    
    def disable_live_monitoring(self) -> None:
        """Disable live console readouts."""
        self._live_monitoring = False
        print("[Health Monitor] Live monitoring DISABLED")
    
    def generate_instant_report(self) -> Dict[str, Any]:
        """
        Generate health report RIGHT NOW without ending session.
        
        Use this to check system health mid-session:
        >>> monitor.generate_instant_report()
        
        Returns:
            Dict with current vital signs, recent events, and warnings
        """
        current_time = time.time()
        session_duration = (current_time - self._session_start) if self._session_start else 0
        
        # Compute current trends
        recent_events = [e for e in self.events if e.severity in ("warn", "error", "critical")]
        
        report = {
            "report_type": "instant_snapshot",
            "session_id": self.session_id,
            "condition": self.condition,
            "timestamp": int(current_time),
            "elapsed_seconds": round(session_duration, 2),
            
            "current_vital_signs": {
                "turns_processed": len(self.turn_fidelity),
                "current_fidelity": round(self.turn_fidelity[-1], 4) if self.turn_fidelity else None,
                "mean_fidelity": round(statistics.mean(self.turn_fidelity), 4) if self.turn_fidelity else None,
                "fidelity_trend": self._compute_trend(self.turn_fidelity),
                "recent_error_signal": round(self.turn_error_signals[-1], 4) if self.turn_error_signals else None,
                "lyapunov_convergence_rate": self._compute_convergence_rate(),
                "intervention_effectiveness_rate": self._compute_effectiveness_rate(),
            },
            
            "event_summary": {
                "total_events": len(self.events),
                "critical": sum(e.severity == "critical" for e in self.events),
                "errors": sum(e.severity == "error" for e in self.events),
                "warnings": sum(e.severity == "warn" for e in self.events),
            },
            
            "recent_issues": [
                {
                    "severity": e.severity,
                    "code": e.code,
                    "message": e.message,
                    "turn": e.turn
                }
                for e in recent_events[-5:]  # Last 5 issues
            ],
            
            "suggested_actions": self._generate_suggestions()
        }
        
        # Print to console if verbose
        if self.verbose:
            self._print_instant_summary(report)
        
        return report
    
    def export_diagnostic_data(
        self, 
        format: str = "json",
        output_path: Optional[str] = None
    ) -> str:
        """
        Export health data in various formats for analysis.
        
        Args:
            format: "json" | "csv" | "summary_text"
            output_path: Where to save (auto-generated if None)
            
        Returns:
            Path to exported file
        
        Example:
            >>> monitor.export_diagnostic_data(format="csv")
            'validation_results/health/diagnostics_session_123.csv'
        """
        if format == "json":
            return self._export_json(output_path)
        elif format == "csv":
            return self._export_csv(output_path)
        elif format == "summary_text":
            return self._export_text_summary(output_path)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'json', 'csv', or 'summary_text'")
    
    def plot_vital_signs(
        self,
        output_path: Optional[str] = None,
        show: bool = True
    ) -> str:
        """
        Generate diagnostic plots of system health over time.
        
        Creates multi-panel figure:
        - Fidelity trajectory
        - Error signal evolution  
        - Lyapunov convergence
        - Intervention timeline
        
        Requires: matplotlib
        
        Args:
            output_path: Where to save plot (auto-generated if None)
            show: Display plot interactively
            
        Returns:
            Path to saved plot file
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError(
                "matplotlib required for plotting. Install with: pip install matplotlib"
            )
        
        if not output_path:
            output_path = self.output_dir / f"health_plot_{self.session_id or 'unknown'}.png"
        else:
            output_path = Path(output_path)
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"TELOS Health Monitor - {self.session_id} ({self.condition})", 
                     fontsize=14, fontweight='bold')
        
        turns = list(range(1, len(self.turn_fidelity) + 1))
        
        # ---- Plot 1: Fidelity Trajectory ----
        ax1 = axes[0, 0]
        if self.turn_fidelity:
            ax1.plot(turns, self.turn_fidelity, 'b-', linewidth=2, label='Fidelity')
            ax1.axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='Good (0.8)')
            ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Warning (0.5)')
            ax1.set_xlabel('Turn')
            ax1.set_ylabel('Telic Fidelity')
            ax1.set_title('Fidelity Trajectory')
            ax1.set_ylim([0, 1.05])
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # ---- Plot 2: Error Signal Evolution ----
        ax2 = axes[0, 1]
        if self.turn_error_signals:
            error_turns = list(range(1, len(self.turn_error_signals) + 1))
            ax2.plot(error_turns, self.turn_error_signals, 'r-', linewidth=2)
            ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='ε_min threshold')
            ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='ε_max threshold')
            ax2.set_xlabel('Turn')
            ax2.set_ylabel('Error Signal')
            ax2.set_title('Distance from Attractor')
            ax2.set_ylim([0, 1.05])
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # ---- Plot 3: Lyapunov Convergence ----
        ax3 = axes[1, 0]
        if self.turn_lyapunov:
            lyap_turns = list(range(1, len(self.turn_lyapunov) + 1))
            ax3.plot(lyap_turns, self.turn_lyapunov, 'purple', linewidth=2)
            
            # Color background: green when decreasing, red when increasing
            for i in range(1, len(self.turn_lyapunov)):
                if self.turn_lyapunov[i] < self.turn_lyapunov[i-1]:
                    ax3.axvspan(i, i+1, alpha=0.1, color='green')
                else:
                    ax3.axvspan(i, i+1, alpha=0.1, color='red')
            
            ax3.set_xlabel('Turn')
            ax3.set_ylabel('V(x) = ||x - â||²')
            ax3.set_title('Lyapunov Function (Stability)')
            ax3.grid(True, alpha=0.3)
            
            # Add legend for background colors
            green_patch = mpatches.Patch(color='green', alpha=0.3, label='Converging (ΔV<0)')
            red_patch = mpatches.Patch(color='red', alpha=0.3, label='Diverging (ΔV>0)')
            ax3.legend(handles=[green_patch, red_patch], loc='upper right')
        
        # ---- Plot 4: Event Timeline ----
        ax4 = axes[1, 1]
        
        # Count events by turn and severity
        event_turns = {}
        for event in self.events:
            if event.turn is not None:
                if event.turn not in event_turns:
                    event_turns[event.turn] = {"error": 0, "warn": 0, "critical": 0}
                if event.severity in event_turns[event.turn]:
                    event_turns[event.turn][event.severity] += 1
        
        if event_turns:
            event_turn_list = sorted(event_turns.keys())
            criticals = [event_turns[t]["critical"] for t in event_turn_list]
            errors = [event_turns[t]["error"] for t in event_turn_list]
            warns = [event_turns[t]["warn"] for t in event_turn_list]
            
            ax4.bar(event_turn_list, criticals, color='darkred', label='Critical', width=0.8)
            ax4.bar(event_turn_list, errors, bottom=criticals, color='red', label='Error', width=0.8)
            ax4.bar(event_turn_list, warns, 
                   bottom=[c+e for c,e in zip(criticals, errors)], 
                   color='orange', label='Warning', width=0.8)
            
            ax4.set_xlabel('Turn')
            ax4.set_ylabel('Event Count')
            ax4.set_title('Health Events Timeline')
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
        else:
            ax4.text(0.5, 0.5, 'No events recorded', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Health Events Timeline')
        
        plt.tight_layout()
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(output_path)
    
    # ========================================
    # Private Export Helpers
    # ========================================
    
    def _export_json(self, output_path: Optional[str] = None) -> str:
        """Export complete diagnostic data as JSON."""
        if not output_path:
            output_path = self.output_dir / f"diagnostics_{self.session_id or 'unknown'}.json"
        else:
            output_path = Path(output_path)
        
        data = {
            "session_id": self.session_id,
            "condition": self.condition,
            "export_timestamp": int(time.time()),
            
            "time_series": {
                "fidelity": self.turn_fidelity,
                "error_signals": self.turn_error_signals,
                "lyapunov": self.turn_lyapunov,
                "latency_ms": self.turn_latencies_ms,
            },
            
            "events": [e.__dict__ for e in self.events],
            
            "statistics": {
                "mean_fidelity": statistics.mean(self.turn_fidelity) if self.turn_fidelity else None,
                "std_fidelity": statistics.pstdev(self.turn_fidelity) if len(self.turn_fidelity) > 1 else None,
                "mean_error": statistics.mean(self.turn_error_signals) if self.turn_error_signals else None,
                "convergence_rate": self._compute_convergence_rate(),
                "effectiveness_rate": self._compute_effectiveness_rate(),
            }
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return str(output_path)
    
    def _export_csv(self, output_path: Optional[str] = None) -> str:
        """Export turn-by-turn data as CSV."""
        import csv
        
        if not output_path:
            output_path = self.output_dir / f"diagnostics_{self.session_id or 'unknown'}.csv"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'turn', 'fidelity', 'error_signal', 'lyapunov', 
                'latency_ms', 'events_this_turn', 'event_severity'
            ])
            
            # Data rows
            max_turns = max(
                len(self.turn_fidelity),
                len(self.turn_error_signals),
                len(self.turn_lyapunov),
                len(self.turn_latencies_ms)
            )
            
            for turn in range(1, max_turns + 1):
                idx = turn - 1
                
                fidelity = self.turn_fidelity[idx] if idx < len(self.turn_fidelity) else None
                error = self.turn_error_signals[idx] if idx < len(self.turn_error_signals) else None
                lyap = self.turn_lyapunov[idx] if idx < len(self.turn_lyapunov) else None
                latency = self.turn_latencies_ms[idx] if idx < len(self.turn_latencies_ms) else None
                
                # Events at this turn
                turn_events = [e for e in self.events if e.turn == turn]
                event_count = len(turn_events)
                severities = ','.join(e.severity for e in turn_events) if turn_events else ''
                
                writer.writerow([
                    turn,
                    round(fidelity, 4) if fidelity is not None else '',
                    round(error, 4) if error is not None else '',
                    round(lyap, 4) if lyap is not None else '',
                    round(latency, 2) if latency is not None else '',
                    event_count,
                    severities
                ])
        
        return str(output_path)
    
    def _export_text_summary(self, output_path: Optional[str] = None) -> str:
        """Export human-readable text summary."""
        if not output_path:
            output_path = self.output_dir / f"summary_{self.session_id or 'unknown'}.txt"
        else:
            output_path = Path(output_path)
        
        lines = []
        lines.append("="*70)
        lines.append(f"TELOS HEALTH MONITOR SUMMARY")
        lines.append("="*70)
        lines.append(f"Session: {self.session_id}")
        lines.append(f"Condition: {self.condition}")
        lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Vital signs
        lines.append("VITAL SIGNS")
        lines.append("-"*70)
        if self.turn_fidelity:
            lines.append(f"  Turns Processed: {len(self.turn_fidelity)}")
            lines.append(f"  Mean Fidelity: {statistics.mean(self.turn_fidelity):.4f}")
            lines.append(f"  Fidelity Trend: {self._compute_trend(self.turn_fidelity)}")
        
        conv_rate = self._compute_convergence_rate()
        if conv_rate is not None:
            lines.append(f"  Lyapunov Convergence: {conv_rate:.1%}")
        
        eff_rate = self._compute_effectiveness_rate()
        if eff_rate is not None:
            lines.append(f"  Intervention Effectiveness: {eff_rate:.1%}")
        
        lines.append("")
        
        # Event summary
        lines.append("EVENTS")
        lines.append("-"*70)
        lines.append(f"  Critical: {sum(e.severity == 'critical' for e in self.events)}")
        lines.append(f"  Errors: {sum(e.severity == 'error' for e in self.events)}")
        lines.append(f"  Warnings: {sum(e.severity == 'warn' for e in self.events)}")
        lines.append(f"  Info: {sum(e.severity == 'info' for e in self.events)}")
        lines.append("")
        
        # Critical/error details
        critical_events = [e for e in self.events if e.severity in ('critical', 'error')]
        if critical_events:
            lines.append("CRITICAL ISSUES")
            lines.append("-"*70)
            for event in critical_events:
                turn_info = f" [T{event.turn}]" if event.turn else ""
                lines.append(f"  [{event.code}]{turn_info} {event.message}")
            lines.append("")
        
        # Suggestions
        suggestions = self._generate_suggestions()
        if suggestions:
            lines.append("SUGGESTED FIXES")
            lines.append("-"*70)
            for i, fix in enumerate(suggestions, 1):
                lines.append(f"  {i}. {fix['issue']}")
                lines.append(f"     → {fix['fix']}")
                lines.append(f"     Priority: {fix['priority']}")
                lines.append("")
        
        lines.append("="*70)
        
        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('\n'.join(lines), encoding='utf-8')
        
        return str(output_path)
    
    def _print_instant_summary(self, report: Dict[str, Any]) -> None:
        """Print instant report to console."""
        print("\n" + "="*60)
        print(f"📊 INSTANT HEALTH CHECK - {report['session_id']}")
        print("="*60)
        
        vital = report['current_vital_signs']
        events = report['event_summary']
        
        print(f"\nTurns: {vital['turns_processed']} | Elapsed: {report['elapsed_seconds']:.1f}s")
        print(f"Current Fidelity: {vital['current_fidelity']:.3f} ({vital['fidelity_trend']})")
        print(f"Events: {events['critical']} critical, {events['errors']} errors, {events['warnings']} warnings")
        
        if report['recent_issues']:
            print(f"\nRecent Issues:")
            for issue in report['recent_issues'][:3]:
                print(f"  [{issue['code']}] {issue['message']}")
        
        print("="*60 + "\n")
    
    # ========================================
    # Modified on_turn to support live monitoring
    # ========================================
    
    def on_turn(self, turn_number: int, turn_record: Dict[str, Any], raw_latency_ms: Optional[float] = None) -> None:
        """Process health checks for a single turn."""
        # ... existing on_turn code ...
        
        # NEW: Live monitoring output
        if self._live_monitoring:
            metrics = turn_record.get("metrics", {})
            action = turn_record.get("governance_action", "none")
            fidelity = metrics.get("telic_fidelity")
            error = metrics.get("error_signal", 0)
            
            print(f"[T{turn_number}] F={fidelity:.3f} | E={error:.3f} | Action={action}")

-----

## Usage Examples

### 1. Enable Live Monitoring During Session

```python
# In your test script or interactive session
steward = UnifiedGovernanceSteward(...)
steward.health.enable_live_monitoring()

steward.start_session()
# Now prints: [T1] F=0.873 | E=0.234 | Action=none
# After each turn automatically

for turn in conversation:
    steward.process_turn(...)
    # Prints live readout

steward.health.disable_live_monitoring()

-----

### 2. Mid-Session Health Check

```python
# During a long session, check health without stopping
steward.start_session()

for i, turn in enumerate(conversation):
    steward.process_turn(...)
    
    # Check health every 10 turns
    if i % 10 == 0:
        report = steward.health.generate_instant_report()
        
        # Check if critical issues detected
        if report['event_summary']['critical'] > 0:
            print("⚠️ Critical issues detected - stopping session")
            break

-----

### 3. Export Data After Session

```python
# After Internal Test 0 completes
steward.end_session()

# Export in multiple formats
steward.health.export_diagnostic_data(format="json")
# → validation_results/health/diagnostics_session_123.json

steward.health.export_diagnostic_data(format="csv")
# → validation_results/health/diagnostics_session_123.csv

steward.health.export_diagnostic_data(format="summary_text")
# → validation_results/health/summary_session_123.txt

-----

### 4. Generate Diagnostic Plots

```python
# Visualize health over time
steward.health.plot_vital_signs(
    output_path="results/health_visualization.png",
    show=True  # Display plot window
)

# Creates 4-panel plot:
# - Fidelity trajectory
# - Error signals
# - Lyapunov convergence (green/red background)
# - Event timeline (stacked bar chart)

-----

### 5. Standalone Health Analysis

```python
from telos_purpose.validation.system_health_monitor import SystemHealthMonitor

# Initialize standalone monitor
monitor = SystemHealthMonitor(verbose=True)

# Run checks
monitor.startup_checks(
    session_id="diagnostic_run",
    condition="telos",
    components={...}
)

# Process turns
for turn_data in session_data:
    monitor.on_turn(turn_data['turn'], turn_data)

# Generate all outputs
monitor.finalize_report()
monitor.export_diagnostic_data(format="json")
monitor.export_diagnostic_data(format="csv")
monitor.plot_vital_signs()

-----

## Added Capabilities Summary

|Method                          |Purpose                        |Output                         |
|--------------------------------|-------------------------------|-------------------------------|
|`enable_live_monitoring()`      |Toggle real-time console output|Prints vital signs each turn   |
|`generate_instant_report()`     |Check health mid-session       |Returns dict, optionally prints|
|`export_diagnostic_data(format)`|Save data for analysis         |JSON/CSV/TXT file              |
|`plot_vital_signs()`            |Visual diagnostics             |4-panel matplotlib figure      |

-----

**For your son:** "We added toggle switches and export buttons to the health monitor. You can turn on live updates (like watching a heart rate monitor), check health mid-session without stopping, and export data as graphs or spreadsheets for analysis."

This gives you full control: silent background monitoring OR verbose real-time readouts OR post-session data export - whatever the situation needs.​​​​​​​​​​​​​​​​