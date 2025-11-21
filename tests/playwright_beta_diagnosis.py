#!/usr/bin/env python3
"""
Playwright-based diagnostic test for TELOS BETA.
Automates user interaction and captures what's actually happening vs what should happen.
"""

from playwright.sync_api import sync_playwright, expect
import json
import time
from datetime import datetime
from pathlib import Path

class BETADiagnostic:
    """Automated diagnostic testing for TELOS BETA using Playwright."""

    def __init__(self, base_url="http://localhost:8501"):
        self.base_url = base_url
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "failures": [],
            "screenshots": []
        }

    def run_diagnostic(self):
        """Run comprehensive diagnostic on BETA."""
        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch(headless=False)  # headless=False to watch
            page = browser.new_page()

            print("🔍 Starting BETA Diagnostic...")
            print(f"📍 Navigating to {self.base_url}")

            # Navigate to BETA
            page.goto(self.base_url)
            page.wait_for_load_state("networkidle")
            time.sleep(2)  # Wait for Streamlit to fully render

            # TEST 1: Check if consent screen appears
            self.test_consent_screen(page)

            # Accept consent if present
            self.accept_consent(page)

            # TEST 2: Navigate to BETA tab
            self.test_beta_tab_navigation(page)

            # TEST 3: Send initial message to establish PA
            self.test_pa_establishment(page)

            # TEST 4: Check if fidelity updates
            self.test_fidelity_calculation(page)

            # TEST 5: Test drift scenario (Thanksgiving conversation)
            self.test_drift_detection(page)

            # TEST 6: Check PA display
            self.test_pa_display(page)

            # TEST 7: Check for fake data
            self.test_fake_data_detection(page)

            # Generate report
            self.generate_report()

            browser.close()

    def test_consent_screen(self, page):
        """Test if consent screen appears."""
        test_name = "Consent Screen Presence"
        print(f"\n🧪 TEST: {test_name}")

        try:
            # Look for consent checkbox
            consent_checkbox = page.locator('input[type="checkbox"]').first
            if consent_checkbox.is_visible(timeout=3000):
                print("  ✅ Consent screen found")
                self.results["tests"].append({
                    "name": test_name,
                    "status": "PASS",
                    "details": "Consent screen present"
                })
            else:
                raise Exception("Consent checkbox not visible")
        except Exception as e:
            print(f"  ❌ FAIL: {e}")
            self.results["tests"].append({
                "name": test_name,
                "status": "FAIL",
                "error": str(e)
            })

    def accept_consent(self, page):
        """Accept beta consent."""
        print("\n📝 Accepting consent...")
        try:
            # Click checkbox
            page.locator('input[type="checkbox"]').first.click()
            time.sleep(0.5)

            # Click continue button
            page.get_by_role("button", name="Continue to Beta").click()
            time.sleep(2)
            print("  ✅ Consent accepted")
        except Exception as e:
            print(f"  ℹ️  No consent needed or already accepted: {e}")

    def test_beta_tab_navigation(self, page):
        """Test navigation to BETA tab."""
        test_name = "BETA Tab Navigation"
        print(f"\n🧪 TEST: {test_name}")

        try:
            # Look for BETA tab
            beta_tab = page.get_by_text("BETA", exact=True).first
            beta_tab.click()
            time.sleep(2)

            # Screenshot
            screenshot_path = f"screenshots/beta_tab_{int(time.time())}.png"
            page.screenshot(path=screenshot_path)
            self.results["screenshots"].append(screenshot_path)

            print("  ✅ BETA tab navigated")
            self.results["tests"].append({
                "name": test_name,
                "status": "PASS"
            })
        except Exception as e:
            print(f"  ❌ FAIL: {e}")
            self.results["tests"].append({
                "name": test_name,
                "status": "FAIL",
                "error": str(e)
            })

    def test_pa_establishment(self, page):
        """Test Primacy Attractor establishment."""
        test_name = "PA Establishment"
        print(f"\n🧪 TEST: {test_name}")

        try:
            # Send message about TELOS project
            message = "Let me tell you about my AI governance project called TELOS"

            # Find text area
            text_area = page.locator('textarea').first
            text_area.fill(message)

            # Find and click send button
            send_button = page.get_by_role("button").filter(has_text="Send").first
            send_button.click()

            print(f"  📤 Sent: {message}")
            time.sleep(3)  # Wait for response

            # Check if PA status changed
            pa_status = page.locator('text=Primacy Attractor Status').first
            if pa_status.is_visible():
                status_text = pa_status.text_content()
                print(f"  📊 PA Status: {status_text}")

                if "Calibrating" in status_text or "Established" in status_text:
                    print("  ✅ PA status detected")
                    self.results["tests"].append({
                        "name": test_name,
                        "status": "PASS",
                        "pa_status": status_text
                    })
                else:
                    raise Exception(f"Unexpected PA status: {status_text}")
            else:
                raise Exception("PA status not visible")

        except Exception as e:
            print(f"  ❌ FAIL: {e}")
            self.results["tests"].append({
                "name": test_name,
                "status": "FAIL",
                "error": str(e)
            })

    def test_fidelity_calculation(self, page):
        """Test if fidelity score actually changes."""
        test_name = "Fidelity Calculation"
        print(f"\n🧪 TEST: {test_name}")

        try:
            # Capture initial fidelity
            fidelity_elements = page.locator('text=Fidelity:').all()
            fidelities = []

            for elem in fidelity_elements:
                text = elem.text_content()
                if "0.850" in text or "0.85" in text:
                    fidelities.append(0.850)
                    print(f"  📊 Found fidelity: {text}")

            if len(fidelities) > 0:
                # All fidelities are 0.850 - SUSPICIOUS
                if all(f == 0.850 for f in fidelities):
                    print("  ❌ FAIL: All fidelities stuck at 0.850 (HARDCODED)")
                    self.results["failures"].append({
                        "issue": "Hardcoded Fidelity",
                        "detail": "Fidelity stuck at 0.850 - not calculating"
                    })
                    self.results["tests"].append({
                        "name": test_name,
                        "status": "FAIL",
                        "error": "Fidelity hardcoded at 0.850"
                    })
                else:
                    print("  ✅ Fidelity varies")
                    self.results["tests"].append({
                        "name": test_name,
                        "status": "PASS"
                    })
            else:
                raise Exception("No fidelity scores found")

        except Exception as e:
            print(f"  ❌ FAIL: {e}")
            self.results["tests"].append({
                "name": test_name,
                "status": "FAIL",
                "error": str(e)
            })

    def test_drift_detection(self, page):
        """Test drift detection with Thanksgiving conversation."""
        test_name = "Drift Detection (Thanksgiving Test)"
        print(f"\n🧪 TEST: {test_name}")

        try:
            # Send off-topic message
            drift_message = "I have Thanksgiving coming up and need to prepare for 18 guests"

            text_area = page.locator('textarea').first
            text_area.fill(drift_message)
            send_button = page.get_by_role("button").filter(has_text="Send").first
            send_button.click()

            print(f"  📤 Sent drift message: {drift_message}")
            time.sleep(3)

            # Check if fidelity dropped
            fidelity_after = page.locator('text=Fidelity:').first.text_content()
            print(f"  📊 Fidelity after drift: {fidelity_after}")

            # Check if intervention occurred
            response_text = page.locator('.stMarkdown').last.text_content()

            if "Thanksgiving" in response_text and "TELOS" not in response_text:
                print("  ❌ FAIL: AI followed drift, no intervention")
                self.results["failures"].append({
                    "issue": "No Drift Intervention",
                    "detail": "AI responded to Thanksgiving instead of redirecting to TELOS"
                })
                self.results["tests"].append({
                    "name": test_name,
                    "status": "FAIL",
                    "error": "No intervention on major drift"
                })
            else:
                print("  ✅ AI redirected or intervened")
                self.results["tests"].append({
                    "name": test_name,
                    "status": "PASS"
                })

        except Exception as e:
            print(f"  ❌ FAIL: {e}")
            self.results["tests"].append({
                "name": test_name,
                "status": "FAIL",
                "error": str(e)
            })

    def test_pa_display(self, page):
        """Test if PA displays real data."""
        test_name = "PA Display (Real vs Placeholder)"
        print(f"\n🧪 TEST: {test_name}")

        try:
            # Look for PA section
            pa_section = page.locator('text=Primacy Attractor').first

            if pa_section.is_visible():
                # Get surrounding text
                parent = pa_section.locator('..')
                pa_text = parent.text_content()

                # Check for placeholder text
                if "Establish conversation purpose" in pa_text or "Topics covered in baseline" in pa_text:
                    print("  ❌ FAIL: PA showing placeholder data")
                    self.results["failures"].append({
                        "issue": "Placeholder PA Data",
                        "detail": "PA not extracted from actual conversation"
                    })
                    self.results["tests"].append({
                        "name": test_name,
                        "status": "FAIL",
                        "error": "Placeholder PA text detected"
                    })
                else:
                    print("  ✅ PA shows custom data")
                    self.results["tests"].append({
                        "name": test_name,
                        "status": "PASS",
                        "pa_content": pa_text[:200]
                    })
            else:
                raise Exception("PA section not visible")

        except Exception as e:
            print(f"  ❌ FAIL: {e}")
            self.results["tests"].append({
                "name": test_name,
                "status": "FAIL",
                "error": str(e)
            })

    def test_fake_data_detection(self, page):
        """Detect fake/demo data in displays."""
        test_name = "Fake Data Detection"
        print(f"\n🧪 TEST: {test_name}")

        fake_indicators = [
            "Counterfactual Analysis",
            "Native LLM Response",
            "TELOS Intervention",
            "Base alignment score: 0.85"
        ]

        fake_found = []
        for indicator in fake_indicators:
            if page.locator(f'text={indicator}').count() > 0:
                fake_found.append(indicator)
                print(f"  🚨 Found fake display: {indicator}")

        if fake_found:
            self.results["failures"].append({
                "issue": "Fake/Demo Data Displayed",
                "detail": f"Found: {', '.join(fake_found)}"
            })
            self.results["tests"].append({
                "name": test_name,
                "status": "FAIL",
                "fake_elements": fake_found
            })
        else:
            print("  ✅ No fake data detected")
            self.results["tests"].append({
                "name": test_name,
                "status": "PASS"
            })

    def generate_report(self):
        """Generate diagnostic report."""
        print("\n" + "="*80)
        print("📊 DIAGNOSTIC REPORT")
        print("="*80)

        total_tests = len(self.results["tests"])
        passed = sum(1 for t in self.results["tests"] if t["status"] == "PASS")
        failed = sum(1 for t in self.results["tests"] if t["status"] == "FAIL")

        print(f"\nTotal Tests: {total_tests}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")

        if self.results["failures"]:
            print(f"\n🚨 CRITICAL FAILURES ({len(self.results['failures'])}):")
            for i, failure in enumerate(self.results["failures"], 1):
                print(f"\n{i}. {failure['issue']}")
                print(f"   {failure['detail']}")

        # Save report
        report_path = Path("tests/diagnostic_results")
        report_path.mkdir(exist_ok=True, parents=True)

        report_file = report_path / f"beta_diagnostic_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n📄 Full report saved: {report_file}")
        print("="*80)


def main():
    """Run diagnostic."""
    print("🏥 TELOS BETA Diagnostic Tool")
    print("Powered by Playwright\n")

    diagnostic = BETADiagnostic()
    diagnostic.run_diagnostic()


if __name__ == "__main__":
    main()
