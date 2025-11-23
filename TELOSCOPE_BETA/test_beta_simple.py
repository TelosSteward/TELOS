#!/usr/bin/env python3
"""
Simple test to check BETA tab state and capture what's visible.
"""

from playwright.sync_api import sync_playwright
import time

def test_beta_state():
    """Check what's visible in BETA tab."""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=500)
        page = browser.new_page()

        try:
            # Navigate to app
            print("📍 Navigating to http://localhost:8501...")
            page.goto("http://localhost:8501")
            page.wait_for_load_state("networkidle")
            time.sleep(2)

            # Try to unlock BETA via localStorage
            print("🔓 Attempting to unlock BETA tab...")
            page.evaluate("""
                window.localStorage.setItem('demo_completed', 'true');
            """)
            page.reload()
            page.wait_for_load_state("networkidle")
            time.sleep(3)

            # Click BETA tab
            print("🔵 Clicking BETA tab...")
            try:
                beta_tab = page.locator('button:has-text("BETA")').first
                beta_tab.click(force=True)
            except:
                beta_tab = page.locator('[role="tab"]:has-text("BETA")').first
                beta_tab.click(force=True)

            # Wait for content to load
            print("⏳ Waiting for BETA tab to load...")
            time.sleep(5)

            # Check what's on the page
            page_content = page.content()

            print("\n📊 BETA TAB CONTENTS:")
            print("=" * 60)

            if "Let's Establish Your Purpose" in page_content:
                print("✓ PA Questionnaire IS visible")
                print("  - Questionnaire title found")
            else:
                print("❌ PA Questionnaire NOT visible")

            if "textarea" in page_content.lower():
                textareas = page.locator("textarea").count()
                print(f"✓ Found {textareas} textarea elements")
            else:
                print("❌ No textarea elements found")

            if "Your Primacy Attractor" in page_content:
                print("✓ PA Summary IS visible (PA already established)")

            if "Message TELOS" in page_content:
                print("✓ Chat interface IS visible")
            else:
                print("❌ Chat interface NOT visible")

            # Take screenshot
            screenshot_path = "/Users/brunnerjf/Desktop/Privacy_PreCommit/TELOSCOPE_BETA/beta_state.png"
            page.screenshot(path=screenshot_path, full_page=True)
            print(f"\n📸 Screenshot saved to: {screenshot_path}")

            # Keep browser open for inspection
            print("\n⏸️  Browser will stay open for 20 seconds for manual inspection...")
            time.sleep(20)

        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            page.screenshot(path="/Users/brunnerjf/Desktop/Privacy_PreCommit/TELOSCOPE_BETA/beta_error.png")
            raise

        finally:
            browser.close()
            print("\n✅ Test complete!")


if __name__ == "__main__":
    test_beta_state()
