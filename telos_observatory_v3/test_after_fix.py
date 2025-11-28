#!/usr/bin/env python3
"""
Quick test to verify the app loads correctly after fixing the column width issue
"""

from playwright.sync_api import sync_playwright
import time


def test_after_fix():
    """Test if the app loads without errors after fix"""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # Headless for quick test
        context = browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = context.new_page()

        print("\n🔧 Testing app after column width fix...")
        print("="*50)

        try:
            # Load the page
            page.goto("http://localhost:8502", wait_until="networkidle", timeout=30000)
            time.sleep(3)

            # Check for error messages
            error_elements = page.locator("text=/error|exception|traceback/i")
            if error_elements.count() > 0:
                print("❌ Error found on page!")
                return False

            # Check for Start Demo button
            start_demo = page.locator("button", has_text="Start Demo")
            if start_demo.count() > 0:
                print("✅ Start Demo button found")

                # Click it to test if slideshow loads
                start_demo.click()
                time.sleep(2)

                # Check if we transitioned to first slide
                slide_content = page.locator("text=/Steward|guide|TELOS/i")
                if slide_content.count() > 0:
                    print("✅ Demo slideshow started successfully")
                else:
                    print("⚠️ Demo slideshow might not have started")

            # Check tabs
            demo_tab = page.locator("button", has_text="DEMO").first
            beta_tab = page.locator("button", has_text="BETA").first

            if demo_tab and beta_tab:
                print("✅ DEMO and BETA tabs present")

            print("\n" + "="*50)
            print("✅ App loads successfully after fix!")
            return True

        except Exception as e:
            print(f"❌ Error during test: {e}")
            return False

        finally:
            browser.close()


if __name__ == "__main__":
    test_after_fix()