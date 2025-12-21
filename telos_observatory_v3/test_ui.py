#!/usr/bin/env python3
"""
UI Testing Script for TELOSCOPE_BETA
Uses Playwright to check the current state of the application
"""

from playwright.sync_api import sync_playwright
import time
from datetime import datetime


def test_teloscope_ui():
    """Test the TELOSCOPE UI and capture screenshots."""

    with sync_playwright() as p:
        # Launch browser in non-headless mode to see what's happening
        browser = p.chromium.launch(headless=False)

        # Create context with viewport
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080}
        )

        page = context.new_page()

        print("\n🔍 Testing TELOSCOPE_BETA UI at localhost:8502")
        print("="*60)

        try:
            # 1. Navigate to main page
            print("\n1. Loading main page...")
            page.goto("http://localhost:8502", wait_until="networkidle")
            time.sleep(3)  # Wait for Streamlit to fully load

            # Take screenshot of initial state
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            page.screenshot(path=f"screenshot_main_{timestamp}.png", full_page=True)
            print("   ✅ Main page loaded - screenshot saved")

            # 2. Check for DEMO tab (should be active by default)
            print("\n2. Checking DEMO tab...")
            demo_button = page.locator("button:has-text('DEMO')")
            if demo_button.count() > 0:
                print("   ✅ DEMO tab found")
                # Check if it's the active tab
                if "primary" in (demo_button.get_attribute("kind") or ""):
                    print("   ✅ DEMO tab is active (as expected)")

            # 3. Check for BETA tab
            print("\n3. Checking BETA tab...")
            beta_button = page.locator("button:has-text('BETA')")
            if beta_button.count() > 0:
                is_disabled = beta_button.is_disabled()
                print(f"   ✅ BETA tab found - {'LOCKED' if is_disabled else 'UNLOCKED'}")

                # If BETA is unlocked, try clicking it
                if not is_disabled:
                    print("   🔄 Clicking BETA tab...")
                    beta_button.click()
                    time.sleep(2)
                    page.screenshot(path=f"screenshot_beta_{timestamp}.png", full_page=True)
                    print("   ✅ BETA tab screenshot saved")

            # 4. Check for progressive demo slideshow
            print("\n4. Checking for progressive demo slideshow...")
            # Look for slide navigation buttons
            prev_button = page.locator("button:has-text('Previous')")
            next_button = page.locator("button:has-text('Next')")

            if prev_button.count() > 0 or next_button.count() > 0:
                print("   ✅ Demo slideshow navigation found")

                # Try clicking Next to advance slide
                if next_button.count() > 0:
                    print("   🔄 Advancing to next slide...")
                    next_button.first.click()
                    time.sleep(2)
                    page.screenshot(path=f"screenshot_slide2_{timestamp}.png", full_page=True)
                    print("   ✅ Slide 2 screenshot saved")

            # 5. Check for admin mode
            print("\n5. Testing admin mode access...")
            page.goto("http://localhost:8502?admin=true", wait_until="networkidle")
            time.sleep(3)

            devops_button = page.locator("button:has-text('DEVOPS')")
            if devops_button.count() > 0:
                print("   ✅ DEVOPS tab found in admin mode")
                devops_button.click()
                time.sleep(2)
                page.screenshot(path=f"screenshot_devops_{timestamp}.png", full_page=True)
                print("   ✅ DEVOPS mode screenshot saved")
            else:
                print("   ❌ DEVOPS tab not found in admin mode")

            # 6. Check for key UI elements
            print("\n6. Checking for key UI elements...")

            # Check for conversation display
            if page.locator("div.message-container").count() > 0:
                print("   ✅ Conversation display found")

            # Check for observation deck toggle
            obs_deck_button = page.locator("button:has-text('Observation Deck')")
            if obs_deck_button.count() > 0:
                print("   ✅ Observation Deck toggle found")

            # Check for alignment lens toggle
            lens_button = page.locator("button:has-text('Alignment Lens')")
            if lens_button.count() > 0:
                print("   ✅ Alignment Lens toggle found")

            print("\n" + "="*60)
            print("✅ UI Testing Complete!")
            print(f"Screenshots saved with timestamp: {timestamp}")

        except Exception as e:
            print(f"\n❌ Error during testing: {e}")
            page.screenshot(path=f"screenshot_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

        finally:
            browser.close()


if __name__ == "__main__":
    test_teloscope_ui()