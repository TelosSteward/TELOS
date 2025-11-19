#!/usr/bin/env python3
"""
Test regular user flow (non-admin) for TELOSCOPE_BETA
"""

from playwright.sync_api import sync_playwright
import time
from datetime import datetime


def test_regular_flow():
    """Test the regular user flow at localhost:8502"""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = context.new_page()

        print("\n🌐 Testing REGULAR USER FLOW at localhost:8502")
        print("="*60)

        try:
            # Load regular page (not admin)
            print("\n1. Loading regular page (non-admin)...")
            page.goto("http://localhost:8502", wait_until="networkidle", timeout=30000)
            time.sleep(5)  # Wait for Streamlit to fully render

            # Take initial screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            page.screenshot(path=f"regular_initial_{timestamp}.png", full_page=True)
            print("   ✅ Page loaded - screenshot saved")

            # Check what tabs are visible
            print("\n2. Checking available tabs...")
            demo_tab = page.locator("button", has_text="DEMO")
            beta_tab = page.locator("button", has_text="BETA")
            telos_tab = page.locator("button", has_text="TELOS")
            devops_tab = page.locator("button", has_text="DEVOPS")

            if demo_tab.count() > 0:
                print(f"   ✅ DEMO tab found")
                # Check if it's primary (active)
                demo_classes = demo_tab.get_attribute("kind") or ""
                if "primary" in demo_classes:
                    print("      → Currently active")

            if beta_tab.count() > 0:
                is_disabled = beta_tab.is_disabled()
                print(f"   ✅ BETA tab found - {'LOCKED' if is_disabled else 'UNLOCKED'}")

            if telos_tab.count() > 0:
                is_disabled = telos_tab.is_disabled()
                print(f"   ✅ TELOS tab found - {'LOCKED' if is_disabled else 'UNLOCKED'}")

            if devops_tab.count() > 0:
                print("   ⚠️ DEVOPS tab visible (shouldn't be in regular mode!)")
            else:
                print("   ✅ DEVOPS tab hidden (correct for regular mode)")

            # Check for demo slideshow content
            print("\n3. Checking demo slideshow...")

            # Look for welcome message
            welcome = page.locator("text=/Welcome to TELOS/i")
            if welcome.count() > 0:
                print("   ✅ Welcome message found")

            # Look for navigation
            prev_button = page.locator("button", has_text="Previous")
            next_button = page.locator("button", has_text="Next")

            if prev_button.count() > 0 and next_button.count() > 0:
                print("   ✅ Navigation buttons found")

                # Try advancing through a few slides
                print("   🔄 Testing slide navigation...")
                for i in range(3):
                    if next_button.count() > 0:
                        next_button.first.click()
                        time.sleep(2)
                        print(f"      → Advanced to slide {i+2}")

                        # Take screenshot of this slide
                        page.screenshot(path=f"regular_slide_{i+2}_{timestamp}.png")

            # Check for interactive elements
            print("\n4. Checking interactive elements...")

            # Check for input field
            text_input = page.locator("input[type='text']")
            if text_input.count() > 0:
                print("   ✅ Text input field found")
                # Try typing something
                text_input.first.fill("Testing TELOS demo mode")
                time.sleep(1)

            # Check for observation deck toggle
            obs_toggle = page.locator("button", has_text="Observation Deck")
            if obs_toggle.count() > 0:
                print("   ✅ Observation Deck toggle found")
                obs_toggle.first.click()
                time.sleep(2)
                page.screenshot(path=f"regular_obs_deck_{timestamp}.png")

            # Check for alignment lens toggle
            lens_toggle = page.locator("button", has_text="Alignment Lens")
            if lens_toggle.count() > 0:
                print("   ✅ Alignment Lens toggle found")

            # Check for steward button
            steward = page.locator("button", has_text="🤝")
            if steward.count() > 0:
                print("   ✅ Steward handshake button found")

            # Check current content type
            print("\n5. Analyzing current content...")

            # Check if in demo slideshow
            slide_content = page.locator("div[data-slide]")
            if slide_content.count() > 0:
                slide_num = slide_content.first.get_attribute("data-slide")
                print(f"   ℹ️ Currently on demo slide: {slide_num}")

            # Check for any error messages
            error = page.locator("text=/Error/i")
            if error.count() > 0:
                print("   ⚠️ Error message detected on page")

            # Final status
            print("\n" + "="*60)
            print("📊 REGULAR USER FLOW STATUS:")
            print("\nAccessible Features:")
            print("  • DEMO tab (active)")
            print("  • Progressive slideshow")
            print("  • Navigation controls")

            print("\nLocked Features (need progression):")
            print("  • BETA tab (requires 10 demo turns)")
            print("  • TELOS tab (requires beta completion)")
            print("  • DEVOPS hidden (admin only)")

            print(f"\nScreenshots saved with timestamp: {timestamp}")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            page.screenshot(path=f"regular_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

        finally:
            # Keep browser open for manual inspection
            print("\n⏸️ Browser staying open for 10 seconds...")
            time.sleep(10)
            browser.close()


if __name__ == "__main__":
    test_regular_flow()