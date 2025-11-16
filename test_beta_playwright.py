#!/usr/bin/env python3
"""
Playwright test script for TELOS Beta flow.
Tests the complete user journey and captures screenshots.
"""

import asyncio
from playwright.async_api import async_playwright
from pathlib import Path
from datetime import datetime

async def test_beta_flow():
    """Test complete beta flow with visual verification."""

    screenshot_dir = Path(".playwright-mcp")
    screenshot_dir.mkdir(exist_ok=True)

    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1920, "height": 1080})
        page = await context.new_page()

        print("🔭 Starting TELOS Beta Testing Flow...")

        # Test 1: Beta Consent Screen
        print("\n📸 Test 1: Beta Consent Screen")
        await page.goto("http://localhost:8504")
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(2000)  # Let animations complete

        screenshot_path = screenshot_dir / f"01_beta_consent_{datetime.now().strftime('%H%M%S')}.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"   ✓ Screenshot saved: {screenshot_path}")

        # Check if consent checkbox is visible
        consent_checkbox = page.locator('input[type="checkbox"]')
        if await consent_checkbox.count() > 0:
            print("   ✓ Consent checkbox found")
        else:
            print("   ⚠ Consent checkbox NOT found")

        # Test 2: Click consent and continue
        print("\n📸 Test 2: Accept Consent")
        try:
            # Click checkbox
            await consent_checkbox.click()
            await page.wait_for_timeout(1000)

            # Click Continue button
            continue_button = page.locator('button:has-text("Continue to Beta")')
            if await continue_button.count() > 0:
                await continue_button.click()
                await page.wait_for_timeout(3000)  # Wait for page transition

                screenshot_path = screenshot_dir / f"02_after_consent_{datetime.now().strftime('%H%M%S')}.png"
                await page.screenshot(path=str(screenshot_path), full_page=True)
                print(f"   ✓ Screenshot saved: {screenshot_path}")
                print("   ✓ Consent accepted successfully")
            else:
                print("   ⚠ Continue button not found")
        except Exception as e:
            print(f"   ⚠ Error during consent: {e}")

        # Test 3: Check for DEMO tab
        print("\n📸 Test 3: Check Tab Navigation")
        try:
            demo_tab = page.locator('button:has-text("DEMO")')
            beta_tab = page.locator('button:has-text("BETA")')
            telos_tab = page.locator('button:has-text("TELOS")')

            demo_count = await demo_tab.count()
            beta_count = await beta_tab.count()
            telos_count = await telos_tab.count()

            print(f"   DEMO tab: {'✓ Found' if demo_count > 0 else '⚠ Not found'}")
            print(f"   BETA tab: {'✓ Found' if beta_count > 0 else '⚠ Not found'}")
            print(f"   TELOS tab: {'✓ Found' if telos_count > 0 else '⚠ Not found'}")

            screenshot_path = screenshot_dir / f"03_tabs_visible_{datetime.now().strftime('%H%M%S')}.png"
            await page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"   ✓ Screenshot saved: {screenshot_path}")
        except Exception as e:
            print(f"   ⚠ Error checking tabs: {e}")

        # Test 4: Click DEMO tab (if visible)
        print("\n📸 Test 4: Navigate to DEMO Tab")
        try:
            if demo_count > 0:
                await demo_tab.first.click()
                await page.wait_for_timeout(2000)

                screenshot_path = screenshot_dir / f"04_demo_tab_{datetime.now().strftime('%H%M%S')}.png"
                await page.screenshot(path=str(screenshot_path), full_page=True)
                print(f"   ✓ Screenshot saved: {screenshot_path}")
                print("   ✓ DEMO tab active")
        except Exception as e:
            print(f"   ⚠ Error navigating to DEMO: {e}")

        # Test 5: Check sidebar visibility
        print("\n📸 Test 5: Sidebar Visibility")
        try:
            sidebar = page.locator('[data-testid="stSidebar"]')
            if await sidebar.count() > 0:
                print("   ✓ Sidebar visible")
            else:
                print("   ⚠ Sidebar not found")

            screenshot_path = screenshot_dir / f"05_sidebar_{datetime.now().strftime('%H%M%S')}.png"
            await page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"   ✓ Screenshot saved: {screenshot_path}")
        except Exception as e:
            print(f"   ⚠ Error checking sidebar: {e}")

        # Test 6: Check BETA tab
        print("\n📸 Test 6: Navigate to BETA Tab")
        try:
            if beta_count > 0:
                await beta_tab.first.click()
                await page.wait_for_timeout(2000)

                screenshot_path = screenshot_dir / f"06_beta_tab_{datetime.now().strftime('%H%M%S')}.png"
                await page.screenshot(path=str(screenshot_path), full_page=True)
                print(f"   ✓ Screenshot saved: {screenshot_path}")
                print("   ✓ BETA tab active")

                # Check for input field
                input_field = page.locator('textarea, input[type="text"]')
                if await input_field.count() > 0:
                    print("   ✓ Input field found in BETA tab")
                else:
                    print("   ⚠ Input field NOT found")
        except Exception as e:
            print(f"   ⚠ Error navigating to BETA: {e}")

        # Test 7: Final state capture
        print("\n📸 Test 7: Final State")
        screenshot_path = screenshot_dir / f"07_final_state_{datetime.now().strftime('%H%M%S')}.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"   ✓ Screenshot saved: {screenshot_path}")

        print("\n✅ Testing complete! Check .playwright-mcp/ for screenshots.")

        # Keep browser open for manual inspection
        print("\n⏸  Browser will stay open for 30 seconds for manual inspection...")
        await page.wait_for_timeout(30000)

        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_beta_flow())
