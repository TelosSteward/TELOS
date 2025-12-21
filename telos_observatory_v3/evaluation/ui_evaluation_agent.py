"""
TELOS UI Evaluation Agent
=========================

Uses Playwright to capture screenshots of the demo interface and evaluate
against best-in-class UI/UX standards for enterprise software.

Evaluation Criteria:
1. Visual Hierarchy - Clear information architecture
2. Color Accessibility - WCAG compliance, contrast ratios
3. Typography - Readability, consistency
4. Layout Consistency - Alignment, spacing, responsive design
5. Professional Appearance - Enterprise-ready aesthetics
6. User Experience - Intuitive navigation, clear feedback
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    from playwright.async_api import async_playwright, Page
except ImportError:
    print("Playwright not installed. Run: pip install playwright && playwright install")
    exit(1)


class UIEvaluationAgent:
    """Agent that evaluates TELOS Observatory UI against enterprise standards."""

    def __init__(self, base_url: str = "http://localhost:8501"):
        self.base_url = base_url
        self.screenshots_dir = Path(__file__).parent / "screenshots" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.evaluation_results: Dict = {
            "timestamp": datetime.now().isoformat(),
            "slides_evaluated": [],
            "overall_scores": {},
            "findings": [],
            "recommendations": []
        }

    async def setup(self):
        """Initialize Playwright browser."""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context(
            viewport={"width": 1920, "height": 1080}
        )
        self.page = await self.context.new_page()
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

    async def teardown(self):
        """Clean up browser resources."""
        await self.browser.close()
        await self.playwright.stop()

    async def navigate_to_demo(self):
        """Navigate to demo mode."""
        await self.page.goto(self.base_url)
        await self.page.wait_for_load_state("networkidle")
        await asyncio.sleep(2)  # Wait for Streamlit to fully render

        # Click Demo tab if available
        try:
            demo_tab = await self.page.query_selector("text=Demo")
            if demo_tab:
                await demo_tab.click()
                await asyncio.sleep(1)
        except:
            pass

    async def capture_slide(self, slide_num: int, description: str) -> str:
        """Capture screenshot of current slide."""
        filename = f"slide_{slide_num:02d}_{description.replace(' ', '_')}.png"
        filepath = self.screenshots_dir / filename
        await self.page.screenshot(path=str(filepath), full_page=True)
        return str(filepath)

    async def click_next(self):
        """Click the Next button to advance slides."""
        try:
            next_btn = await self.page.query_selector("text=Next")
            if next_btn:
                await next_btn.click()
                await asyncio.sleep(1.5)  # Wait for animation
                return True
        except:
            pass
        return False

    async def toggle_alignment_lens(self):
        """Toggle the Alignment Lens visibility."""
        try:
            lens_btn = await self.page.query_selector("text=Show Alignment Lens")
            if lens_btn:
                await lens_btn.click()
                await asyncio.sleep(1)  # Wait for animation
                return True

            # Try hide button if already visible
            hide_btn = await self.page.query_selector("text=Hide Alignment Lens")
            if hide_btn:
                await hide_btn.click()
                await asyncio.sleep(0.5)
                return True
        except:
            pass
        return False

    async def evaluate_color_contrast(self) -> Dict:
        """Evaluate color contrast ratios (basic check)."""
        # Extract key colors from the page
        colors = await self.page.evaluate("""
            () => {
                const elements = document.querySelectorAll('*');
                const colors = new Set();
                elements.forEach(el => {
                    const style = window.getComputedStyle(el);
                    colors.add(style.color);
                    colors.add(style.backgroundColor);
                    colors.add(style.borderColor);
                });
                return Array.from(colors).slice(0, 20);
            }
        """)

        return {
            "extracted_colors": colors,
            "assessment": "Color palette extracted for manual WCAG review",
            "recommendation": "Verify contrast ratios meet WCAG 2.1 AA standards (4.5:1 for text)"
        }

    async def evaluate_typography(self) -> Dict:
        """Evaluate typography consistency."""
        fonts = await self.page.evaluate("""
            () => {
                const elements = document.querySelectorAll('*');
                const fonts = new Set();
                const sizes = new Set();
                elements.forEach(el => {
                    const style = window.getComputedStyle(el);
                    fonts.add(style.fontFamily);
                    sizes.add(style.fontSize);
                });
                return {
                    fonts: Array.from(fonts).slice(0, 10),
                    sizes: Array.from(sizes).slice(0, 15)
                };
            }
        """)

        return {
            "font_families": fonts.get("fonts", []),
            "font_sizes": fonts.get("sizes", []),
            "assessment": f"Found {len(fonts.get('fonts', []))} font families",
            "recommendation": "Limit to 2-3 font families for professional consistency"
        }

    async def evaluate_layout_consistency(self) -> Dict:
        """Evaluate layout alignment and spacing."""
        layout = await self.page.evaluate("""
            () => {
                const containers = document.querySelectorAll('[style*="max-width"]');
                const widths = new Set();
                containers.forEach(el => {
                    widths.add(window.getComputedStyle(el).maxWidth);
                });
                return {
                    max_widths: Array.from(widths),
                    container_count: containers.length
                };
            }
        """)

        return {
            "max_widths_found": layout.get("max_widths", []),
            "containers": layout.get("container_count", 0),
            "assessment": "Checking for consistent max-width containers",
            "recommendation": "Use consistent 700px max-width across all slides"
        }

    async def evaluate_slide(self, slide_num: int) -> Dict:
        """Comprehensive evaluation of a single slide."""
        slide_eval = {
            "slide_number": slide_num,
            "screenshots": [],
            "evaluations": {}
        }

        # Capture main slide view
        screenshot = await self.capture_slide(slide_num, "main_view")
        slide_eval["screenshots"].append(screenshot)

        # Try to capture Alignment Lens view
        if await self.toggle_alignment_lens():
            await asyncio.sleep(1)
            lens_screenshot = await self.capture_slide(slide_num, "alignment_lens")
            slide_eval["screenshots"].append(lens_screenshot)
            await self.toggle_alignment_lens()  # Hide it again

        # Run evaluations
        slide_eval["evaluations"]["colors"] = await self.evaluate_color_contrast()
        slide_eval["evaluations"]["typography"] = await self.evaluate_typography()
        slide_eval["evaluations"]["layout"] = await self.evaluate_layout_consistency()

        return slide_eval

    async def run_full_evaluation(self, max_slides: int = 13) -> Dict:
        """Run complete evaluation across all demo slides."""
        print("=" * 60)
        print("TELOS UI EVALUATION AGENT")
        print("=" * 60)
        print(f"Target: {self.base_url}")
        print(f"Screenshots: {self.screenshots_dir}")
        print()

        await self.setup()

        try:
            await self.navigate_to_demo()
            print("Navigated to demo mode")

            for slide_num in range(1, max_slides + 1):
                print(f"\nEvaluating Slide {slide_num}...")

                slide_eval = await self.evaluate_slide(slide_num)
                self.evaluation_results["slides_evaluated"].append(slide_eval)

                print(f"  - Screenshots: {len(slide_eval['screenshots'])}")

                # Advance to next slide
                if slide_num < max_slides:
                    if not await self.click_next():
                        print(f"  - Could not advance past slide {slide_num}")
                        break

        except Exception as e:
            print(f"Error during evaluation: {e}")
            self.evaluation_results["error"] = str(e)

        finally:
            await self.teardown()

        # Generate summary
        self._generate_summary()

        # Save results
        results_file = self.screenshots_dir / "evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_file}")

        return self.evaluation_results

    def _generate_summary(self):
        """Generate evaluation summary and recommendations."""
        slides_count = len(self.evaluation_results["slides_evaluated"])

        self.evaluation_results["summary"] = {
            "slides_captured": slides_count,
            "total_screenshots": sum(
                len(s["screenshots"])
                for s in self.evaluation_results["slides_evaluated"]
            )
        }

        # Best-in-class criteria checklist
        self.evaluation_results["best_practices_checklist"] = {
            "visual_hierarchy": {
                "criterion": "Clear information hierarchy with gold accent color",
                "standard": "Enterprise dashboards use max 3-4 color zones",
                "status": "REVIEW_SCREENSHOTS"
            },
            "color_accessibility": {
                "criterion": "WCAG 2.1 AA compliant contrast ratios",
                "standard": "4.5:1 for normal text, 3:1 for large text",
                "status": "REVIEW_SCREENSHOTS"
            },
            "typography": {
                "criterion": "Consistent font hierarchy",
                "standard": "Max 2-3 font families, clear size scale",
                "status": "REVIEW_SCREENSHOTS"
            },
            "layout_consistency": {
                "criterion": "Consistent container widths and spacing",
                "standard": "700px max-width, 15-30px spacing",
                "status": "REVIEW_SCREENSHOTS"
            },
            "navigation_clarity": {
                "criterion": "Clear Next/Previous/Alignment Lens buttons",
                "standard": "3-button layout, consistent positioning",
                "status": "REVIEW_SCREENSHOTS"
            },
            "animation_quality": {
                "criterion": "Smooth fade-in animations",
                "standard": "0.3-1.0s duration, ease-in-out timing",
                "status": "REVIEW_SCREENSHOTS"
            },
            "professional_aesthetics": {
                "criterion": "Enterprise-ready appearance",
                "standard": "Dark theme, gold accents, clean borders",
                "status": "REVIEW_SCREENSHOTS"
            },
            "steward_attractor_naming": {
                "criterion": "Correct terminology (Steward Attractor, not AI Primacy)",
                "standard": "User Primacy Attractor + Steward Attractor",
                "status": "VERIFY_IN_SCREENSHOTS"
            },
            "alignment_lens_border": {
                "criterion": "Border reflects User Fidelity (not Primacy State)",
                "standard": "Yellow border for 0.69 user fidelity",
                "status": "VERIFY_IN_SCREENSHOTS"
            }
        }

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Slides Evaluated: {slides_count}")
        print(f"Screenshots Captured: {self.evaluation_results['summary']['total_screenshots']}")
        print(f"\nBest Practices Checklist:")
        for key, item in self.evaluation_results["best_practices_checklist"].items():
            print(f"  [{item['status']}] {item['criterion']}")


async def main():
    """Run the UI evaluation agent."""
    agent = UIEvaluationAgent()
    results = await agent.run_full_evaluation(max_slides=10)

    print("\n" + "=" * 60)
    print("MANUAL REVIEW REQUIRED")
    print("=" * 60)
    print(f"Review screenshots in: {agent.screenshots_dir}")
    print("\nKey items to verify:")
    print("1. Steward Attractor naming appears correctly")
    print("2. Alignment Lens border matches User Fidelity color")
    print("3. No 'Dual Primacy Attractors' header visible")
    print("4. Consistent 700px container widths")
    print("5. Professional dark theme with gold accents")


if __name__ == "__main__":
    asyncio.run(main())
