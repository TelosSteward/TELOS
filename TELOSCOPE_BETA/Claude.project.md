# TELOS Observatory V3 - Project Documentation

## TELOS Observatory V3 UI Continuity Schema

**Established:** 2025-01-17
**Git Reference:** Commit `54e2b63`
**Purpose:** Ensure seamless visual and interaction continuity from Demo Mode → Beta Mode

---

## 1. Color Coding System (4-Tier Fidelity)

All fidelity scores use this consistent color scheme across Demo and Beta:

| Range | Color | Hex | Status | Usage |
|-------|-------|-----|--------|-------|
| ≥0.85 | **Green** | `#4CAF50` | Good/Aligned | High fidelity, on-purpose |
| 0.70-0.85 | **Yellow** | `#FFD700` | Mild Drift | Slight deviation, monitoring |
| 0.50-0.70 | **Orange** | `#FFA500` | Moderate Drift | Noticeable drift, intervention tier |
| <0.50 | **Red** | `#FF4444` | Severe Drift | Critical, blocking intervention |

**Application:**
- User Fidelity box: border + text color
- AI Fidelity box: border + text color
- Primacy State box: border + text color
- PA column headers: gradient backgrounds
- Event log entries: border-left accent colors
- Drift visualization rings: concentric circles

---

## 2. Animation Standards

### Timing
- **Fade-in duration:** 1.0 seconds (`animation: fadeIn 1.0s ease-out forwards`)
- **Staggered delays:** 0.3s increments for sequential elements
- **Observation Deck reveal:** 1.0s fade + translateY(20px)

### Keyframes
```css
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
```

### Application
- All Q&A slide transitions
- Intro slide reveals (slides 0, 1, 2)
- Observation Deck toggle
- Alignment Lens expansion
- Unique `data-slide` attributes force animation retriggering

---

## 3. Layout & Sizing Standards

### Fidelity Display Boxes
- **Fixed width:** `200px`
- **Max-width:** `100%` (responsive shrinking)
- **Display:** `inline-block`
- **Padding:** `15px`
- **Border:** `2px solid [color]`
- **Border-radius:** `8px`
- **Text alignment:** Centered

### Column Structure
- **Observation Deck fidelity row:** Flexbox (`display: flex; justify-content: space-between`)
- **PA columns:** Streamlit columns `[50, 1, 50]` (left PA, spacer, right PA)
- **Alignment Lens grid:** 3-column equal layout for top row, custom for bottom

### Concentric Ring Visualization
- 4 distinct `<div>` circles (not CSS radial-gradient)
- Red outer (75-100%), Orange (50-75%), Yellow (25-50%), Green center (0-25%)
- Border on outer ring: `2px solid #FFD700`
- Dots positioned absolutely with precise percentages

---

## 4. Typography Standards

### Text Colors
- **Primary body text:** `#e0e0e0` (light gray, high contrast)
- **Secondary/detail text:** `#ddd` (87% brightness, improved from #888)
- **Steward responses:** `#FFD700` (gold, always in sidebar)
- **Headers:** `#FFD700` (Observatory branding gold)
- **Fidelity scores:** Match tier color (green/yellow/orange/red)

### Font Sizing Hierarchy
- **Main headers:** `48px` (TELOSCOPE), `26px` (Observation Deck)
- **Subtitles:** `22px` (Alignment Lens), `18px` (descriptors)
- **Fidelity scores:** `48px` (large display numbers), `24px` (Obs Deck)
- **Labels:** `14px` (box headers), `12px` (status text), `10px-11px` (detail)
- **Body text:** `16px` (PA content), `18px` (Live Governance Metrics)

### Font Weight
- **Bold:** Headers, scores, status indicators
- **Regular:** Body text, explanations

---

## 5. Component Consistency (Demo → Beta)

### Critical Requirements
These elements MUST remain identical between Demo and Beta modes:

1. **Visual Identity**
   - Same color coding for all fidelity tiers
   - Identical box styling (borders, gradients, shadows)
   - Consistent gold branding (#FFD700)

2. **Interaction Patterns**
   - 1.0s fade animations on all transitions
   - Toggle behaviors (Show/Hide Observation Deck, Alignment Lens)
   - Auto-scroll on expansion with smooth scrollIntoView
   - Navigation button placement and styling

3. **Information Architecture**
   - Same layout for fidelity displays
   - Identical PA presentation structure
   - Consistent event log formatting
   - Unified drift visualization approach

4. **Steward Integration**
   - Gold text (#FFD700) in sidebar panel always
   - Same invocation patterns
   - Consistent help/guidance presentation

### Why This Matters
Users transitioning from Demo → Beta should feel they're using **the same application with live data**, not switching to a different interface. Visual and interaction continuity builds trust and reduces cognitive load.

---

## 6. Interactive Toggle Components (Observation Deck & Alignment Lens)

### Animation Specifications

**Observation Deck (Slide 3)**
```css
@keyframes obsDeckFadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
animation: obsDeckFadeIn 1.0s ease-in-out forwards;
```

**Alignment Lens (Slide 6)**
```css
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
animation: fadeIn 1.0s ease-in-out forwards;
```

**Critical Parameters:**
- **Duration:** 1.0s (consistent across all components)
- **Timing Function:** `ease-in-out` (smooth acceleration/deceleration)
- **Transform Distance:** `translateY(20px)` (NOT 10px - needs more dramatic motion)
- **Fill Mode:** `forwards` (maintains final state)

### Scroll Behavior

**Observation Deck**
- **Anchor Position:** Right BEFORE bottom navigation buttons
- **Anchor ID:** `observation-deck-anchor`
- **Scroll Positioning:** `block: 'start'` (anchor at top of viewport)
- **Delay:** 100ms setTimeout for Streamlit compatibility
- **Result:** Bottom nav buttons immediately visible, zero manual scrolling

**Alignment Lens**
- **Anchor Position:** Right BEFORE lens content
- **Anchor ID:** `alignment-lens-anchor`
- **Scroll Positioning:** `block: 'start'`
- **Delay:** 100ms setTimeout
- **Result:** Lens content and controls immediately accessible

**Scroll Implementation:**
```javascript
setTimeout(function() {
    window.parent.document.getElementById('[anchor-id]').scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
}, 100);
```

### Reset Behavior

**When Navigating Away (Previous/Next buttons):**

Both Observation Deck and Alignment Lens MUST reset to hidden state when user navigates to different slide.

**Implementation:**
```python
# In navigation button handlers
if st.button("Previous"):
    st.session_state.demo_obs_deck_visible = False  # Reset
    st.session_state.slide_7_drift_visible = False  # Reset
    st.session_state.demo_slide_index -= 1
    st.rerun()
```

**Reset triggers:**
- Top Previous button (resets before slide change)
- Top Next button (resets before slide change)
- Bottom Previous button (resets before slide change)
- Bottom Next button (resets before slide change)

**When Returning to Slide:**
- Observation Deck: Starts hidden (default state)
- Alignment Lens: Starts hidden (default state)
- User must click "Show" button to reveal again
- Clean state ensures consistent UX

### Toggle Button States

**Observation Deck (3-button layout)**
```
[← Previous] [Show/Hide Observation Deck] [Next →]
```
- Previous: Resets visibility, navigates to slide 2
- Toggle: Shows "Hide" when visible, "Show" when hidden
- Next: Resets visibility, navigates to slide 4

**Alignment Lens (3-button layout)**
```
[← Previous] [Show/Hide Alignment Lens] [Next →]
```
- Previous: Resets visibility, navigates to slide 5
- Toggle: Shows "Hide" when visible, "Show" when hidden
- Next: Resets visibility, navigates to slide 7

**Duplicate Navigation at Bottom:**
Both components render duplicate nav buttons at bottom when expanded:
- Same functionality as top buttons
- Accessible without scrolling back up
- Consistent styling and behavior

### State Management

**Session State Keys:**
- `demo_obs_deck_visible` (Boolean) - Observation Deck visibility
- `slide_7_drift_visible` (Boolean) - Alignment Lens visibility
- `last_demo_slide` (Integer) - Track current slide for reset logic

**Default State:**
- Both start as `False` (hidden)
- Initialize on first render if key doesn't exist
- Explicitly reset to `False` on navigation away

### Why These Specifics Matter

1. **20px transform** creates dramatic, professional motion (10px felt too subtle)
2. **ease-in-out** provides balanced, smooth animation (vs ease-out's abrupt start)
3. **Anchor before buttons** ensures immediate accessibility (vs content anchor)
4. **Explicit reset** is reliable (vs tracking previous slide state)
5. **100ms delay** allows Streamlit DOM to render before scroll

### Testing Checklist

When modifying these components:
- [ ] Fade animation is smooth (1.0s ease-in-out)
- [ ] Transform distance is 20px (not 10px)
- [ ] Auto-scroll brings buttons into immediate view
- [ ] Navigation away resets visibility to hidden
- [ ] Returning to slide shows default hidden state
- [ ] Toggle button text updates correctly (Show/Hide)
- [ ] Both top and bottom nav buttons work identically
- [ ] Behavior matches between Observation Deck and Alignment Lens

---

## 7. Files Implementing This Schema

### Primary Components
- `components/conversation_display.py` - Demo Mode Q&A slides, Alignment Lens
- `components/observation_deck.py` - Observation Deck rendering (to be updated for Beta)
- `demo_mode/telos_framework_demo.py` - Demo slide content and configuration

### Shared Resources
- Color constants should be extracted to shared config
- Animation keyframes should be reusable across components
- Layout utilities for consistent box rendering

---

## 8. Future Maintenance

When updating UI elements:

1. **Check both Demo and Beta** - Changes must propagate consistently
2. **Test responsive behavior** - Validate on mobile, tablet, desktop
3. **Verify color contrast** - WCAG AA compliance for accessibility
4. **Maintain animation timing** - 1.0s standard unless specific UX reason
5. **Update this schema** - Document any new patterns or changes

---

## Git History

Key commits establishing this schema:
- `54e2b63` - Fix fidelity box sizing at native resolution
- `ae27fa6` - Make Primacy State box same size as fidelity boxes
- `75060ad` - Fix HTML rendering issue in Observation Deck
- `69c5654` - Change Observation Deck to all green for perfect fidelity
- `7b545c0` - Fix Observation Deck fidelity alignment
- `2a96655` - Animation and alignment improvements

---

**Last Updated:** 2025-01-17
**Schema Version:** 1.0
**Status:** ✅ Active - Implement consistently across Demo and Beta
