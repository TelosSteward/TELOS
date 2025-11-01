# TELOS Observatory User Testing Plan

**Purpose**: Validate Observatory UI usability and identify bugs before grant application submission

**Timeline**: 3-5 days before finalization

**Target**: 5-10 test users with varied backgrounds

---

## Testing Objectives

### Primary Goals
1. **Usability Validation**: Can users navigate and understand the interface?
2. **Bug Identification**: Find and fix technical issues before demo
3. **Clarity Testing**: Are research results and methodology comprehensible?
4. **Value Proposition**: Do users understand TELOS's significance?

### Success Criteria
- ✅ Users can browse and view studies independently
- ✅ Users understand counterfactual methodology
- ✅ No critical bugs or UI breakage
- ✅ Users can articulate TELOS value proposition after 10-minute exploration

---

## Test Participant Recruitment

### Ideal Mix (5-10 participants)

**Category 1: Domain Experts (2-3 people)**
- AI safety researchers
- Machine learning engineers
- Academic researchers in governance/ethics
- **Why**: Validate technical accuracy and research rigor

**Category 2: Grant Reviewers (1-2 people)**
- People who have reviewed grants or funding applications
- Professors or lab directors with funding experience
- **Why**: Test if value proposition is clear and compelling

**Category 3: Technical Adjacent (1-2 people)**
- Software engineers not in AI
- Data scientists
- **Why**: Test if interface is intuitive without deep AI knowledge

**Category 4: Non-Technical Stakeholders (1-2 people)**
- Policy researchers
- Science communicators
- Educated generalists
- **Why**: Test if research is accessible to broader audience

### Recruitment Channels
- AI safety community contacts
- Academic networks
- Professional connections
- Twitter/LinkedIn reach-out
- AI safety forums (LessWrong, EA Forum)

---

## Testing Protocol

### Setup Instructions for Participants

**Pre-Test Email**:
```
Subject: User Testing - TELOS AI Governance Observatory (10-15 min)

Hi [Name],

I'm conducting user testing for TELOS Observatory, an AI governance research
platform I'm submitting for grant funding. I'd appreciate 10-15 minutes of
your time to explore the interface and provide feedback.

Setup:
1. Open: http://localhost:8502 (or deployed link if available)
2. No preparation needed - fresh eyes are valuable
3. Think aloud as you explore
4. I'll ask a few questions afterward

Scheduling: [Calendly link or specific time options]

Thanks!
[Your name]
```

### Testing Session Structure (10-15 minutes)

**Phase 1: Open Exploration (5 minutes)**
- Give minimal instruction: "This is an AI governance measurement tool. Explore freely."
- Observe without intervening
- Note: Where do they click first? What confuses them?

**Phase 2: Task-Based Testing (5 minutes)**

Ask participants to:
1. **Find a study with effective governance** (ΔF > 0)
2. **Explain what 'ΔF' means** in their own words
3. **View a research brief** for any study
4. **Describe what TELOS does** based on the "About" tab
5. **Identify the effectiveness rate** for ShareGPT dataset

**Phase 3: Feedback Questions (5 minutes)**

See "Feedback Collection Form" below.

---

## Feedback Collection Form

### Usability Questions (1-5 scale)

1. **Navigation**: How easy was it to find information?
   - 1 (Very Difficult) → 5 (Very Easy)

2. **Clarity**: How well did you understand the research methodology?
   - 1 (Very Confusing) → 5 (Very Clear)

3. **Visual Design**: How professional does the interface feel?
   - 1 (Unprofessional) → 5 (Very Professional)

4. **Information Density**: Was the amount of information...
   - 1 (Too Little) → 3 (Just Right) → 5 (Too Much)

### Open-Ended Questions

1. **First Impression**: What was your immediate reaction when you opened the app?

2. **Confusing Elements**: What was most confusing or unclear?

3. **Value Proposition**: In your own words, what does TELOS do and why does it matter?

4. **Grant Worthiness**: Based on what you saw, does this seem worthy of $150k funding? Why/why not?

5. **Missing Information**: What questions do you have that the UI didn't answer?

6. **Suggestions**: If you could change one thing, what would it be?

### Technical Issues

- **Bugs Encountered**: List any errors, broken features, or unexpected behavior

- **Browser/Device**: What browser and OS were you using?

- **Performance**: Any slowness or lag?

---

## Bug Tracking Template

### Critical Bugs (Fix Immediately)
- **B001**: [Description]
  - Steps to reproduce:
  - Browser/OS:
  - Impact: [High/Medium/Low]
  - Status: [Open/In Progress/Fixed]

### Medium Priority (Fix Before Submission)
- **B002**: [Description]

### Low Priority (Nice to Have)
- **B003**: [Description]

### UI/UX Improvements (Not Bugs)
- **UX001**: [Description]

---

## Testing Checklist

### Before Testing Sessions
- [ ] Observatory running on accessible URL (localhost or deployed)
- [ ] All 56 studies loading correctly
- [ ] Statistics calculating properly
- [ ] Research briefs rendering
- [ ] Counterfactual charts displaying
- [ ] No console errors on page load
- [ ] Tested on Chrome, Firefox, Safari
- [ ] Mobile responsiveness checked (optional)

### During Testing
- [ ] Record session (with permission) or take detailed notes
- [ ] Note exact quotes from participants
- [ ] Track time spent on each section
- [ ] Document confusion points
- [ ] Capture bug reproduction steps

### After Each Session
- [ ] Log bugs in tracking template
- [ ] Note recurring feedback themes
- [ ] Update UI immediately for critical issues
- [ ] Send thank you to participant

---

## Analysis Plan

### After 3-5 Sessions

**Look for Patterns**:
- Do 3+ people mention the same confusion point? → Priority fix
- Do 3+ people misunderstand the same concept? → Clarify copy
- Do multiple people struggle with same task? → UI redesign needed

**Quantitative Thresholds**:
- **Navigation** average < 3.0: Major usability problem
- **Clarity** average < 3.0: Methodology explanation needs work
- **Visual Design** average < 3.0: Polish UI before screenshots

**Qualitative Red Flags**:
- "I don't understand what this is for" → Value prop unclear
- "This looks like a school project" → Professionalization needed
- "Is this real data?" → Credibility/trust issue

### Iteration Protocol

**Critical Issues** (Fix same day):
- Interface broken
- Critical information missing
- Value proposition incomprehensible

**High Priority** (Fix within 24 hours):
- Recurring confusion (3+ participants)
- Visual/professionalism concerns
- Key tasks difficult to complete

**Medium Priority** (Fix before screenshots):
- Minor UI polish
- Copy improvements
- Nice-to-have features

**Low Priority** (Post-submission):
- Advanced features
- Edge case improvements
- Optimization

---

## Specific Testing Scenarios

### Scenario 1: Domain Expert Review
**Participant**: AI safety researcher

**Custom Questions**:
1. Does the counterfactual methodology seem rigorous?
2. Are there obvious flaws in the validation approach?
3. Would you trust these results in a research paper?
4. How does this compare to other governance evaluation tools you've seen?

### Scenario 2: Grant Reviewer Perspective
**Participant**: Professor with funding experience

**Custom Questions**:
1. Does $150k seem justified for what's demonstrated?
2. What additional evidence would make this more compelling?
3. Does the "research infrastructure" framing make sense?
4. How does this compare to other grant applications you've reviewed?

### Scenario 3: Non-Technical Stakeholder
**Participant**: Policy researcher or generalist

**Custom Questions**:
1. Can you explain what ΔF means without looking at documentation?
2. Did you understand the medical/financial domain examples?
3. Would this be useful for regulatory compliance?
4. What would make this more accessible to non-technical audiences?

---

## Documentation Protocol

### Create After Testing

**File**: `telos_outreach/USER_TESTING_RESULTS.md`

**Contents**:
```markdown
# User Testing Results Summary

**Dates**: [Date range]
**Participants**: 5-10 testers
**Sessions Completed**: X/Y

## Participant Demographics
- Domain experts: X
- Grant reviewers: X
- Technical adjacent: X
- Non-technical: X

## Quantitative Results

| Metric | Average Score | Notes |
|--------|---------------|-------|
| Navigation | X.X / 5 | |
| Clarity | X.X / 5 | |
| Visual Design | X.X / 5 | |
| Information Density | X.X / 5 | |

## Key Findings

### What Worked Well
1. [Finding from multiple participants]
2. [Finding from multiple participants]

### Critical Issues (Fixed)
1. **Issue**: [Description]
   - **Impact**: X participants affected
   - **Fix**: [What was changed]

### Recurring Feedback Themes
1. [Theme mentioned by 3+ participants]
2. [Theme mentioned by 3+ participants]

## Value Proposition Clarity

**Can participants articulate TELOS purpose?**
- Yes: X / Y
- Partial: X / Y
- No: X / Y

**Example articulations** (direct quotes):
> "TELOS is..."

## Grant Worthiness Assessment

**Would you fund this?** (from grant reviewer perspective)
- Yes: X / Y
- Maybe: X / Y
- No: X / Y

**Reasons cited**:
- Positive: [List]
- Concerns: [List]

## Changes Made Based on Testing

1. **[Change]**: [Why we made it]
2. **[Change]**: [Why we made it]

## Conclusion

[Overall assessment of Observatory readiness for grant submission]
```

---

## Next Steps After Testing

### Immediate (Same Day as Final Test)
1. Fix all critical bugs
2. Polish UI based on recurring feedback
3. Clarify any confusing copy
4. Update screenshots if UI changed significantly

### Short-Term (1-2 days)
1. Complete `USER_TESTING_RESULTS.md` documentation
2. Add testing summary to grant application
3. Update "About" tab with clarifications from testing
4. Final QA pass on all features

### Grant Application Integration
1. Add section: "Observatory validated with 10 independent user tests"
2. Include testing results summary in appendix
3. Reference user feedback as evidence of usability
4. Cite specific improvements made based on testing

---

## Testing Success Indicators

### Ready for Submission When:
- ✅ 5+ independent users tested successfully
- ✅ No critical bugs remaining
- ✅ Average clarity score ≥ 3.5/5
- ✅ 80%+ of users can articulate value proposition
- ✅ All domain experts validate methodology rigor
- ✅ Visual design receives ≥ 3.0/5 average
- ✅ Testing results documented

---

## Contact Script for Participants

**Recruitment Message** (personalized):
```
Hi [Name],

I'm [Your Name], working on TELOS - an AI governance measurement platform
I'm submitting for LTFF funding ($150k grant application).

Before submission, I'm conducting user testing to ensure the Observatory
interface is clear and professional. Would you be willing to spend 10-15
minutes exploring the platform and providing feedback?

Your background in [their expertise] would be particularly valuable for
[specific aspect: methodology validation / usability / value proposition].

Available times: [options]

Thanks for considering!
```

**Thank You Message**:
```
Thanks for testing TELOS Observatory!

Your feedback on [specific insight they provided] was incredibly helpful.
I'm making changes to [what you're fixing based on their feedback].

I'll keep you posted on the grant application outcome. If LTFF funds this,
the open-source release will be available for [their use case].

Best,
[Your name]
```

---

## Testing Metrics Target

**Minimum Viable Testing**:
- 5 participants
- 3+ different background categories
- Average clarity ≥ 3.0/5
- Zero critical bugs

**Ideal Testing**:
- 10 participants
- All 4 background categories represented
- Average clarity ≥ 4.0/5
- Average navigation ≥ 4.0/5
- 90%+ can articulate value proposition
- 2+ domain experts validate rigor

---

**Status**: Ready to begin recruitment
**Next**: Identify 5-10 potential participants and schedule sessions
**Timeline**: Complete testing 2-3 days before grant submission deadline
