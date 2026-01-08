---
name: slide
description: Create Markdown-based presentation slides using Slidev framework. Use when users ask to create slides, presentations, or slide decks. SCOPE: Only creates NEW presentations—cannot modify existing slides or output Keynote/PowerPoint formats (.pptx, .key).
---

# Slide Skill (Slidev)

## Pre-flight

Before proceeding, ask:

**"Are you planning to edit this presentation in Keynote or PowerPoint afterwards?"**

- If YES → Stop. Explain: "Slidev creates browser-based presentations from Markdown. For Keynote/PowerPoint output, use those apps directly or a different tool."
- If NO → Continue.

**"Is this a new presentation, or do you want to modify an existing one?"**

- If MODIFY → Stop. This skill only creates new slides. Request the user provide content to recreate.
- If NEW → Continue.

## Phase 1: Research & Design Outline

### 1.1 Gather Requirements

Ask briefly:
- Topic
- Audience (technical/non-technical)
- Approximate slide count (default: 10-15)
- Style preference (minimal, corporate, playful, dark)
- Special content needs (code demos, diagrams, equations)

### 1.2 Research Topic

Delegate topic research to a subagent:

```
task references/subagent_topic_research.md "Research the topic: {TOPIC}"
```

The subagent will return key points, statistics, quotes, and suggested structure.

### 1.3 Art Direction

Delegate visual design to a subagent:

```
task references/subagent_art_direction.md "Design art direction for: {TOPIC}, audience: {AUDIENCE}, style: {STYLE}"
```

### 1.4 Create Outline Document

Combine research and art direction into an outline with:
- Slide-by-slide breakdown (title + key points for each)
- Art direction (theme, color scheme, visual style)
- Content types per slide (text, code, table, image, quote)

Example format:
```
# Presentation Outline: [Topic]

## Art Direction
- Theme: seriph
- Style: [minimal/corporate/playful]
- Color accents: [colors]

## Slides

1. **Title Slide** (layout: cover)
   - Title: ...
   - Subtitle: ...

2. **Agenda** (layout: default)
   - Bullet points of topics

3. **[Topic 1]** (layout: two-cols)
   - Left: key points
   - Right: code example

...
```

### 1.5 Discuss with User

Present the outline to the user. Ask for feedback:
- "Does this structure cover what you need?"
- "Any slides to add, remove, or reorganize?"
- "Happy with the visual direction?"

Wait for approval before proceeding to Phase 2.

## Phase 2: Create Slides (Per-Page)

### 2.1 Initialize Project

Create project directory with `slides.md` containing only the headmatter:

```markdown
---
theme: seriph
title: [Title]
transition: slide-left
---
```

Start the server:
```bash
npx @slidev/cli slides.md &
```

### 2.2 Create Each Slide with Visual Verification

For each slide in the outline:

1. **Append** the slide content to `slides.md`
2. **Delegate verification** to a subagent:
   ```
   task references/subagent_visual_verification.md "Verify slide {N} at http://localhost:3030/{N}"
   ```
3. Review the subagent's status report

If subagent reports ERROR:
1. Check terminal output for error details
2. Fix the slide content
3. Re-run verification task

Only proceed to next slide after current one passes.

## Phase 3: Final Verification

After all slides created:

```bash
for i in {1..N}; do echo "Slide $i: $(curl -s http://localhost:3030/$i | grep -o 'An error occurred' || echo 'OK')"; done
```

Confirm all slides pass. Present final result to user.

## Syntax Quick Reference

**Slide separator:** `---`

**Layouts:** `default`, `center`, `cover`, `two-cols`, `image-right`, `quote`, `section`, `fact`, `end`

**Two columns:**
```markdown
---
layout: two-cols
---

Left content

::right::

Right content
```

**Code with highlighting:**
```markdown
```python {2,3}
line 1
line 2  # highlighted
```
```

**Tables:** Use standard Markdown tables (avoid Mermaid pie/timeline charts—they may not render)

**LaTeX:** Inline `$E=mc^2$`, block `$$\int_0^1 x dx$$`

**Presenter notes:** `<!-- Speaker notes here -->`

## Reference Files

- [references/syntax.md](references/syntax.md) - Full syntax guide
- [references/layouts.md](references/layouts.md) - All layout options
- [references/themes.md](references/themes.md) - Theme gallery
