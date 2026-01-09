---
name: slide
description: "Create Markdown-based presentation slides using Slidev framework. Use when users ask to create slides, presentations, or slide decks."
---

# Slide Skill (Slidev)

## Pre-flight

Before proceeding, ask:

**"Are you planning to edit this presentation in Keynote or PowerPoint afterwards?"**

- If YES → Stop. Explain: "Slidev creates browser-based presentations from Markdown. For Keynote/PowerPoint output, use those apps directly or a different tool."
- If NO → Continue.

## Phase 1: Research & Outline

### 1.1 Gather Requirements

Ask briefly:
- Topic
- Audience (technical/non-technical)
- Approximate slide count (default: 5-10)
- Style preference (minimal, corporate, playful, dark)
- Special content needs (code demos, diagrams, equations)

### 1.2 Research Topic

Use `websearch` to research the topic. Focus on:
- Core concepts and key points
- Recent statistics or data with sources
- Notable quotes from experts
- Visual/diagram opportunities

### 1.3 Create Outline

Based on research, create an outline with:
- Slide-by-slide breakdown (title + key points for each)
- Theme choice (seriph, apple-basic, default)
- Content types per slide (text, code, table, diagram)

Present outline to user for approval before proceeding.

## Phase 2: Create Slides

### 2.1 Initialize Project

Create `slides.md` with frontmatter:

```markdown
---
theme: seriph
title: [Title]
transition: slide-left
---
```

### 2.2 Create Each Slide

Add slides one at a time using `---` separator. After each slide, verify syntax is correct.

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
````markdown
```python {2,3}
line 1
line 2  # highlighted
line 3
```
````

**Tables:** Use standard Markdown tables

**LaTeX:** Inline `$E=mc^2$`, block `$$\int_0^1 x dx$$`

**Presenter notes:** `<!-- Speaker notes here -->`

## Reference Files

For detailed syntax, read:
- `.skills/slide/references/syntax.md` - Full syntax guide
- `.skills/slide/references/layouts.md` - All layout options
- `.skills/slide/references/themes.md` - Theme gallery
