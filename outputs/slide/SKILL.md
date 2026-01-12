---
name: slide
description: "Create Markdown-based presentation slides using Slidev. Use when users ask to create slides, presentations, or slide decks."
---

# Slide Skill (Slidev)

## Pre-flight Check

Ask: **"Will you edit this in Keynote/PowerPoint later?"**
- YES: Stop. Recommend using those apps directly.
- NO: Continue.

## Workflow

### Phase 1: Research (Optional)

If the topic needs research, use the builtin research subagent:

```bash
task research --prompt "Research [TOPIC] for a presentation. Find key concepts, recent statistics, and expert insights."
```

Use explicit flags to avoid positional argument ambiguity:

```bash
task --subagent research --prompt "Research [TOPIC] for a presentation. Find key concepts, recent statistics, and expert insights."
```

### Phase 2: Requirements (Main Agent)

Gather from user:
- Topic and key message
- Audience: technical / general / executive
- Slide count (default: 5-8)
- Style: dark / light / minimal / corporate / playful

### Phase 3: Art Direction (Subagent)

Launch art direction subagent to create design guide and outline:

```bash
task --subagent .skills/slide/subagents/art_direction.md --prompt "Create design guide for: [TOPIC]. Audience: [AUDIENCE]. Style: [STYLE]. Slide count: [N]."
```

The subagent returns a YAML design guide containing:
- Theme selection
- Color palette
- Typography guidance
- Slide-by-slide outline with layouts and content

### Phase 4: User Approval

Present the outline to the user:
- Show slide titles and structure
- Confirm design direction
- **Get approval before proceeding**

### Phase 5: Parallel Slide Creation (Subagents)

Launch one slide_creator subagent per slide **in parallel**:

```bash
task --subagent .skills/slide/subagents/slide_creator.md --prompt "Slide 1: [SPEC]. Design: [DESIGN_GUIDE]"
task --subagent .skills/slide/subagents/slide_creator.md --prompt "Slide 2: [SPEC]. Design: [DESIGN_GUIDE]"
task --subagent .skills/slide/subagents/slide_creator.md --prompt "Slide 3: [SPEC]. Design: [DESIGN_GUIDE]"
# ... one per slide
```

Each subagent:
1. Creates standalone temp file with slide content
2. Renders to PNG for visual verification
3. Views the image to ensure quality
4. Iterates until the slide looks correct
5. Returns verified slide markdown

### Phase 6: Assembly & Preview (Main Agent)

Collect all slide outputs and assemble into final presentation:

```bash
mkdir -p artifacts
cat > artifacts/slides.md << 'EOF'
---
theme: [from design guide]
title: [presentation title]
transition: [from design guide]
---

[Slide 1 content]

---

[Slide 2 content]

---

[... all slides ...]
EOF
```

Clean up temp files and start preview server:

```bash
rm -rf artifacts/temp
cd artifacts && npx slidev slides.md --port 3030 &
```

Tell user: **"Preview at http://localhost:3030"**

### Phase 7: Export (Optional)

If user wants PDF:

```bash
cd artifacts && npx slidev export slides.md --output slides.pdf
```

## Reference Files

Read these from the skill directory:

- `.skills/slide/references/syntax.md` - Full Slidev syntax guide
- `.skills/slide/references/layouts.md` - Layout options and usage
- `.skills/slide/references/themes.md` - Theme gallery and recommendations
