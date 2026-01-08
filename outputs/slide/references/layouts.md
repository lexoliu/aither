# Slidev Layouts Reference

## Basic Layouts

### default

Standard slide with content.

```yaml
---
layout: default
---
```

### center

Content centered both horizontally and vertically.

```yaml
---
layout: center
---
```

### cover

Title slide with large centered heading.

```yaml
---
layout: cover
background: /cover.jpg
---
```

## Content Layouts

### two-cols

Two-column layout with `::right::` separator.

```markdown
---
layout: two-cols
---

# Left Column

Content here

::right::

# Right Column

More content
```

### two-cols-header

Two columns with shared header.

```markdown
---
layout: two-cols-header
---

# Header spans both columns

::left::

Left content

::right::

Right content
```

## Image Layouts

### image

Full-bleed image as background.

```yaml
---
layout: image
image: /path/to/image.jpg
---
```

### image-right

Image on right, content on left.

```yaml
---
layout: image-right
image: /diagram.png
---

# Explanation

Content appears on the left side.
```

### image-left

Image on left, content on right.

```yaml
---
layout: image-left
image: /photo.jpg
---
```

## Special Layouts

### quote

Centered quote with attribution.

```markdown
---
layout: quote
---

# "Quote text here"

— Attribution
```

### section

Section divider with large centered text.

```yaml
---
layout: section
---

# Part 2: Implementation
```

### fact

Large centered fact/statistic.

```markdown
---
layout: fact
---

# 100M+

Users worldwide
```

### statement

Bold statement layout.

```yaml
---
layout: statement
---
```

### intro

Introduction slide for speakers.

```yaml
---
layout: intro
---
```

### end

Closing slide.

```yaml
---
layout: end
---
```

## Embed Layouts

### iframe

Full-page iframe embed.

```yaml
---
layout: iframe
url: https://example.com
---
```

### iframe-right

Iframe on right, content on left.

```yaml
---
layout: iframe-right
url: https://example.com
---

# Description

Content here
```

### iframe-left

Iframe on left, content on right.

```yaml
---
layout: iframe-left
url: https://example.com
---
```

## Layout Quick Reference

| Layout | Use Case |
|--------|----------|
| `default` | Regular content slides |
| `center` | Centered content, announcements |
| `cover` | Title/intro slides |
| `two-cols` | Comparisons, code + output |
| `image-right` | Image with explanation |
| `quote` | Quotations |
| `section` | Section dividers |
| `fact` | Statistics, key numbers |
| `end` | Thank you / closing |
