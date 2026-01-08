# Slidev Themes Reference

## Using Themes

Set theme in headmatter:

```yaml
---
theme: seriph
---
```

Install theme:

```bash
npm install @slidev/theme-<name>
```

## Official Themes

### default

Built-in minimal theme. No installation required.

### seriph

Modern, clean design with serif typography. Good for most presentations.

```bash
npm install @slidev/theme-seriph
```

## Community Themes

### apple-basic

Apple Keynote-inspired design.

```bash
npm install slidev-theme-apple-basic
```

### dracula

Dark theme with Dracula color scheme.

```bash
npm install slidev-theme-dracula
```

### geist

Vercel/Geist design system.

```bash
npm install slidev-theme-geist
```

### penguin

Playful theme with illustrations.

```bash
npm install slidev-theme-penguin
```

### bricks

Clean, professional slides.

```bash
npm install slidev-theme-bricks
```

### unicorn

Colorful gradient theme.

```bash
npm install slidev-theme-unicorn
```

### shibainu

Cute theme with shiba inu illustrations.

```bash
npm install slidev-theme-shibainu
```

### academic

Academic/research presentation style.

```bash
npm install slidev-theme-academic
```

## Theme Recommendations

| Use Case | Recommended Theme |
|----------|-------------------|
| Tech talks | `seriph`, `geist` |
| Corporate | `default`, `bricks` |
| Academic | `academic`, `default` |
| Dark mode | `dracula` |
| Playful | `penguin`, `unicorn` |
| Apple-style | `apple-basic` |

## Custom Styling

Override theme styles with scoped CSS:

```markdown
<style>
h1 {
  background: linear-gradient(45deg, #4e8cff, #34d399);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
</style>
```

Or UnoCSS classes:

```markdown
# Title {.bg-gradient-to-r .from-blue-500 .to-green-400}
```
