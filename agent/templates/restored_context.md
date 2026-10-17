These files were named by the handoff above and have been re-read for you. Do not read them again unless you need a part that was truncated.
{% for file in files %}

### {{ file.path }}{% if file.truncated %} (truncated){% endif %}

```
{{ file.content }}
```
{% endfor %}
{% if omitted > 0 %}

{{ omitted }} further file(s) named by the handoff were left out of this restoration to stay inside the context budget. Read them on demand.
{% endif %}
