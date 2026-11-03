{% for command in commands %}{{ command.name }}() { printf '%s\n' {{ command.message }} >&2; return 1; }
{% endfor %}{{ script }}