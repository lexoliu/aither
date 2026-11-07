//! The parse finds every command line a script would run, and carries what a
//! command policy needs to judge it: the rules of the policies an embedder
//! writes on it (kills by name, a foreground `sleep`, file reads) are
//! expressed here against the parse alone.

use super::*;

fn parsed(script: &str) -> ShellScript {
    ShellScript::parse(script).unwrap_or_else(|error| panic!("`{script}` should parse: {error}"))
}

fn programs(script: &str) -> Vec<String> {
    parsed(script)
        .commands()
        .iter()
        .map(|command| command.program().to_string())
        .collect()
}

/// Programs that signal processes selected by name, user or other pattern.
const KILLERS: &[&str] = &["pkill", "killall", "killall5", "skill"];

/// A kill-by-name rule written against the parse: the program that kills by
/// name or pattern, if any.
fn kill_by_name(script: &str) -> Option<String> {
    let script = parsed(script);
    for command in script.commands() {
        let arguments = command.arguments();
        let by_name = match command.program() {
            "fuser" => arguments.iter().any(|argument| {
                argument == "--kill"
                    || argument
                        .strip_prefix('-')
                        .is_some_and(|flags| !flags.starts_with('-') && flags.contains('k'))
            }),
            "taskkill" => arguments.iter().any(|argument| {
                let selector = argument.trim_start_matches(['/', '-']).to_ascii_lowercase();
                argument.starts_with(['/', '-']) && matches!(selector.as_str(), "im" | "fi")
            }),
            name => KILLERS.contains(&name),
        };
        if by_name {
            return Some(command.program().to_string());
        }
    }
    let lookups = script
        .commands()
        .iter()
        .filter(|command| matches!(command.program(), "pgrep" | "pidof"));
    for lookup in lookups {
        // `kill $(pgrep NAME)`.
        let substituted_into_kill = lookup
            .substituted_into()
            .is_some_and(|line| script.line(line).any(|command| command.program() == "kill"));
        // `pgrep NAME | xargs kill`.
        let piped_into_kill = lookup.pipeline().is_some_and(|producer| {
            script.commands().iter().any(|command| {
                command.program() == "kill"
                    && command.pipeline().is_some_and(|consumer| {
                        consumer.pipeline == producer.pipeline && consumer.stage > producer.stage
                    })
                    && script
                        .wrappers(command)
                        .any(|wrapper| wrapper.program() == "xargs")
            })
        });
        if substituted_into_kill || piped_into_kill {
            return Some("kill of pgrep/pidof output".to_string());
        }
    }
    None
}

/// A foreground-`sleep` rule written against the parse.
fn sleeps_in_the_foreground(script: &str) -> bool {
    parsed(script)
        .commands()
        .iter()
        .any(|command| !command.is_background() && command.program() == "sleep")
}

/// A file-read rule written against the parse: the positional words of
/// `cat`, `head` and `tail`, `tail`'s follow forms excluded.
fn file_reads(script: &str) -> Vec<String> {
    let mut reads = Vec::new();
    for command in parsed(script).commands() {
        if !matches!(command.program(), "cat" | "head" | "tail") {
            continue;
        }
        let follows = command.program() == "tail"
            && command.arguments().iter().any(|argument| {
                argument == "-f"
                    || argument.starts_with("--follow")
                    || argument
                        .strip_prefix('-')
                        .is_some_and(|flags| !flags.starts_with('-') && flags.contains('f'))
            });
        if !follows {
            reads.extend(
                command
                    .arguments()
                    .iter()
                    .filter(|argument| !argument.starts_with('-'))
                    .cloned(),
            );
        }
    }
    reads
}

#[test]
fn commands_are_found_through_shell_syntax() {
    for script in [
        "pkill foo",
        "/usr/bin/pkill -f foo",
        "'pkill' foo",
        "p''kill foo",
        "\\pkill foo",
        "PKILL foo.exe",
        "echo hi && pkill foo",
        "echo hi || pkill foo",
        "echo hi; pkill foo",
        "echo hi\npkill foo",
        "sleep 1 & pkill foo",
        "cat x | pkill foo",
        "! pkill foo",
        "(pkill foo)",
        "{ pkill foo; }",
        "if pkill foo; then echo a; fi",
        "if true; then pkill foo; fi",
        "if false; then :; else pkill foo; fi",
        "while pkill foo; do :; done",
        "until true; do pkill foo; done",
        "for i in 1 2; do pkill foo; done",
        "for ((i=0;i<1;i++)); do pkill foo; done",
        "case x in x) pkill foo;; esac",
        "f() { pkill foo; }; f",
        "function f { pkill foo; }",
        "echo $(pkill foo)",
        "echo `pkill foo`",
        "echo \"$(pkill foo)\"",
        "echo ${x:-$(pkill foo)}",
        "echo $(( $(pkill foo) + 1 ))",
        "x=$(pkill foo)",
        "FOO=1 pkill foo",
        "cat <(pkill foo)",
        "echo hi > $(pkill foo)",
        "[[ $(pkill foo) ]]",
        "for x in $(pkill foo); do :; done",
        "case $(pkill foo) in x) :;; esac",
        "echo $(echo $(echo $(pkill foo)))",
        "cat <<EOF\n$(pkill foo)\nEOF",
    ] {
        assert!(
            programs(script).iter().any(|program| program == "pkill"),
            "`{script}` runs pkill: {:?}",
            programs(script)
        );
    }
}

#[test]
fn commands_are_found_behind_wrappers_and_in_shell_strings() {
    for script in [
        "env pkill foo",
        "env FOO=1 BAR=2 pkill foo",
        "env -u HOME pkill foo",
        "exec pkill foo",
        "command pkill foo",
        "builtin pkill foo",
        "sudo -u root pkill foo",
        "sudo -Eu root pkill foo",
        "doas pkill foo",
        "nohup pkill foo &",
        "nice -n 5 pkill foo",
        "nice -5 pkill foo",
        "ionice -c 3 pkill foo",
        "setsid pkill foo",
        "stdbuf -o0 pkill foo",
        "timeout -s KILL 5 pkill foo",
        "timeout --signal=KILL 5 pkill foo",
        "time pkill foo",
        "echo foo | xargs -n1 pkill",
        "echo foo | xargs -I {} pkill {}",
        "busybox pkill foo",
        "env -- pkill foo",
        "sudo env nice timeout 3 /usr/bin/pkill foo",
        "find . -exec pkill {} \\;",
        "find . -name x -execdir pkill {} +",
        "sh -c 'pkill foo'",
        "bash -lc 'pkill foo'",
        "bash -o pipefail -c 'pkill foo'",
        "sh -c \"sh -c 'pkill foo'\"",
        "su -c 'pkill foo' root",
        "eval pkill foo",
        "eval 'echo a; pkill foo'",
        "sh <<< 'pkill foo'",
        "bash <<EOF\npkill foo\nEOF",
        "sh -c 'echo $(pkill foo)'",
    ] {
        assert!(
            programs(script).iter().any(|program| program == "pkill"),
            "`{script}` runs pkill: {:?}",
            programs(script)
        );
    }
}

#[test]
fn words_that_only_name_a_program_are_not_commands() {
    for script in [
        "echo pkill",
        "echo 'pkill foo' killall",
        "rg killall file",
        "cat pkill.txt",
        "ls /usr/bin/pkill",
        "which pkill",
        "command -v pkill",
        "command -V pkill",
        "echo pkill | cat",
        "env echo pkill",
        "timeout 5 echo pkill",
        "xargs echo pkill",
        "sh -c 'echo pkill'",
        "find . -name pkill",
        "find . -exec echo pkill {} \\;",
        "sh script.sh pkill",
        "echo \"$(echo pkill)\"",
        "x=pkill",
        "mypkill foo",
        "cat <<EOF\npkill foo\nEOF",
        "sh script.sh <<EOF\npkill foo\nEOF",
        "git commit -m 'stop using pkill'",
        "$tool foo",
        "$(echo pkill) foo",
        "\"${tool}\" foo",
        "",
        "   ",
    ] {
        assert!(
            !programs(script).iter().any(|program| program == "pkill"),
            "`{script}` does not run pkill: {:?}",
            programs(script)
        );
    }
}

#[test]
fn a_command_knows_its_wrapper_interpreter_and_line() {
    let script = parsed("sudo -u root env FOO=1 kill 1; sh -c 'kill 2'");
    let names: Vec<_> = script
        .commands()
        .iter()
        .map(ShellCommand::program)
        .collect();
    assert_eq!(names, ["sudo", "env", "kill", "sh", "kill"]);
    let commands = script.commands();
    assert_eq!(
        commands[0].argv(),
        ["sudo", "-u", "root", "env", "FOO=1", "kill", "1"]
    );
    assert_eq!(commands[2].argv(), ["kill", "1"]);
    assert_eq!(commands[2].arguments(), ["1"]);
    let wrappers: Vec<_> = script
        .wrappers(&commands[2])
        .map(ShellCommand::program)
        .collect();
    assert_eq!(wrappers, ["env", "sudo"]);
    assert_eq!(
        script.line(commands[0].line()).count(),
        3,
        "a wrapper and what it runs are one line"
    );
    assert_eq!(commands[4].interpreter(), Some(commands[3].id()));
    assert_eq!(commands[4].wrapper(), None);
    assert_ne!(commands[4].line(), commands[3].line());
}

#[test]
fn a_command_knows_its_pipeline_stage_and_substitution() {
    let script = parsed("a | sudo b | c; d; kill $(e | f)");
    let by_name = |name: &str| {
        script
            .commands()
            .iter()
            .find(|command| command.program() == name)
            .unwrap()
    };
    let a = by_name("a").pipeline().unwrap();
    let b = by_name("b").pipeline().unwrap();
    assert_eq!(by_name("sudo").pipeline(), Some(b));
    assert_eq!((a.pipeline, a.stage, b.stage), (b.pipeline, 0, 1));
    assert_eq!(by_name("c").pipeline().unwrap().stage, 2);
    assert_eq!(
        by_name("d").pipeline(),
        None,
        "a lone command is in no pipeline"
    );
    let e = by_name("e");
    assert_eq!(e.substituted_into(), Some(by_name("kill").line()));
    assert_ne!(e.pipeline().unwrap().pipeline, a.pipeline);
    assert_eq!(by_name("kill").substituted_into(), None);
}

#[test]
fn a_command_in_an_async_position_is_background() {
    let script = parsed("sleep 30 & (sleep 1; echo one) & wait");
    let background: Vec<_> = script
        .commands()
        .iter()
        .map(|command| (command.program(), command.is_background()))
        .collect();
    assert_eq!(
        background,
        [
            ("sleep", true),
            ("sleep", true),
            ("echo", true),
            ("wait", false)
        ]
    );
}

#[test]
fn a_kill_by_name_rule_is_expressible() {
    for script in [
        "pkill foo",
        "killall5 -9",
        "skill -KILL foo",
        "fuser -k 8080/tcp",
        "fuser -km /mnt",
        "taskkill /IM foo.exe /F",
        "sudo env nice timeout 3 /usr/bin/pkill foo",
        "sh -c 'killall foo'",
    ] {
        assert!(kill_by_name(script).is_some(), "`{script}` kills by name");
    }
    for script in [
        "kill $(pgrep foo)",
        "kill -9 $(pgrep -f foo)",
        "kill `pidof foo`",
        "kill \"$(pgrep foo)\"",
        "kill $(pgrep foo | head -1)",
        "/bin/kill $(/usr/bin/pgrep foo)",
        "sudo kill -9 $(pidof foo)",
        "env X=1 kill $(pidof foo)",
        "sh -c 'kill $(pgrep foo)'",
        "echo a && kill $(pgrep foo)",
        "pgrep foo | xargs kill",
        "pgrep -f foo | xargs -r kill -9",
        "pidof foo | sudo xargs kill",
        "pgrep foo | xargs sudo kill",
        "pgrep foo | tr '\\n' ' ' | xargs kill",
        "/usr/bin/pgrep foo | /usr/bin/xargs /bin/kill",
        "sudo pgrep foo | xargs kill",
    ] {
        assert_eq!(
            kill_by_name(script).as_deref(),
            Some("kill of pgrep/pidof output"),
            "`{script}` kills what a lookup by name found"
        );
    }
    for script in [
        "kill 123",
        "sleep 30 & kill $!",
        "kill $(cat pidfile)",
        "kill \"$pid\"",
        "echo $(pgrep foo)",
        "kill $(echo $(pgrep foo))",
        "pgrep foo | wc -l",
        "pgrep foo; kill 123",
        "pidof foo | xargs echo",
        "cat pids | xargs kill",
        "kill $(cat pidfile) && pgrep foo",
        "xargs kill",
        "fuser 8080/tcp",
        "fuser -v /mnt",
        "taskkill /PID 42 /F",
        "man pkill",
        "type killall",
    ] {
        assert_eq!(kill_by_name(script), None, "`{script}` kills by PID");
    }
}

#[test]
fn a_foreground_sleep_rule_is_expressible() {
    for script in [
        "sleep 5",
        "sleep 60; tail -50 out.txt",
        "timeout 5 sleep 3",
        "env sleep 1",
        "until [ -f done ]; do sleep 5; done",
        "sh -c 'sleep 2'",
        "sleep 1 & sleep 2",
    ] {
        assert!(sleeps_in_the_foreground(script), "`{script}` sleeps");
    }
    for script in [
        "sleep 30 & echo $! > child.pid; wait",
        "(sleep 1; echo one > one.txt) & wait",
        "echo 'sleep'",
        "cat sleep.txt",
        "sleepy 5",
    ] {
        assert!(!sleeps_in_the_foreground(script), "`{script}` does not");
    }
}

#[test]
fn a_file_read_rule_is_expressible() {
    assert_eq!(file_reads("cat a.txt"), ["a.txt"]);
    assert_eq!(file_reads("sleep 60; tail -50 out.txt"), ["out.txt"]);
    assert_eq!(file_reads("head -n 5 a.txt b.txt"), ["5", "a.txt", "b.txt"]);
    assert_eq!(file_reads("tail -fn5 log.txt"), [] as [String; 0]);
    assert_eq!(file_reads("tail --follow=name log.txt"), [] as [String; 0]);
    assert_eq!(file_reads("echo tail out.txt"), [] as [String; 0]);
    assert_eq!(file_reads("cat 'tail -f x'"), ["tail -f x"]);
}

#[test]
fn invalid_shell_is_a_parse_error() {
    for script in [
        "echo 'unterminated",
        "if true",
        "echo $(",
        "sh -c 'if'",
        "fi",
    ] {
        assert!(
            ShellScript::parse(script).is_err(),
            "`{script}` is not valid shell"
        );
    }
}

#[test]
fn runaway_nesting_is_a_parse_error() {
    let script = format!("{}pkill foo", "eval ".repeat(MAX_DEPTH + 2));
    let error = ShellScript::parse(&script).expect_err("nesting is bounded");
    assert!(error.reason().contains("nested"), "{error}");
}
