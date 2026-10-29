//! Refuses commands that kill processes by name or pattern.
//!
//! A command that runs `pkill` or `killall` signals whatever happens to share
//! a name, including processes the agent never started. The terminal tools
//! parse every command as shell before it runs and refuse the ones that invoke
//! such a program, so the agent has to look up a PID and kill that PID.
//!
//! The check walks the syntax tree rather than matching text, so a word that
//! merely contains a killer's name (`echo pkill`, `grep killall`, `pkill.txt`)
//! is not refused, while the program is found through pipelines, lists,
//! compound commands, command and process substitutions, wrappers (`env`,
//! `sudo`, `timeout`, `xargs`, `find -exec`, ...) and the script strings of
//! `sh -c`, `eval` and shells fed through a here-document.
//!
//! Static analysis only sees names that are written out. A program name that is
//! computed at run time (`$tool`, `$(echo pkill)`) or a script piped into a
//! shell's stdin cannot be resolved and is not refused.

use std::cell::Cell;
use std::fmt::{self, Display};

use askama::Template;
use brush_parser::ast::{
    self, AndOrList, Assignment, AssignmentValue, Command, CommandPrefixOrSuffixItem,
    CompoundCommand, CompoundList, ExtendedTestExpr, IoFileRedirectTarget, IoRedirect, Pipeline,
    SimpleCommand,
};
use brush_parser::word::{ParameterExpr, WordPiece, WordPieceWithSource};
use brush_parser::{Parser, ParserOptions, SourceInfo};

/// Deepest nesting of shell strings (`sh -c '... sh -c ...'`) and
/// substitutions the check follows before it refuses the command.
const MAX_DEPTH: usize = 32;

/// Why a command was refused before it ran.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandRefusal {
    /// The command invokes a program that kills by name or pattern.
    KillByName {
        /// File name of the refused program, as written in the command.
        program: String,
    },
    /// The command, or a shell string inside it, is not valid shell, so it
    /// cannot be checked.
    Unparseable {
        /// What the parser reported.
        reason: String,
    },
}

#[derive(Template)]
#[template(path = "kill_by_name_refused.txt")]
struct KillByNameTemplate<'a> {
    program: &'a str,
}

#[derive(Template)]
#[template(path = "unparseable_command_refused.txt")]
struct UnparseableTemplate<'a> {
    reason: &'a str,
}

impl Display for CommandRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rendered = match self {
            Self::KillByName { program } => KillByNameTemplate { program }.render(),
            Self::Unparseable { reason } => UnparseableTemplate { reason }.render(),
        };
        f.write_str(&rendered.map_err(|_| fmt::Error)?)
    }
}

impl std::error::Error for CommandRefusal {}

/// Refuses `script` when any command in it kills by name or pattern.
///
/// # Errors
/// Returns [`CommandRefusal::KillByName`] when a refused program is invoked,
/// and [`CommandRefusal::Unparseable`] when the script is not valid shell.
pub fn check(script: &str) -> Result<(), CommandRefusal> {
    script_check(
        script,
        Cx {
            depth: 0,
            lookups: &Cell::new(false),
        },
    )
}

fn unparseable(reason: impl Display) -> CommandRefusal {
    CommandRefusal::Unparseable {
        reason: reason.to_string(),
    }
}

/// Traversal state: nesting depth, and whether a program that looks processes
/// up by name (`pgrep`, `pidof`) was met anywhere below.
#[derive(Clone, Copy)]
struct Cx<'a> {
    depth: usize,
    lookups: &'a Cell<bool>,
}

impl Cx<'_> {
    const fn deeper(self) -> Self {
        Self {
            depth: self.depth + 1,
            lookups: self.lookups,
        }
    }
}

fn script_check(script: &str, cx: Cx<'_>) -> Result<(), CommandRefusal> {
    if cx.depth > MAX_DEPTH {
        return Err(unparseable(format!(
            "shell strings are nested more than {MAX_DEPTH} levels deep"
        )));
    }
    let program = Parser::new(
        script.as_bytes(),
        &ParserOptions::default(),
        &SourceInfo::default(),
    )
    .parse_program()
    .map_err(unparseable)?;
    for list in &program.complete_commands {
        compound_list(list, cx)?;
    }
    Ok(())
}

fn compound_list(list: &CompoundList, cx: Cx<'_>) -> Result<(), CommandRefusal> {
    for item in &list.0 {
        and_or_list(&item.0, cx)?;
    }
    Ok(())
}

fn and_or_list(list: &AndOrList, cx: Cx<'_>) -> Result<(), CommandRefusal> {
    pipeline(&list.first, cx)?;
    for next in &list.additional {
        match next {
            ast::AndOr::And(next) | ast::AndOr::Or(next) => pipeline(next, cx)?,
        }
    }
    Ok(())
}

fn pipeline(pipeline: &Pipeline, cx: Cx<'_>) -> Result<(), CommandRefusal> {
    let mut command_lines = Vec::new();
    for command in &pipeline.seq {
        if let Some(argv) = self::command(command, cx)? {
            command_lines.push(argv);
        }
    }
    // `pgrep NAME | xargs kill` kills by name without looking at a PID.
    for (at, producer) in command_lines.iter().enumerate() {
        if chain(producer).iter().any(|name| is_name_lookup(name))
            && command_lines[at + 1..]
                .iter()
                .any(|consumer| xargs_kill(&chain(consumer)))
        {
            return Err(kill_of_lookup());
        }
    }
    Ok(())
}

fn kill_of_lookup() -> CommandRefusal {
    CommandRefusal::KillByName {
        program: "kill of pgrep/pidof output".to_string(),
    }
}

/// Programs that find PIDs by name or pattern.
fn is_name_lookup(name: &str) -> bool {
    matches!(name, "pgrep" | "pidof")
}

/// The program names a command line runs, looking through wrappers.
fn chain(mut argv: &[String]) -> Vec<String> {
    let mut names = Vec::new();
    while let Some((program, arguments)) = argv.split_first() {
        let name = file_name(program);
        let inner = wrapped(&name, arguments);
        names.push(name);
        match inner {
            Some(inner) => argv = inner,
            None => break,
        }
    }
    names
}

fn xargs_kill(chain: &[String]) -> bool {
    chain
        .iter()
        .position(|name| name == "xargs")
        .is_some_and(|at| chain[at + 1..].iter().any(|name| name == "kill"))
}

/// Checks `command`; a simple command returns its resolved command line.
fn command(command: &Command, cx: Cx<'_>) -> Result<Option<Vec<String>>, CommandRefusal> {
    match command {
        Command::Simple(simple) => simple_command(simple, cx).map(Some),
        Command::Compound(compound, redirects) => {
            compound_command(compound, cx)?;
            if let Some(redirects) = redirects {
                for redirect in &redirects.0 {
                    io_redirect(redirect, cx)?;
                }
            }
            Ok(None)
        }
        Command::Function(function) => {
            word(&function.fname, cx)?;
            compound_command(&function.body.0, cx)?;
            if let Some(redirects) = &function.body.1 {
                for redirect in &redirects.0 {
                    io_redirect(redirect, cx)?;
                }
            }
            Ok(None)
        }
        Command::ExtendedTest(test) => extended_test(&test.expr, cx).map(|()| None),
    }
}

fn extended_test(expr: &ExtendedTestExpr, cx: Cx<'_>) -> Result<(), CommandRefusal> {
    match expr {
        ExtendedTestExpr::And(left, right) | ExtendedTestExpr::Or(left, right) => {
            extended_test(left, cx)?;
            extended_test(right, cx)
        }
        ExtendedTestExpr::Not(inner) | ExtendedTestExpr::Parenthesized(inner) => {
            extended_test(inner, cx)
        }
        ExtendedTestExpr::UnaryTest(_, operand) => word(operand, cx).map(drop),
        ExtendedTestExpr::BinaryTest(_, left, right) => {
            word(left, cx)?;
            word(right, cx).map(drop)
        }
    }
}

fn compound_command(compound: &CompoundCommand, cx: Cx<'_>) -> Result<(), CommandRefusal> {
    match compound {
        CompoundCommand::Arithmetic(arithmetic) => text(&arithmetic.expr.value, cx).map(drop),
        CompoundCommand::ArithmeticForClause(clause) => {
            for expr in [&clause.initializer, &clause.condition, &clause.updater]
                .into_iter()
                .flatten()
            {
                text(&expr.value, cx)?;
            }
            compound_list(&clause.body.list, cx)
        }
        CompoundCommand::BraceGroup(group) => compound_list(&group.list, cx),
        CompoundCommand::Subshell(subshell) => compound_list(&subshell.list, cx),
        CompoundCommand::ForClause(clause) => {
            for value in clause.values.iter().flatten() {
                word(value, cx)?;
            }
            compound_list(&clause.body.list, cx)
        }
        CompoundCommand::CaseClause(clause) => {
            word(&clause.value, cx)?;
            for item in &clause.cases {
                for pattern in &item.patterns {
                    word(pattern, cx)?;
                }
                if let Some(body) = &item.cmd {
                    compound_list(body, cx)?;
                }
            }
            Ok(())
        }
        CompoundCommand::IfClause(clause) => {
            compound_list(&clause.condition, cx)?;
            compound_list(&clause.then, cx)?;
            for other in clause.elses.iter().flatten() {
                if let Some(condition) = &other.condition {
                    compound_list(condition, cx)?;
                }
                compound_list(&other.body, cx)?;
            }
            Ok(())
        }
        CompoundCommand::WhileClause(clause) | CompoundCommand::UntilClause(clause) => {
            compound_list(&clause.0, cx)?;
            compound_list(&clause.1.list, cx)
        }
    }
}

/// A simple command's program and arguments with every word unquoted, and
/// the scripts it receives on stdin through here-documents and here-strings.
struct Invocation {
    argv: Vec<String>,
    stdin_scripts: Vec<String>,
    /// Whether a word of the command substitutes the output of `pgrep`/`pidof`.
    lookup_in_words: bool,
}

fn simple_command(simple: &SimpleCommand, cx: Cx<'_>) -> Result<Vec<String>, CommandRefusal> {
    let mut invocation = Invocation {
        argv: Vec::new(),
        stdin_scripts: Vec::new(),
        lookup_in_words: false,
    };
    // Words are traversed with their own flag: a lookup program inside a
    // word's substitution feeds the command's arguments.
    let in_words = Cell::new(false);
    let word_cx = Cx {
        depth: cx.depth,
        lookups: &in_words,
    };
    let prefix = simple.prefix.iter().flat_map(|prefix| &prefix.0);
    let suffix = simple.suffix.iter().flat_map(|suffix| &suffix.0);
    for item in prefix {
        // Assignments before the program set its environment; they are not
        // part of the command line.
        if let CommandPrefixOrSuffixItem::AssignmentWord(assignment, _) = item {
            assignment_value(assignment, word_cx)?;
        } else {
            affix_item(item, &mut invocation, word_cx)?;
        }
    }
    if let Some(program) = &simple.word_or_name {
        let program = word(program, word_cx)?;
        invocation.argv.push(program);
    }
    for item in suffix {
        affix_item(item, &mut invocation, word_cx)?;
    }
    invocation.lookup_in_words = in_words.get();
    invoked(
        &invocation.argv,
        &invocation.stdin_scripts,
        invocation.lookup_in_words,
        cx,
    )?;
    Ok(invocation.argv)
}

fn affix_item(
    item: &CommandPrefixOrSuffixItem,
    invocation: &mut Invocation,
    cx: Cx<'_>,
) -> Result<(), CommandRefusal> {
    match item {
        CommandPrefixOrSuffixItem::Word(argument) => {
            let argument = word(argument, cx)?;
            invocation.argv.push(argument);
        }
        CommandPrefixOrSuffixItem::AssignmentWord(assignment, argument) => {
            assignment_value(assignment, cx)?;
            // `declare x=1` style words keep their text as an argument.
            let argument = word(argument, cx)?;
            invocation.argv.push(argument);
        }
        CommandPrefixOrSuffixItem::IoRedirect(redirect) => {
            if let Some(script) = stdin_script(redirect, cx)? {
                invocation.stdin_scripts.push(script);
            }
            io_redirect(redirect, cx)?;
        }
        CommandPrefixOrSuffixItem::ProcessSubstitution(_, subshell) => {
            compound_list(&subshell.list, cx)?;
        }
    }
    Ok(())
}

/// The text a here-document or here-string feeds to stdin.
fn stdin_script(redirect: &IoRedirect, cx: Cx<'_>) -> Result<Option<String>, CommandRefusal> {
    match redirect {
        IoRedirect::HereDocument(_, document) => Ok(Some(document.doc.value.clone())),
        IoRedirect::HereString(_, operand) => word(operand, cx).map(Some),
        IoRedirect::File(..) | IoRedirect::OutputAndError(..) => Ok(None),
    }
}

fn assignment_value(assignment: &Assignment, cx: Cx<'_>) -> Result<(), CommandRefusal> {
    match &assignment.value {
        AssignmentValue::Scalar(value) => word(value, cx).map(drop),
        AssignmentValue::Array(values) => {
            for (key, value) in values {
                if let Some(key) = key {
                    word(key, cx)?;
                }
                word(value, cx)?;
            }
            Ok(())
        }
    }
}

fn io_redirect(redirect: &IoRedirect, cx: Cx<'_>) -> Result<(), CommandRefusal> {
    match redirect {
        IoRedirect::File(_, _, target) => match target {
            IoFileRedirectTarget::Filename(target) | IoFileRedirectTarget::Duplicate(target) => {
                word(target, cx).map(drop)
            }
            IoFileRedirectTarget::Fd(_) => Ok(()),
            IoFileRedirectTarget::ProcessSubstitution(_, subshell) => {
                compound_list(&subshell.list, cx)
            }
        },
        IoRedirect::HereDocument(_, document) => {
            if document.requires_expansion {
                // A here-document body is free text; one that is not a
                // valid word cannot hold a substitution worth following.
                match text(&document.doc.value, cx) {
                    Err(refusal @ CommandRefusal::KillByName { .. }) => return Err(refusal),
                    Err(CommandRefusal::Unparseable { .. }) | Ok(_) => {}
                }
            }
            Ok(())
        }
        IoRedirect::HereString(_, operand) | IoRedirect::OutputAndError(operand, _) => {
            word(operand, cx).map(drop)
        }
    }
}

/// Follows the substitutions inside `word` and returns its text with quotes
/// removed. A part computed at run time contributes nothing to the text.
fn word(word: &ast::Word, cx: Cx<'_>) -> Result<String, CommandRefusal> {
    text(&word.value, cx)
}

fn text(source: &str, cx: Cx<'_>) -> Result<String, CommandRefusal> {
    let pieces =
        brush_parser::word::parse(source, &ParserOptions::default()).map_err(unparseable)?;
    let mut literal = String::new();
    pieces_text(&pieces, &mut literal, cx)?;
    Ok(literal)
}

fn pieces_text(
    pieces: &[WordPieceWithSource],
    literal: &mut String,
    cx: Cx<'_>,
) -> Result<(), CommandRefusal> {
    for WordPieceWithSource { piece, .. } in pieces {
        match piece {
            WordPiece::Text(part)
            | WordPiece::SingleQuotedText(part)
            | WordPiece::AnsiCQuotedText(part) => literal.push_str(part),
            WordPiece::EscapeSequence(part) => {
                literal.push_str(part.strip_prefix('\\').unwrap_or(part));
            }
            WordPiece::DoubleQuotedSequence(inner)
            | WordPiece::GettextDoubleQuotedSequence(inner) => {
                pieces_text(inner, literal, cx)?;
            }
            WordPiece::TildePrefix(_) => {}
            WordPiece::CommandSubstitution(inner)
            | WordPiece::BackquotedCommandSubstitution(inner) => script_check(inner, cx.deeper())?,
            WordPiece::ArithmeticExpression(expr) => {
                text(&expr.value, cx.deeper())?;
            }
            WordPiece::ParameterExpansion(expansion) => parameter_text(expansion, cx)?,
        }
    }
    Ok(())
}

/// Follows the words a parameter expansion carries (`${x:-$(cmd)}`).
fn parameter_text(expansion: &ParameterExpr, cx: Cx<'_>) -> Result<(), CommandRefusal> {
    let embedded: [Option<&str>; 2] = match expansion {
        ParameterExpr::UseDefaultValues { default_value, .. }
        | ParameterExpr::AssignDefaultValues { default_value, .. } => {
            [default_value.as_deref(), None]
        }
        ParameterExpr::IndicateErrorIfNullOrUnset { error_message, .. } => {
            [error_message.as_deref(), None]
        }
        ParameterExpr::UseAlternativeValue {
            alternative_value, ..
        } => [alternative_value.as_deref(), None],
        ParameterExpr::RemoveSmallestSuffixPattern { pattern, .. }
        | ParameterExpr::RemoveLargestSuffixPattern { pattern, .. }
        | ParameterExpr::RemoveSmallestPrefixPattern { pattern, .. }
        | ParameterExpr::RemoveLargestPrefixPattern { pattern, .. }
        | ParameterExpr::UppercaseFirstChar { pattern, .. }
        | ParameterExpr::UppercasePattern { pattern, .. }
        | ParameterExpr::LowercaseFirstChar { pattern, .. }
        | ParameterExpr::LowercasePattern { pattern, .. } => [pattern.as_deref(), None],
        ParameterExpr::Substring { offset, length, .. } => [
            Some(&offset.value),
            length.as_ref().map(|expr| &*expr.value),
        ],
        ParameterExpr::ReplaceSubstring {
            pattern,
            replacement,
            ..
        } => [Some(pattern), replacement.as_deref()],
        ParameterExpr::Parameter { .. }
        | ParameterExpr::ParameterLength { .. }
        | ParameterExpr::Transform { .. }
        | ParameterExpr::VariableNames { .. }
        | ParameterExpr::MemberKeys { .. } => [None, None],
    };
    for source in embedded.into_iter().flatten() {
        text(source, cx.deeper())?;
    }
    Ok(())
}

/// Programs that signal processes selected by name, user or other pattern.
const KILLERS: &[&str] = &["pkill", "killall", "killall5", "skill"];

/// Lower-cased file name of `program` without a Windows executable suffix.
fn file_name(program: &str) -> String {
    let name = program.rsplit(['/', '\\']).next().unwrap_or(program);
    let mut name = name.to_ascii_lowercase();
    if let Some(extension_at) = name.len().checked_sub(".exe".len())
        && name.is_char_boundary(extension_at)
        && name[extension_at..] == *".exe"
    {
        name.truncate(extension_at);
    }
    name
}

/// Checks one resolved command line; `stdin_scripts` are the scripts its
/// redirects feed to a shell reading stdin.
fn invoked(
    argv: &[String],
    stdin_scripts: &[String],
    lookup_in_words: bool,
    cx: Cx<'_>,
) -> Result<(), CommandRefusal> {
    let Some((program, arguments)) = argv.split_first() else {
        return Ok(());
    };
    let name = file_name(program);
    if kills_by_name(&name, arguments) {
        return Err(CommandRefusal::KillByName { program: name });
    }
    if is_name_lookup(&name) {
        cx.lookups.set(true);
    }
    if name == "kill" && lookup_in_words {
        return Err(kill_of_lookup());
    }
    if let Some(inner) = wrapped(&name, arguments) {
        invoked(inner, stdin_scripts, lookup_in_words, cx)?;
    }
    if name == "find" {
        for inner in find_exec_commands(arguments) {
            invoked(inner, &[], false, cx)?;
        }
    }
    if name == "eval" && !arguments.is_empty() {
        script_check(&arguments.join(" "), cx.deeper())?;
    }
    if is_shell(&name) {
        match shell_script(arguments) {
            ShellInput::Script(script) => script_check(script, cx.deeper())?,
            ShellInput::Stdin => {
                for script in stdin_scripts {
                    script_check(script, cx.deeper())?;
                }
            }
            ShellInput::File => {}
        }
    }
    Ok(())
}

/// Whether `name` with `arguments` selects the processes it signals by name,
/// by owner or by what they use rather than by PID.
fn kills_by_name(name: &str, arguments: &[String]) -> bool {
    match name {
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
        _ => KILLERS.contains(&name),
    }
}

fn is_shell(name: &str) -> bool {
    matches!(name, "sh" | "bash" | "dash" | "ash" | "ksh" | "zsh" | "su")
}

/// Where a shell takes its commands from.
enum ShellInput<'a> {
    /// A string given with `-c` (or `su -c`).
    Script(&'a str),
    /// Standard input.
    Stdin,
    /// A script file.
    File,
}

fn shell_script(arguments: &[String]) -> ShellInput<'_> {
    let mut remaining = arguments.iter();
    let mut reads_stdin = false;
    let mut script_file = false;
    while let Some(argument) = remaining.next() {
        if argument == "--command" {
            return remaining
                .next()
                .map_or(ShellInput::Stdin, |script| ShellInput::Script(script));
        }
        if let Some(flags) = argument
            .strip_prefix(['-', '+'])
            .filter(|f| !f.starts_with('-'))
        {
            if argument.starts_with('-') && flags.contains('c') {
                return remaining
                    .next()
                    .map_or(ShellInput::Stdin, |script| ShellInput::Script(script));
            }
            reads_stdin |= flags.contains('s');
            if flags.ends_with(['o', 'O']) {
                remaining.next();
            }
        } else if argument != "--" {
            script_file = true;
            break;
        }
    }
    if script_file && !reads_stdin {
        ShellInput::File
    } else {
        ShellInput::Stdin
    }
}

/// How a command wrapper takes its own options before the command it runs.
struct WrapperSyntax {
    /// Short options that take a value.
    short_values: &'static str,
    /// Long options that take a value when it is not joined with `=`.
    long_values: &'static [&'static str],
    /// Positional arguments before the command.
    positionals: usize,
    /// Whether leading `NAME=value` words belong to the wrapper.
    assignments: bool,
}

const fn wrapper(
    short_values: &'static str,
    long_values: &'static [&'static str],
    positionals: usize,
    assignments: bool,
) -> WrapperSyntax {
    WrapperSyntax {
        short_values,
        long_values,
        positionals,
        assignments,
    }
}

/// What a command wrapper runs, or `None` when `name` is not a wrapper or
/// runs nothing (`command -v pkill` only looks the name up).
fn wrapped<'a>(name: &str, arguments: &'a [String]) -> Option<&'a [String]> {
    let syntax = match name {
        "env" => wrapper("uC", &["unset", "chdir"], 0, true),
        "sudo" => wrapper(
            "ughpCDRTU",
            &["user", "group", "host", "prompt", "chdir", "chroot"],
            0,
            true,
        ),
        "doas" => wrapper("uC", &[], 0, true),
        "nice" => wrapper("n", &["adjustment"], 0, false),
        "ionice" => wrapper("cnp", &["class", "classdata", "pid"], 0, false),
        "stdbuf" => wrapper("ioe", &["input", "output", "error"], 0, false),
        "timeout" => wrapper("sk", &["signal", "kill-after"], 1, false),
        "time" => wrapper("fo", &["format", "output"], 0, false),
        "xargs" => wrapper(
            "IinPLsdEa",
            &[
                "replace",
                "max-args",
                "max-procs",
                "max-lines",
                "max-chars",
                "delimiter",
                "eof",
                "arg-file",
            ],
            0,
            false,
        ),
        "exec" => wrapper("a", &[], 0, false),
        "command" => {
            if arguments
                .iter()
                .any(|argument| argument == "-v" || argument == "-V")
            {
                return None;
            }
            wrapper("", &[], 0, false)
        }
        "builtin" | "nohup" | "setsid" | "busybox" | "toybox" => wrapper("", &[], 0, false),
        _ => return None,
    };
    let mut index = 0;
    while let Some(argument) = arguments.get(index) {
        index += 1;
        if argument == "--" {
            break;
        }
        if let Some(long) = argument.strip_prefix("--") {
            if !long.contains('=') && syntax.long_values.contains(&long) {
                index += 1;
            }
        } else if let Some(flags) = argument.strip_prefix('-').filter(|flags| !flags.is_empty()) {
            // getopt: the first value option takes the rest of the cluster,
            // or the next word when nothing follows it.
            let value_at = flags
                .char_indices()
                .find(|(_, letter)| syntax.short_values.contains(*letter));
            if let Some((at, letter)) = value_at
                && at + letter.len_utf8() == flags.len()
            {
                index += 1;
            }
        } else if syntax.assignments && argument.contains('=') {
        } else {
            index -= 1;
            break;
        }
    }
    index += syntax.positionals;
    arguments.get(index..).filter(|inner| !inner.is_empty())
}

/// The command lines `find` runs through `-exec`, `-execdir`, `-ok` and
/// `-okdir`, each ending at a lone `;` or `+`.
fn find_exec_commands(arguments: &[String]) -> Vec<&[String]> {
    let mut commands = Vec::new();
    let mut index = 0;
    while index < arguments.len() {
        if matches!(
            arguments[index].as_str(),
            "-exec" | "-execdir" | "-ok" | "-okdir"
        ) {
            let start = index + 1;
            let end = arguments[start..]
                .iter()
                .position(|argument| argument == ";" || argument == "+")
                .map_or(arguments.len(), |offset| start + offset);
            commands.push(&arguments[start..end]);
            index = end;
        }
        index += 1;
    }
    commands
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refused_program(script: &str) -> Option<String> {
        match check(script) {
            Err(CommandRefusal::KillByName { program }) => Some(program),
            Err(other) => panic!("`{script}` should parse: {other}"),
            Ok(()) => None,
        }
    }

    #[test]
    fn refuses_killers_by_name_and_path() {
        for script in [
            "pkill foo",
            "killall foo",
            "/usr/bin/pkill -f foo",
            "/usr/bin/killall -9 foo",
            "./killall foo",
            "'pkill' foo",
            "p''kill foo",
            "\\pkill foo",
            "killall5 -9",
            "skill -KILL foo",
            "PKILL foo.exe",
            "fuser -k 8080/tcp",
            "fuser -km /mnt",
            "taskkill /IM foo.exe /F",
        ] {
            assert!(
                refused_program(script).is_some(),
                "`{script}` must be refused"
            );
        }
    }

    #[test]
    fn refuses_killers_nested_in_shell_syntax() {
        for script in [
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
        ] {
            assert!(
                refused_program(script).is_some(),
                "`{script}` must be refused"
            );
        }
    }

    #[test]
    fn refuses_killers_behind_wrappers() {
        for script in [
            "env pkill foo",
            "env FOO=1 BAR=2 killall foo",
            "env -u HOME pkill foo",
            "exec pkill foo",
            "command pkill foo",
            "builtin pkill foo",
            "sudo pkill foo",
            "sudo -u root pkill foo",
            "sudo -E killall foo",
            "sudo -Eu root pkill foo",
            "doas pkill foo",
            "nohup pkill foo",
            "nohup pkill foo &",
            "nice pkill foo",
            "nice -n 5 pkill foo",
            "nice -5 pkill foo",
            "ionice -c 3 pkill foo",
            "setsid pkill foo",
            "stdbuf -o0 pkill foo",
            "timeout 5 pkill foo",
            "timeout -s KILL 5 pkill foo",
            "timeout --signal=KILL 5 killall foo",
            "time pkill foo",
            "xargs pkill",
            "echo foo | xargs -n1 pkill",
            "echo foo | xargs -I {} pkill {}",
            "busybox pkill foo",
            "env -- pkill foo",
            "sudo env nice timeout 3 /usr/bin/pkill foo",
            "find . -exec pkill {} \\;",
            "find . -name x -execdir killall {} +",
        ] {
            assert!(
                refused_program(script).is_some(),
                "`{script}` must be refused"
            );
        }
    }

    #[test]
    fn refuses_killers_inside_shell_strings() {
        for script in [
            "sh -c 'pkill foo'",
            "bash -c \"killall foo\"",
            "bash -lc 'pkill foo'",
            "bash -o pipefail -c 'pkill foo'",
            "sh -c 'echo a; /usr/bin/pkill foo'",
            "sh -c \"sh -c 'pkill foo'\"",
            "zsh -c 'pkill foo'",
            "su -c 'pkill foo' root",
            "sudo sh -c 'pkill foo'",
            "env FOO=1 bash -c 'killall foo'",
            "eval pkill foo",
            "eval 'echo a; pkill foo'",
            "sh <<< 'pkill foo'",
            "bash <<EOF\npkill foo\nEOF",
            "cat <<EOF\n$(pkill foo)\nEOF",
            "sh -c 'echo $(pkill foo)'",
        ] {
            assert!(
                refused_program(script).is_some(),
                "`{script}` must be refused"
            );
        }
    }

    #[test]
    fn names_the_refused_program() {
        assert_eq!(
            refused_program("/usr/bin/pkill foo").as_deref(),
            Some("pkill")
        );
        assert_eq!(
            refused_program("sudo killall foo").as_deref(),
            Some("killall")
        );
    }

    #[test]
    fn allows_words_that_only_contain_a_killer_name() {
        for script in [
            "echo pkill",
            "echo 'pkill foo' killall",
            "grep killall file",
            "grep -r pkill .",
            "cat pkill.txt",
            "cat /tmp/killall",
            "ls /usr/bin/pkill",
            "man pkill",
            "which pkill",
            "type killall",
            "command -v pkill",
            "command -V killall",
            "pgrep foo",
            "pidof foo",
            "echo pkill | cat",
            "env echo pkill",
            "sudo echo killall",
            "timeout 5 echo pkill",
            "xargs echo pkill",
            "sh -c 'echo pkill'",
            "bash -c 'grep killall file'",
            "find . -name pkill",
            "find . -exec echo pkill {} \\;",
            "sh script.sh pkill",
            "echo \"$(echo pkill)\"",
            "x=pkill",
            "mypkill foo",
            "pkill_helper",
            "fuser 8080/tcp",
            "fuser -v /mnt",
            "taskkill /PID 42 /F",
            "cat <<EOF\npkill foo\nEOF",
            "sh script.sh <<EOF\npkill foo\nEOF",
            "git commit -m 'stop using pkill'",
            "",
            "   ",
        ] {
            assert_eq!(check(script), Ok(()), "`{script}` must not be refused");
        }
    }

    #[test]
    fn refuses_kills_of_pids_found_by_name() {
        for script in [
            "kill $(pgrep foo)",
            "kill -9 $(pgrep -f foo)",
            "kill `pgrep foo`",
            "kill \"$(pgrep foo)\"",
            "kill $(pidof foo)",
            "kill `pidof foo`",
            "kill $(pgrep foo | head -1)",
            "kill $(/usr/bin/pgrep foo)",
            "/bin/kill $(pgrep foo)",
            "sudo kill $(pgrep foo)",
            "sudo kill -9 $(pidof foo)",
            "env X=1 kill $(pidof foo)",
            "sh -c 'kill $(pgrep foo)'",
            "echo a && kill $(pgrep foo)",
            "pgrep foo | xargs kill",
            "pgrep -f foo | xargs kill -9",
            "pgrep -f foo | xargs -r kill -9",
            "pidof foo | xargs kill",
            "pidof foo | sudo xargs kill",
            "pgrep foo | xargs sudo kill",
            "pgrep foo | tr '\\n' ' ' | xargs kill",
            "/usr/bin/pgrep foo | /usr/bin/xargs /bin/kill",
            "sudo pgrep foo | xargs kill",
        ] {
            assert_eq!(
                refused_program(script).as_deref(),
                Some("kill of pgrep/pidof output"),
                "`{script}` must be refused"
            );
        }
    }

    #[test]
    fn allows_kill_by_pid() {
        for script in [
            "kill 123",
            "kill -9 123",
            "/bin/kill -TERM 123",
            "sleep 30 & kill $!",
            "kill $(cat pidfile)",
            "kill -9 $(cat /tmp/x.pid)",
            "kill $pid",
            "kill \"$pid\"",
            "kill 1 2 3",
            "echo $(pgrep foo)",
            "pgrep foo | wc -l",
            "pgrep foo; kill 123",
            "pidof foo | xargs echo",
            "cat pids | xargs kill",
            "kill $(cat pidfile) && pgrep foo",
            "kill -0 123 2>/dev/null",
            "xargs kill",
            "sudo kill 123",
        ] {
            assert_eq!(check(script), Ok(()), "`{script}` must not be refused");
        }
    }

    #[test]
    fn names_that_are_computed_at_run_time_are_not_resolved() {
        for script in ["$tool foo", "$(echo pkill) foo", "\"${tool}\" foo"] {
            assert_eq!(check(script), Ok(()), "`{script}` cannot be resolved");
        }
    }

    #[test]
    fn refuses_what_cannot_be_parsed() {
        for script in [
            "echo 'unterminated",
            "if true",
            "echo $(",
            "sh -c 'if'",
            "fi",
        ] {
            assert!(
                matches!(check(script), Err(CommandRefusal::Unparseable { .. })),
                "`{script}` must be refused as unparseable"
            );
        }
    }

    #[test]
    fn refuses_runaway_nesting() {
        let script = format!("{}pkill foo", "eval ".repeat(MAX_DEPTH + 2));
        assert!(matches!(
            check(&script),
            Err(CommandRefusal::Unparseable { .. })
        ));
    }

    #[test]
    fn refusal_tells_the_agent_to_kill_by_pid() {
        let kill = CommandRefusal::KillByName {
            program: "pkill".to_string(),
        }
        .to_string();
        assert!(kill.contains("pkill"), "{kill}");
        assert!(kill.contains("PID"), "{kill}");
        let parse = CommandRefusal::Unparseable {
            reason: "syntax error".to_string(),
        }
        .to_string();
        assert!(parse.contains("syntax error"), "{parse}");
    }
}
