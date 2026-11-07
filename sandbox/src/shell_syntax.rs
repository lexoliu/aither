//! Shell scripts parsed into the command lines they would run.
//!
//! [`ShellScript::parse`] walks the syntax tree rather than matching text, so
//! a word that merely contains a program's name (`echo pkill`, `cat
//! sleep.txt`) is not a command, while commands are found through pipelines,
//! lists, compound commands, command and process substitutions, wrappers
//! (`env`, `sudo`, `timeout`, `xargs`, `find -exec`, ...) and the script
//! strings of `sh -c`, `eval` and shells fed through a here-document.
//!
//! Static analysis only sees names that are written out. A program name that
//! is computed at run time (`$tool`, `$(echo pkill)`) or a script piped into a
//! shell's stdin cannot be resolved and yields no command.

use std::cell::{Cell, RefCell};
use std::fmt::Display;

use brush_parser::ast::{
    self, AndOrList, Assignment, AssignmentValue, Command, CommandPrefixOrSuffixItem,
    CompoundCommand, CompoundList, ExtendedTestExpr, IoFileRedirectTarget, IoRedirect, Pipeline,
    SimpleCommand,
};
use brush_parser::word::{ParameterExpr, WordPiece, WordPieceWithSource};
use brush_parser::{Parser, ParserOptions, SourceInfo};

/// Deepest nesting of shell strings (`sh -c '... sh -c ...'`) and
/// substitutions the parse follows before it gives up.
const MAX_DEPTH: usize = 32;

/// A script that is not valid shell, or nests shell strings and
/// substitutions too deeply to follow.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{reason}")]
pub struct ShellParseError {
    reason: String,
}

impl ShellParseError {
    fn new(reason: impl Display) -> Self {
        Self {
            reason: reason.to_string(),
        }
    }

    /// What the parser reported.
    #[must_use]
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

/// Identifies one command of a [`ShellScript`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CommandId(usize);

/// Identifies one simple command of a [`ShellScript`] as written: a program
/// with its words, which may run further commands through wrappers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LineId(usize);

/// Identifies one pipeline of a [`ShellScript`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineId(usize);

/// Where a command line sits in a pipeline of two or more commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineStage {
    /// The pipeline.
    pub pipeline: PipelineId,
    /// The command's position in it, from 0 for the command that writes
    /// first.
    pub stage: usize,
}

/// One command line a script would run, with its words unquoted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellCommand {
    id: CommandId,
    argv: Vec<String>,
    program: String,
    background: bool,
    line: LineId,
    wrapper: Option<CommandId>,
    interpreter: Option<CommandId>,
    substituted_into: Option<LineId>,
    pipeline: Option<PipelineStage>,
}

impl ShellCommand {
    /// This command's id.
    #[must_use]
    pub const fn id(&self) -> CommandId {
        self.id
    }

    /// The program and its arguments, quotes removed. A word part computed
    /// at run time contributes nothing to its word.
    #[must_use]
    pub fn argv(&self) -> &[String] {
        &self.argv
    }

    /// The program's file name, lower-cased and without a Windows
    /// executable suffix: `/usr/bin/PKILL.exe` is `pkill`.
    #[must_use]
    pub fn program(&self) -> &str {
        &self.program
    }

    /// The arguments after the program.
    #[must_use]
    pub fn arguments(&self) -> &[String] {
        &self.argv[1..]
    }

    /// Whether the command sits in an `&`-asynchronous position, so the
    /// script does not wait on it. Everything nested in a detached command
    /// is detached as well.
    #[must_use]
    pub const fn is_background(&self) -> bool {
        self.background
    }

    /// The simple command, as written, this command belongs to: the
    /// command itself, or the wrapper that runs it.
    #[must_use]
    pub const fn line(&self) -> LineId {
        self.line
    }

    /// The command that runs this one from its own arguments: `env`,
    /// `sudo`, `timeout`, `xargs`, `find -exec` and the like.
    #[must_use]
    pub const fn wrapper(&self) -> Option<CommandId> {
        self.wrapper
    }

    /// The shell or `eval` command that runs this one from a script string
    /// (`sh -c`, `eval`, or a shell reading a here-document).
    #[must_use]
    pub const fn interpreter(&self) -> Option<CommandId> {
        self.interpreter
    }

    /// The command line whose words this command's output is substituted
    /// into, through a command or process substitution, the nearest one when
    /// substitutions nest: `kill $(pgrep foo)` substitutes `pgrep` into the
    /// `kill` line.
    #[must_use]
    pub const fn substituted_into(&self) -> Option<LineId> {
        self.substituted_into
    }

    /// Where the command's line sits in a pipeline of two or more commands;
    /// a wrapped command shares its wrapper's stage.
    #[must_use]
    pub const fn pipeline(&self) -> Option<PipelineStage> {
        self.pipeline
    }
}

/// A shell script parsed into every command line it would run, in source
/// order, a command's substitutions ahead of the command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellScript {
    commands: Vec<ShellCommand>,
}

impl ShellScript {
    /// Parses `script`.
    ///
    /// # Errors
    /// Returns [`ShellParseError`] when the script, or a shell string inside
    /// it, is not valid shell, or nests deeper than the parse follows.
    pub fn parse(script: &str) -> Result<Self, ShellParseError> {
        let walk = Walk::default();
        script_walk(
            script,
            Cx {
                walk: &walk,
                depth: 0,
                backgrounded: false,
                substituted_into: None,
                interpreter: None,
            },
        )?;
        Ok(Self {
            commands: walk.commands.into_inner(),
        })
    }

    /// Every command the script would run.
    #[must_use]
    pub fn commands(&self) -> &[ShellCommand] {
        &self.commands
    }

    /// The command `id` names.
    ///
    /// # Panics
    /// Panics when `id` comes from another script.
    #[must_use]
    pub fn get(&self, id: CommandId) -> &ShellCommand {
        &self.commands[id.0]
    }

    /// The commands of `line`: its program, then what it wraps.
    pub fn line(&self, line: LineId) -> impl Iterator<Item = &ShellCommand> {
        self.commands
            .iter()
            .filter(move |command| command.line == line)
    }

    /// The wrappers that run `command`, innermost first.
    pub fn wrappers<'a>(
        &'a self,
        command: &'a ShellCommand,
    ) -> impl Iterator<Item = &'a ShellCommand> + 'a {
        std::iter::successors(command.wrapper.map(|id| self.get(id)), |wrapper| {
            wrapper.wrapper.map(|id| self.get(id))
        })
    }
}

/// What the walk has found so far.
#[derive(Default)]
struct Walk {
    commands: RefCell<Vec<ShellCommand>>,
    lines: Cell<usize>,
    pipelines: Cell<usize>,
}

impl Walk {
    fn next_line(&self) -> LineId {
        let line = self.lines.get();
        self.lines.set(line + 1);
        LineId(line)
    }

    fn next_pipeline(&self) -> PipelineId {
        let pipeline = self.pipelines.get();
        self.pipelines.set(pipeline + 1);
        PipelineId(pipeline)
    }
}

/// Traversal state: the findings, nesting depth, and what every command
/// found below inherits.
#[derive(Clone, Copy)]
struct Cx<'a> {
    walk: &'a Walk,
    depth: usize,
    /// Whether the commands below sit in an `&`-asynchronous position. Once
    /// a list is detached, everything below it runs detached.
    backgrounded: bool,
    /// The line a substitution below feeds.
    substituted_into: Option<LineId>,
    /// The command whose script string is being walked.
    interpreter: Option<CommandId>,
}

impl Cx<'_> {
    const fn deeper(self) -> Self {
        Self {
            depth: self.depth + 1,
            ..self
        }
    }
}

/// A command line found in a script, ahead of being recorded.
struct Invocation<'a> {
    argv: &'a [String],
    /// The scripts its redirects feed to a shell reading stdin.
    stdin_scripts: &'a [String],
    line: LineId,
    wrapper: Option<CommandId>,
    pipeline: Option<PipelineStage>,
}

impl<'a> Invocation<'a> {
    /// The command line `wrapper`, this invocation's command, runs from its
    /// arguments: the same line, at the same pipeline stage.
    const fn wrapped(&self, argv: &'a [String], wrapper: CommandId) -> Self {
        Self {
            argv,
            stdin_scripts: self.stdin_scripts,
            line: self.line,
            wrapper: Some(wrapper),
            pipeline: self.pipeline,
        }
    }
}

fn script_walk(script: &str, cx: Cx<'_>) -> Result<(), ShellParseError> {
    if cx.depth > MAX_DEPTH {
        return Err(ShellParseError::new(format!(
            "shell strings are nested more than {MAX_DEPTH} levels deep"
        )));
    }
    let program = Parser::new(
        script.as_bytes(),
        &ParserOptions::default(),
        &SourceInfo::default(),
    )
    .parse_program()
    .map_err(ShellParseError::new)?;
    for list in &program.complete_commands {
        compound_list(list, cx)?;
    }
    Ok(())
}

fn compound_list(list: &CompoundList, cx: Cx<'_>) -> Result<(), ShellParseError> {
    for item in &list.0 {
        // `cmd &` detaches the command: the script does not wait on it, and
        // everything it nests runs detached as well.
        let item_cx = Cx {
            backgrounded: cx.backgrounded || matches!(item.1, ast::SeparatorOperator::Async),
            ..cx
        };
        and_or_list(&item.0, item_cx)?;
    }
    Ok(())
}

fn and_or_list(list: &AndOrList, cx: Cx<'_>) -> Result<(), ShellParseError> {
    pipeline(&list.first, cx)?;
    for next in &list.additional {
        match next {
            ast::AndOr::And(next) | ast::AndOr::Or(next) => pipeline(next, cx)?,
        }
    }
    Ok(())
}

fn pipeline(pipeline: &Pipeline, cx: Cx<'_>) -> Result<(), ShellParseError> {
    let id = (pipeline.seq.len() > 1).then(|| cx.walk.next_pipeline());
    for (stage, command) in pipeline.seq.iter().enumerate() {
        let stage = id.map(|pipeline| PipelineStage { pipeline, stage });
        self::command(command, stage, cx)?;
    }
    Ok(())
}

fn command(
    command: &Command,
    stage: Option<PipelineStage>,
    cx: Cx<'_>,
) -> Result<(), ShellParseError> {
    match command {
        Command::Simple(simple) => simple_command(simple, stage, cx),
        Command::Compound(compound, redirects) => {
            compound_command(compound, cx)?;
            if let Some(redirects) = redirects {
                for redirect in &redirects.0 {
                    io_redirect(redirect, cx)?;
                }
            }
            Ok(())
        }
        Command::Function(function) => {
            word(&function.fname, cx)?;
            compound_command(&function.body.0, cx)?;
            if let Some(redirects) = &function.body.1 {
                for redirect in &redirects.0 {
                    io_redirect(redirect, cx)?;
                }
            }
            Ok(())
        }
        Command::ExtendedTest(test) => extended_test(&test.expr, cx),
    }
}

fn extended_test(expr: &ExtendedTestExpr, cx: Cx<'_>) -> Result<(), ShellParseError> {
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

fn compound_command(compound: &CompoundCommand, cx: Cx<'_>) -> Result<(), ShellParseError> {
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

fn simple_command(
    simple: &SimpleCommand,
    stage: Option<PipelineStage>,
    cx: Cx<'_>,
) -> Result<(), ShellParseError> {
    let line = cx.walk.next_line();
    // A substitution in any of the line's words feeds the line.
    let word_cx = Cx {
        substituted_into: Some(line),
        ..cx
    };
    let mut argv = Vec::new();
    let mut stdin_scripts = Vec::new();
    let prefix = simple.prefix.iter().flat_map(|prefix| &prefix.0);
    let suffix = simple.suffix.iter().flat_map(|suffix| &suffix.0);
    for item in prefix {
        // Assignments before the program set its environment; they are not
        // part of the command line.
        if let CommandPrefixOrSuffixItem::AssignmentWord(assignment, _) = item {
            assignment_value(assignment, word_cx)?;
        } else {
            affix_item(item, &mut argv, &mut stdin_scripts, word_cx)?;
        }
    }
    if let Some(program) = &simple.word_or_name {
        argv.push(word(program, word_cx)?);
    }
    for item in suffix {
        affix_item(item, &mut argv, &mut stdin_scripts, word_cx)?;
    }
    invoked(
        &Invocation {
            argv: &argv,
            stdin_scripts: &stdin_scripts,
            line,
            wrapper: None,
            pipeline: stage,
        },
        cx,
    )
}

fn affix_item(
    item: &CommandPrefixOrSuffixItem,
    argv: &mut Vec<String>,
    stdin_scripts: &mut Vec<String>,
    cx: Cx<'_>,
) -> Result<(), ShellParseError> {
    match item {
        CommandPrefixOrSuffixItem::Word(argument) => {
            argv.push(word(argument, cx)?);
        }
        CommandPrefixOrSuffixItem::AssignmentWord(assignment, argument) => {
            assignment_value(assignment, cx)?;
            // `declare x=1` style words keep their text as an argument.
            argv.push(word(argument, cx)?);
        }
        CommandPrefixOrSuffixItem::IoRedirect(redirect) => {
            if let Some(script) = stdin_script(redirect, cx)? {
                stdin_scripts.push(script);
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
fn stdin_script(redirect: &IoRedirect, cx: Cx<'_>) -> Result<Option<String>, ShellParseError> {
    match redirect {
        IoRedirect::HereDocument(_, document) => Ok(Some(document.doc.value.clone())),
        IoRedirect::HereString(_, operand) => word(operand, cx).map(Some),
        IoRedirect::File(..) | IoRedirect::OutputAndError(..) => Ok(None),
    }
}

fn assignment_value(assignment: &Assignment, cx: Cx<'_>) -> Result<(), ShellParseError> {
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

fn io_redirect(redirect: &IoRedirect, cx: Cx<'_>) -> Result<(), ShellParseError> {
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
                // A here-document body is free text; one that is not a valid
                // word cannot hold a substitution worth following, so the
                // commands found before the parse failed are all it has.
                let _ = text(&document.doc.value, cx);
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
fn word(word: &ast::Word, cx: Cx<'_>) -> Result<String, ShellParseError> {
    text(&word.value, cx)
}

fn text(source: &str, cx: Cx<'_>) -> Result<String, ShellParseError> {
    let pieces = brush_parser::word::parse(source, &ParserOptions::default())
        .map_err(ShellParseError::new)?;
    let mut literal = String::new();
    pieces_text(&pieces, &mut literal, cx)?;
    Ok(literal)
}

fn pieces_text(
    pieces: &[WordPieceWithSource],
    literal: &mut String,
    cx: Cx<'_>,
) -> Result<(), ShellParseError> {
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
            | WordPiece::BackquotedCommandSubstitution(inner) => script_walk(inner, cx.deeper())?,
            WordPiece::ArithmeticExpression(expr) => {
                text(&expr.value, cx.deeper())?;
            }
            WordPiece::ParameterExpansion(expansion) => parameter_text(expansion, cx)?,
        }
    }
    Ok(())
}

/// Follows the words a parameter expansion carries (`${x:-$(cmd)}`).
fn parameter_text(expansion: &ParameterExpr, cx: Cx<'_>) -> Result<(), ShellParseError> {
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

/// Records one resolved command line and follows what it runs: the command
/// a wrapper runs, `find -exec` commands, and the scripts of `eval` and of
/// shells.
fn invoked(invocation: &Invocation<'_>, cx: Cx<'_>) -> Result<(), ShellParseError> {
    let Some((program, arguments)) = invocation.argv.split_first() else {
        return Ok(());
    };
    let name = file_name(program);
    let id = {
        let mut commands = cx.walk.commands.borrow_mut();
        let id = CommandId(commands.len());
        commands.push(ShellCommand {
            id,
            argv: invocation.argv.to_vec(),
            program: name.clone(),
            background: cx.backgrounded,
            line: invocation.line,
            wrapper: invocation.wrapper,
            interpreter: cx.interpreter,
            substituted_into: cx.substituted_into,
            pipeline: invocation.pipeline,
        });
        id
    };
    if let Some(inner) = wrapped(&name, arguments) {
        invoked(&invocation.wrapped(inner, id), cx)?;
    }
    if name == "find" {
        for inner in find_exec_commands(arguments) {
            invoked(
                &Invocation {
                    stdin_scripts: &[],
                    ..invocation.wrapped(inner, id)
                },
                cx,
            )?;
        }
    }
    let interpreted = Cx {
        interpreter: Some(id),
        ..cx.deeper()
    };
    if name == "eval" && !arguments.is_empty() {
        script_walk(&arguments.join(" "), interpreted)?;
    }
    if is_shell(&name) {
        match shell_script(arguments) {
            ShellInput::Script(script) => script_walk(script, interpreted)?,
            ShellInput::Stdin => {
                for script in invocation.stdin_scripts {
                    script_walk(script, interpreted)?;
                }
            }
            ShellInput::File => {}
        }
    }
    Ok(())
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
#[path = "shell_syntax/tests.rs"]
mod tests;
