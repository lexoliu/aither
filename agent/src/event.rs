//! Agent events for streaming execution.

use crate::context_window::ContextWindowPhase;
use crate::error::AgentError;
use crate::hook::CheckpointReason;
use aither_core::llm::ToolResult;

pub use aither_sandbox::BackgroundReason;

/// Reason why an agent run is paused or resumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunPauseReason {
    /// Waiting for a terminal permission approval.
    PermissionRequest,
    /// Waiting for an `ask_user` response.
    AskUser,
    /// Waiting for a `request_workspace` response.
    WorkspaceRequest,
    /// The context window is full and the run cannot continue.
    ///
    /// Emitted when usage reaches
    /// [`ContextPolicy::pause_at`](crate::ContextPolicy::pause_at) and no
    /// compaction can relieve it: either none was in flight, or the one that
    /// was failed. Sending the request anyway would have the provider reject
    /// it for length, so the run stops with the reason named instead.
    ContextExhausted,
}

/// Events emitted during agent execution.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// Agent run started with a stable run identifier.
    RunStart {
        /// Unique identifier for this run invocation.
        run_id: String,
        /// Maximum number of iterations configured for this run.
        max_iterations: usize,
    },

    /// Text chunk from the LLM response.
    Text(String),

    /// Reasoning step (for models with extended thinking).
    Reasoning(String),

    /// Incremental tool call assembly progress from the model stream.
    ///
    /// Emitted as the model streams tool call name and arguments, before the
    /// full call is assembled. UI layers can use this to show early feedback.
    ToolCallDelta {
        /// Tool call identifier.
        id: String,
        /// Tool name (may be empty in very early deltas for some providers).
        name: String,
        /// Incremental JSON arguments fragment for this delta.
        arguments_fragment: String,
    },

    /// Tool is about to be called.
    ToolCallStart {
        /// Unique identifier for this tool call.
        id: String,
        /// Name of the tool being called.
        name: String,
        /// JSON-encoded arguments.
        arguments: String,
    },

    /// Tool execution completed.
    ToolCallEnd {
        /// Unique identifier matching the start event.
        id: String,
        /// Name of the tool that was called.
        name: String,
        /// Structured result of the tool execution.
        result: ToolResult,
    },

    /// Agent turn completed (may have more turns if tools were called).
    TurnComplete {
        /// The turn number (0-indexed).
        turn: usize,
        /// Whether this turn included tool calls.
        has_tool_calls: bool,
    },

    /// Runtime checkpoint emitted after mutating agent state.
    Checkpoint {
        /// Unique identifier for this run invocation.
        run_id: String,
        /// Why the checkpoint was emitted.
        reason: CheckpointReason,
        /// Current turn number (1-indexed within the agent loop).
        turn: usize,
        /// Current lifecycle phase of the assembled context window.
        phase: ContextWindowPhase,
        /// Number of recent conversation messages currently in memory.
        message_count: usize,
    },

    /// A terminal command was promoted to a background task.
    BackgroundTaskStarted {
        /// Background task identifier.
        task_id: String,
        /// Inline preview of current output.
        output_preview: String,
        /// File path or URL where full output can be recovered.
        output_file: String,
        /// Why the command was moved to the background.
        reason: BackgroundReason,
    },

    /// A background task completed and its result was reinjected into context.
    BackgroundTaskCompleted {
        /// Background task identifier.
        task_id: String,
        /// Structured XML/text payload injected back into context.
        result: String,
    },

    /// A terminal task is waiting for stdin from the user or host runtime.
    TerminalInputNeeded {
        /// Optional task identifier for the waiting terminal.
        task_id: Option<String>,
        /// Human-readable notice explaining why input is needed.
        notice: String,
    },

    /// The run is now paused waiting for external input.
    RunPaused {
        /// Why the run paused.
        reason: RunPauseReason,
        /// Tool call identifier responsible for the pause.
        tool_call_id: String,
    },

    /// The run resumed after an external input arrived.
    RunResumed {
        /// Why the run had paused.
        reason: RunPauseReason,
        /// Tool call identifier responsible for the pause.
        tool_call_id: String,
    },

    /// A skill was activated for the current run.
    SkillActivated {
        /// Skill name.
        name: String,
        /// Explicitly allowed tools declared by the skill.
        allowed_tools: Option<Vec<String>>,
        /// Resource files made available by the skill.
        resource_paths: Option<Vec<String>>,
    },

    /// Agent finished processing successfully.
    Complete {
        /// The final response text.
        final_text: String,
        /// Total number of turns taken.
        turns: usize,
    },

    /// Token usage information from LLM.
    Usage(aither_core::llm::Usage),

    /// Error occurred during execution.
    Error(AgentError),
}

impl AgentEvent {
    /// Creates a new run-start event.
    #[must_use]
    pub fn run_start(run_id: impl Into<String>, max_iterations: usize) -> Self {
        Self::RunStart {
            run_id: run_id.into(),
            max_iterations,
        }
    }

    /// Creates a new text event.
    #[must_use]
    pub fn text(content: impl Into<String>) -> Self {
        Self::Text(content.into())
    }

    /// Creates a new reasoning event.
    #[must_use]
    pub fn reasoning(content: impl Into<String>) -> Self {
        Self::Reasoning(content.into())
    }

    /// Creates a new tool call delta event.
    #[must_use]
    pub fn tool_delta(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments_fragment: impl Into<String>,
    ) -> Self {
        Self::ToolCallDelta {
            id: id.into(),
            name: name.into(),
            arguments_fragment: arguments_fragment.into(),
        }
    }

    /// Creates a new tool call start event.
    #[must_use]
    pub fn tool_start(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self::ToolCallStart {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }

    /// Creates a new tool call end event with success.
    #[must_use]
    pub fn tool_success(
        id: impl Into<String>,
        name: impl Into<String>,
        result: impl Into<String>,
    ) -> Self {
        Self::ToolCallEnd {
            id: id.into(),
            name: name.into(),
            result: ToolResult::text(result),
        }
    }

    /// Creates a new tool call end event with failure.
    #[must_use]
    pub fn tool_failure(
        id: impl Into<String>,
        name: impl Into<String>,
        error: impl Into<String>,
    ) -> Self {
        Self::ToolCallEnd {
            id: id.into(),
            name: name.into(),
            result: ToolResult::error(error),
        }
    }

    /// Creates a new turn complete event.
    #[must_use]
    pub const fn turn_complete(turn: usize, has_tool_calls: bool) -> Self {
        Self::TurnComplete {
            turn,
            has_tool_calls,
        }
    }

    /// Creates a new checkpoint event.
    #[must_use]
    pub fn checkpoint(
        run_id: impl Into<String>,
        reason: CheckpointReason,
        turn: usize,
        phase: ContextWindowPhase,
        message_count: usize,
    ) -> Self {
        Self::Checkpoint {
            run_id: run_id.into(),
            reason,
            turn,
            phase,
            message_count,
        }
    }

    /// Creates a background-task-started event.
    #[must_use]
    pub fn background_task_started(
        task_id: impl Into<String>,
        output_preview: impl Into<String>,
        output_file: impl Into<String>,
        reason: BackgroundReason,
    ) -> Self {
        Self::BackgroundTaskStarted {
            task_id: task_id.into(),
            output_preview: output_preview.into(),
            output_file: output_file.into(),
            reason,
        }
    }

    /// Creates a background-task-completed event.
    #[must_use]
    pub fn background_task_completed(
        task_id: impl Into<String>,
        result: impl Into<String>,
    ) -> Self {
        Self::BackgroundTaskCompleted {
            task_id: task_id.into(),
            result: result.into(),
        }
    }

    /// Creates a terminal-input-needed event.
    #[must_use]
    pub fn terminal_input_needed(task_id: Option<String>, notice: impl Into<String>) -> Self {
        Self::TerminalInputNeeded {
            task_id,
            notice: notice.into(),
        }
    }

    /// Creates a run-paused event.
    #[must_use]
    pub fn run_paused(reason: RunPauseReason, tool_call_id: impl Into<String>) -> Self {
        Self::RunPaused {
            reason,
            tool_call_id: tool_call_id.into(),
        }
    }

    /// Creates a run-resumed event.
    #[must_use]
    pub fn run_resumed(reason: RunPauseReason, tool_call_id: impl Into<String>) -> Self {
        Self::RunResumed {
            reason,
            tool_call_id: tool_call_id.into(),
        }
    }

    /// Creates a skill-activated event.
    #[must_use]
    pub fn skill_activated(
        name: impl Into<String>,
        allowed_tools: Option<Vec<String>>,
        resource_paths: Option<Vec<String>>,
    ) -> Self {
        Self::SkillActivated {
            name: name.into(),
            allowed_tools,
            resource_paths,
        }
    }

    /// Creates a new completion event.
    #[must_use]
    pub fn complete(final_text: impl Into<String>, turns: usize) -> Self {
        Self::Complete {
            final_text: final_text.into(),
            turns,
        }
    }

    /// Creates a new error event.
    #[must_use]
    pub const fn error(error: AgentError) -> Self {
        Self::Error(error)
    }

    /// Returns `true` if this is a completion event.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        matches!(self, Self::Complete { .. })
    }

    /// Returns `true` if this is an error event.
    #[must_use]
    pub const fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }

    /// Returns `true` if this is a terminal event (complete or error).
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        self.is_complete() || self.is_error()
    }
}
