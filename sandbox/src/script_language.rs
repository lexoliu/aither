//! The language of a script a command runs.

use std::borrow::Cow;

use askama::Template;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::tool_python::PYTHON_COMMAND;

/// The language of a script `terminal` or `monitor` runs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ScriptLanguage {
    /// The script is shell source.
    #[default]
    Shell,
    /// The script is Python source, run from a file in the sandbox's
    /// temporary directory so a traceback carries its line numbers.
    Python,
}

#[derive(Template)]
#[template(path = "python_script.sh", escape = "none")]
struct PythonScriptTemplate<'a> {
    interpreter: &'a str,
    task_id: &'a str,
    code: &'a str,
}

impl ScriptLanguage {
    /// The shell script that runs `script` for the job `task_id`.
    ///
    /// Shell source runs as it is. Python source is written to a file under
    /// the sandbox's `TMPDIR`, which the job removes when it ends, and run
    /// with the `python3` the sandbox's `PATH` resolves: the managed
    /// interpreter on a platform that provides one.
    ///
    /// # Errors
    ///
    /// Fails when the wrapper template does not render.
    pub(crate) fn launch_script<'a>(
        self,
        script: &'a str,
        task_id: &str,
    ) -> Result<Cow<'a, str>, askama::Error> {
        match self {
            Self::Shell => Ok(Cow::Borrowed(script)),
            Self::Python => PythonScriptTemplate {
                interpreter: PYTHON_COMMAND,
                task_id,
                code: script,
            }
            .render()
            .map(Cow::Owned),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_runs_as_written() {
        let script = ScriptLanguage::Shell
            .launch_script("echo hi", "t1")
            .unwrap();
        assert_eq!(script, "echo hi");
    }

    #[test]
    fn python_is_written_to_a_file_of_its_task() {
        let script = ScriptLanguage::Python
            .launch_script("print(1)", "t1")
            .unwrap();
        assert!(script.contains("aither-t1.py"));
        assert!(script.contains("print(1)"));
    }

    #[test]
    fn language_is_lowercase_on_the_wire() {
        assert_eq!(
            serde_json::to_string(&ScriptLanguage::Python).unwrap(),
            "\"python\""
        );
        assert_eq!(ScriptLanguage::default(), ScriptLanguage::Shell);
    }
}
