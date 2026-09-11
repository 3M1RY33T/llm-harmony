use std::path::Path;
use std::process::Command;

fn uid() -> u32 {
    // SAFETY: getuid cannot fail and has no side effects.
    unsafe { libc::getuid() }
}

pub fn bootstrap_argv(uid: u32, plist: &Path) -> Vec<String> {
    vec!["bootstrap".into(), format!("gui/{uid}"), plist.display().to_string()]
}

pub fn kickstart_argv(uid: u32, label: &str) -> Vec<String> {
    vec!["kickstart".into(), "-k".into(), format!("gui/{uid}/{label}")]
}

pub fn bootout_argv(uid: u32, label: &str) -> Vec<String> {
    vec!["bootout".into(), format!("gui/{uid}/{label}")]
}

fn run(args: Vec<String>) -> Result<(), String> {
    let out = Command::new("launchctl")
        .args(&args)
        .output()
        .map_err(|e| format!("launchctl: {e}"))?;
    if out.status.success() {
        return Ok(());
    }
    Err(format!(
        "launchctl {} failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr).trim()
    ))
}

pub fn bootstrap(plist: &Path) -> Result<(), String> {
    run(bootstrap_argv(uid(), plist))
}

/// `-k` restarts it if already running, which makes `start` idempotent.
pub fn kickstart(label: &str) -> Result<(), String> {
    run(kickstart_argv(uid(), label))
}

/// Start the agent, bootstrapping it first if launchd has forgotten it.
///
/// `stop` is `bootout`, and `bootout` does not pause a service — it **removes
/// it from the domain**. So the obvious pair, stop then start, did not work:
/// `kickstart` answered *Could not find service … in domain* on an agent whose
/// plist was sitting right there on disk. Found running Task 10's own Step 2,
/// which exists to press the Start button on a provider that is down.
///
/// Bootstrapping only when kickstart reports that specific absence keeps this
/// from papering over a genuinely broken agent: any other failure is passed
/// through untouched.
pub fn start(label: &str, plist: &Path) -> Result<(), String> {
    match kickstart(label) {
        Ok(()) => Ok(()),
        Err(e) if e.contains("Could not find service") => {
            if !plist.exists() {
                return Err(format!(
                    "{e}; and no agent exists at {} to load",
                    plist.display()
                ));
            }
            bootstrap(plist)?;
            kickstart(label)
        }
        Err(e) => Err(e),
    }
}

pub fn bootout(label: &str) -> Result<(), String> {
    run(bootout_argv(uid(), label))
}

pub fn is_loaded(label: &str) -> bool {
    Command::new("launchctl")
        .args(["print", &format!("gui/{}/{label}", uid())])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// The agent's last exit status, if launchd knows one.
///
/// Turns "did not answer" into "your command exited 127", which is the
/// difference between a diagnosable failure and a hunt through logs.
pub fn last_exit_code(label: &str) -> Option<i32> {
    let out = Command::new("launchctl")
        .args(["print", &format!("gui/{}/{label}", uid())])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .find_map(|l| l.trim().strip_prefix("last exit code = "))
        .and_then(|v| v.trim().parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// gui/<uid> is the modern domain. `launchctl load` is deprecated and
    /// silently does the wrong thing under some session types.
    #[test]
    fn commands_target_the_gui_domain_of_the_current_user() {
        assert_eq!(
            bootstrap_argv(501, Path::new("/x/a.plist")),
            vec!["bootstrap", "gui/501", "/x/a.plist"]
        );
        assert_eq!(
            kickstart_argv(501, "dev.llm-harmony.llamacpp"),
            vec!["kickstart", "-k", "gui/501/dev.llm-harmony.llamacpp"]
        );
        assert_eq!(
            bootout_argv(501, "dev.llm-harmony.llamacpp"),
            vec!["bootout", "gui/501/dev.llm-harmony.llamacpp"]
        );
    }

    /// The recovery path in `start` only helps when there is a plist to load.
    /// Naming the file beats repeating launchctl's own message.
    #[test]
    fn starting_something_with_no_agent_on_disk_says_which_file_is_missing() {
        let e = start("dev.llm-harmony.nothing", Path::new("/nonexistent.plist"))
            .expect_err("there is no such agent");
        assert!(e.contains("/nonexistent.plist"), "{e}");
    }
}