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
}
