use std::path::PathBuf;

use crate::config::ProviderConfig;

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn log_stem(label: &str) -> String {
    match std::env::var_os("HOME") {
        Some(h) => format!("{}/.local/state/llm-harmony/{label}", PathBuf::from(h).display()),
        None => format!("/tmp/{label}"),
    }
}

/// A LaunchAgent that runs the provider's declared command.
///
/// Run through `/bin/sh -lc` deliberately: the declared command is what the
/// user would type, complete with `~`, `$HOME` and quoting, and splitting it
/// into an argv here would mean reimplementing shell parsing badly.
///
/// `KeepAlive` is unconditional, and that is deliberate. The obvious-looking
/// `KeepAlive.SuccessfulExit = false` -- restart only on failure -- does not
/// supervise anything useful here: measured 2026-09-09, `llama-server` traps
/// SIGTERM, logs "cleaning up before exit...", and exits **0**. launchd
/// correctly treated a killed server as a clean exit and left it down.
///
/// Unconditional restart is safe because a deliberate stop does not go through
/// the exit code at all: `launchctl bootout` removes the job from the domain,
/// so there is nothing left for KeepAlive to revive. The qualifier was
/// guarding against a problem `bootout` already solves, at the cost of the
/// supervision this exists to provide.
///
/// `ThrottleInterval` bounds the damage when the command is simply wrong -- a
/// start that exits 127 would otherwise be respawned as fast as launchd can
/// manage it.
///
/// `PATH` is captured at install time rather than inherited. Found the hard way
/// 2026-09-09: `launchctl getenv PATH` is unset, so an agent gets the bare
/// `/usr/bin:/bin:/usr/sbin:/sbin`, and `~/.llamacpp/serve-pool` died eight
/// times with `exec: llama-server: not found` even though it is on the user's
/// PATH at /opt/homebrew/bin. A login shell does not help: `/bin/sh -lc` reads
/// `/etc/profile`, never the user's shell rc. Installing with the environment
/// you would have run it in is the honest fix.
pub fn for_provider(p: &ProviderConfig, path_env: &str) -> Result<String, String> {
    let Some(command) = p.start.as_deref() else {
        return Err(format!(
            "{} declares no `start` command; add one to config to start it",
            p.kind
        ));
    };
    let label = p.label();
    Ok(format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{label}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/sh</string>
    <string>-lc</string>
    <string>{command}</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict><key>PATH</key><string>{path_env}</string></dict>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>{log}.out.log</string>
  <key>StandardErrorPath</key><string>{log}.err.log</string>
</dict>
</plist>
"#,
        label = escape(&label),
        command = escape(command),
        path_env = escape(path_env),
        log = escape(&log_stem(&label)),
    ))
}

/// `~/Library/LaunchAgents/<label>.plist`
pub fn path(label: &str) -> Option<PathBuf> {
    Some(
        PathBuf::from(std::env::var_os("HOME")?)
            .join("Library/LaunchAgents")
            .join(format!("{label}.plist")),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ProviderKind;

    fn cfg(start: Option<&str>) -> ProviderConfig {
        ProviderConfig {
            kind: ProviderKind::LlamaCpp,
            url: "http://127.0.0.1:8080".into(),
            start: start.map(str::to_string),
            launchd_label: None,
            kv_dtype_bytes: None,
        }
    }

    #[test]
    fn the_agent_runs_the_declared_command_through_a_shell() {
        let xml = for_provider(&cfg(Some("~/.llamacpp/serve-pool --port 8080")), "/usr/bin").unwrap();
        assert!(xml.contains("dev.llm-harmony.llamacpp"), "{xml}");
        assert!(xml.contains("/bin/sh"), "a shell, so ~ and flags behave as typed: {xml}");
        assert!(xml.contains("serve-pool --port 8080"), "{xml}");
    }

    /// Unconditional, because a killed llama-server exits 0 -- it traps
    /// SIGTERM and shuts down cleanly. `SuccessfulExit: false` therefore
    /// supervises nothing, which is what happened live on 2026-09-09.
    /// A deliberate stop is `bootout`, which removes the job outright.
    #[test]
    fn the_agent_restarts_unconditionally() {
        let xml = for_provider(&cfg(Some("x")), "/usr/bin").unwrap();
        assert!(xml.contains("<key>KeepAlive</key><true/>"), "{xml}");
        assert!(
            !xml.contains("SuccessfulExit"),
            "a provider that exits 0 when killed would never come back: {xml}"
        );
    }

    #[test]
    fn a_provider_with_no_start_command_cannot_produce_an_agent() {
        assert!(for_provider(&cfg(None), "/usr/bin").is_err());
    }

    /// A launchd agent inherits none of the user's PATH. Found live
    /// 2026-09-09: serve-pool died with `exec: llama-server: not found`
    /// because /opt/homebrew/bin was absent.
    #[test]
    fn the_agent_carries_the_path_it_was_installed_with() {
        let xml = for_provider(&cfg(Some("serve-pool")), "/opt/homebrew/bin:/usr/bin").unwrap();
        assert!(xml.contains("EnvironmentVariables"), "{xml}");
        assert!(xml.contains("/opt/homebrew/bin"), "{xml}");
    }

    /// A start command that exits 127 must not be respawned as fast as
    /// launchd can manage it.
    #[test]
    fn a_failing_command_is_throttled() {
        let xml = for_provider(&cfg(Some("x")), "/usr/bin").unwrap();
        assert!(xml.contains("ThrottleInterval"), "{xml}");
    }

    /// A `&` in a path would otherwise produce a plist launchd silently
    /// refuses to load.
    #[test]
    fn the_command_is_escaped_for_xml() {
        let xml = for_provider(&cfg(Some("run --a=1 && echo <ok>")), "/usr/bin").unwrap();
        assert!(xml.contains("&amp;&amp;"), "{xml}");
        assert!(xml.contains("&lt;ok&gt;"), "{xml}");
        assert!(!xml.contains("&& echo <ok>"), "raw markup leaked: {xml}");
    }
}
