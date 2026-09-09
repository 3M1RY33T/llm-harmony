use llm_harmony::memory::{port_from_url, Machine};

#[test]
fn machine_total_matches_sysctl_hw_memsize() {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .expect("sysctl runs on macOS");
    let expected: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().unwrap();

    let m = Machine::read().unwrap();
    assert_eq!(m.total_bytes, expected);
}

#[test]
fn machine_reports_plausible_usage() {
    let m = Machine::read().unwrap();
    assert!(m.used_bytes > 0);
    assert!(m.used_bytes <= m.total_bytes);
    assert!(m.swap_used_bytes <= m.swap_total_bytes);
}

#[test]
fn port_is_parsed_out_of_a_config_url() {
    assert_eq!(port_from_url("http://127.0.0.1:1234"), Some(1234));
    assert_eq!(port_from_url("http://127.0.0.1:11434/"), Some(11434));
    assert_eq!(port_from_url("http://localhost"), None);
    assert_eq!(port_from_url("garbage"), None);
}

/// Nothing binds port 1, so this must be None rather than a panic.
#[test]
fn an_unbound_port_has_no_pid() {
    assert_eq!(llm_harmony::memory::pid_listening_on(1), None);
}
