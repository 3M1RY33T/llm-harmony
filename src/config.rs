use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::provider::ProviderKind;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderConfig {
    pub kind: ProviderKind,
    pub url: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub providers: Vec<ProviderConfig>,
}

/// The on-disk shape. Kept separate from `Config` so `kind` can be validated
/// with a message that names the offending string.
#[derive(Debug, Deserialize)]
struct RawConfig {
    #[serde(default)]
    provider: Vec<RawProvider>,
}

#[derive(Debug, Deserialize)]
struct RawProvider {
    kind: String,
    url: String,
}

impl Config {
    /// Zero-config behaviour: every provider on its conventional port.
    pub fn defaults() -> Self {
        Config {
            providers: ProviderKind::ALL
                .into_iter()
                .map(|kind| ProviderConfig {
                    kind,
                    url: format!("http://127.0.0.1:{}", kind.default_port()),
                })
                .collect(),
        }
    }

    pub fn from_toml(s: &str) -> Result<Self, String> {
        let raw: RawConfig = toml::from_str(s).map_err(|e| format!("invalid config: {e}"))?;
        if raw.provider.is_empty() {
            return Ok(Config::defaults());
        }
        let providers = raw
            .provider
            .into_iter()
            .map(|p| {
                let kind = p.kind.parse::<ProviderKind>()?;
                Ok(ProviderConfig {
                    kind,
                    url: p.url.trim_end_matches('/').to_string(),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Config { providers })
    }

    /// `~/.config/llm-harmony/config.toml`, or defaults if it is absent.
    pub fn load(explicit: Option<&Path>) -> Result<Self, String> {
        let path = match explicit {
            Some(p) => p.to_path_buf(),
            None => match Self::default_path() {
                Some(p) => p,
                None => return Ok(Config::defaults()),
            },
        };
        match std::fs::read_to_string(&path) {
            Ok(s) => Config::from_toml(&s),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && explicit.is_none() => {
                Ok(Config::defaults())
            }
            Err(e) => Err(format!("cannot read {}: {e}", path.display())),
        }
    }

    fn default_path() -> Option<PathBuf> {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config/llm-harmony/config.toml"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_cover_all_four_providers() {
        let c = Config::defaults();
        assert_eq!(c.providers.len(), 4);
        let lms = c.providers.iter().find(|p| p.kind == ProviderKind::LmStudio).unwrap();
        assert_eq!(lms.url, "http://127.0.0.1:1234");
    }

    #[test]
    fn toml_replaces_defaults_entirely() {
        let c = Config::from_toml(
            r#"
            [[provider]]
            kind = "ollama"
            url = "http://127.0.0.1:99"
            "#,
        )
        .unwrap();
        assert_eq!(c.providers.len(), 1, "an explicit config is not merged with defaults");
        assert_eq!(c.providers[0].url, "http://127.0.0.1:99");
    }

    #[test]
    fn empty_toml_falls_back_to_defaults() {
        assert_eq!(Config::from_toml("").unwrap().providers.len(), 4);
    }

    #[test]
    fn unknown_kind_is_a_config_error() {
        let err = Config::from_toml(
            r#"
            [[provider]]
            kind = "vllm-mlx"
            url = "http://127.0.0.1:8000"
            "#,
        )
        .unwrap_err();
        assert!(err.contains("vllm-mlx"), "error should name the bad kind, got: {err}");
    }

    #[test]
    fn trailing_slashes_are_stripped_so_urls_join_cleanly() {
        let c = Config::from_toml(
            r#"
            [[provider]]
            kind = "ollama"
            url = "http://127.0.0.1:11434/"
            "#,
        )
        .unwrap();
        assert_eq!(c.providers[0].url, "http://127.0.0.1:11434");
    }
}
