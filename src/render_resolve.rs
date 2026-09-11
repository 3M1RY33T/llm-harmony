use crate::estimate::estimator::{Basis, Estimate};
use crate::memory::Machine;
use crate::render::human_bytes;
use crate::resolve::decide::Decision;

pub fn render(d: &Decision, e: &Estimate, machine: &Machine, reserve: u64) -> String {
    let free = machine.total_bytes.saturating_sub(machine.used_bytes);
    let headroom = free.saturating_sub(reserve);
    let mut out = String::new();

    match d {
        Decision::Ready { base_url, provider_model_id, provider, resident, inferred } => {
            out.push_str(&format!("  ready      {base_url}  {provider_model_id}\n"));
            out.push_str(&format!(
                "  provider   {provider} ({})\n",
                if *resident { "model already resident" } else { "model not resident" }
            ));
            match (e.bytes, e.basis) {
                (Some(b), Basis::Measured) => out.push_str(&format!(
                    "  predicted  {}  (measured, {} observation{})\n",
                    human_bytes(b),
                    e.samples,
                    if e.samples == 1 { "" } else { "s" }
                )),
                (Some(b), Basis::Computed) => out.push_str(&format!(
                    "  estimate   {} (computed from the artifact)\n",
                    crate::render::human_bytes(b)
                )),
                (Some(b), Basis::Declared) => out.push_str(&format!(
                    "  predicted  {}  (DECLARED — weights only, a floor not a peak)\n",
                    human_bytes(b)
                )),
                _ if *resident => out.push_str("  predicted  n/a — already loaded\n"),
                _ => out.push_str("  predicted  unknown\n"),
            }
            out.push_str(&format!("  headroom   {}\n", human_bytes(headroom)));
            out.push_str(&format!(
                "  identity   {}\n",
                if *inferred {
                    "INFERRED — derived, not reported by the provider"
                } else {
                    "artifact path reported by the provider"
                }
            ));
        }
        Decision::Wait { reason, eta_s } => {
            out.push_str(&format!("  wait       {reason}\n"));
            out.push_str(&format!("  eta        ~{eta_s}s\n"));
        }
        Decision::Deny { reason, alternatives } => {
            out.push_str(&format!("  deny       {reason}\n"));
            if !alternatives.is_empty() {
                out.push_str("  elsewhere  (offered, not chosen — a different build is a\n");
                out.push_str("             different model; see docs/field-notes.md)\n");
                for a in alternatives {
                    out.push_str(&format!("               {a}\n"));
                }
            }
        }
    }
    out
}
