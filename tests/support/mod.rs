use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

/// A canned-response HTTP server. Routes map a request path to
/// `(status, body)`. Any path not in the map gets a 404 -- except when
/// `always_200` is set, which reproduces LM Studio's behaviour of answering
/// unknown paths with a 200 and an error body.
pub struct StubServer {
    port: u16,
    stop: Arc<AtomicBool>,
}

impl StubServer {
    pub fn start(routes: HashMap<String, (u16, String)>) -> Self {
        Self::start_inner(routes, false)
    }

    /// Verified against LM Studio 2026-09-09: unknown paths return
    /// `200 {"error":"Unexpected endpoint or method. (GET /path)"}`.
    pub fn start_lmstudio_style(routes: HashMap<String, (u16, String)>) -> Self {
        Self::start_inner(routes, true)
    }

    fn start_inner(routes: HashMap<String, (u16, String)>, always_200: bool) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let port = listener.local_addr().unwrap().port();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = stop.clone();

        thread::spawn(move || {
            for stream in listener.incoming() {
                if stop_thread.load(Ordering::Relaxed) {
                    break;
                }
                let Ok(stream) = stream else { continue };
                handle(stream, &routes, always_200);
            }
        });

        StubServer { port, stop }
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }
}

impl Drop for StubServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = std::net::TcpStream::connect(("127.0.0.1", self.port));
    }
}

fn handle(mut stream: TcpStream, routes: &HashMap<String, (u16, String)>, always_200: bool) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    let path = request_line.split_whitespace().nth(1).unwrap_or("/").to_string();

    let (status, body) = match routes.get(&path) {
        Some((s, b)) => (*s, b.clone()),
        None if always_200 => (
            200,
            format!(r#"{{"error":"Unexpected endpoint or method. (GET {path})"}}"#),
        ),
        None => (404, "404 page not found".to_string()),
    };

    let response = format!(
        "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

/// Convenience for the common single-route case.
pub fn routes(pairs: &[(&str, &str)]) -> HashMap<String, (u16, String)> {
    pairs
        .iter()
        .map(|(p, b)| (p.to_string(), (200u16, b.to_string())))
        .collect()
}
pub mod tree;
