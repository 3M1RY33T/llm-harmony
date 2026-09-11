use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
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

    /// A route that serves raw bytes with an honest Content-Length. The JSON
    /// routes above all set `application/json`, which a file transfer is not.
    pub fn start_bytes(path: &str, body: &str) -> Self {
        Self::start_raw(path.to_string(), body.to_string(), body.len())
    }

    /// A route that *promises* `declared` bytes and delivers the body instead.
    /// Exactly the shape of an interrupted transfer, which is the failure a
    /// download must not publish as a finished file.
    pub fn start_truncating(path: &str, body: &str, declared: usize) -> Self {
        Self::start_raw(path.to_string(), body.to_string(), declared)
    }

    fn start_raw(path: String, body: String, declared_len: usize) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let port = listener.local_addr().unwrap().port();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = stop.clone();
        thread::spawn(move || {
            for stream in listener.incoming() {
                if stop_thread.load(Ordering::Relaxed) {
                    break;
                }
                let Ok(mut stream) = stream else { continue };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                if reader.read_line(&mut line).is_err() {
                    continue;
                }
                let asked = line.split_whitespace().nth(1).unwrap_or("/").to_string();
                let asked = asked.split('?').next().unwrap_or("/").to_string();
                let head = if asked == path {
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {declared_len}\r\nConnection: close\r\n\r\n"
                    )
                } else {
                    "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_string()
                };
                let _ = stream.write_all(head.as_bytes());
                if asked == path {
                    let _ = stream.write_all(body.as_bytes());
                }
                let _ = stream.flush();
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

/// A stub that remembers what it was asked, for testing the verbs that *write*.
/// `StubServer` proves an adapter reads a body correctly; this one proves it
/// sent the right one.
pub struct RecordingServer {
    port: u16,
    stop: Arc<AtomicBool>,
    seen: Arc<Mutex<Vec<Request>>>,
}

#[derive(Debug, Clone)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub json: serde_json::Value,
}

impl RecordingServer {
    pub fn start() -> Self {
        Self::start_with(HashMap::new())
    }

    /// Canned bodies for paths the adapter reads before it writes -- Ollama
    /// asks `/api/show` what a model can do before choosing an endpoint.
    pub fn start_with(canned: HashMap<String, String>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let port = listener.local_addr().unwrap().port();
        let stop = Arc::new(AtomicBool::new(false));
        let seen: Arc<Mutex<Vec<Request>>> = Arc::new(Mutex::new(Vec::new()));

        let (stop_t, seen_t) = (stop.clone(), seen.clone());
        thread::spawn(move || {
            for stream in listener.incoming() {
                if stop_t.load(Ordering::Relaxed) {
                    break;
                }
                let Ok(stream) = stream else { continue };
                record(stream, &seen_t, &canned);
            }
        });

        RecordingServer { port, stop, seen }
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    pub fn last_request(&self) -> Request {
        self.seen.lock().unwrap().last().cloned().expect("a request was made")
    }
}

impl Drop for RecordingServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = std::net::TcpStream::connect(("127.0.0.1", self.port));
    }
}

fn record(
    mut stream: TcpStream,
    seen: &Arc<Mutex<Vec<Request>>>,
    canned: &HashMap<String, String>,
) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut line = String::new();
    if reader.read_line(&mut line).is_err() {
        return;
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("GET").to_string();
    let path = parts.next().unwrap_or("/").to_string();

    let mut len = 0usize;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).is_err() || header.trim().is_empty() {
            break;
        }
        if let Some(v) = header.to_ascii_lowercase().strip_prefix("content-length:") {
            len = v.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0u8; len];
    if len > 0 {
        let _ = reader.read_exact(&mut body);
    }
    let json = serde_json::from_slice(&body).unwrap_or(serde_json::Value::Null);
    seen.lock().unwrap().push(Request { method, path: path.clone(), json });

    let body = canned.get(&path).cloned().unwrap_or_else(|| "{}".to_string());
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

pub mod tree;
