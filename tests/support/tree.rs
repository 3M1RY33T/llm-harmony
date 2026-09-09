use std::path::{Path, PathBuf};

/// A throwaway directory tree for scanner tests.
pub struct Tree {
    pub root: PathBuf,
}

impl Tree {
    pub fn new(name: &str) -> Tree {
        let root = std::env::temp_dir()
            .join(format!("llm-harmony-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("create test tree");
        Tree { root }
    }

    pub fn file(&self, rel: &str, bytes: usize) -> PathBuf {
        let p = self.root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, vec![0u8; bytes]).unwrap();
        p
    }

    pub fn write(&self, rel: &str, content: &str) -> PathBuf {
        let p = self.root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, content).unwrap();
        p
    }

    pub fn link(&self, rel: &str, target: &Path) -> PathBuf {
        let p = self.root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::os::unix::fs::symlink(target, &p).unwrap();
        p
    }
}

impl Drop for Tree {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}
