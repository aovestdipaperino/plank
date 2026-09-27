//! Where documents are saved: plank's RAM disk in the real component, a map
//! in tests.

use std::collections::BTreeMap;

/// A place CSV files live. Paths are names at the disk's root.
pub trait Disk {
    /// A file's text.
    ///
    /// # Errors
    /// When it does not exist or the host refused.
    fn read(&self, path: &str) -> Result<String, String>;
    /// Stores a file.
    ///
    /// # Errors
    /// When the host refused (grant, quota).
    fn write(&mut self, path: &str, text: &str) -> Result<(), String>;
    /// The `.csv` files at the root, sorted.
    ///
    /// # Errors
    /// When the host refused.
    fn list(&self) -> Result<Vec<String>, String>;
}

/// An in-memory disk, for tests.
#[derive(Debug, Default)]
pub struct MemDisk(BTreeMap<String, String>);

impl Disk for MemDisk {
    fn read(&self, path: &str) -> Result<String, String> {
        self.0
            .get(path)
            .cloned()
            .ok_or_else(|| format!("no such file: /{path}"))
    }
    fn write(&mut self, path: &str, text: &str) -> Result<(), String> {
        self.0.insert(path.to_string(), text.to_string());
        Ok(())
    }
    fn list(&self) -> Result<Vec<String>, String> {
        Ok(self.0.keys().filter(|k| k.ends_with(".csv")).cloned().collect())
    }
}

/// plank's RAM disk, through the `fs` capability's host functions.
#[cfg(target_arch = "wasm32")]
#[derive(Debug, Default)]
pub struct PlankDisk;

#[cfg(target_arch = "wasm32")]
mod host {
    use extism_pdk::*;

    #[host_fn]
    extern "ExtismHost" {
        pub fn plank_fs_read(path: String) -> Vec<u8>;
        pub fn plank_fs_write(path: String, bytes: Vec<u8>) -> String;
        pub fn plank_fs_list(dir: String) -> Vec<u8>;
    }
}

#[cfg(target_arch = "wasm32")]
fn untag(reply: Vec<u8>) -> Result<Vec<u8>, String> {
    match reply.split_first() {
        Some((0, data)) => Ok(data.to_vec()),
        Some((_, msg)) => Err(String::from_utf8_lossy(msg).into_owned()),
        None => Err("empty reply from host".into()),
    }
}

#[cfg(target_arch = "wasm32")]
impl Disk for PlankDisk {
    fn read(&self, path: &str) -> Result<String, String> {
        let reply =
            unsafe { host::plank_fs_read(path.to_string()) }.map_err(|e| e.to_string())?;
        String::from_utf8(untag(reply)?).map_err(|_| format!("{path} is not UTF-8"))
    }
    fn write(&mut self, path: &str, text: &str) -> Result<(), String> {
        let err = unsafe { host::plank_fs_write(path.to_string(), text.as_bytes().to_vec()) }
            .map_err(|e| e.to_string())?;
        if err.is_empty() {
            Ok(())
        } else {
            Err(err)
        }
    }
    fn list(&self) -> Result<Vec<String>, String> {
        let reply =
            unsafe { host::plank_fs_list("/".to_string()) }.map_err(|e| e.to_string())?;
        let json = String::from_utf8_lossy(&untag(reply)?).into_owned();
        // Flat array of {"name":..,"size":..}; names never contain `"` in
        // practice, but a name that does is skipped rather than mangled.
        let mut names: Vec<String> = json
            .split("\"name\":\"")
            .skip(1)
            .filter_map(|s| s.split_once('"').map(|(n, _)| n.to_string()))
            .filter(|n| n.ends_with(".csv") && !n.contains('\\'))
            .collect();
        names.sort();
        Ok(names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_disk_round_trips_and_lists_only_csv() {
        let mut d = MemDisk::default();
        d.write("b.csv", "x").unwrap();
        d.write("a.csv", "y").unwrap();
        d.write("notes.txt", "z").unwrap();
        assert_eq!(d.read("a.csv").unwrap(), "y");
        assert_eq!(d.list().unwrap(), ["a.csv", "b.csv"]);
        assert!(d.read("missing.csv").is_err());
    }
}
