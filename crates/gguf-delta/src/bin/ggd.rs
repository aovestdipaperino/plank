//! `ggd`: create and inspect GGUF weight deltas.
//!
//! ```text
//! ggd create <base.gguf> <target.gguf> <out.ggd> [--label NAME] [--hash-base]
//! ggd info <file.ggd>
//! ```

use std::path::Path;
use std::process::ExitCode;

use gguf_delta::{CreateOptions, find_base, read_chunks, read_header, write_delta};

const USAGE: &str = "usage:
  ggd create <base.gguf> <target.gguf> <out.ggd> [--label NAME] [--hash-base]
      write the weight delta from base to target (same-layout GGUFs)
  ggd info <file.ggd>
      describe a weight delta and whether its base is reachable";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let rc = match args.first().map(String::as_str) {
        Some("create") => run_create(&args[1..]),
        Some("info") => run_info(&args[1..]),
        Some("-h" | "--help" | "help") => {
            println!("{USAGE}");
            0
        }
        _ => {
            eprintln!("{USAGE}");
            2
        }
    };
    ExitCode::from(u8::try_from(rc).unwrap_or(1))
}

#[allow(clippy::cast_precision_loss)]
fn run_create(args: &[String]) -> i32 {
    let mut positional: Vec<&String> = Vec::new();
    let mut opts = CreateOptions::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--label" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("ggd: --label needs a value");
                    return 2;
                };
                opts.label = Some(v.clone());
            }
            "--hash-base" => opts.hash_base = true,
            a if a.starts_with("--") => {
                eprintln!("ggd: unknown option {a}\n{USAGE}");
                return 2;
            }
            _ => positional.push(&args[i]),
        }
        i += 1;
    }
    let [base, target, out] = positional[..] else {
        eprintln!("{USAGE}");
        return 2;
    };
    let started = std::time::Instant::now();
    match write_delta(Path::new(base), Path::new(target), Path::new(out), &opts) {
        Ok(r) => {
            let ratio = if r.bytes_spanned == 0 {
                0.0
            } else {
                r.payload_bytes as f64 / r.bytes_spanned as f64 * 100.0
            };
            println!(
                "wrote {out}: label {}, {} tensors changed in {} chunks, {} of {} spanned bytes differ, {} bytes compressed ({ratio:.1}% of span), {:.1}s",
                r.label,
                r.tensors_changed,
                r.chunks,
                r.bytes_changed,
                r.bytes_spanned,
                r.payload_bytes,
                started.elapsed().as_secs_f64()
            );
            0
        }
        Err(e) => {
            eprintln!("ggd: {e}");
            1
        }
    }
}

fn run_info(args: &[String]) -> i32 {
    let [path] = args else {
        eprintln!("{USAGE}");
        return 2;
    };
    let delta = Path::new(path);
    let (h, first) = match read_header(delta) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("ggd: {e}");
            return 1;
        }
    };
    let chunks = match read_chunks(delta, &h, first) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("ggd: {e}");
            return 1;
        }
    };
    let base = find_base(&h, delta, &[]);
    let layout = base
        .as_ref()
        .ok()
        .and_then(|b| gguf_delta::gguf::layout(b).ok());
    println!("GGUF weight delta {}", delta.display());
    print!("{}", gguf_delta::describe(&h, &chunks, layout.as_ref()));
    match base {
        Ok(b) => println!("  base resolves:  {}", b.display()),
        Err(why) => println!("  base resolves:  no\n{why}"),
    }
    0
}
