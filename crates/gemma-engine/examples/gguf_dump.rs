//! Prints the metadata Gemma support depends on, for checking a real GGUF
//! against the keys and control spellings plank assumes.
fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: gguf_dump <file.gguf>");
    let mut f = std::fs::File::open(&path).expect("open");
    let ct = candle_core::quantized::gguf_file::Content::read(&mut f).expect("read gguf");
    let mut keys: Vec<_> = ct
        .metadata
        .keys()
        .filter(|k| {
            !k.starts_with("tokenizer.ggml.tokens")
                && !k.starts_with("tokenizer.ggml.merges")
                && !k.starts_with("tokenizer.ggml.token_type")
                && !k.starts_with("tokenizer.ggml.scores")
        })
        .collect();
    keys.sort();
    for k in keys {
        println!("{k} = {:?}", ct.metadata[k]);
    }
    let tok = gemma_engine::tokenizer::GemmaTokenizer::from_gguf(&ct).expect("tokenizer");
    for s in gemma_engine::template::CONTROL_SPELLINGS {
        println!("control {s:?} -> {:?}", tok.control_id(s));
    }
    let mut names: Vec<_> = ct
        .tensor_infos
        .keys()
        .filter(|n| n.starts_with("blk.0.") || !n.starts_with("blk."))
        .collect();
    names.sort();
    for n in names {
        let i = &ct.tensor_infos[n];
        println!("tensor {n} {:?} {:?}", i.shape, i.ggml_dtype);
    }
}
