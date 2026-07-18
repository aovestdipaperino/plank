//! Startup banner: the plank sage, in ASCII.

/// ASCII rendition of the wild-haired sage shown at startup.
pub const LOGO: &str = r"
                .-~~^~-.,_,.-~~~-.
          _,-~~    ..-~~-..    ~~-,_
       ,-~   _,-~~          ~~-,_   ~-,
     ,~    ,~     __     __     ~,    ~,
    /     /      /  \~-~/  \      \     \
   ;     ;      | (@) | (@) |      ;     ;
   |     |       \__/ ^ \__/       |     |
   ;      \      _,-~\_/~-,_      /      ;
    \      ~,   / /~~-----~~\ \  ,~      /
     ~,      ~-,| | ,~~~~~, | |,-~     ,~
       ~-,      \_\_|     |_/_/     ,-~
          ~~-,    \  \~~~/  /   ,-~~
              \    ~, ||| ,~   /
               \     \|||/    /
                ~-..__~~~__..-~
                  p l a n k
";

/// Writes the logo followed by a version line.
#[must_use]
pub fn banner() -> String {
    format!("{LOGO}      v{}\n", env!("CARGO_PKG_VERSION"))
}

#[cfg(test)]
mod tests {
    #[test]
    fn banner_contains_name_and_version() {
        let b = super::banner();
        assert!(b.contains("p l a n k"));
        assert!(b.contains(env!("CARGO_PKG_VERSION")));
    }
}
