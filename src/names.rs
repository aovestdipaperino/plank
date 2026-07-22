// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Memorable session names: a funny adjective plus a celebrity, e.g.
//! `deadly-einstein`. Half the celebrity pool is scientists; the other half is
//! historical / pop / sport figures, so a name is science-flavoured ~50% of
//! the time. On a filename collision the caller appends a short guid.

use std::sync::atomic::{AtomicU64, Ordering};

/// Funny adjectives (50).
const ADJECTIVES: &[&str] = &[
    "deadly", "sneaky", "wobbly", "grumpy", "sparkly", "dizzy", "cranky", "jolly", "snazzy",
    "bumbling", "cheeky", "zesty", "quirky", "feisty", "dapper", "goofy", "plucky", "sassy",
    "nifty", "rowdy", "spunky", "witty", "breezy", "cosmic", "turbo", "funky", "groovy", "mellow",
    "peppy", "zippy", "salty", "crispy", "fluffy", "sleepy", "sneezy", "bouncy", "cuddly", "wacky",
    "zany", "giddy", "snappy", "chunky", "spicy", "dorky", "loopy", "perky", "goopy", "wiggly",
    "squishy", "jazzy",
];

/// Scientists (75) — half the celebrity pool.
const SCIENTISTS: &[&str] = &[
    "einstein",
    "curie",
    "newton",
    "darwin",
    "tesla",
    "hawking",
    "turing",
    "feynman",
    "galileo",
    "bohr",
    "faraday",
    "pasteur",
    "mendel",
    "lovelace",
    "hopper",
    "sagan",
    "franklin",
    "heisenberg",
    "planck",
    "maxwell",
    "kepler",
    "copernicus",
    "dirac",
    "noether",
    "ramanujan",
    "fermi",
    "goodall",
    "euler",
    "gauss",
    "archimedes",
    "pauling",
    "bell",
    "boyle",
    "hubble",
    "oppenheimer",
    "schrodinger",
    "rutherford",
    "pavlov",
    "linnaeus",
    "galvani",
    "hertz",
    "kelvin",
    "watt",
    "volta",
    "ampere",
    "ohm",
    "joule",
    "avogadro",
    "dalton",
    "boltzmann",
    "hooke",
    "cavendish",
    "lavoisier",
    "priestley",
    "herschel",
    "halley",
    "brahe",
    "fibonacci",
    "pythagoras",
    "euclid",
    "humboldt",
    "fleming",
    "jenner",
    "lister",
    "salk",
    "mcclintock",
    "meitner",
    "chandrasekhar",
    "raman",
    "becquerel",
    "roentgen",
    "thomson",
    "chadwick",
    "pauli",
    "hahn",
];

/// Historical / pop / sport figures (75) — the other half of the pool.
const OTHERS: &[&str] = &[
    "caesar",
    "napoleon",
    "cleopatra",
    "gandhi",
    "lincoln",
    "churchill",
    "mandela",
    "bowie",
    "prince",
    "madonna",
    "elvis",
    "dylan",
    "jordan",
    "pele",
    "ali",
    "serena",
    "messi",
    "bolt",
    "gretzky",
    "federer",
    "shakespeare",
    "mozart",
    "beethoven",
    "picasso",
    "dali",
    "hemingway",
    "tolkien",
    "chaplin",
    "hitchcock",
    "houdini",
    "socrates",
    "plato",
    "aristotle",
    "columbus",
    "magellan",
    "tubman",
    "hendrix",
    "lennon",
    "jagger",
    "beyonce",
    "nadal",
    "ronaldo",
    "phelps",
    "owens",
    "ruth",
    "montana",
    "brady",
    "kobe",
    "maradona",
    "zidane",
    "hannibal",
    "attila",
    "genghis",
    "joan",
    "victoria",
    "washington",
    "jefferson",
    "roosevelt",
    "kennedy",
    "tut",
    "sinatra",
    "cash",
    "marley",
    "cobain",
    "mercury",
    "aretha",
    "springsteen",
    "tyson",
    "jeter",
    "woods",
    "senna",
    "agassi",
    "navratilova",
    "biles",
    "hamilton",
];

/// A random-ish 64-bit value seeded from wall clock, pid, and a per-process
/// bump counter so repeated calls within one process still differ. Good enough
/// for picking names and short guids; real filename uniqueness is enforced by
/// the caller checking the directory.
#[allow(clippy::cast_possible_truncation)] // truncation is fine for entropy
fn entropy() -> u64 {
    static BUMP: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    let pid = u64::from(std::process::id());
    let bump = BUMP.fetch_add(1, Ordering::Relaxed);
    splitmix64(nanos ^ (pid << 32) ^ bump.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// One round of the splitmix64 PRNG.
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Picks one item from `pool` using `r`.
#[allow(clippy::cast_possible_truncation)] // pool lengths are tiny
fn pick<'a>(pool: &[&'a str], r: u64) -> &'a str {
    pool[(r % pool.len() as u64) as usize]
}

/// Generates one `adjective-celebrity` session slug. The celebrity is a
/// scientist ~50% of the time, otherwise from the historical/pop/sport pool.
#[must_use]
pub fn session_slug() -> String {
    let adj = pick(ADJECTIVES, entropy());
    let r = entropy();
    let celeb = if r & 1 == 0 {
        pick(SCIENTISTS, r >> 1)
    } else {
        pick(OTHERS, r >> 1)
    };
    format!("{adj}-{celeb}")
}

/// A short 8-hex-char disambiguation suffix (a compact guid) for the rare case
/// where a generated slug already names an existing file.
#[must_use]
#[allow(clippy::cast_possible_truncation)] // low 32 bits are plenty
pub fn guid8() -> String {
    format!("{:08x}", entropy() as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slug_is_adjective_dash_celebrity_from_the_pools() {
        for _ in 0..200 {
            let slug = session_slug();
            let (adj, celeb) = slug.split_once('-').expect("adjective-celebrity");
            assert!(ADJECTIVES.contains(&adj), "unknown adjective: {adj}");
            assert!(
                SCIENTISTS.contains(&celeb) || OTHERS.contains(&celeb),
                "unknown celebrity: {celeb}"
            );
            // Slug is a clean filename stem.
            assert!(slug.bytes().all(|b| b.is_ascii_lowercase() || b == b'-'));
        }
    }

    #[test]
    fn scientists_are_about_half_the_names() {
        let mut science = 0;
        let n = 4000;
        for _ in 0..n {
            let celeb = session_slug().split_once('-').unwrap().1.to_owned();
            if SCIENTISTS.contains(&celeb.as_str()) {
                science += 1;
            }
        }
        // ~50%; allow a wide band so the test is not flaky.
        assert!(
            (1600..2400).contains(&science),
            "science share off: {science}/{n}"
        );
    }

    #[test]
    fn pools_are_sized_and_unique() {
        assert_eq!(ADJECTIVES.len(), 50);
        assert_eq!(SCIENTISTS.len(), 75);
        assert_eq!(OTHERS.len(), 75); // 150 celebrities total, 50% science
        let mut all: Vec<&str> = ADJECTIVES
            .iter()
            .chain(SCIENTISTS)
            .chain(OTHERS)
            .copied()
            .collect();
        let n = all.len();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), n, "duplicate word across pools");
    }

    #[test]
    fn guid8_is_eight_hex_chars() {
        let g = guid8();
        assert_eq!(g.len(), 8);
        assert!(g.bytes().all(|b| b.is_ascii_hexdigit()));
    }
}
