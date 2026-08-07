use std::io::{self, Write};

/// Simple trial-division primality test.
fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }
    let mut d = 5u64;
    while d <= n / d {
        if n % d == 0 {
            return false;
        }
        d += 2;
        if d <= n / d && n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

fn main() {
    print!("Enter a number to check: ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();

    let n: u64 = match input.trim().parse() {
        Ok(v) => v,
        Err(_) => {
            println!("That's not a valid number.");
            return;
        }
    };

    if is_prime(n) {
        println!("{} is prime.", n);
    } else {
        println!("{} is not prime.", n);
    }
}

#[cfg(test)]
mod tests {
    use super::is_prime;

    #[test]
    fn small_primes() {
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(is_prime(5));
        assert!(is_prime(7));
        assert!(is_prime(97));
        assert!(is_prime(7919));
    }

    #[test]
    fn small_composites() {
        assert!(!is_prime(1));
        assert!(!is_prime(4));
        assert!(!is_prime(6));
        assert!(!is_prime(9));
        assert!(!is_prime(100));
        assert!(!is_prime(7917));
    }

    #[test]
    fn large_prime_no_overflow() {
        // Regression: `d * d <= n` overflowed u64 for inputs whose sqrt
        // exceeds 2^32, panicking in debug and hanging in release.
        assert!(is_prime(18446744073709551557));
    }
}
