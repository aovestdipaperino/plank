//! RFC 4180 CSV, small: quoted fields, doubled quotes, embedded commas and
//! newlines, CRLF or LF in, LF out.

/// What a parse produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parsed {
    /// Every complete record.
    pub rows: Vec<Vec<String>>,
    /// The 1-based line where an unterminated quoted field began, if any.
    /// Records before it are kept.
    pub error: Option<usize>,
}

/// Parses CSV text.
#[must_use]
pub fn parse(text: &str) -> Parsed {
    let mut rows = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut quote_line = 0;
    let mut line = 1;
    let mut any = false; // current record has content
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if in_quotes {
            match c {
                '"' if chars.peek() == Some(&'"') => {
                    chars.next();
                    field.push('"');
                }
                '"' => in_quotes = false,
                '\n' => {
                    line += 1;
                    field.push('\n');
                }
                c => field.push(c),
            }
            continue;
        }
        match c {
            '"' => {
                in_quotes = true;
                quote_line = line;
                any = true;
            }
            ',' => {
                row.push(std::mem::take(&mut field));
                any = true;
            }
            '\r' if chars.peek() == Some(&'\n') => {}
            '\n' => {
                line += 1;
                row.push(std::mem::take(&mut field));
                rows.push(std::mem::take(&mut row));
                any = false;
            }
            c => {
                field.push(c);
                any = true;
            }
        }
    }
    if in_quotes {
        return Parsed {
            rows,
            error: Some(quote_line),
        };
    }
    if any || !field.is_empty() {
        row.push(field);
        rows.push(row);
    }
    Parsed { rows, error: None }
}

/// Writes records as CSV, quoting only fields that need it.
#[must_use]
pub fn write(rows: &[Vec<String>]) -> String {
    let mut out = String::new();
    for row in rows {
        for (i, f) in row.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            if f.contains([',', '"', '\n', '\r']) {
                out.push('"');
                out.push_str(&f.replace('"', "\"\""));
                out.push('"');
            } else {
                out.push_str(f);
            }
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(v: &[&[&str]]) -> Vec<Vec<String>> {
        v.iter()
            .map(|r| r.iter().map(|s| (*s).to_string()).collect())
            .collect()
    }

    #[test]
    fn plain_fields_and_line_endings() {
        let p = parse("a,b\r\nc,d\n");
        assert_eq!(p.rows, rows(&[&["a", "b"], &["c", "d"]]));
        assert_eq!(p.error, None);
    }

    #[test]
    fn quoted_fields_with_commas_quotes_and_newlines() {
        let p = parse("\"x,y\",\"say \"\"hi\"\"\",\"two\nlines\"\n");
        assert_eq!(p.rows, rows(&[&["x,y", "say \"hi\"", "two\nlines"]]));
    }

    #[test]
    fn empty_input_and_trailing_empty_field() {
        assert!(parse("").rows.is_empty());
        assert_eq!(parse("a,\n").rows, rows(&[&["a", ""]]));
    }

    #[test]
    fn unterminated_quote_keeps_what_parsed_and_names_the_line() {
        let p = parse("a,b\nc,\"open\nstill open");
        assert_eq!(p.rows, rows(&[&["a", "b"]]));
        assert_eq!(p.error, Some(2));
    }

    #[test]
    fn a_blank_line_is_an_empty_record() {
        assert!(parse("").rows.is_empty());
        assert_eq!(parse("a\n").rows, rows(&[&["a"]]));
        assert_eq!(parse("a\n\n").rows, rows(&[&["a"], &[""]]));
        assert_eq!(
            parse("a,b\n\nc,d\n").rows,
            rows(&[&["a", "b"], &[""], &["c", "d"]])
        );
        assert_eq!(
            parse("a,b\r\n\r\nc,d").rows,
            rows(&[&["a", "b"], &[""], &["c", "d"]])
        );
    }

    #[test]
    fn write_quotes_only_what_needs_it_and_round_trips() {
        let r = rows(&[&["a", "b,c", "d\"e", "f\ng", ""]]);
        let text = write(&r);
        assert_eq!(text, "a,\"b,c\",\"d\"\"e\",\"f\ng\",\n");
        assert_eq!(parse(&text).rows, r);
    }
}
