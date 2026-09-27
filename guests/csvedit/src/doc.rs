//! The table being edited: a header row and a body, always rectangular and
//! never empty, with a modified flag.

use crate::csv;

/// Spreadsheet-style name for column `i`: A..Z, then AA, AB, ...
fn column_name(mut i: usize) -> String {
    let mut s = String::new();
    loop {
        s.insert(0, char::from(b'A' + u8::try_from(i % 26).unwrap_or(0)));
        if i < 26 {
            return s;
        }
        i = i / 26 - 1;
    }
}

/// A CSV document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsvDoc {
    header: Vec<String>,
    body: Vec<Vec<String>>,
    modified: bool,
}

impl CsvDoc {
    /// `cols` named columns and `rows` empty rows. Both at least 1.
    #[must_use]
    pub fn new_blank(cols: usize, rows: usize) -> Self {
        let cols = cols.max(1);
        Self {
            header: (0..cols).map(column_name).collect(),
            body: vec![vec![String::new(); cols]; rows.max(1)],
            modified: false,
        }
    }

    /// Parses text; the first record is the header. Ragged rows are padded,
    /// and missing header names become column letters. A parse error comes
    /// back as its line number, with the document marked modified.
    #[must_use]
    pub fn from_text(text: &str) -> (Self, Option<usize>) {
        let parsed = csv::parse(text);
        let mut rows = parsed.rows.into_iter();
        let mut header = rows.next().unwrap_or_default();
        let mut body: Vec<Vec<String>> = rows.collect();
        let width = body.iter().map(Vec::len).chain([header.len(), 1]).max().unwrap_or(1);
        for i in header.len()..width {
            header.push(column_name(i));
        }
        for r in &mut body {
            r.resize(width, String::new());
        }
        if body.is_empty() {
            body.push(vec![String::new(); width]);
        }
        (Self { header, body, modified: parsed.error.is_some() }, parsed.error)
    }

    /// The document as CSV text, header first.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut all = Vec::with_capacity(self.body.len() + 1);
        all.push(self.header.clone());
        all.extend(self.body.iter().cloned());
        csv::write(&all)
    }

    #[must_use]
    pub fn header(&self) -> &[String] {
        &self.header
    }
    #[must_use]
    pub fn body(&self) -> &[Vec<String>] {
        &self.body
    }
    #[must_use]
    pub fn width(&self) -> usize {
        self.header.len()
    }
    #[must_use]
    pub fn height(&self) -> usize {
        self.body.len()
    }
    #[must_use]
    pub fn cell(&self, row: usize, col: usize) -> &str {
        self.body.get(row).and_then(|r| r.get(col)).map_or("", String::as_str)
    }

    pub fn set_cell(&mut self, row: usize, col: usize, value: String) {
        if let Some(c) = self.body.get_mut(row).and_then(|r| r.get_mut(col))
            && *c != value
        {
            *c = value;
            self.modified = true;
        }
    }

    pub fn insert_row(&mut self, at: usize) {
        let at = at.min(self.body.len());
        self.body.insert(at, vec![String::new(); self.width()]);
        self.modified = true;
    }

    pub fn delete_row(&mut self, at: usize) {
        if at >= self.body.len() {
            return;
        }
        if self.body.len() == 1 {
            self.body[0].iter_mut().for_each(String::clear);
        } else {
            self.body.remove(at);
        }
        self.modified = true;
    }

    pub fn insert_col(&mut self, at: usize) {
        let at = at.min(self.width());
        let name = column_name(self.width());
        self.header.insert(at, name);
        for r in &mut self.body {
            r.insert(at, String::new());
        }
        self.modified = true;
    }

    pub fn delete_col(&mut self, at: usize) {
        if at >= self.width() {
            return;
        }
        if self.width() == 1 {
            self.header[0] = column_name(0);
            for r in &mut self.body {
                r[0].clear();
            }
        } else {
            self.header.remove(at);
            for r in &mut self.body {
                r.remove(at);
            }
        }
        self.modified = true;
    }

    pub fn rename_col(&mut self, col: usize, name: String) {
        if let Some(h) = self.header.get_mut(col)
            && *h != name
        {
            *h = name;
            self.modified = true;
        }
    }

    #[must_use]
    pub fn is_modified(&self) -> bool {
        self.modified
    }

    pub fn mark_saved(&mut self) {
        self.modified = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_doc_has_named_columns_and_empty_body() {
        let d = CsvDoc::new_blank(3, 3);
        assert_eq!(d.header(), ["A", "B", "C"]);
        assert_eq!(d.height(), 3);
        assert_eq!(d.cell(0, 0), "");
        assert!(!d.is_modified());
    }

    #[test]
    fn ragged_rows_pad_to_the_widest() {
        let (d, err) = CsvDoc::from_text("h1,h2\nx\ny,z,extra\n");
        assert_eq!(err, None);
        assert_eq!(d.width(), 3);
        assert_eq!(d.header(), ["h1", "h2", "C"]);
        assert_eq!(d.cell(0, 1), "");
        assert_eq!(d.cell(1, 2), "extra");
    }

    #[test]
    fn edits_mark_modified_and_save_clears_it() {
        let mut d = CsvDoc::new_blank(2, 1);
        d.set_cell(0, 1, "v".into());
        assert!(d.is_modified());
        assert_eq!(d.to_text(), "A,B\n,v\n");
        d.mark_saved();
        assert!(!d.is_modified());
    }

    #[test]
    fn rows_and_columns_insert_and_delete() {
        let (mut d, _) = CsvDoc::from_text("a,b\n1,2\n3,4\n");
        d.insert_row(1);
        assert_eq!(d.height(), 3);
        assert_eq!(d.cell(1, 0), "");
        d.delete_row(0);
        assert_eq!(d.cell(1, 0), "3");
        d.insert_col(1);
        assert_eq!(d.header(), ["a", "C", "b"]);
        d.delete_col(0);
        assert_eq!(d.header(), ["C", "b"]);
        d.rename_col(0, "mid".into());
        assert_eq!(d.to_text(), "mid,b\n,\n,4\n");
    }

    #[test]
    fn deletes_never_leave_an_empty_table() {
        let mut d = CsvDoc::new_blank(1, 1);
        d.delete_row(0);
        assert_eq!(d.height(), 1, "the last row is cleared, not removed");
        d.delete_col(0);
        assert_eq!(d.width(), 1, "the last column is cleared, not removed");
    }

    #[test]
    fn a_parse_error_marks_the_doc_modified() {
        let (d, err) = CsvDoc::from_text("a\n\"open");
        assert_eq!(err, Some(2));
        assert!(d.is_modified());
    }
}
