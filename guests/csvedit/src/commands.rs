//! Command ids. Turbo Vision leaves `[1000, 65535]` to applications.

use turbo_vision::core::command::CommandId;

pub const CMD_NEW: CommandId = 1000;
pub const CMD_OPEN: CommandId = 1001;
pub const CMD_SAVE: CommandId = 1002;
pub const CMD_SAVE_AS: CommandId = 1003;
pub const CMD_EXIT: CommandId = 1004;
pub const CMD_EDIT_CELL: CommandId = 1005;
pub const CMD_RENAME_COL: CommandId = 1006;
pub const CMD_ROW_INS: CommandId = 1007;
pub const CMD_ROW_DEL: CommandId = 1008;
pub const CMD_COL_INS: CommandId = 1009;
pub const CMD_COL_DEL: CommandId = 1010;
pub const CMD_DLG_OK: CommandId = 1011;
pub const CMD_DLG_CANCEL: CommandId = 1012;
pub const CMD_DISCARD: CommandId = 1013;
pub const CMD_OPEN_PICK: CommandId = 1014;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_and_in_the_application_range() {
        let mut all = [
            CMD_NEW,
            CMD_OPEN,
            CMD_SAVE,
            CMD_SAVE_AS,
            CMD_EXIT,
            CMD_EDIT_CELL,
            CMD_RENAME_COL,
            CMD_ROW_INS,
            CMD_ROW_DEL,
            CMD_COL_INS,
            CMD_COL_DEL,
            CMD_DLG_OK,
            CMD_DLG_CANCEL,
            CMD_DISCARD,
            CMD_OPEN_PICK,
        ];
        assert!(all.iter().all(|c| *c >= 1000));
        all.sort_unstable();
        let n = all.len();
        let mut v = all.to_vec();
        v.dedup();
        assert_eq!(v.len(), n);
    }
}
