// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! `plank-console` — a Turbo Vision monitor for plank model-token streams.

mod cmd;

use turbo_vision::app::Application;
use turbo_vision::core::command::{CM_CASCADE, CM_CLOSE, CM_NEXT, CM_QUIT, CM_TILE};
use turbo_vision::core::event::{KB_ALT_X, KB_F6, KB_F10};
use turbo_vision::core::geometry::Rect;
use turbo_vision::core::menu_data::{Menu, MenuItem};
use turbo_vision::views::menu_bar::{MenuBar, SubMenu};
use turbo_vision::views::status_line::{StatusItem, StatusLine};

fn main() -> turbo_vision::core::error::Result<()> {
    let mut app = Application::new()?;
    let (width, height) = app.terminal.size();

    let mut menu_bar = MenuBar::new(Rect::new(0, 0, width, 1));

    let file_menu = SubMenu::new(
        "~F~ile",
        Menu::from_items(vec![
            MenuItem::new("~O~pen capture...", cmd::CM_OPEN_CAPTURE, 0, 0),
            MenuItem::new("~S~ave As...", cmd::CM_SAVE_AS, 0, 0),
            MenuItem::separator(),
            MenuItem::new("E~x~it", CM_QUIT, 0, 0),
        ]),
    );
    let view_menu = SubMenu::new(
        "~V~iew",
        Menu::from_items(vec![
            MenuItem::new("Show ~t~hinking", cmd::CM_SHOW_THINKING, 0, 0),
            MenuItem::new("~M~arkdown", cmd::CM_SHOW_MARKDOWN, 0, 0),
            MenuItem::new("~C~lear window", cmd::CM_CLEAR_WINDOW, 0, 0),
        ]),
    );
    let window_menu = SubMenu::new(
        "~W~indow",
        Menu::from_items(vec![
            MenuItem::new("~N~ext", CM_NEXT, 0, 0),
            MenuItem::new("~T~ile", CM_TILE, 0, 0),
            MenuItem::new("C~a~scade", CM_CASCADE, 0, 0),
            MenuItem::new("~C~lose", CM_CLOSE, 0, 0),
        ]),
    );

    menu_bar.add_submenu(file_menu);
    menu_bar.add_submenu(view_menu);
    menu_bar.add_submenu(window_menu);
    app.set_menu_bar(menu_bar);

    let status_line = StatusLine::new(
        Rect::new(0, height - 1, width, height),
        vec![
            StatusItem::new("~F6~ Next", KB_F6, CM_NEXT),
            StatusItem::new("~F10~ Menu", KB_F10, 0),
            StatusItem::new("~Alt-X~ Exit", KB_ALT_X, CM_QUIT),
        ],
    );
    app.set_status_line(status_line);

    app.run();
    Ok(())
}
