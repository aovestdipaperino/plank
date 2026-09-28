# ChatBGT, a sample plank profile

Run it from the repository root:

    plank --profile examples/profiles/chatbgt

The first launch offers to install it into `~/.plank/profiles/chatbgt/`; after
that the same command, or `plank --profile chatbgt` from anywhere, starts it
directly. Edit the installed copy with `/edit-profile`.

To give it a logo, drop a PNG beside this file and add `"logo": "chatbgt.png"`
to the `profile` block. A missing or undecodable PNG falls back to the plank
logo, so the profile still runs while you iterate on the art.

This is a toy. The real ChatBGT profile lives in the private
`aovestdipaperino/private-profiles` repository under `ChatBGT/`, and it uses
the grid bridge (`docs/WASM-PLUGINS.md`) to open its tables in csvedit.
