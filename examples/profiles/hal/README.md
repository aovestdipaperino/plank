# HAL, a plank mail and calendar profile

Run it:

    plank --plugin-dir examples/profiles/hal --profile hal

Install it:

    /install-profile ./examples/profiles/hal

That copies it to `~/.plank/profiles/hal/`, after which `plank --profile hal`
finds it anywhere. Remove it with `rm -rf ~/.plank/profiles/hal`.

## The mail server is not built yet

`.mcp.json` names `plank-mail-mcp`, an MCP server that does not exist yet: an
IMAP and CalDAV client authenticating by XOAUTH2 against Google and Microsoft,
with refresh tokens in the macOS Keychain. Until it is built, plank starts HAL
and reports one MCP server that failed to start. Everything else about the
profile — its prompt, its identity, its tool containment — works.

The entry is here rather than absent because it is the contract that server has
to satisfy: the command name, the config path, and the fact that HAL expects
mail and calendar tools to arrive over MCP.

Password and app-password authentication are deliberately out of scope, so
Fastmail, iCloud and self-hosted Dovecot are not supported. Note also that
Microsoft has been retiring IMAP OAuth for many tenants in favour of Graph;
confirm your tenant still permits IMAP before relying on this.

## What HAL can and cannot do

The `tools.builtin` allow-list gives HAL five builtins — `read`, `more`,
`glob`, `search` and `ask` — and withholds every other one, `bash`, `edit` and
`write` included. It reads files it is pointed at and cannot run a shell or
change your working tree.

The allow-list governs **builtins only**. Every tool the mail MCP server
exposes is available to HAL regardless of what is listed here. If that server
can send mail, so can HAL when asked.
