/* gguf_delta.h - C ABI for .ggd weight deltas between same-layout GGUF files.
 *
 * A delta records the byte spans where a target GGUF differs from a base GGUF
 * of identical layout. `ggd_materialize` turns base + delta back into a full
 * GGUF you can hand to any loader. The base is looked up beside the delta by
 * the filename recorded in it; the originals are never written.
 *
 * Every function that can fail takes an error buffer: on failure it returns
 * NULL or a non-zero code and writes a NUL-terminated message into `err`
 * (truncated to `err_len`). Pass NULL/0 to skip the message.
 *
 * Strings returned by the ggd_*() accessors are owned by the handle and stay
 * valid until ggd_close(). Paths are plain NUL-terminated byte strings.
 */
#ifndef GGUF_DELTA_H
#define GGUF_DELTA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* An opened delta: header and chunk table, parsed once. */
typedef struct ggd_delta ggd_delta;

/* What ggd_create found. All counts are in bytes except the first two. */
typedef struct ggd_report {
    uint64_t tensors_changed;
    uint64_t chunks;
    uint64_t bytes_spanned;   /* bytes covered by chunks */
    uint64_t bytes_changed;   /* bytes that actually differ */
    uint64_t payload_bytes;   /* compressed bytes written */
} ggd_report;

/* Version of this library as "major.minor.patch". Static storage. */
const char *ggd_version(void);

/* Open a .ggd. Returns NULL on error. */
ggd_delta *ggd_open(const char *path, char *err, size_t err_len);
void ggd_close(ggd_delta *d);

/* Header fields. */
const char *ggd_label(const ggd_delta *d);
const char *ggd_base_name(const ggd_delta *d);          /* filename of the base */
const char *ggd_base_model(const ggd_delta *d);         /* general.name of the base */
const char *ggd_base_source_url(const ggd_delta *d);    /* general.source.url, or "" */
const char *ggd_base_source_revision(const ggd_delta *d);
const char *ggd_target_name(const ggd_delta *d);
const char *ggd_target_sha256(const ggd_delta *d);      /* 64 hex chars */
uint64_t ggd_base_size(const ggd_delta *d);             /* bytes; the target is the same size */

/* Chunk table. ggd_chunk fills the out-params for chunk `i` and returns 0,
 * or -1 when `i` is out of range. Any out-param may be NULL. */
size_t ggd_chunk_count(const ggd_delta *d);
int ggd_chunk(const ggd_delta *d, size_t i,
              uint64_t *offset, uint64_t *len, uint64_t *payload_len);

/* Locate the base beside the delta (<delta dir>/<base name>) and verify its
 * size and header hash. Writes the path into `out` (NUL-terminated,
 * truncated to out_len). Returns 0, or -1 with a message listing what was
 * tried. */
int ggd_find_base(const ggd_delta *d, char *out, size_t out_len,
                  char *err, size_t err_len);

/* Whether `base_path` has the size and header the delta was cut against.
 * 0 when it does, -1 with a short reason otherwise. */
int ggd_check_base(const ggd_delta *d, const char *base_path,
                   char *err, size_t err_len);

/* The whole load: find the base beside the delta, clone it (APFS clonefile
 * on macOS, reflink or copy elsewhere) to `out_path`, apply the delta. On
 * success `out_path` is a complete GGUF identical to the original target.
 * Nothing partial is left behind on failure. Returns 0 or -1. */
int ggd_materialize(const ggd_delta *d, const char *out_path,
                    char *err, size_t err_len);

/* Apply the delta in place onto `clone_path`, which must already hold a byte
 * copy of the base. Each chunk's base bytes are verified before they are
 * replaced. Returns 0 or -1. */
int ggd_apply(const ggd_delta *d, const char *clone_path,
              char *err, size_t err_len);

/* Create `out_path` from two same-layout GGUFs. `label` may be NULL to derive
 * one from the filenames; `hash_base` non-zero also records the base's
 * whole-file SHA-256 (slow). `report` may be NULL. Returns 0 or -1. */
int ggd_create(const char *base_path, const char *target_path,
               const char *out_path, const char *label, int hash_base,
               ggd_report *report, char *err, size_t err_len);

#ifdef __cplusplus
}
#endif

#endif /* GGUF_DELTA_H */
