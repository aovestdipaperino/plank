/* load.c - materialize a .ggd into a GGUF from C.
 *
 *   cc -I include examples/load.c -L target/release -lgguf_delta -o load
 *   ./load model-variant.ggd out/model-variant.gguf
 *
 * The base GGUF must sit beside the .ggd under the name recorded in it.
 */
#include <stdio.h>
#include <stdlib.h>
#include "gguf_delta.h"

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <delta.ggd> <out.gguf>\n", argv[0]);
        return 2;
    }
    char err[1024];
    ggd_delta *d = ggd_open(argv[1], err, sizeof err);
    if (!d) {
        fprintf(stderr, "cannot open %s: %s\n", argv[1], err);
        return 1;
    }
    printf("%s: label %s, base %s (%s), %zu chunks\n", argv[1], ggd_label(d),
           ggd_base_name(d), ggd_base_model(d), ggd_chunk_count(d));

    char base[4096];
    if (ggd_find_base(d, base, sizeof base, err, sizeof err) != 0) {
        fprintf(stderr, "%s\n", err);
        ggd_close(d);
        return 1;
    }
    printf("base resolves to %s\n", base);

    if (ggd_materialize(d, argv[2], err, sizeof err) != 0) {
        fprintf(stderr, "materialize failed: %s\n", err);
        ggd_close(d);
        return 1;
    }
    printf("wrote %s (sha256 should be %s)\n", argv[2], ggd_target_sha256(d));
    ggd_close(d);
    return 0;
}
