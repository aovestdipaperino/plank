# minisign fixtures

Produced by **minisign 0.12 itself**, not by plank. A signature verifier tested
only against its own signer can share a misreading of the format with it and
pass every test, so these are ground truth from the reference implementation.

The secret keys are deliberately **not** here: nothing in the test suite signs,
and a committed secret key is a thing that eventually gets reused somewhere it
matters. Regenerate all of it with:

```sh
minisign -G -p pub.key -s sec.key -W
printf 'hello plugin artifact\n' > artifact.wasm
minisign -S -s sec.key -m artifact.wasm                              # ED, prehashed (the default)
minisign -S -l -s sec.key -m artifact.wasm -x artifact.legacy.minisig # Ed, legacy
minisign -G -p other.pub -s other.sec -W                             # a second publisher
minisign -S -s other.sec -m artifact.wasm -x artifact.other.minisig
```

`artifact.wasm` is not real WASM — nothing loads it. These tests are about the
signature format, and using a 21-byte file keeps the fixture readable.
