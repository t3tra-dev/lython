# aarch64 ELF emits its LSDA type table with EIGHT-byte entries and the reader
# indexes FOUR, so every raise on that target reaches `ly_eh_lookup_site`'s
# refuse block and aborts inside the unwinder. Loud, not silent -- but a
# program as small as this one cannot run there.
#
# MEASURED, not inferred: `DriverTest.EveryTargetsExceptionTableIsTheOneTheReaderReads`
# builds a target machine per triple and reads the encoding the object-file
# lowering would use. It is 0x9b (indirect|pcrel|sdata4) for x86-64 ELF, both
# Darwin targets and armv7 ELF, and 0x9c (indirect|pcrel|sdata8) for aarch64
# ELF. The test asserts 0x9c for that target so this stays a KNOWN value
# rather than a surprise.
#
# ⛔ WHY IT IS NOT FIXED HERE. The decode differs only in width -- the
# pc-relative add and the indirect load are the same -- but the width has to
# reach `ly_eh_action_walk`, which takes the type-table BASE and not the
# encoding; plumbing it means a sixth parameter, a wider memo slot and a wider
# `out` buffer in `ly_eh_lookup_site`. That is a small change and an untestable
# one from here: no aarch64 Linux runs this suite, and a wrong pointer built
# inside a personality is a crash in the unwinder rather than a failed test.
# Whoever has the machine should take it; the shape of the repair is above.
#
# ⭐ x86-64 ELF got here by a DIFFERENT route and is fixed: its encoding
# follows the RELOCATION MODEL (0x03 `udata4` under Reloc::Static, 0x9b under
# PIC), and the driver now spells PIC rather than defaulting. aarch64 is 0x9c
# under both models, so nothing about the relocation model helps it.
try:
    raise ValueError("x")
except ValueError as e:
    print("caught", e)
