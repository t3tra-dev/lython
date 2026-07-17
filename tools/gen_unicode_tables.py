#!/usr/bin/env python3
"""Generate the Unicode Character Database tables shipped with the runtime.

Reads UnicodeData.txt / SpecialCasing.txt / CaseFolding.txt /
DerivedCoreProperties.txt (Unicode 16.0.0, the CPython 3.14 database) and
emits:

  src/lython/runtime/modules/_ucd.mlir   -- two-stage lookup tables plus the
                                            generated accessor functions the
                                            handwritten unicodedata.mlir and
                                            builtins.mlir primitives call
  tests/unit/UnicodeTablesData.inc       -- the same tables as C++ arrays so
                                            the unit tests can validate the
                                            checked-in data without a JIT

Both outputs are checked into the repository: builds must not touch the
network, so this script is run manually when the UCD version changes and the
regenerated files are committed. UCD source files are cached under
tools/.ucd-cache/<version>/ (downloaded on first run, or supply --ucd-dir).

Encoding scheme (mirrors CPython's Tools/unicode/makeunicodedata.py two-stage
compression, re-derived here rather than copied):

  * ctype records: (upper, lower, fold, decimal, digit, flags). Case fields
    hold a signed code-point delta, or -- when the matching EXT_* flag bit is
    set -- (offset << 8) | length into the extended-case code-point pool for
    multi-character full mappings ("ss" -> "SS" etc.).
  * info records: (category, numeric_index). numeric_index points into the
    f64 numeric-value pool, -1 when the character has no numeric value.
  * Each record kind gets its own two-stage index: index1[cp >> 7] names a
    128-entry block in index2, index2 holds record indices.

Flag bits (keep in sync with unicodedata.mlir and UnicodeTablesTests.cpp):
  0 ALPHA (category Lu/Ll/Lt/Lm/Lo)     5 CASED (DerivedCoreProperties)
  1 LOWER (DCP Lowercase)               6 CASE_IGNORABLE (DCP)
  2 UPPER (DCP Uppercase)               7 EXT_UPPER
  3 TITLE (category Lt)                 8 EXT_LOWER
  4 SPACE (Zs or bidi WS/B/S)           9 EXT_FOLD

Category enum order matches CPython's _PyUnicode_CategoryNames so indices
stay recognizable: Cn=0 ... Co=29.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

UCD_VERSION = "16.0.0"
UCD_BASE_URL = f"https://www.unicode.org/Public/{UCD_VERSION}/ucd/"
UCD_FILES = [
    "UnicodeData.txt",
    "SpecialCasing.txt",
    "CaseFolding.txt",
    "DerivedCoreProperties.txt",
]

NUM_CODEPOINTS = 0x110000
SHIFT = 7
BLOCK = 1 << SHIFT

CATEGORIES = [
    "Cn", "Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Me", "Nd",
    "Nl", "No", "Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po", "Sm",
    "Sc", "Sk", "So", "Zs", "Zl", "Zp", "Cc", "Cf", "Cs", "Co",
]
CATEGORY_INDEX = {name: index for index, name in enumerate(CATEGORIES)}

FLAG_ALPHA = 1 << 0
FLAG_LOWER = 1 << 1
FLAG_UPPER = 1 << 2
FLAG_TITLE = 1 << 3
FLAG_SPACE = 1 << 4
FLAG_CASED = 1 << 5
FLAG_CASE_IGNORABLE = 1 << 6
FLAG_EXT_UPPER = 1 << 7
FLAG_EXT_LOWER = 1 << 8
FLAG_EXT_FOLD = 1 << 9

ALPHA_CATEGORIES = {"Lu", "Ll", "Lt", "Lm", "Lo"}
SPACE_BIDI = {"WS", "B", "S"}


def fetch_ucd(ucd_dir: Path) -> dict[str, list[str]]:
    ucd_dir.mkdir(parents=True, exist_ok=True)
    contents: dict[str, list[str]] = {}
    for name in UCD_FILES:
        path = ucd_dir / name
        if not path.exists():
            url = UCD_BASE_URL + name
            print(f"downloading {url}", file=sys.stderr)
            with urllib.request.urlopen(url) as response:
                path.write_bytes(response.read())
        contents[name] = path.read_text(encoding="utf-8").splitlines()
    return contents


def parse_ranges(lines: list[str], wanted: set[str]) -> dict[str, set[int]]:
    """DerivedCoreProperties-style 'XXXX[..YYYY] ; Prop' membership sets."""
    result: dict[str, set[int]] = {name: set() for name in wanted}
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(";")]
        if len(fields) < 2 or fields[1] not in wanted:
            continue
        code_range = fields[0]
        if ".." in code_range:
            lo_text, hi_text = code_range.split("..")
            lo, hi = int(lo_text, 16), int(hi_text, 16)
        else:
            lo = hi = int(code_range, 16)
        result[fields[1]].update(range(lo, hi + 1))
    return result


class CharData:
    __slots__ = (
        "category", "bidi", "decimal", "digit", "numeric",
        "upper", "lower", "fold",
    )

    def __init__(self) -> None:
        self.category = "Cn"
        self.bidi = ""
        self.decimal = -1
        self.digit = -1
        self.numeric: float | None = None
        self.upper: list[int] | None = None  # None = identity
        self.lower: list[int] | None = None
        self.fold: list[int] | None = None


def parse_unicode_data(lines: list[str]) -> list[CharData]:
    table: list[CharData | None] = [None] * NUM_CODEPOINTS
    range_first: tuple[int, list[str]] | None = None
    for line in lines:
        if not line.strip():
            continue
        fields = line.split(";")
        cp = int(fields[0], 16)
        name = fields[1]
        if name.endswith(", First>"):
            range_first = (cp, fields)
            continue
        if name.endswith(", Last>"):
            assert range_first is not None
            start, first_fields = range_first
            for code in range(start, cp + 1):
                table[code] = make_char(first_fields, code, in_range=True)
            range_first = None
            continue
        table[cp] = make_char(fields, cp, in_range=False)
    filled = [entry if entry is not None else CharData() for entry in table]
    return filled


def make_char(fields: list[str], cp: int, in_range: bool) -> CharData:
    entry = CharData()
    entry.category = fields[2]
    entry.bidi = fields[4]
    if fields[6]:
        entry.decimal = int(fields[6])
    if fields[7]:
        entry.digit = int(fields[7])
    if fields[8]:
        text = fields[8]
        if "/" in text:
            numerator, denominator = text.split("/")
            entry.numeric = float(int(numerator)) / float(int(denominator))
        else:
            entry.numeric = float(int(text))
    if not in_range:
        # Simple one-to-one case mappings; ranges carry none.
        if fields[12]:
            entry.upper = [int(fields[12], 16)]
        if fields[13]:
            entry.lower = [int(fields[13], 16)]
    return entry


def apply_special_casing(table: list[CharData], lines: list[str]) -> None:
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(";")]
        if len(fields) < 4:
            continue
        # A fifth non-empty field is a condition list: conditional mappings
        # are either locale-dependent (excluded: casing is locale-independent
        # here, as in CPython's str) or Final_Sigma (handled in code).
        if len(fields) >= 5 and fields[4]:
            continue
        cp = int(fields[0], 16)
        lower = [int(part, 16) for part in fields[1].split()]
        upper = [int(part, 16) for part in fields[3].split()]
        entry = table[cp]
        entry.lower = lower if lower != [cp] else None
        entry.upper = upper if upper != [cp] else None


def apply_case_folding(table: list[CharData], lines: list[str]) -> None:
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(";")]
        if len(fields) < 3 or fields[1] not in ("C", "F"):
            continue
        cp = int(fields[0], 16)
        mapping = [int(part, 16) for part in fields[2].split()]
        table[cp].fold = mapping if mapping != [cp] else None


class ExtendedCasePool:
    def __init__(self) -> None:
        self.values: list[int] = []
        self.index: dict[tuple[int, ...], int] = {}

    def pack(self, mapping: list[int]) -> int:
        key = tuple(mapping)
        offset = self.index.get(key)
        if offset is None:
            offset = len(self.values)
            self.values.extend(mapping)
            self.index[key] = offset
        return (offset << 8) | len(mapping)


def encode_case(
    cp: int, mapping: list[int] | None, pool: ExtendedCasePool, ext_flag: int
) -> tuple[int, int]:
    """Return (field value, flags contribution)."""
    if mapping is None:
        return 0, 0
    if len(mapping) == 1:
        delta = mapping[0] - cp
        if -(1 << 30) < delta < (1 << 30):
            return delta, 0
    return pool.pack(mapping), ext_flag


def two_stage(record_ids: list[int]) -> tuple[list[int], list[int]]:
    index1: list[int] = []
    index2: list[int] = []
    blocks: dict[tuple[int, ...], int] = {}
    for start in range(0, NUM_CODEPOINTS, BLOCK):
        block = tuple(record_ids[start:start + BLOCK])
        block_id = blocks.get(block)
        if block_id is None:
            block_id = len(blocks)
            blocks[block] = block_id
            index2.extend(block)
        index1.append(block_id)
    return index1, index2


def build() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ucd-dir",
        type=Path,
        default=Path(__file__).parent / ".ucd-cache" / UCD_VERSION,
        help="directory holding (or receiving) the UCD source files",
    )
    args = parser.parse_args()
    sources = fetch_ucd(args.ucd_dir)

    table = parse_unicode_data(sources["UnicodeData.txt"])
    apply_special_casing(table, sources["SpecialCasing.txt"])
    apply_case_folding(table, sources["CaseFolding.txt"])
    core = parse_ranges(
        sources["DerivedCoreProperties.txt"],
        {"Lowercase", "Uppercase", "Cased", "Case_Ignorable"},
    )

    pool = ExtendedCasePool()
    ctype_records: list[tuple[int, ...]] = []
    ctype_index: dict[tuple[int, ...], int] = {}
    ctype_ids = [0] * NUM_CODEPOINTS
    info_records: list[tuple[int, int]] = []
    info_index: dict[tuple[int, int], int] = {}
    info_ids = [0] * NUM_CODEPOINTS
    numeric_values: list[float] = []
    numeric_index: dict[float, int] = {}

    for cp, entry in enumerate(table):
        flags = 0
        if entry.category in ALPHA_CATEGORIES:
            flags |= FLAG_ALPHA
        if cp in core["Lowercase"]:
            flags |= FLAG_LOWER
        if cp in core["Uppercase"]:
            flags |= FLAG_UPPER
        if entry.category == "Lt":
            flags |= FLAG_TITLE
        if entry.category == "Zs" or entry.bidi in SPACE_BIDI:
            flags |= FLAG_SPACE
        if cp in core["Cased"]:
            flags |= FLAG_CASED
        if cp in core["Case_Ignorable"]:
            flags |= FLAG_CASE_IGNORABLE

        upper, upper_flag = encode_case(cp, entry.upper, pool, FLAG_EXT_UPPER)
        lower, lower_flag = encode_case(cp, entry.lower, pool, FLAG_EXT_LOWER)
        fold, fold_flag = encode_case(cp, entry.fold, pool, FLAG_EXT_FOLD)
        flags |= upper_flag | lower_flag | fold_flag

        ctype = (upper, lower, fold, entry.decimal, entry.digit, flags)
        record_id = ctype_index.get(ctype)
        if record_id is None:
            record_id = len(ctype_records)
            ctype_records.append(ctype)
            ctype_index[ctype] = record_id
        ctype_ids[cp] = record_id

        if entry.numeric is None:
            numeric_id = -1
        else:
            numeric_id = numeric_index.get(entry.numeric)
            if numeric_id is None:
                numeric_id = len(numeric_values)
                numeric_values.append(entry.numeric)
                numeric_index[entry.numeric] = numeric_id
        info = (CATEGORY_INDEX[entry.category], numeric_id)
        info_id = info_index.get(info)
        if info_id is None:
            info_id = len(info_records)
            info_records.append(info)
            info_index[info] = info_id
        info_ids[cp] = info_id

    # The default record used for out-of-range lookups must be the Cn/no-case
    # record; code point 0x10FFFF is permanently unassigned, so its record is
    # exactly that.
    default_ctype = ctype_ids[0x10FFFF]
    default_info = info_ids[0x10FFFF]
    assert ctype_records[default_ctype] == (0, 0, 0, -1, -1, 0)
    assert info_records[default_info] == (0, -1)

    ctype_index1, ctype_index2 = two_stage(ctype_ids)
    info_index1, info_index2 = two_stage(info_ids)
    assert len(ctype_records) < (1 << 16) and len(info_records) < (1 << 16)
    assert max(ctype_index1) < (1 << 16) and max(info_index1) < (1 << 16)

    return {
        "ctype_records": ctype_records,
        "ctype_index1": ctype_index1,
        "ctype_index2": ctype_index2,
        "info_records": info_records,
        "info_index1": info_index1,
        "info_index2": info_index2,
        "numeric_values": numeric_values,
        "ext_case": pool.values,
        "default_ctype": default_ctype,
        "default_info": default_info,
    }


def format_i64_attr(value: int) -> str:
    return str(value)


def mlir_global(name: str, values: list[int], element: str) -> str:
    body = ", ".join(str(value) for value in values)
    return (
        f"  memref.global \"private\" constant @{name} : "
        f"memref<{len(values)}x{element}> = dense<[{body}]>\n"
    )


def mlir_f64_global(name: str, values: list[float]) -> str:
    body = ", ".join(format_float(value) for value in values)
    return (
        f"  memref.global \"private\" constant @{name} : "
        f"memref<{len(values)}xf64> = dense<[{body}]>\n"
    )


def format_float(value: float) -> str:
    text = repr(value)
    if "e" not in text and "." not in text and "inf" not in text:
        text += ".0"
    return text


def two_stage_accessor(
    name: str,
    index1: str, index1_len: int,
    index2: str, index2_len: int,
    records: str, records_len: int,
    fields: int,
    default_record: int,
) -> str:
    """A generated accessor: cp -> the record's fields, sign-extended."""
    results = ", ".join(["i64"] * fields)
    lines = [
        f"  func.func private @{name}(%cp: i64) -> ({results}) {{\n",
        "    %zero = arith.constant 0 : i64\n",
        "    %limit = arith.constant 1114112 : i64\n",
        "    %in_lo = arith.cmpi sge, %cp, %zero : i64\n",
        "    %in_hi = arith.cmpi slt, %cp, %limit : i64\n",
        "    %in_range = arith.andi %in_lo, %in_hi : i1\n",
        "    %safe_cp = arith.select %in_range, %cp, %zero : i64\n",
        f"    %shift = arith.constant {SHIFT} : i64\n",
        f"    %low_mask = arith.constant {BLOCK - 1} : i64\n",
        f"    %block_size = arith.constant {BLOCK} : i64\n",
        "    %block = arith.shrui %safe_cp, %shift : i64\n",
        "    %block_index = arith.index_cast %block : i64 to index\n",
        f"    %index1_ref = memref.get_global @{index1} : memref<{index1_len}xi16>\n",
        "    %block_id_raw = memref.load %index1_ref[%block_index] : "
        f"memref<{index1_len}xi16>\n",
        "    %block_id = arith.extui %block_id_raw : i16 to i64\n",
        "    %low = arith.andi %safe_cp, %low_mask : i64\n",
        "    %row_base = arith.muli %block_id, %block_size : i64\n",
        "    %row = arith.addi %row_base, %low : i64\n",
        "    %row_index = arith.index_cast %row : i64 to index\n",
        f"    %index2_ref = memref.get_global @{index2} : memref<{index2_len}xi16>\n",
        "    %record_raw = memref.load %index2_ref[%row_index] : "
        f"memref<{index2_len}xi16>\n",
        "    %record_in_range = arith.extui %record_raw : i16 to i64\n",
        f"    %default_record = arith.constant {default_record} : i64\n",
        "    %record = arith.select %in_range, %record_in_range, "
        "%default_record : i64\n",
        f"    %fields = arith.constant {fields} : i64\n",
        "    %base = arith.muli %record, %fields : i64\n",
        f"    %records_ref = memref.get_global @{records} : memref<{records_len}xi32>\n",
    ]
    names = []
    for field in range(fields):
        lines += [
            f"    %offset_{field} = arith.constant {field} : i64\n",
            f"    %slot_{field} = arith.addi %base, %offset_{field} : i64\n",
            f"    %slot_index_{field} = arith.index_cast %slot_{field} : i64 to index\n",
            f"    %raw_{field} = memref.load %records_ref[%slot_index_{field}] : "
            f"memref<{records_len}xi32>\n",
            f"    %value_{field} = arith.extsi %raw_{field} : i32 to i64\n",
        ]
        names.append(f"%value_{field}")
    lines.append(f"    func.return {', '.join(names)} : {results}\n")
    lines.append("  }\n")
    return "".join(lines)


def emit_mlir(data: dict, path: Path) -> None:
    ctype_flat = [value for record in data["ctype_records"] for value in record]
    info_flat = [value for record in data["info_records"] for value in record]
    category_chars = [ord(ch) for name in CATEGORIES for ch in name]
    ext_len = max(len(data["ext_case"]), 1)
    ext_values = data["ext_case"] or [0]
    numeric_values = data["numeric_values"]

    out = [
        "// GENERATED FILE - DO NOT EDIT.\n",
        f"// Unicode Character Database {UCD_VERSION} tables, generated by\n",
        "// tools/gen_unicode_tables.py (see that file for the record and\n",
        "// flag-bit encoding). Regenerate and commit when the UCD version\n",
        "// changes; builds never touch the network.\n",
        "module {\n",
        mlir_global("__ly_ucd_ctype_index1", data["ctype_index1"], "i16"),
        mlir_global("__ly_ucd_ctype_index2", data["ctype_index2"], "i16"),
        mlir_global("__ly_ucd_ctype_records", ctype_flat, "i32"),
        mlir_global("__ly_ucd_info_index1", data["info_index1"], "i16"),
        mlir_global("__ly_ucd_info_index2", data["info_index2"], "i16"),
        mlir_global("__ly_ucd_info_records", info_flat, "i32"),
        mlir_global("__ly_ucd_ext_case", ext_values, "i32"),
        mlir_f64_global("__ly_ucd_numeric_values", numeric_values),
        mlir_global("__ly_ucd_category_names", category_chars, "i8"),
        "\n",
        "  // (upper, lower, fold, decimal, digit, flags) for a code point;\n",
        "  // out-of-range values resolve to the unassigned (Cn) record.\n",
        two_stage_accessor(
            "__ly_ucd_ctype",
            "__ly_ucd_ctype_index1", len(data["ctype_index1"]),
            "__ly_ucd_ctype_index2", len(data["ctype_index2"]),
            "__ly_ucd_ctype_records", len(ctype_flat),
            6, data["default_ctype"],
        ),
        "\n",
        "  // (category enum, numeric-value index or -1) for a code point.\n",
        two_stage_accessor(
            "__ly_ucd_info",
            "__ly_ucd_info_index1", len(data["info_index1"]),
            "__ly_ucd_info_index2", len(data["info_index2"]),
            "__ly_ucd_info_records", len(info_flat),
            2, data["default_info"],
        ),
        "\n",
        "  // Code point %j of an extended-case mapping packed as\n",
        "  // (offset << 8) | length.\n",
        "  func.func private @__ly_ucd_ext_cp(%packed: i64, %j: i64) -> i64 {\n",
        "    %eight = arith.constant 8 : i64\n",
        "    %offset = arith.shrui %packed, %eight : i64\n",
        "    %slot = arith.addi %offset, %j : i64\n",
        "    %slot_index = arith.index_cast %slot : i64 to index\n",
        f"    %pool = memref.get_global @__ly_ucd_ext_case : memref<{ext_len}xi32>\n",
        f"    %raw = memref.load %pool[%slot_index] : memref<{ext_len}xi32>\n",
        "    %value = arith.extui %raw : i32 to i64\n",
        "    func.return %value : i64\n",
        "  }\n",
        "\n",
        "  // ASCII byte %j (0/1) of the two-letter name of category enum\n",
        "  // %cat. An accessor rather than a shared global: cross-module\n",
        "  // memref.get_global does not verify, function declarations do.\n",
        "  func.func private @__ly_ucd_category_char(%cat: i64, %j: i64) -> i64 {\n",
        "    %two = arith.constant 2 : i64\n",
        "    %base = arith.muli %cat, %two : i64\n",
        "    %slot = arith.addi %base, %j : i64\n",
        "    %slot_index = arith.index_cast %slot : i64 to index\n",
        f"    %names = memref.get_global @__ly_ucd_category_names : "
        f"memref<{len(category_chars)}xi8>\n",
        f"    %raw = memref.load %names[%slot_index] : "
        f"memref<{len(category_chars)}xi8>\n",
        "    %value = arith.extui %raw : i8 to i64\n",
        "    func.return %value : i64\n",
        "  }\n",
        "\n",
        "  func.func private @__ly_ucd_numeric_value(%idx: i64) -> f64 {\n",
        "    %idx_index = arith.index_cast %idx : i64 to index\n",
        f"    %pool = memref.get_global @__ly_ucd_numeric_values : "
        f"memref<{len(numeric_values)}xf64>\n",
        f"    %value = memref.load %pool[%idx_index] : "
        f"memref<{len(numeric_values)}xf64>\n",
        "    func.return %value : f64\n",
        "  }\n",
        "}\n",
    ]
    path.write_text("".join(out), encoding="utf-8")
    print(f"wrote {path}", file=sys.stderr)


def cpp_array(name: str, ctype: str, values: list) -> str:
    if not values:
        values = [0]
    body = ", ".join(
        format_float(value) if isinstance(value, float) else str(value)
        for value in values
    )
    return f"inline const {ctype} {name}[] = {{{body}}};\n"


def emit_cpp(data: dict, path: Path) -> None:
    ctype_flat = [value for record in data["ctype_records"] for value in record]
    info_flat = [value for record in data["info_records"] for value in record]
    category_names = ", ".join(f"\"{name}\"" for name in CATEGORIES)
    out = [
        "// GENERATED FILE - DO NOT EDIT.\n",
        f"// Unicode {UCD_VERSION} tables, generated by "
        "tools/gen_unicode_tables.py.\n",
        "// C++ mirror of src/lython/runtime/modules/_ucd.mlir so unit tests\n",
        "// can validate the checked-in data without a JIT.\n",
        "#include <cstdint>\n",
        "\n",
        "namespace lython_ucd {\n",
        "\n",
        f"inline constexpr int kShift = {SHIFT};\n",
        f"inline constexpr int kDefaultCType = {data['default_ctype']};\n",
        f"inline constexpr int kDefaultInfo = {data['default_info']};\n",
        "inline constexpr std::int32_t kFlagAlpha = 1 << 0;\n",
        "inline constexpr std::int32_t kFlagLower = 1 << 1;\n",
        "inline constexpr std::int32_t kFlagUpper = 1 << 2;\n",
        "inline constexpr std::int32_t kFlagTitle = 1 << 3;\n",
        "inline constexpr std::int32_t kFlagSpace = 1 << 4;\n",
        "inline constexpr std::int32_t kFlagCased = 1 << 5;\n",
        "inline constexpr std::int32_t kFlagCaseIgnorable = 1 << 6;\n",
        "inline constexpr std::int32_t kFlagExtUpper = 1 << 7;\n",
        "inline constexpr std::int32_t kFlagExtLower = 1 << 8;\n",
        "inline constexpr std::int32_t kFlagExtFold = 1 << 9;\n",
        f"inline const char *const kCategoryNames[] = {{{category_names}}};\n",
        "\n",
        cpp_array("kCTypeIndex1", "std::uint16_t", data["ctype_index1"]),
        cpp_array("kCTypeIndex2", "std::uint16_t", data["ctype_index2"]),
        cpp_array("kCTypeRecords", "std::int32_t", ctype_flat),
        cpp_array("kInfoIndex1", "std::uint16_t", data["info_index1"]),
        cpp_array("kInfoIndex2", "std::uint16_t", data["info_index2"]),
        cpp_array("kInfoRecords", "std::int32_t", info_flat),
        cpp_array("kExtCase", "std::int32_t", data["ext_case"]),
        cpp_array("kNumericValues", "double", data["numeric_values"]),
        "\n",
        "} // namespace lython_ucd\n",
    ]
    path.write_text("".join(out), encoding="utf-8")
    print(f"wrote {path}", file=sys.stderr)


def main() -> None:
    data = build()
    root = Path(__file__).resolve().parent.parent
    emit_mlir(data, root / "src" / "lython" / "runtime" / "modules" / "_ucd.mlir")
    emit_cpp(data, root / "tests" / "unit" / "UnicodeTablesData.inc")
    ctype_count = len(data["ctype_records"])
    info_count = len(data["info_records"])
    print(
        f"ctype: {ctype_count} records, index2 {len(data['ctype_index2'])}; "
        f"info: {info_count} records, index2 {len(data['info_index2'])}; "
        f"ext pool {len(data['ext_case'])}; "
        f"numeric values {len(data['numeric_values'])}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
