# 例外処理の実装とその費用

Python の例外がどう投げられ、どう捕まえられるか。何に費用がかかっているか。
これからどうするか。

## 1. unwinder の方式 (2026-10-24 の P2 まで。いまは境界と `LYTHON_ABLATE_RETURN_PROPAGATION` だけ。§5)

- **伝播は Itanium の unwinder だけを借りる。** 例外オブジェクトはランタイムの
  `g_current_parts` に置き、`_Unwind_RaiseException`/`_Unwind_ForcedUnwind` に
  自前の carrier を渡す。各フレームは自前の personality (`LyEH_Personality`) が
  LSDA を読んで捌く。
- **C++ ランタイムは使わない。** `__cxa_throw` も RTTI も通らないので、lldb の
  `break set -E c++` は Python の raise では止まらない。止めるなら
  `b LyEH_ThrowException`。
- **traceback は伝播の途中で積む。** 例外が通過するフレームの landing pad が、
  通常経路ではなく例外経路でだけ積む。CPython の `PyTraceBack_Here` と同じ形である。
- **同じ関数で捕まる raise は分岐にする。** unwinder を通らない
  (`branchLocalRaisesToTheirHandler`、リンク後の LLVM IR)。
  - raise: `LyEH_RecordException` で例外を記録し、ハンドラの dispatch へ分岐する。
  - 再送出 (`finally` の終わり、`__exit__` が握りつぶさなかった `with`、どの節にも
    合わなかった `except`): `LyEH_RecordRethrow` で native catch を閉じ、分岐する。

## 2. CPython との比較

| | CPython | Itanium (C++) |
|---|---|---|
| 伝播 | 戻り値 (NULL/-1) を呼び出しごとに確かめる | unwinder がスタックを歩く (探索と後始末の 2 段) |
| 例外がないときの費用 | 呼び出しごとに 1 分岐 | ほぼ 0 |
| 例外の費用 | 例外の確保、フレームごとの表引き、traceback | フレームごとに像の特定、CFI 解釈、personality、レジスタ復元 |
| C のフレームを跨ぐか | 跨がない (境界で unraisable) | unwind 表があれば跨ぐ |

## 3. 測定 (2026-10-24、macOS 26、Apple M 系)

- **費用の大半は Lython の外にある。** 呼んだ関数からの raise を 5M 回。

  | どこ | 割合 |
  |---|---|
  | dyld (アドレスがどの像に属するかの照会) | 83% |
  | libunwind | 13% |
  | Lython のコード | 3.5% |

- **照会 1 回の費用。**

  | アドレスの場所 | 1 回 |
  |---|---|
  | 実行ファイル | 108 ns |
  | 共有キャッシュの中 (libunwind 自身) | 340 ns |

- ⛔ **unwinder を差し替えても速くならない。** LLVM の libunwind を静的にリンクすると
  9 倍遅くなった。Homebrew 版には compact unwind の速い経路がない。

## 4. 変えたこと (2026-10-24)

- **P0: Darwin では 1 段で unwind する** (`unwindInOnePhase`、`_Unwind_ForcedUnwind`)。
  - 探索段は C++ のためにある。何も捕まえない throw を、スタックを壊さず
    terminate させるためである。Python のプログラムには入口に必ず handler がある。
  - だから探索段は、全フレームをもう一度歩いて dyld に照会し直すだけになっていた。
  - personality は、探索段がしていた判定 (例外のクラスが action chain に合うか) を
    後始末段で自分で行う。
  - 結果: 関数を跨ぐ raise が 35〜40% 速くなった。
  - ⛔ ELF では有効にしていない。`LyRunPythonMain` の C++ personality が forced
    unwind を libstdc++ の規則で読むが、それをここで確かめる手段がなかった
    (podman の VM が使えない)。
- **P1: 同じ関数で捕まる再送出も分岐にする** (`LyEH_RecordRethrow`)。
  `finally`、`with`、合わなかった内側の `except` を通る raise が 22〜32 倍速くなった。

| 例外の形 | CPython | 前 | 後 |
|---|---|---|---|
| 同じ関数で raise → except | 0.225 s | 0.019 s | 0.017 s |
| `finally` を通って同じ関数で捕まる | 0.275 s | 0.938 s | 0.039 s |
| `with` を通る | 0.370 s | 0.951 s | 0.043 s |
| 内側の合わない `except` を通る | 0.250 s | 0.924 s | 0.029 s |
| 呼んだ関数から raise | 0.154 s | 0.640 s | 0.280 s |

## 5. P2: 例外を戻り値で伝える (2026-10-24、全ターゲット)

### 仕組み

`installPythonErrorReturns` (Cleanup/EH.cpp) が、例外の辺を invoke ではなく分岐で作る。

- **raise。** プリミティブは戻り値版 (`LyEH_ThrowException$ret` など、
  `kReturningRaiseSuffix`) に置き換える。戻り値版は例外を記録し、`LyEH_Propagating`
  を立てて戻る。
- **呼び出しの後。** 伝播中の例外を抱えて戻りうる呼び出しのすぐ後で、フラグを確かめる。
  - try の marker が付いた呼び出し: その try のハンドラへ分岐する。フラグを下ろし、
    このフレームを traceback に積む。catch pad がしていたことと同じである。
  - それ以外: 関数から戻る。cleanup pad が積んでいたフレームは積んでから戻る。
- **try の構造は invoke を作っていたものをそのまま使う。** 例外の辺での所有権の解放、
  ハンドラ、dispatch の連鎖、traceback は同じコードで、分岐で辿り着くだけである。
- **境界は unwinder のまま。**
  - `LyRunPythonMain` は `__main__` の後でフラグを見る。立っていれば
    `LyEH_RethrowCurrent` で投げ直し、既存の landing pad が報告する。
  - JS の export (`LyJs_Dispatch`/`LyJs_Release`) は `LyEH_RaisePending` で同じことをする。
- **安全網。** リンク後に `verifyPropagationChecked` が確かめる。フラグを抱えて戻りうる
  関数を呼んだ後は、すぐにフラグを確かめるか、そのまま戻るかのどちらかでなければならない。
  そうでない呼び出しがあればコンパイルを止める。
- **追わなくてよいもの:**
  - 解放 (`__del__` の例外はその場で "Exception ignored" になる)。
  - 例外を記録するだけの関数。
  - ポインタ越しの呼び出し (Python に届くのは slot hook からだけ)。
- `LYTHON_ABLATE_RETURN_PROPAGATION=1` で invoke の方式に戻せる (比較用)。

### 測定 (macOS arm64、AOT、同じバイナリで比較)

| | invoke | 戻り値 | CPython |
|---|---|---|---|
| 呼んだ関数から raise | 0.274 s | **0.017 s** | 0.151 s |
| ベンチマーク exceptions | 0.273 s | **0.027 s** | 0.145 s |

- **例外の起きない経路は、10 本の長いベンチマークの幾何平均で 2% 遅い。**
  - 大半は ±1% に収まる。
  - nbody は 17% 遅い。小さなランタイム関数 1 つの inline cost が 220 から 230 になり、
    閾値 225 を跨いだためである (`-Rpass-missed=inline`)。
  - generators は 4〜5% 遅い。
  - ⛔ inline の閾値を上げたり `inlinehint` を付けたりはしない。nbody が戻る代わりに
    sort が 19% 遅くなった。崖が動くだけである。
- **実行ファイルは 1〜9% 小さい** (landing pad と LSDA がない)。

### ターゲットごとの確認

| ターゲット | 確認したこと | 結果 |
|---|---|---|
| macOS arm64 | 両スイート、リーク検査 | 違いなし |
| macOS x86_64 (Rosetta) | 例外のベンチマーク | 同じ出力 |
| Linux aarch64 / x86_64 (podman) | 全 golden を invoke と戻り値の両方で | 両方式で同じ |
| Windows (Wine) | 例外関係の golden | invoke は 3 件がハング。戻り値は全件通る |
| WASI (wasmtime) | 例外のベンチマーク | 同じ出力、2〜3 倍速い |

### デバッガ

- unwind 表、フレームポインタ、行テーブルは変わらない。
- raise を止めるなら `b LyEH_ThrowException$ret` (戻り値の方式)。
- 未着手: lldb の `ly-exc` / `ly-bt` コマンド。
