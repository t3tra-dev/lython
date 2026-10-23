# 例外処理の実装とその費用

Python の例外がどう投げられ、どう捕まえられるか。何に費用がかかっているか。
これからどうするか。

## 1. いまの仕組み

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

## 5. 残り: 関数を跨ぐ raise (P2)

- **Itanium のままでは下げきれない。** 2 フレームを跨ぐ raise は CPython の約 2 倍のまま
  である。1 段でも、unwind の開始と各 cleanup からの再開のたびに、共有キャッシュの
  照会が入る。
- **下げるなら、Lython の関数どうしの伝播を戻り値に変える** (失敗フラグを分岐で確かめる)。
  - unwinder を残すのは C のフレームを跨ぐ境界だけ: ctypes コールバック、シグナル
    ハンドラ。
  - 代償として、例外の起きない経路に、投げうる呼び出しごとに 1 分岐が増える。
    `ly.eh.cannot_raise` の推論で減らせる。
  - 切り替える前に、コーパスとベンチマークで測る。
- **デバッガとの連携は伝播の方式によらず保てる。**
  - unwind 表、フレームポインタ、行テーブルはそのまま出す (lldb のバックトレースと
    プロファイラ用)。
  - raise を 1 つの `noinline` の入口に集める。
  - lldb のコマンドを用意する: `ly-exc` (現在の例外と Python の traceback)、
    `ly-bt` (Python 単位のバックトレース)。CPython の `Tools/gdb/libpython.py` と同じ形。
