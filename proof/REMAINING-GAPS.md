# 残りのギャップ

`proof/` 49 モジュール 10400 行、`--safe`、postulate / TERMINATING / hole ゼロ、`make` / `make redcheck` 通過時点。

`GAP-ANALYSIS.md` は **モデルと C++ 実装の差** を扱う。この文書は **モデル内部で何が未証明か** を扱う。

---

## 0. 棚卸しの方法と、この会話で 3 回起きた計測失敗

型検査器は「書いた定理が正しい」ことしか言わない。**「その定理に中身があるか」は言わない。** それを見る道具は 2 つある。

1. **inhabitation census** — 各述語が trace で一度でも構築されたか。ゼロなら、その述語についての全定理は空虚かもしれない。
2. **変異テスト** — 前提や補題を壊したとき、実際に赤くなるか。

どちらも計測器であり、**この会話で 3 回壊れた**。3 回とも「壊れた計測器が合格を報告する」形だった:

| # | 何が壊れたか | 誤った出力 |
|---|---|---|
| 1 | glob `src/*/*Trace.agda` が外れた | 全述語 0 = 「全部空虚」 |
| 2 | `grep ": *$r\b"` の `\b` が BSD grep で無効 | 同上 |
| 3 | 変異 sed が当たらない / 変異先ファイルだけを検査 | 「ACCEPTED = 前提が不要」 |

3 番は特に危険で、「前提を消しても通った」は**前提が不要という結論と区別がつかない**。以後の規律:

- **変異が実際にファイルを変えたか diff で確認する。**
- **波及先を含めて `make` 全体を回す** (変異先ファイル単体は通ってしまう)。
- **センサスは既知の陽性で反応を確認してから読む。**

---

## 現在の census

未構築の述語は **ゼロ**。命令ステップ 5 規則すべて、並行ステップ、`WFRC` (6+)、`NameSiteCoherent` (7)、`Conflict` / `Race` / `Earlier` / `HappensBefore`、`Invalidity` 4 構成子すべて、`Leaked`、`Orphaned`、`Aggregate`、`ErasedHasNoRuntimeOp`、`ReachedZero` が実体化されている。

---

## 閉じたもの

### 1. 保存 — `Proof.Program.Preservation` ★

**`reachable-preserves-WF : f ⊢ s —→* u → WF s → WF u`。** 命令 5 規則 + 終端子 5 規則 × 5 フィールド、全オブジェクトについて。開発中で最大の未証明項目だった。

証明に至るまでに 2 つ変更した。**どちらも不変条件を弱めるのではなく IR を直す方向**で:

- **`WFRC.no-stale-owner` を `strongAt` から `Holds` (メンバシップ) に変えた。** `strongAt` は site の**最初の**エントリしか見ない。`vacate` はそれを消して次を露出させるので、最初のエントリについての性質は露出した側について何も言わない。`drop` では保存不能だった。変異テストで確認済み: `strongAt` に戻すと落ちる。
- **5 規則に前提を足した。** `dup` に `life c ≡ live` (死んだオブジェクトの dup は復活)、`new` にヒープ 3 条件と ghost 側の freshness (存在しない記憶域への参照を生む)、`move`/`drop` に SSA 条件 (`bindVar` はシャドウし `unbindVar` は最初の 1 つしか消さないので、シャドウされた束縛が site を失った参照を所有し続ける — over-release の形そのもの)。各前提の必要性は変異テストで個別に確認済み。

不変条件は `WFRC` だけでは足りず、**`backed`** (所有名は自分の site を占有し、自分の entity を保持している) を加えた `WF` になった。`move` は環境の事実 (`lookupVar es src ≡ …`) を知って機械を操作 (`vacate (siteOf src)`) するので、両者を繋ぐものが要る。

前提を足して規則を充足不能にする逃げ道は `Proof.Program.Run` が塞ぐ — 5 規則すべてを具体ブロック上で導出しており、`WF s₀` から `reachable-preserves-WF` で `WF s₆` を得ている。**定理は使われている。**

### 1b. ⭐ ブロック引数は MOVE である

**残っていた唯一の設計判断で、下した。** `moveArgs` は被演算子の名前を解除し、その owner site をパラメータへ移す。カウンタは動かない — ブロック引数のコストはゼロである。

dup にしなかった理由: ループ搬送値が毎周 retain/release の対を払うことになるが、その参照はどこへも行っていない。move は無料である。

結果として `Proof.Program.Trace` の結論が反転した。以前は「分岐後 `x` と `p` が 1 オブジェクトの 2 所有名で、count が食い違う」を示していた。いまは:

- `x-is-gone : entityOf envAfterBr x ≡ nothing` — 被演算子の名前は消える
- `no-longer-aliased : ¬ Aliases envAfterBr x p` — 反証できる
- `counts-agree` — 所有名 1、owner site 1
- `counter-untouched` — カウンタは不変

**出荷済み SIGSEGV が turn する状態は到達不能になった。** 変異テストで確認済み: 被演算子を残す (dup に戻す) と保存定理が落ち、site を relocate せず追加だけにしても落ちる。

そして保存が終端子まで通ったので、`—→*` 上の定理が初めて言えるようになった。

### 2. `siteOf` のスレッド添字化

`PState` が `onThread` を持ち、`siteOf : ThreadId → Var → OwnerSite`。逐次規則が第 2 スレッドの owner site を名指せるようになったので、並行層が site map を触る命令をスケジュールできる。`Concurrent.Trace` は `borrow` から実際の `dup` に書き直した。

### 3. イベントが命令から導出される

`instrEvent : ThreadId → Policy → Instr → Env → Maybe Event`。`sched-step` の自由変数 `e` は消えた。`move` と `borrow` は `nothing` を返す — 「移動は無料」が counter だけでなく履歴上でも真になる。`rcFootprint` は `Concurrent.Event` に移し、二重定義を防いだ。

### 4. checker 4/4、`Valid` 決定 — `Proof.Lython.Decide`

| 不正 | checker | 健全 | 完全 | 沈黙⇒安全 |
|---|---|---|---|---|
| dangling borrow | `danglingAnchor` | ✅ | ✅ | ✅ |
| premature reclaim | `prematureReclaim?` | ✅ | ✅ | ✅ |
| leak | `leaked?` | ✅ | ✅ | ✅ |
| refcount race | `needsAtomic?` | ✅ | ✅ | ✅ |

`silence-means-valid : AllChecksSilent es m → Valid es m` — しかも **有限チェック**である。義務は `Var` / `ObjId` 全体ではなく、状態自身が提供する 3 つのリスト (`names es`、`objectsNamed es`、`objectsOwned (sites m)`) 上に量化されている。各 membership 補題が「不正はそのリストの要素についてしか起こり得ない」を言うので、リストを見ることと全部を見ることが等価になる。

`Proof.Program.Coherence.state-is-valid` が到達可能な状態 `s₂` で実際に `Valid` を確立している — **走らせて示すのであって、証明して示すのではない**。1 ステップ後の `s₆` は `¬ Valid` であり、制約が実効的であることも示してある。

`needsAtomic?` は escape analysis を必要としない — owner site を最初からスレッド添字付きにした設計上の見返りで、ここで回収した。

### 5. リーク方向 — `Proof.Program.Coherence`

`Leaked es m o` = site は保持しているが所有名がない。`coherent-has-no-leaks : NameSiteCoherent es m → ∀ o → ¬ Leaked es m o`。全オブジェクトに量化してあるのが要点。

具体例は `Run.s₅` の機械 × `Run.s₆` の環境 — 名前がスコープを抜けたのに site が残った状態、すなわち `drop` を出し忘れたコンパイラが作る状態。ステップ関係では到達不能 (`step-drop` は名前と site を同時に消す) であり、**それが正しい**: リークはモデルの欠陥ではなくコンパイラの欠陥である。

### 6. race freedom — `Proof.Concurrent.RaceFree` ★

**権限代数なしで閉じた。** この IR が出せるメモリトラフィックは 3 種しかなく、2 種は競合し得ない:

- `allocate` — アクセスモードがないので `Conflict.is-access` が落ちる
- `move` / `borrow` — イベントを出さない
- refcount rmw — 唯一競合しうる種類で、常にオブジェクト自身の割り当ての word 0

したがって race freedom は「refcount 更新は必要な場所で atomic か」1 問に帰着し、それは `needsAtomic?` が決定する。

- `emission-is-not-a-race : FollowsTheChecker m pol → ¬ RefcountRace m o (rcEvent t pol o)`
- `two-emissions-do-not-conflict` — `Conflict` そのものを反証（より強い）
- `allocate-never-conflicts` — allocate は競合の片側になれない
- `immortals-race-free` — **ポリシーによらず**。`{0,1,2}` に atomic は不要であることの証明

### 7. 集約と多重度 — `Proof.RC.Aggregate`

`Aggregate ss p ks c` (フィールド path)、`releaseFields`、そして多重度: 1 つの子が親の 2 フィールドに入っていれば count は 2 で、1 フィールドの解放は 1 だけ下げる。**owner を集合でなくリストで数えていたことがここで効く** — 集合なら 3 フィールドの子が 1 に潰れ、1 回の解放が 3 回分に見える。

`Orphaned ss p c` = 親の count が 0 なのに子が親のフィールドから保持されている状態。未帰属リーク 2 系統の形で、**名前ではなく個数**で表現されている。

### 8. `WellShaped` — `Proof.Object.Shaped`

`ShapedBox` が記述子と証人を束ね、`allocShaped` が証人を `refl` で作る。**誤ったペアリングは型で不可能**になった。

**実行時消去はできない**、努力不足ではなく: Agda の `@0` は消去された等式で関連値を輸送できない (`/tmp` で確認済み)。`wordIx` の結果型 `Ix (sizes b)` は `spans-box` を必要とする。真の消去には添字型が証明に依存しなくなる再設計が要る。

### 9. QTT 層 — `Proof.QTT.Trace`

`ErasedHasNoRuntimeOp` + 5 反証、「量は所有を決めない」の定理化。

### 10. ⭐ 権限代数なしで済んだ理由 — `Proof.Concurrent.Event`

**この IR のあらゆるアクセスは「1 オブジェクトの 1 ワード」である。** one-lane レイアウトのおかげで指せる先が他に無い。したがって 2 つのアクセスが重なるのは同じオブジェクトの同じワードのときだけで、代数は `aligned-blocks-disjoint` という算術補題 1 本に潰れる。

- `different-words-do-not-conflict` — 別ワードは競合しない
- `fields-never-touch-the-refcount` — `HeaderWords` が正なのでフィールドは word 0 に来ない
- `different-fields-do-not-conflict` — 別フィールドは競合しない
- `different-objects-do-not-conflict` — **identity が provenance であることの見返り**。アドレスなら 2 オブジェクトが同じ値を持ちうるのでこの定理は偽になる

### 11. race freedom を履歴全体へ — `Proof.Concurrent.RaceFree`

`Emitted` が「この IR が出しうるイベントは 3 種 (inert / refcount / payload) だけ」を言い、`instrEvent-shape` がそれを証明する。`FromProgram` の provenance を `⇒*` 上の帰納で保存し、`history-is-race-free` が結論する。

**`Policy = ObjId → Maybe Atomicity`** に変えた。`nothing` は「メモリ操作を出さない」で、immortal の正しい答えである (`bumpUp immortal ≡ immortal` なので書くものが無い)。`Concurrent.Trace` に 3 ポリシーを並べてある: naive は義務を果たさず (`naive-does-not-follow`)、atomic と eliding は果たす。

### 12. `aggregate_release` が操作になった

`ObjCell` にクラスのアリティを持たせた。`fieldIds` が列挙し、`aggregateRelease` は**誰からもリストを受け取らない**。

### 13. `Valid` が有限チェックになった

義務は状態自身が提供する 3 リスト上に量化され、`Proof.Program.Coherence.state-is-valid` が到達可能な状態 `s₂` で実際に確立する。**走らせて示すのであって証明して示すのではない。**

---

## 残っているもの

### A′. `PayloadSeparated` は仮説であり、証明ではない

**これは意図的で、コンパイラの義務ではない。** 1 オブジェクトの同じフィールドに 2 スレッドが書くのは、GIL なしの CPython と同じくソースプログラムのデータ競合であり、どんな lowering も防げない。lowering にできるのは**自分でトラフィックを増やさないこと**で、それが `FollowsTheChecker` — こちらは `sharedPair` により site map から決定される。

権限代数を導入してもこの線は動かない。動くのは「どのバイト分割が両立するか」の答え方だけで、この IR では答えが「同じオブジェクトの同じワード」に潰れている。

### C′. `FollowsTheChecker` / `AllShared` は全 `ObjId` に量化されている

不正性 4 種の検査は有限化済み。残るのは発行側の義務で、これは「これから書く命令」についてであって状態についてではないため、対応する列挙が状態側に無い。コンパイラは発行地点ごとに分割して果たすので、同じ義務を分けているだけである。

### F. 精製 (refinement) — Agda 内では閉じられない

**`proof/` は `src/lython/` を一切拘束していない。** これは proof/ 内の作業では閉じられず、コンパイラ側の変更が要る。移せる形になっているものは以下:

| モデル側 | コンパイラ側に要る表現 |
|---|---|
| `danglingAnchor` | `Mode.borrowed` がアンカー名を持つこと |
| `prematureReclaim?` | 解放時点で env を参照できること |
| `leaked?` | 所有名数と site 数を別々に数えること |
| `needsAtomic?` / `sharedPair` | owner site がスレッド添字付きであること |
| `moveArgs` | ブロック引数が被演算子の名前を消すこと |
| `aggregateRelease` | オブジェクトがクラスのアリティを持つこと |

6 つとも健全かつ完全 (あるいは操作として導出済み) で、依存しているのは**表現上の決定 1 つずつ**である。最も安いのは依然 `danglingAnchor`。

### `Aggregate` は「未設定」と「不在」を区別しない

`Aggregate` は site map 上の path なので、クラスが宣言していてもフィールドが未設定なら `field′` エントリが無く、path も通らない。解放については正しい (解放するものが無い) が、到達可能性解析には足りない。区別にはセル内にフィールドスロットそのものが要る。

### `WellShaped` の実行時消去

Agda の `@0` が消去された等式で関連値を輸送することを禁じており、`wordIx` の結果型 `Ix (sizes b)` が `spans-box` を必要とする。真の消去には添字型が証明に依存しなくなる `Proof.Object.Box` の再設計が要る。誤ペアリング防止 (安全性の内容) は `Proof.Object.Shaped` で達成済み。

---

## 優先順位

| # | ギャップ | 性質 | 規模 |
|---|---|---|---|
| F | 精製 | **proof/ の外** | 大 |
| C′ | 発行側義務の有限化 | 表現の変更 | 中 |
| — | フィールドスロット | 表現の変更 | 中 |
| — | `WellShaped` の消去 | `Box` の再設計 | 中 |

**A (権限代数) は閉じた。** 必要だったのは代数ではなく、ワード整列ブロックの非重複補題 1 本だった。

**「既に述べたことが述べたほど強くない」種類のギャップは残っていない。**
