# 残りのギャップ

`proof/` 51 モジュール 11388 行、`--safe`、postulate / TERMINATING / hole ゼロ、`make` / `make redcheck` 通過時点。

`GAP-ANALYSIS.md` は **モデルと C++ 実装の差** を扱う。この文書は **モデル内部で何が未証明か** を扱う。

---

## 0. 棚卸しの方法と、この会話で 5 回起きた計測失敗

型検査器は「書いた定理が正しい」ことしか言わない。**「その定理に中身があるか」は言わない。** それを見る道具は 2 つある。

1. **inhabitation census** — 各述語が trace で一度でも構築されたか。ゼロなら、その述語についての全定理は空虚かもしれない。
2. **変異テスト** — 前提や補題を壊したとき、実際に赤くなるか。

どちらも計測器であり、**この会話で 5 回壊れた**。最初の 3 回は「壊れた計測器が合格を報告する」形、4 回目は逆向き、5 回目はまた「0 を報告する」形だった:

| # | 何が壊れたか | 誤った出力 |
|---|---|---|
| 1 | glob `src/*/*Trace.agda` が外れた | 全述語 0 = 「全部空虚」 |
| 2 | `grep ": *$r\b"` の `\b` が BSD grep で無効 | 同上 |
| 3 | 変異 sed が当たらない / 変異先ファイルだけを検査 | 「ACCEPTED = 前提が不要」 |
| 4 | 前提を**削除**する変異が arity を変え、パターンマッチ側が壊れた | 「REJECTED = 前提が必要」 |
| 5 | フェーズ別 IR 走査で `.strip() == "}"` が**ネストした region の閉じ括弧**に一致し、全 body が最初の内側 region で切れた | 11 フェーズで `FromSlot = 0` |

3 番は特に危険で、「前提を消しても通った」は**前提が不要という結論と区別がつかない**。

4 番は 1〜3 と逆向きの故障で、同じくらい重い。前提を消すとコンストラクタの arity が変わり、**それを掴んでいる全パターンマッチが壊れる**。`make` は落ちるが、落ちた理由は「証明がその前提を使っている」ではなく「行数が合わない」である。**dead weight が load-bearing に見える。**

以後の規律:

- **変異が実際にファイルを変えたか diff で確認する。**
- **波及先を含めて `make` 全体を回す** (変異先ファイル単体は通ってしまう)。
- **センサスは既知の陽性で反応を確認してから読む。**
- **前提の変異は arity を保つ。** 削除ではなく `o ≡ o` のような空虚な命題へ**弱める**。規則側と、それを受ける証明モジュールの telescope の**両方**を同時に弱める (片方だけでは呼び出し側の型が合わずに落ち、また偽の REJECTED になる)。
- **両方向の対照を取る。** REJECTED しか出せない計測器は何も測っていない。実際にこの規律で `step-init` の `fresh` が ACCEPTED (= dead weight) として出た — 削除変異では REJECTED と誤報していたものである。
- **走査器に「手で読んだ事実」への assertion を埋め込む。** 5 番はこれで捕まった: 最終フェーズの IR は手で読んで `FromSlot` がちょうど 1 個あると分かっていたので、走査器がそれを満たさなければ表を出さずに `REFUSE` する。表を眺めて気づくのを期待するのではなく、**計測器自身に落ちてもらう。**

---

## 現在の census

未構築の述語は **ゼロ**。命令ステップ 5 規則すべて、並行ステップ、`WFRC` (6+)、`NameSiteCoherent` (7 + 保存定理)、`Conflict` / `Race` / `Earlier` / `HappensBefore`、`Invalidity` 4 構成子すべて、`Leaked`、`Orphaned`、`Aggregate`、`ErasedHasNoRuntimeOp`、`ReachedZero` が実体化されている。

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

具体例は `Run.s₅` の機械 × `Run.s₆` の環境 — 名前がスコープを抜けたのに site が残った状態、すなわち `drop` を出し忘れたコンパイラが作る状態。

**「到達不能」は #16 まで主張であってコメントに書いてあるだけだった。**

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

### 14. ⭐ 初期化ウィンドウ — `alloc` と `init` の分離

`new` が確保と初期化を 1 命令でやっていたため、**その間の状態が表現できなかった**。そこは実際に出荷済み欠陥のあった場所である: `boxRuntimeObject` は `memref.alloc` の**後**に word 0/1 を書くので、ハンドル定義位置に打たれた retain は未初期化の refcount を読む (`Ly_IncRef observed non-positive refcount`、golden 3 件)。

命令を 2 つに割った。`alloc` は記憶域と所有する名前を作り、**セルを作らない**。`init` がヘッダを書く。間の状態では `lookupObj` が `nothing` を返すので、カウンタと life に関する `WFRC` の全フィールドが空虚になる — これはウィンドウの正しい記述であって、逃げではない。

得られた定理 (`Proof.Program.Ownership`):

| 定理 | 内容 |
|---|---|
| `no-dup-in-the-initialisation-window` | ウィンドウ内の `dup` に**ステップが存在しない** |
| `no-drop-in-the-initialisation-window` | 同じく `drop` |
| `dup-resumes-after-init` | ウィンドウは**閉じる** (これが無いと上 2 つは「`dup` が一度も踏めないモデル」でも成り立つ) |
| `no-init-when-shared` | 2 名が所有する記憶域は初期化できない (カウンタに 1 を書くので) |

`Proof.Program.Run` で実物を出している。`sHoisted` は**コンパイラがかつて出していた IR** — retain をハンドル定義位置へ吊り上げた形 — で、`hoisted-retain-has-no-step` がそれに導出が無いことを示す。`window-is-well-formed` が対になっていて、**ウィンドウ自体は正当な状態である** (壊れているのは確保ではなく、その間の incref である) ことを言う。

C++ 側では `prefixIsInitializedAtDefinition` (ABI/EntityHeaderPrefix.h) が「所有権マーカーが立つ時点で prefix は書かれている」という**規約**でこれを守っている。規約は誰かが producer を足すまで持つ。モデル側は producer について何も言わないので、まだ存在しない producer にも効く。**fuzz では届かない**: ウィンドウは数命令幅で、実際に踏んだ 3 入力は 1 本の boxing 経路を通ったものだった。

### 15. ⭐ 記録された所有権 vs 実際の所有権 — `Proof.Program.Recorded`

モデルの所有権は `Binding.mode` ただ 1 つで、構成上つねに真実だった。コンパイラには 2 つある。**その食い違いが、今セッションで直した欠陥の根本原因である。**

lowering パスは意味論を読めない。読むのは属性 (`ly.ownership.*`) であって、そこから判断する。`boxRuntimeObject` が payload retain を出しながら box を owned と記録しなかったのは「retain の漏れ」ではない — retain は出ていた。**記録の漏れ**であり、後で box を borrow と読んだパスは**間違った台帳を正しく読んでいた**。

`edgeRetain : Maybe Mode → Var → Var → List Instr` が `isOwnedIncoming` そのもので、**型が結論である**: 台帳しか受け取らず `Env` を受け取らない。パス内部をどれだけ注意深く書いてもこの欠陥には届かない。義務は台帳を書く側にある。

| 定理 | 内容 |
|---|---|
| `faithful-edge-is-free` | 台帳が忠実なら owned 被演算子への発行は `[]` |
| `unrecorded-ownership-emits-a-retain` | 未記録は borrow 経路に落ち、retain を出す (`nothing` = 「所有でない」はコンパイラの実挙動) |
| `the-unrecorded-retain-bumps-the-counter` | その retain はカウンタを 1 上げる — ステップを添えて述べてある |
| `no-drop-of-a-borrow` | 逆向きの誤記録。borrow への release に**ステップが無い** |
| `not-recording-breaks-faithfulness` | 所有を取って記録せずに返ると、その後何をしても不忠実 |

`Proof.Program.Run` に実物がある。`ledgerAsShipped` (= 空) は retain を出し、`ledgerRepaired` (= 記録した版) は同じパスに何も出させない。`attrsSayBorrow` は**1 つの台帳と 2 つの真実**で、片方にだけ忠実である。

削った命題を 1 つ記録しておく。`the-pass-cannot-see-the-difference` を書いたが、前提 2 つから `md₁ ≡ md₂` が即座に従うので **`refl` の言い換え**だった。一般形は補題を要さない — `edgeRetain` の型がそれである。

### 16. ⭐ リークを到達可能性に結びつけた — `Proof.Program.Leak`

`Proof.Program.Coherence` は自分でこう書いていた: 「これは 7 つの証人であって保存定理ではない」。`Leaked` は定義でき `coherent-has-no-leaks` は coherence をリーク自由性に変換していたが、**プログラムを走らせても coherence が保たれることは何も言っていなかった** — つまりリーク自由性は誰かが書き下した 7 状態についてしか成り立っていなかった。`WFRC` が `Preservation` 以前に抱えていたのと同じ穴で、しかも最も重い場所にある: スイートの leak 段が存在するのは golden 7 件が**リークしながら緑だった**からであり、リークはクラッシュも fuzz も見つけない唯一の欠陥である。

`bump : Bool → ℕ → ℕ` を置いて、名前側と site 側を**同じ形**にした。これは `ownedCount` を `with` ではなく `if isOwned (mode b) ∧ …` で書いた設計上の見返りである — 名前を束縛することと site を占めることは数に対する同じ操作 (述語が成り立つとき 1 足す) で、同じ `if` で綴られているので 1 回の場合分けで両方が reduce する。`with` なら 2 つの補助関数になり、どの単一の分割でも両方には届かない。

要になった補題 2 本:

| 補題 | 内容 |
|---|---|
| `ownedCount-unbind` | `lookupVar` は最初の項目を返し `unbindVar` は最初の項目を消すので、消える項目は規則が引いた項目そのもの。だから等式になる (SSA 前提は不要) |
| `logicalRC-vacate` | 保持している site の vacate はちょうど 1 減らし他は動かさない — `OwnerSite` が別々に証明している 2 つを 1 つの形に合わせたもの |

定理は `no-reachable-state-leaks`、そして `Coherence.leak-is-unreachable` が具体例に適用した形である。**「経路が見つからない」ではない**: 経路があれば定理が `the-leak` を反駁する。

`WF` は仮説であり落とせない。`logicalRC-vacate` は vacate される site が本当にそのオブジェクトを保持していたことを要求し、それを規則の環境前提から機械の事実に変えるのは `WFES.backed` だけである。

変異テスト (arity 保存、両方向対照付き):

| 変異 | 結果 |
|---|---|
| コメントのみ（対照） | ACCEPTED |
| `NameSiteCoherent` を空虚な述語に | REJECTED |
| `after-removal` の `held` を弱め呼び出し側も調整 | REJECTED |
| **`br` を DUP に (vacate せず occupy)** | REJECTED |

最後の 1 つが要点である。**ブロック引数が move であることがリーク自由性を支えている。**

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

### ⛔ 「リークは境界にある」は**実測で反証された** (2026-07-30)

`no-reachable-state-leaks` から「リークには不整合な開始状態か、この関係が持たない遷移 — **関数入口**か**スコープ退出** — が必要」と予測した。**実測したら、実在するリークはどちらでもなかった。**

計測 (`leaks --atExit`、AOT、`print(0)` を基準に差分):

| 段 | 何をしたか | 結果 |
|---|---|---|
| 計器検証 | `hello` / `tuple_one_lane_interior` | net 0 / net 11 root・11520 B — 両方向で生きている |
| 構文切り分け | 16 プローブ | **3 漏れ / 13 清** — `+`、`*`、`e.args[0]` |
| スケーリング | 要素読み数を振る | root は**読み数に比例** (`concat_r1/r2/r4` = 1/2/4)。生産側ではなく消費側 |
| 有界性 | ループ 10 / 50 回 | **10 / 50 root、線形・飽和なし** |
| IR 対照 | リテラルタプル | `lit_r1`/`lit_r4` = 0。**`LyObject_FromSlot` を 1 度も呼ばない** (静的解決) |
| フェーズ帰属 | `LYTHON_IR_DUMP=all` | retain は **`runtime-lowering`**、release は **`refcount-insertion`** |

**根本原因**: `LyObject_FromSlot` (`runtime/modules/builtins.mlir`) は `memref.store %one, %box[0]` で refcount を 1 に初期化して返し、`ly.ownership.owned_results = [0]` とも宣言している。呼び出し側は既に参照を持っている。ところが lowering がその上に `Ly_IncRef` を重ねて 2 にし、`refcount-insertion` が release 1 つで 1 に戻す。**net +1 / ボックス化スロット読み 1 回。** release は同一値・同一ブロック鎖の正しい位置に**存在する**。

つまり配置でも境界でもなく、**余分な retain** — 意味論が要求していない操作の発行である。

**なぜ定理が捕まえなかったか。** モデルの `dup` は**原子的**である: カウンタの増加、owner site の占有、名前の束縛が構成上まとめて起きる。だから**冗長な `dup` は `WFRC` も coherence も保存する** (`Dup.preserves` と `instr-preserves-coherence` が証明している)。モデルは余分な retain を**安全だと言う**。

コンパイラの余分な retain は**素のカウンタ増加**である — site も名前も伴わない。モデルにはその規則が無く、現在の設計では持てない。カウンタと ghost 状態が構成上一緒に動くからである。

**これがギャップである: モデルはランタイム操作と ghost 帳簿を束ねているので、「発行されたが帳簿に載っていない retain」が表現できない。** そしてこのコンパイラの最大のリークはそこにある。

`Proof.Program.Recorded.the-unrecorded-retain-bumps-the-counter` は余分な retain を**本物の `step-dup`** としてモデル化しているので、「カウンタが 1 高い」は捉えるが結論を「整合かつ well-formed な状態」にしてしまう。**算術は合っていて帰結が違う。**

閉じるには `dup` を 2 つに割る必要がある — `py.incref` (カウンタだけ) と、site を占め名前を束縛する側。`alloc`/`init` の分割と同型で、同じ理由 (1 命令が 2 つのことをしていて、その間が表現できない) である。

### ⭐ モデルが先に正解を出した 1 件 — read-back token (2026-07-30)

境界予測は外れたが、**同じ日に見つかった 2 件目の実在リークはモデル側が既に正しい不変条件を持っていた。**

コンテナの同じスロットを 2 回読むと同じハンドルが再構成され、lowering が**同じ SSA 値に所有トークンを 2 つ**載せていた。所有権は SSA 値ごとに追跡されるので、retain 2・release 1 で内側の実体一式が漏れる (内側長 2 で 2 root / 10368 B、70 で 69 root / 14656 B、そして**飽和する** — per-value map を外から見た形)。

`WFES.backed` が破られていた不変条件そのものである: **所有名は自分の site を占める** = 1 名 1 トークン。そして修理は `step-borrow` — 再読み出しは名前を増やし site を占めない — であり、**モデルはこの答えを実測より先に持っていた**。

対比が有益なので両方残す:

| | 境界予測 | read-back token |
|---|---|---|
| モデルの主張 | 「リークは関数入口かスコープ退出」 | 「所有名は自分の site を占める」 |
| 実測 | **外れ** (配置でも境界でもない over-retain) | **当たり** |
| 差 | coherence 保存から導いた**新しい**予測 | 既存フィールドが直接述べている**不変条件** |

読み取れること: **モデルの既存の不変条件は当たり、モデルから新たに導いた予測は外れた。** 前者は定理が支えているが、後者は「関係が持たない遷移は 2 つだけ」という**列挙の完全性**に依存していて、そこは証明されていなかった。

### ⭐ `WFRC` は `backing` について何も言わない

`step-init` の `fresh` (`lookupObj (objects m) o ≡ nothing`) を空虚な命題へ弱めても**ツリー全体が通る** — 保存証明はこの前提を使っていない (arity 保存変異で実測)。

前提自体は規則に残してある。無いと `init` は「オブジェクトを作る」と同じくらい容易に「オブジェクトを上書きする」を意味する: `lookupObj` は最初の一致を返すので、2 枚目のセルが 1 枚目を隠し、**同じ記憶域が別の `backing` 記述子を持つ** — `dealloc` に渡されるブロックが変わる。

**それが `WFRC` に見えないことがギャップである。** どのフィールドも `backing` に触れていないので、「再初期化がデアロケータを差し替える」は不可視。GAP-ANALYSIS §2.4 の「デアロケータ選択が見えない」と同じ穴の、状態側の口である。

### 初期化ウィンドウの一意所有は前提であって不変量ではない

`step-init` の `alone : logicalRC (sites m) o ≡ 1` は前提として渡している。実際にはウィンドウを通して**保たれる**はずである — `alloc` が 1 にし、`move` と `moveArgs` は動かさず、`dup`/`drop` はセルを要求するので踏めない。`WFES` のフィールドに昇格すれば前提が要らなくなり、「ウィンドウは単一所有である」がヘッダ書き込みが競合しない理由にもなる。5 モジュール全部にケースが 1 つずつ増える。

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
