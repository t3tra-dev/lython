# unittest の移植: 何が要り、どこまでできるか

CPython がそうしているように、回帰テストを ctest から Python で書かれた
unittest のテストに移す (利用者の目標、2026-10-18)。Lython 版の unittest は
CPython の Lib/unittest を逐語的に移すのではなく、well-typed に書き直す
(利用者の判断、2026-10-19)。この文書は、その前提で何が足りないか、どこまで
できるか、次に何をするかを記録する。

## 結論 (2026-10-20、lang-gaps fa5c535a で計測)

- **型付きに書き直した unittest の核は、回避策を入れれば今日動く。** 試作
  (`tests/probe/unittest_port/unittest.py`、1,127 行) は TestCase、
  TestResult、TestSuite、TestLoader、TextTestRunner、main と util を持ち、
  26 か所の回避策を入れると `lyc jit` でコンパイルされて動く。
  - 16 本のテスト (12 本が成功、3 本がスキップ、1 本が予期した失敗) を走らせると、
    stdout、stderr (所要時間を除く)、終了コードが CPython と一致する。
  - 9 本のうち 8 本が失敗するテストファイルでは、どのテストが FAIL か ERROR か、
    要約の行、終了コードが一致する。
- **回避策なしでは動かない。止めているのは主に欠陥である。** 回避策が
  避けているのは 27 件のプローブで、20 件が欠陥、7 件が未対応。
  欠陥の多くは「正しく型の付くコードが、別の綴りなら通るのに拒否される」か
  「エミッタを通ったあと lowering や所有権検査で落ちる」ものだった。
- **3 つの部分は回避策でも動かせていない。** subTest (ジェネレータが
  コンパイルされない)、cleanup の実行 (破棄している)、メソッドとしての
  `TestCase.run` (モジュール関数 `_run_case` に移した)。
- **黙って誤動作する欠陥のうち 1 件は unittest の出力に出る。**
  @contextmanager のハンドラで見える traceback に `with` の本体のフレームが
  入らないので、失敗報告にテストのフレームが出ない。結果の判定は正しい。
- **1 ファイルのコンパイルは約 5 秒かかる。** golden の 1 件は約 0.7 秒。
  内訳 (RelWithDebInfo) は lowering 3.3 秒、エミッタ 1.3 秒、JIT ビルド
  0.5 秒、実行 4 ミリ秒、最大メモリ約 0.5 GB。
- **設計判断が要る未対応は 3 つあった。** 利用者が決めた (2026-10-20、下の「方針」)。
  1. テストクラスの見つけ方 (`__subclasses__` か登録か)
  2. docstring を実行時に残すか
  3. builtin とクラスを Callable の値として渡せるようにするか
- **標準ライブラリの不足が効くのは失敗メッセージの質で、実行できるかどうかではない。**
  - difflib と pprint (差分の表示)
  - re と fnmatch (assertRegex と -k)
  - warnings、logging、argparse

## 方針と初段の目標 (利用者の判断、2026-10-20)

- サブクラスは `__subclasses__` を静的に答える。テストクラスは
  `unittest.main()` が `TestCase.__subclasses__()` から見つける (登録表は作らない)。
- docstring は、読まれる場合は残す。読まれなければ DCE で消す。
- ビルトイン関数は Callable の値として扱えるようにする。
- **初段の目標は、この方針で回避なしの unittest を動かすこと。** 受け入れ条件は
  次の課題 A のとおり。

### 進捗

| 日付 | 直したプローブ | 何を直したか |
| --- | --- | --- |
| 2026-10-20 | `ut_bug_inherited_method_loses_its_module_globals`、`ut_bug_computed_getattr_classmethod_arm_in_importer`、`ut_bug_inherited_generic_method_annotation_in_importer` | A1 (継承したメソッド本体のスコープ)。サブクラス向けに複製した継承メソッドの本体を、定義元のモジュールで解決する。プログラムのクラスの classmethod をインスタンス経由で呼ぶとき、クラスの裸の名前に書き換えない。ジェネリッククラスの特殊化の、後回しにしたメソッド本体で型パラメータを束縛する (3 件目の引き金は継承ではなく、関数本体の中からの呼び出しだった)。 |
| 2026-10-20 | `ut_bug_class_value_call_checks_every_subclass_init`、`ut_bug_computed_getattr_checks_subclass_init`、`ut_bug_computed_getattr_generic_method_with_defaults`、`ut_bug_class_value_call_checks_unrelated_subclasses` の半分 | A2 (振り分けの枝の静的検査)。クラス値の呼び出し、仮想ディスパッチ、計算された名前の呼び出しで、呼び出しを受けられない枝が CPython と同じ TypeError を実行時に投げる (`initialize_locals` の順序と文言、3.14 の "Did you mean" を含む)。ジェネリックの型パラメータを、省いた既定値から解く。 |
| 2026-10-20 | `ut_bug_class_value_class_attribute_through_instance`、`ut_bug_class_value_class_attribute_shadowed`、`ut_unsupported_class_attribute_overridden_read_through_base`、`ut_unsupported_instance_shadows_class_attribute`、`ut_unsupported_object_with_class_value_union_field_held`、`ut_bug_class_value_call_checks_unrelated_subclasses` の残り | A5 の一部 (クラス属性)。クラス値を持つクラス属性をセルに置く。サブクラスが上書きしたクラス属性を基底の型から読むディスパッチで、注釈に書けない型 (`type[X]`、`int \| None`) を受ける。インスタンス経由の代入 (`self.maxDiff = None`) を、クラス属性の系統に未設定で始まるフィールドを足して受け、読みは「設定されていればフィールド、なければ `type(obj).attr`」にする (`py.attr.is_set`)。`type[X] \| None` のフィールドを箱に入れる。 |
| 2026-10-20 | `ut_bug_set_display_solves_no_type_parameter`、`ut_bug_isinstance_and_keeps_dead_arm_in_generic_method`、`ut_bug_generic_method_local_of_generic_class` | A4 (ジェネリクス)。集合の表示が、型パラメータを含む期待型 (`set[T]`) で作られないようにする。`and`/`or` の分岐を、決まる被演算子で静的に決める (実行時に決まる被演算子の後に決まるものも、その腕を刈る)。フィールドの事前走査でジェネリックメソッドの型パラメータを束縛し、未束縛の型パラメータでジェネリッククラスを特殊化しない。途中で、フィールドに入ったサブクラスのないクラス値を `issubclass` が拒否していたのも直した。 |
| 2026-10-20 | `ut_bug_type_alias_of_an_imported_module`、`ut_bug_field_typed_with_own_subclass`、`ut_bug_class_values_into_varargs`、`ut_bug_imported_function_list_of_one_subclass_value`、`ut_bug_and_over_hasattr_of_a_class_value`、`ut_unsupported_hasattr_on_class_value` | A5 の残り。取り込んだモジュールの型エイリアスをそのモジュールのスコープで束縛し、`type X = ...` 文も宣言として受ける。基底の本体が名指すサブクラスを、基底より先に生成しない。可変長引数のタプルにクラス値を入れる。コンテナ表示の期待型で、サブクラスのクラス値を基底のクラス型に合うと見る。`hasattr` を bool と型付けし、クラス値への `hasattr` を実行時のクラスで答える (その名前を宣言するクラスから派生しているか)。 |
| 2026-10-20 | `ut_unsupported_computed_constant_of_imported_module`、`ut_unsupported_generic_factory_result_returned`、`ut_unsupported_fixed_tuple_concatenation`、`ut_unsupported_exc_info_outside_handler`、`ut_bug_subclass_into_optional_method_parameter` | 未対応 4 件と A5 の 1 件。取り込んだモジュールの、他の定数から計算した int と str の定数を畳み込む。結果だけが名指す型パラメータを、結果に期待される型から解く (`skipUnless` が `skipIf(...)` を返す形)。長さの決まったタプルを要素ごとに連結する。except ブロックの外の `sys.exc_info()` を、typeshed どおり省略可能な三つ組として実行時に答える (新しい `py.except.handling`)。union の引数に、どれかのメンバーに合う値を渡せる。途中で、呼び出し可能なインスタンスの union を呼ぶ形も通した (`test(result)`)。 |

進捗の途中で決めたこと・見つけたこと:

- **型の合わない枝は Lython の TypeError を投げる (意図した逸脱)。** クラス値の
  呼び出しで、引数の数は合うが型が `__init__` の宣言と合わないクラスに当たると、
  CPython はそのままオブジェクトを作る。Lython ではそのフィールドを持つ
  オブジェクトを作れないので、理由を書いた TypeError を投げる。プログラム全体を
  拒否していた以前より狭い。`wb_a_class_value_call_whose_argument_the_init_cannot_hold`
  に記録した (expect-wrong)。
- **既存のクラッシュを 1 件直した。** ジェネリック関数の、リテラルでない既定値
  (`xs: list[int] = [1]` など) を省いて呼ぶと、誰も書き込んでいないセルを読んで
  segfault していた。既定値は特殊化ごとではなく関数ごとに 1 つのセルにし、`def`
  文の位置で 1 回評価する (CPython と同じく、全呼び出しで同じオブジェクトを共有する)。
- **既存の誤動作を 1 件直した。** クラス属性と同名のフィールドを `__init__` が
  条件付きでしか書かないとき、書かなかったインスタンスの読みが AttributeError を
  投げていた (CPython はクラス属性を返す)。上のインスタンス経由の代入と同じ読み方で直った。
- **黙った誤動作を 1 件直した。** インスタンスの本体はボックスを先に、bool と
  クラス値の語を後に置いていたので、ボックスのフィールドを足したサブクラスでは
  基底の bool の語の位置がずれた。基底の型から書いた値をサブクラスの型で読むと
  別の語を読み、クラス値のフィールドではクラッシュした (unittest では
  `_testRunEntered` が見えず、クラスの後始末が 2 回走った)。フィールドごとに 1 つの
  ボックス大のセルを、フィールドの番号の位置に置くように直した。
- **コンパイル時間の退行を 1 件、入れる前に直した。** タプル連結の型付けが
  `a + b + c ...` の連鎖で被演算子を二度ずつ推論し、traceback を取り込むだけの
  プログラムが 25 秒かかった。

## 計測の方法

1. **試作** (`tests/probe/unittest_port/`)。CPython 3.14 の case.py、
   result.py、suite.py、loader.py、runner.py、main.py、util.py を、型の付く
   形で 1 つのモジュールに書き直したもの。
   - テストファイルの隣に `unittest.py` を置くと、CPython でも標準のものより先に
     読まれる。そのため同じテストファイルを CPython と Lython の両方で走らせて比べられる。
   - Lython が拒否した箇所ごとに最小の再現を作り、試作には回避策を入れて先へ進めた。
     回避策には `WORKAROUND(<プローブ名>)` が付いていて、プローブが直れば外せる。
   - 手での切り分けで見つからなかった 3 件は、拒否の文言が出続ける限り定義や文を
     消していく自動縮小 (約 300〜900 回のコンパイル) で最小形を得た。
2. **逐語的な移植を想定したギャップ探針** (113 本、前回 2026-10-18 の計測の
   再実行)。67 本が CPython と一致、46 本が拒否、黙った誤動作は 0 本。
   拒否の多くは注釈のない引数や `object` に対する演算で、型付きに書き直せば
   出てこない形である。
3. **残差の記録** (`tests/probe/ut_*.py`、82 本)。この調査と、直前の
   `*args`/`**kwargs` の作業で残ったものを、1 件 1 プログラムで置いた。
   - ファイル名の接頭辞は次のとおり。
     - `ut_bug_*` (42 本): 欠陥
     - `ut_unsupported_*` (33 本): 未対応
     - `ut_mod_*` (7 本): 複数モジュールにまたがる再現の補助モジュール
   - 各ファイルの先頭には、何が起きるか、分類、計測した木、CPython の出力を書いた。
   - 差分ベースライン (`tests/probe/differential-baseline.txt`) に登録してあり、
     `tests/probe/tools/differential.py` が動きを報告する。内訳は GAP 67、
     WRONG-DECLARED 6、EXTRA 1、SKIP 1 (stderr だけの差)、AGREE 7 (補助モジュール)。
   - 直れば GAP や WRONG-DECLARED から AGREE へ動く。

## 型付きの unittest の形 (CPython からの逸脱)

試作で選んだ形。どれも型を静的に決めるための逸脱で、回避策とは別のもの。
`unittest.py` の冒頭にも書いてある。

- **skip と expectedFailure はメソッドを包む。** CPython は関数に
  `__unittest_skip__` などの属性を付ける。
  - ここでは `skip(reason)` が、テストメソッドを `SkipTest` を投げる関数で置き換える。
  - `expectedFailure` は、失敗を `_ExpectedFailure` に、成功を
    `_UnexpectedSuccess` に変える関数で包む。
  - ジェネリックなデコレータファクトリ (`def skip[T: TestCase](reason)`) はそのまま書ける。
  - クラスに付ける `@skip` はまだ書けない (`ut_unsupported_class_decorator`)。
- **assertEqual は引数の静的な型で振り分ける。** CPython は実行時のクラスを
  キーにした `_type_equality_funcs` を引く。ここでは
  `def assertEqual[T](self, first: T, second: T)` の中で `isinstance` により
  分ける。特殊化された T に合わない枝は、型検査の前に刈られる。
- **assertRaises の 2 つの形は名前を分ける。** 1 つの名前で、context manager
  を返すことと呼び出しを転送することの両方はできないので、呼び出しの形は
  `assertRaisesCall[E, **P]` とした。
- **cleanup は引数を閉じ込めたクロージャで持つ。** `addCleanup[**P]` が
  `lambda: function(*args, **kwargs)` を積む。CPython の (関数, args, kwargs)
  の三つ組は、型が呼び出しごとに違うので 1 つのリストに入らない。
- **テストクラスは明示的に渡す。** `unittest.main([TestA, TestB])`。
  - CPython の `loadTestsFromModule` は、モジュールの属性を `dir()` で総なめにし、
    `isinstance(obj, type)` で選ぶ。
  - 異なる型の値を 1 つずつ取り出すこの形は、静的には書けない。
  - `__subclasses__` が入れば、`unittest.main()` の引数は要らなくなる (次の課題 B2)。
- **traceback から unittest 自身のフレームを除くのはファイル名で行う。**
  CPython はフレームのグローバルにある `__unittest` を見る。ここには
  フレームオブジェクトがないので、`traceback.format_tb` の各項目の
  ファイル名で判断する。
- **差分の表示がない。** difflib と pprint がないので、assertListEqual などは
  最初に違う要素までしか示さない。

## 欠陥 (42 件)

種類の欄の意味は次のとおり。

| 種類 | 意味 |
| --- | --- |
| 黙って | CPython と違う結果を出して正常終了する (8 件) |
| 遅い拒否 | エミッタを通ったあと lowering や所有権検査で落ちる。最も早い静的境界で拒否していない (14 件) |
| 拒否 | 正しく型の付くコードを、エミッタが拒否する (20 件) |

### unittest の核に効くもの (22 件)

試作の回避策が避けているものと、試作の出力に出るもの。

| プローブ | 種類 | 何が起きるか |
| --- | --- | --- |
| `ut_bug_inherited_method_loses_its_module_globals` | 拒否 | 取り込んだクラスのメソッドが、main 側のサブクラス向けにもう一度コンパイルされ、元のモジュールの関数を見失う。TestCase.run が setUp を呼ぶ形そのもの。 |
| `ut_bug_inherited_generic_method_annotation_in_importer` | 遅い拒否 | 同じ系統。継承したジェネリックメソッドが作るジェネリッククラスの `self.x: E \| None` 注釈が、E の束縛なしで読まれる。 |
| `ut_bug_computed_getattr_classmethod_arm_in_importer` | 拒否 | 同じ系統。`getattr(test, name)()` の classmethod の枝が、取り込む側のスコープで `TestCase` を探す。 |
| `ut_bug_class_value_call_checks_unrelated_subclasses` | 拒否 | `self.failureException(msg)` が、プログラム中のすべての例外クラスの `__init__` で型検査される。テストファイルに独自の `__init__` を持つ例外が 1 つあるだけで fail() が通らない。 |
| `ut_bug_class_value_call_checks_every_subclass_init` | 拒否 | 同じ系統。ローダの `cls(name)` が、`__init__` の違うサブクラスのために拒否される。 |
| `ut_bug_computed_getattr_checks_subclass_init` | 拒否 | 同じ系統。`getattr(self, name)()` の `__init__` の枝が、実行時の TypeError ではなく静的な拒否になる。 |
| `ut_bug_optional_held_across_yield_in_with` | 遅い拒否 | ジェネレータが `None` を含み得る値を、`with` の中の yield をまたいで持つと、所有権検査で落ちる。 |
| `ut_bug_class_body_field_object_into_inner_contextmanager` | 遅い拒否 | クラス本体で注釈だけ宣言したフィールドを持つオブジェクトを、ジェネレータが yield をまたいで内側の contextmanager に渡すと、所有権検査で落ちる。呼ばれていなくても落ちる。subTest の形。 |
| `ut_bug_exc_info_after_calling_a_callable_from_a_list` | 遅い拒否 | リストから取った Callable を try の中で呼び、ハンドラが traceback を読む (`sys.exc_info()`、`traceback.format_exception`) と、所有権検査で落ちる。cleanup を回す形。 |
| `ut_bug_join_of_tuple_with_narrowed_optional` | 遅い拒否 | `None` でないと絞った `str \| None` をタプル表示に入れて `str.join` に渡すと、所有権検査で落ちる。 |
| `ut_bug_generic_method_local_of_generic_class` | 遅い拒否 | ジェネリックメソッドの中で `ctx = Ctx[E](...)` をローカルに置くと、未解決の E で Ctx が特殊化される。呼ばれていなくても、取り込むだけで落ちる。 |
| `ut_bug_set_display_solves_no_type_parameter` | 拒否 | 集合の表示 `{1, 2}` からジェネリックメソッドの T が解けない (リストなら解ける)。assertSetEqual の形。 |
| `ut_bug_isinstance_and_keeps_dead_arm_in_generic_method` | 拒否 | `isinstance(a, list) and isinstance(b, list)` だと、特殊化で死んだ枝が刈られない (`if` を入れ子にすれば刈られる)。 |
| `ut_bug_type_alias_of_an_imported_module` | 拒否 | 取り込んだモジュールの型エイリアス (`_SysExcInfoType` の形) が、存在しないクラスとして扱われる。 |
| `ut_bug_subclass_into_optional_method_parameter` | 拒否 | メソッドの `Base \| None` 引数にサブクラスを渡せない (関数なら通る)。`TestCase.run(result: TestResult \| None)` に TextTestResult を渡す形。 |
| `ut_bug_field_typed_with_own_subclass` | 拒否 | 自分のサブクラスを型に持つフィールド (`_SubTest \| None`) を宣言すると、そのサブクラスが親のフィールドを失う。 |
| `ut_bug_class_value_class_attribute_through_instance` | 遅い拒否 | クラス属性に置いたクラス値 (`failureException`) をインスタンス経由で読むと、lowering で落ちる。 |
| `ut_bug_class_value_class_attribute_shadowed` | 遅い拒否 | 同じクラス属性へのインスタンス経由の代入が、エミッタの検査をすり抜けて lowering で落ちる。 |
| `ut_bug_class_values_into_varargs` | 遅い拒否 | クラス値を `*classes: type[Base]` に渡すと lowering で落ちる (`list[type[Base]]` なら通る)。`unittest.main(TestA, TestB)` の形。 |
| `ut_bug_imported_function_list_of_one_subclass_value` | 遅い拒否 | 取り込んだ関数の `list[type[TestCase]]` 引数に `[Mine]` (サブクラス 1 つ) を渡すと、lowering で落ちる (2 つなら通る)。 |
| `ut_bug_and_over_hasattr_of_a_class_value` | 拒否 | bool どうしの `and` が、片方が `hasattr(cls, ...)` だと拒否される。 |
| `ut_bug_contextmanager_handler_traceback_lacks_body_frames` | 黙って | @contextmanager のハンドラで見える traceback に、`with` の本体のフレームが入らない。unittest の失敗報告にテストのフレームが出ない。 |

### その他の欠陥 (20 件)

| プローブ | 種類 | 何が起きるか |
| --- | --- | --- |
| `ut_bug_generator_repr_names_its_class` | 黙って | `repr(gen())` が関数名ではなく `GeneratorType` を名乗る。 |
| `ut_bug_explicit_base_str_calls_the_override` | 黙って | `BaseException.__str__(e)` が e の上書きした `__str__` を呼ぶ。 |
| `ut_bug_int_function_held_as_float_callable` | 黙って | int を返す関数を `Callable[[], float]` に入れると、呼び出しが偽の TypeError を投げる。 |
| `ut_bug_unhashable_key_error_not_wrapped` | 黙って | `__hash__` の TypeError を、CPython 3.14 のように "cannot use 'Key' as a dict key" で包まない。 |
| `ut_bug_format_exception_only_drops_module` | 黙って | `traceback.format_exception_only(e)` が、取り込んだモジュールの例外にモジュール名を付けない (捕まらなかった例外の表示は付ける)。 |
| `ut_bug_uncaught_line_ignores_overridden_str` | 黙って (stderr) | 捕まらなかった例外の最後の行が、`str(exc)` ではなく保存された引数を出す。 |
| `ut_bug_class_base_named_before_its_definition` | 黙って (寛容) | 後で定義されるクラスを基底にしたクラスが、CPython の NameError なしに動く。 |
| `ut_bug_computed_getattr_generic_method_with_defaults` | 拒否 | 計算された名前の呼び出しで、引数がすべて既定値のジェネリックメソッドの枝の型パラメータが解けない。 |
| `ut_bug_generic_class_built_from_type_parameter` | 拒否 | `-> Box[T]` のジェネリックメソッドで `Box(v)` が `Box[T]` と推論されない (`Box[T](v)` なら通る)。 |
| `ut_bug_generic_call_mixing_int_and_float` | 拒否 | `collect(0, 1.5)` で T が float に解けず、デコレータファクトリ向けの的外れな診断が出る。 |
| `ut_bug_object_lambda_into_container_method` | 拒否 | `fs.append(lambda: value())` で、list.append の引数型がラムダの期待型にならない。 |
| `ut_bug_return_list_of_base_from_subclass_display` | 拒否 | `-> list[Base]` の関数で `return [Sub()]` が拒否される。 |
| `ut_bug_generator_method_fixed_tuple_argument` | 拒否 | ジェネレータメソッドの `tuple[int, ...]` 引数に固定長タプルを渡せない。 |
| `ut_bug_unreachable_yield_still_makes_a_generator` | 拒否 | `if False: yield` だけの関数が、空のジェネレータにならない。 |
| `ut_bug_qualified_generator_type_annotation` | 拒否 | `types.GeneratorType[...]` とモジュール名付きで書いた注釈が解決されない。 |
| `ut_bug_oserror_with_errno` | 拒否 (内部の文言) | `OSError(2, "...")` がコンパイラ内部の文言で拒否される。 |
| `ut_bug_union_of_exceptions_into_exception_parameter` | 遅い拒否 | `ValueError \| KeyError` を `Exception` 引数に渡すと、lowering で落ちる。 |
| `ut_bug_object_in_str` | 遅い拒否 | `x in s` (`x: object`) が lowering で落ちる。 |
| `ut_bug_union_dispatch_then_narrow_ownership` | 遅い拒否 | union 値を振り分け呼び出しに使った後で isinstance で絞ると、所有権検査で落ちる。 |
| `ut_bug_tuple_of_optional_class_results` | 遅い拒否 | `Box \| None` を返す呼び出し 2 つをタプルに入れると、所有権検査で落ちる。 |

## 未対応 (33 件)

### 設計判断が要るもの

| プローブ | 何が要るか |
| --- | --- |
| `ut_unsupported_subclasses` | `cls.__subclasses__()`。プログラムのクラスはコンパイル時にすべて分かるので、答えは静的に出せる。`unittest.main()` を引数なしにする最短の道。 |
| `ut_unsupported_init_subclass_super` | `__init_subclass__` と `super()`。テストクラスを登録表に集める、もう 1 つの道。 |
| `ut_unsupported_docstring` | `__doc__` を実行時に残すか。`-v` の出力の 2 行目 (shortDescription) に使われる。 |
| `ut_unsupported_builtin_function_as_callable` | `print` などの builtin を Callable の値にする。拒否が lowering まで遅れている点は欠陥でもある。 |
| `ut_unsupported_bound_builtin_method_as_callable` | `log.append` を Callable の値にする。`addCleanup(log.append, x)` の形。拒否の文言はコンパイラ内部のもの。 |
| `ut_unsupported_class_as_callable` | クラスを Callable の値にする。`assertRaisesCall(TypeError, int, "x")` の形。 |
| `ut_unsupported_function_name_through_value` | Callable の値から `__name__` を読む。assertRaises の呼び出し形の失敗文言に使われる。 |

### unittest の書き方に効くもの

| プローブ | 何が起きるか |
| --- | --- |
| `ut_unsupported_class_decorator` | クラスに付けるデコレータ (`@skip` をクラスに)。 |
| `ut_unsupported_class_attribute_overridden_read_through_base` | サブクラスが上書きするクラス属性 (`maxDiff = None` など) を、基底の型から読む。 |
| `ut_unsupported_instance_shadows_class_attribute` | インスタンス経由でクラス属性を上書きする代入。 |
| `ut_unsupported_object_with_class_value_union_field_held` | `type[X] \| None` のフィールドを持つオブジェクトを、別のオブジェクトやリストに入れる。 |
| `ut_unsupported_hasattr_on_class_value` | クラス値への `hasattr(cls, 'runTest')`。 |
| `ut_unsupported_exc_info_outside_handler` | ハンドラから呼んだ関数の中の `sys.exc_info()`。 |
| `ut_unsupported_generic_factory_result_returned` | 別のジェネリックファクトリの結果を返すファクトリ (`skipUnless` が `skipIf(...)` を返す形)。 |
| `ut_unsupported_held_decorator_factory` | 名前に保持したデコレータファクトリの結果を `@t` で使う。 |
| `ut_unsupported_decorated_staticmethod` | staticmethod、classmethod、property、ジェネレータに付けたプログラムのデコレータ。 |
| `ut_unsupported_computed_constant_of_imported_module` | 取り込んだモジュールで、他の定数から計算した定数には注釈が要る。 |
| `ut_unsupported_fixed_tuple_concatenation` | 固定長タプルの連結 `(i,) + pair` の要素型が `object` になる。 |
| `ut_unsupported_object_sentinel` | `object()` を番兵にする。 |
| `ut_unsupported_traceback_attribute` | `e.__traceback__`。 |
| `ut_unsupported_sys_modules` | `sys.modules` (モジュール単位の fixture と、名前からのロード)。 |

### `*args` / `**kwargs` の転送で意図して拒否しているもの

`*args` / `**kwargs` の転送は、「型付き unpack で足りる。呼び出し側から逆算できる場合は
除く」という利用者の判断 (2026-10-19) に沿って作った。次の形は、長さやキーが実行時に
しか分からず、呼び出しの形が決まらないので拒否している。

| プローブ | 形 |
| --- | --- |
| `ut_unsupported_dynamic_star_into_positional_parameters` | `add(*xs)` が名前付きの位置引数を埋める。 |
| `ut_unsupported_mapping_into_named_parameters` | `add(**d)` が名前付きの引数を埋める。 |
| `ut_unsupported_dynamic_star_into_builtin` | `print(*xs)`。 |
| `ut_unsupported_dynamic_unpack_into_paramspec_forwarder` | ParamSpec の転送関数へ長さ不定の unpack を渡す。 |
| `ut_unsupported_paramspec_forwarder_as_value` | ParamSpec の転送関数を値として使う。 |
| `ut_unsupported_paramspec_decorator` | 典型的な ParamSpec デコレータ (内側の `inner(*args: P.args, **kwargs: P.kwargs)` を値として返す)。内側の def を、デコレートされた関数のシグネチャで特殊化する仕組みが要る。 |

### その他

- `ut_unsupported_eq_only_in_subclass`
- `ut_unsupported_generic_protocol`
- `ut_unsupported_iter_of_non_indexable`
- `ut_unsupported_reversed_heterogeneous_tuple`
- `ut_unsupported_range_count`
- `ut_unsupported_bytes_count_int`

## 標準ライブラリ

unittest 自身が取り込むもののうち、Lython にないもの。

| モジュール | unittest での用途 | 無いときの代わり |
| --- | --- | --- |
| difflib | assertMultiLineEqual などの差分表示 (`ndiff`) | 最初に違う要素だけを示す |
| pprint | 差分表示の前の整形 (`pformat`) | `repr` |
| re | assertRegex、assertRaisesRegex、loader のモジュール名の判定 | 部分文字列の比較 |
| fnmatch | `-k` のパターン | 部分文字列の比較 |
| warnings | assertWarns、runner の警告フィルタ | 無し |
| logging | assertLogs | 無し |
| argparse | `unittest.main` のコマンドライン | 手書きの解析 |
| signal | `-c` (Ctrl-C で止める) | 無し |
| collections.Counter | assertCountEqual | 契約が `list[str] \| str \| None` しか受けない |
| os.path.commonprefix | 失敗文言の省略 | 自前の実装 |

## 回帰テストの移行で動かせるもの・動かせないもの

| 今のテスト | 件数 | 移せるか |
| --- | --- | --- |
| `tests/golden/cases` (stdout の完全一致) | 937 | 移せる。値を print する形を `assertEqual` に書き換える手作業になる。副作用の順序を見るもの (try/finally など) は、記録用のリストを比べる形になる。 |
| `tests/golden/errors` (診断と終了コード) | 123 | 移せない。コンパイル時の診断を見るものは、実行時の `compile()` も subprocess もないので、Python のテストからコンパイラを呼べない。捕まらなかった例外の表示と終了コードを見るものは、プロセスの外からしか見えない。 |
| `tests/unit` (C++ API) | 267 | 移せない。エミッタやドライバを直接叩くもの。 |
| leak gate、examples smoke | — | 別の仕組みのまま。 |

テストを動かすのが、テストされる側のコンパイラだという点には注意が要る。
assertEqual の比較そのものが誤ってコンパイルされれば、失敗すべきテストが通る。
CPython も自分のインタプリタでテストを動かしているので、構造は同じである。
ただし Lython では、同じテストファイルを CPython でも走らせて結果を突き合わせる
ことができる (試作はそのために CPython でも動く形にしてある)。移行の後も、
「CPython で通り、Lython でも通る」ことを確かめる差分の段を残す。

## 次の課題

並びは、前の段が次の段の前提になる順にした。

### A. 試作が回避策なしで動き、CPython と同じ結果を出す

試作の `WORKAROUND(...)` を外せるようにする。

受け入れ条件:

- `tests/probe/unittest_port/` の各テストファイルを、回避策なしの `unittest.py` で走らせる。
- 出力と終了コードが CPython と同じになる (所要時間の行は除く)。
- subTest、cleanup、メソッドとしての `TestCase.run` を戻して、同じことが言える。

上の表の「unittest の核に効くもの」22 件を、仕組みごとにまとめた。

1. **継承したメソッド本体のスコープ** (3 件)。サブクラスや振り分けのために
   もう一度コンパイルするメソッド本体は、名前を定義元のモジュールのスコープで
   解決する。
2. **振り分けの枝の静的検査** (3 件、`ut_bug_computed_getattr_generic_method_with_defaults`
   も同じ機構)。クラス値の呼び出しと計算された名前の呼び出しで、候補のうち
   引数の合わない枝は、静的に拒否しない。CPython と同じ TypeError を実行時に
   投げる枝にする。
3. **所有権検査で落ちるもの** (4 件)。
   - yield をまたぐ Optional
   - クラス本体だけで宣言したフィールドを持つオブジェクト
   - リストから取った Callable と traceback を読むハンドラ
   - 絞った Optional のタプル

   `ut_bug_union_dispatch_then_narrow_ownership` と
   `ut_bug_tuple_of_optional_class_results` も同じ層にある。
4. **ジェネリクス** (3 件)。
   - ジェネリックメソッドのローカルの事前型付け
   - 集合の表示からの T
   - `and` でつないだ isinstance での枝刈り
5. **型・属性・引数** (8 件)。
   - 取り込んだモジュールの型エイリアス
   - メソッドの Optional 引数へのサブクラス
   - 自分のサブクラスを型に持つフィールド
   - クラス値のクラス属性 (読み・書き)
   - クラス値の `*args`
   - 取り込んだ関数へのサブクラス 1 つのリスト
   - `hasattr` との `and`
6. **traceback** (1 件)。@contextmanager のハンドラで見える traceback に、
   `with` の本体のフレームを残す (`ut_bug_format_exception_only_drops_module` も
   失敗報告に出る)。

### B. unittest を標準ライブラリにして、テストを ctest から走らせる

1. **埋め込み**。`src/lython/runtime/lib/unittest.py` として埋め込む。
   埋め込みの glob は平らなので、まずは 1 つのモジュールにする
   (CPython はパッケージ)。
2. **テストクラスの発見**。`type.__subclasses__()` を静的に答える
   (`ut_unsupported_subclasses`)。これで `unittest.main()` が、`__main__` で定義された
   TestCase のサブクラスを引数なしで見つけられるようになる。
3. **ctest への登録**。`tests/python/test_*.py` を 1 ファイル 1 テストで登録し、
   `lyc jit` の終了コードで判定する。同じファイルを CPython でも走らせる段を
   置く。
4. **コンパイル時間**。1 ファイル約 5 秒のうち 3.3 秒が lowering。unittest
   本体の大半はテストファイルによらず同じなので、ファイルあたりの費用を
   `LYTHON_PERF=1` の内訳から詰める。

### C. 失敗メッセージと便利機能を CPython に近づける

1. docstring を残すかどうかの判断 (`-v` の 2 行目)。
2. difflib と pprint (差分の表示)。
3. re と fnmatch (assertRegex、assertRaisesRegex、`-k`)。
4. builtin とクラスを Callable の値にする (`addCleanup(log.append, x)`、
   `assertRaisesCall(TypeError, int, "x")`)。
5. クラスに付ける `@skip`、setUpModule、warnings (assertWarns)、
   logging (assertLogs)、argparse。

### D. テストを移す

1. 新しく書く回帰テストから unittest の形にする。
2. 既存の golden は、値を見るものから順に書き換える。
3. コンパイル時の診断を見る golden と、C++ のユニットテストは ctest に残す。
