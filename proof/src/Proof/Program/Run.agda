{-# OPTIONS --safe #-}

-- Every instruction rule, taken as an actual step.
--
-- Proof.Program.Trace exhibits the branch and its alias, and nothing else: the
-- only derivations it builds are TERMINATOR steps. So `new`, `move`, `dup`,
-- `drop` and `borrow` -- the five rules the entire reference-counting story is
-- about -- had never been executed. Two changes had been made to close
-- obstructions to `WFRC` (`step-new` constructs its cell, `step-drop` requires
-- the object live) and neither had ever been exercised.
--
-- This module runs all six instructions of one block and reads both counts at
-- every point. Every state is written in NORMAL FORM rather than as the rule's
-- own right-hand side, so each `step-…` term has to prove that the rule really
-- produces the state written here. Defining the states as the rules' outputs
-- would make every derivation typecheck by construction and check nothing.

module Proof.Program.Run where

open import Data.List using (List; []; _∷_)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc)
open import Data.Product using (_×_; _,_; Σ; proj₁; proj₂)
open import Data.Integer using (+_)
open import Data.Vec using ([]; _∷_)
open import Relation.Binary.PropositionalEquality using (_≡_; refl)
open import Relation.Nullary using (¬_)

open import Proof.Memory.Heap using (Heap)
open import Proof.Memory.Lython using (LythonSig; i64)
open import Proof.Memory.Descriptor LythonSig using (Desc)
-- ⭐ Renamed on import rather than renaming the instruction. `Instr.alloc` IS
-- this operation seen from the program layer, so the two SHOULD have the same
-- name; what a reader needs here is to know which level a given occurrence is
-- at, and that is what the rename says.
open import Proof.MemRef.Dialect LythonSig renaming (alloc to allocBlock)
open import Proof.RC.Object
open import Proof.RC.OwnerSite using (OwnerSite; local; SiteMap; logicalRC)
open import Proof.RC.Machine LythonSig
open import Proof.RC.Ops LythonSig using (reclaim; RCFault)
open import Proof.Prelude using (Result; ok; err)
open import Proof.Program.Syntax
open import Proof.Program.Env
open import Proof.Program.Step LythonSig
open import Proof.Program.Ownership LythonSig
  using (no-dup-in-the-initialisation-window)
open import Proof.Program.Preservation LythonSig
open import Proof.RC.Invariant LythonSig using (WFRC; counted-exact)
open import Data.Empty using (⊥; ⊥-elim)
open import Proof.Lython.Invalid LythonSig
  using (StillNamed; named-by; PrematureReclaim; premature; Invalidity;
         premature-reclaim; DanglingBorrow; dangling-borrow)
open import Proof.Lython.Detect LythonSig
  using (danglingAnchor; danglingAnchor-sound; stillNamed?; prematureReclaim?;
         prematureReclaim?-sound)

------------------------------------------------------------------------
-- The program.
--
--   bb0(): alloc x ; init x ; dup y x ; borrow z y ; move w y ; drop w ; drop x
--          ; unwind
--
-- The `borrow` is deliberately anchored at `y`, which the `move` two
-- instructions later removes. That is not an accident of the example: it is the
-- one obligation a borrow carries, and running it here means the dangling
-- borrow at the end is produced BY THE STEP RELATION rather than written into
-- an environment by hand.

x y z w : Var
x = 0
y = 1
z = 2
w = 3

bb0 : BlockId
bb0 = 0

fullBody : List Instr
fullBody = alloc x 7 ∷ init x ∷ dup y x ∷ borrow z y ∷ move w y ∷ drop w ∷ drop x ∷ []

prog : Function
prog = function (block bb0 [] fullBody unwind ∷ []) bb0

------------------------------------------------------------------------
-- The allocation the object will live in, and the object.

allocated : Heap × Desc 1
allocated = allocBlock [] 0 i64 1 8

h : Heap
h = proj₁ allocated

bk : Desc 1
bk = proj₂ allocated

theObj : ObjId
theObj = obj 0 0

------------------------------------------------------------------------
-- The states, in normal form.

s₀ : PState
s₀ = pstate 0 bb0 fullBody [] (machine h [] [])

-- ⭐ INSIDE THE INITIALISATION WINDOW. `x` owns storage; the object table is
-- still empty, so there is nothing here with a refcount.
sWindow : PState
sWindow = pstate 0 bb0 (init x ∷ dup y x ∷ borrow z y ∷ move w y ∷ drop w ∷ drop x ∷ [])
                 ((x , bind theObj owned) ∷ [])
                 (machine h [] ((local 0 x , theObj) ∷ []))

s₁ : PState
s₁ = pstate 0 bb0 (dup y x ∷ borrow z y ∷ move w y ∷ drop w ∷ drop x ∷ [])
            ((x , bind theObj owned) ∷ [])
            (machine h ((theObj , cell live (counted 1) bk 0) ∷ [])
                     ((local 0 x , theObj) ∷ []))

s₂ : PState
s₂ = pstate 0 bb0 (borrow z y ∷ move w y ∷ drop w ∷ drop x ∷ [])
            ((y , bind theObj owned) ∷ (x , bind theObj owned) ∷ [])
            (machine h ((theObj , cell live (counted 2) bk 0) ∷ [])
                     ((local 0 y , theObj) ∷ (local 0 x , theObj) ∷ []))

s₃ : PState
s₃ = pstate 0 bb0 (move w y ∷ drop w ∷ drop x ∷ [])
            ((z , bind theObj (borrowed y))
             ∷ (y , bind theObj owned) ∷ (x , bind theObj owned) ∷ [])
            (machine h ((theObj , cell live (counted 2) bk 0) ∷ [])
                     ((local 0 y , theObj) ∷ (local 0 x , theObj) ∷ []))

s₄ : PState
s₄ = pstate 0 bb0 (drop w ∷ drop x ∷ [])
            ((w , bind theObj owned)
             ∷ (z , bind theObj (borrowed y)) ∷ (x , bind theObj owned) ∷ [])
            (machine h ((theObj , cell live (counted 2) bk 0) ∷ [])
                     ((local 0 w , theObj) ∷ (local 0 x , theObj) ∷ []))

s₅ : PState
s₅ = pstate 0 bb0 (drop x ∷ [])
            ((z , bind theObj (borrowed y)) ∷ (x , bind theObj owned) ∷ [])
            (machine h ((theObj , cell live (counted 1) bk 0) ∷ [])
                     ((local 0 x , theObj) ∷ []))

s₆ : PState
s₆ = pstate 0 bb0 []
            ((z , bind theObj (borrowed y)) ∷ [])
            (machine h ((theObj , cell finalizing (counted 0) bk 0) ∷ []) [])

------------------------------------------------------------------------
-- The six derivations. ⭐ This is what the module exists for: before it, none
-- of these five constructors had ever been applied.

-- `alloc`: storage and a name, and NO CELL. The object table of `sWindow` is
-- `[]`, which is the window as a value rather than as a description.
run-alloc : prog ⊢ s₀ —→ᵢ sWindow
run-alloc = step-alloc refl refl refl refl refl

-- `init`: the header write. The rule CONSTRUCTS `cell live (counted 1) bk 0` --
-- a change made to remove an obstruction to `WFRC.live-positive` -- and its
-- third premise is `logicalRC (sites m) o ≡ 1`, the condition that the storage
-- being initialised has exactly one owner.
run-init : prog ⊢ sWindow —→ᵢ s₁
run-init = step-init refl refl refl

run-dup : prog ⊢ s₁ —→ᵢ s₂
run-dup = step-dup refl refl refl

run-borrow : prog ⊢ s₂ —→ᵢ s₃
run-borrow = step-borrow refl

run-move : prog ⊢ s₃ —→ᵢ s₄
run-move = step-move refl refl

-- `drop`: the third premise is `life c ≡ live`, added so that dropping a dead
-- object could not resurrect it. Also used here for the first time.
run-drop-w : prog ⊢ s₄ —→ᵢ s₅
run-drop-w = step-drop refl refl refl refl

run-drop-x : prog ⊢ s₅ —→ᵢ s₆
run-drop-x = step-drop refl refl refl refl

-- and they compose.
whole-block : prog ⊢ s₀ —→* s₆
whole-block =
  more (by-instr run-alloc)
  (more (by-instr run-init)
  (more (by-instr run-dup)
  (more (by-instr run-borrow)
  (more (by-instr run-move)
  (more (by-instr run-drop-w)
  (more (by-instr run-drop-x) done))))))

------------------------------------------------------------------------
-- ⭐ THE INITIALISATION WINDOW, read off this run.
--
-- `sWindow` is the state between `alloc` and `init`: `x` owns the storage, one
-- site holds it, and the object table is empty. Every one of these is computed
-- from the state the step relation produced, not written down.

window-owns-the-storage : entityOf (env sWindow) x ≡ just theObj
window-owns-the-storage = refl

window-has-one-owner : logicalRC (sites (mach sWindow)) theObj ≡ 1
window-has-one-owner = refl

window-has-no-object : countOf (mach sWindow) theObj ≡ nothing
window-has-no-object = refl

window-has-no-life : lifeOf (mach sWindow) theObj ≡ nothing
window-has-no-life = refl

-- ⭐ THE IR THE COMPILER USED TO EMIT: the retain hoisted to the handle's
-- definition, which is inside the window. Same environment, same machine, one
-- instruction reordered.
sHoisted : PState
sHoisted = pstate 0 bb0 (dup y x ∷ init x ∷ borrow z y ∷ move w y ∷ drop w ∷ drop x ∷ [])
                  (env sWindow) (mach sWindow)

-- and it has NO STEP. Not "a step that is unsound", not "a step whose result
-- fails the invariant" -- the relation has no derivation, so a compiler that
-- emits this is emitting an operation the semantics does not have. That is the
-- `Ly_IncRef observed non-positive refcount` crash as an absence.
hoisted-retain-has-no-step : ¬ (Σ PState λ u → prog ⊢ sHoisted —→ᵢ u)
hoisted-retain-has-no-step = no-dup-in-the-initialisation-window refl refl

-- The same retain one instruction later is fine, and `run-dup` above is the
-- derivation. Stated here so the pair reads as "too early" rather than
-- "forbidden".
same-retain-after-init : prog ⊢ s₁ —→ᵢ s₂
same-retain-after-init = run-dup

------------------------------------------------------------------------
-- Both counts, at every point.
--
-- The runtime counter and the ghost count are read separately and compared. An
-- implementation that moved one without the other satisfies one column and not
-- the other, and that is the entire class of defect this layer exists to catch.

counter : ∀ (s : PState) → Maybe RuntimeCount
counter s = countOf (mach s) theObj

ghost : ∀ (s : PState) → ℕ
ghost s = logicalRC (sites (mach s)) theObj

counters : (counter s₁ ≡ just (counted 1))
         × (counter s₂ ≡ just (counted 2))
         × (counter s₃ ≡ just (counted 2))
         × (counter s₄ ≡ just (counted 2))
         × (counter s₅ ≡ just (counted 1))
         × (counter s₆ ≡ just (counted 0))
counters = refl , refl , refl , refl , refl , refl

ghosts : (ghost s₁ ≡ 1) × (ghost s₂ ≡ 2) × (ghost s₃ ≡ 2)
       × (ghost s₄ ≡ 2) × (ghost s₅ ≡ 1) × (ghost s₆ ≡ 0)
ghosts = refl , refl , refl , refl , refl , refl

-- ⭐ Stated as the agreement itself, so that it is one equation per state rather
-- than two facts a reader has to compare. This is `WFRC.counted-exact` at
-- `theObj`, checked by computation along a run of the step relation.
counts-agree : (counter s₁ ≡ just (counted (ghost s₁)))
             × (counter s₂ ≡ just (counted (ghost s₂)))
             × (counter s₃ ≡ just (counted (ghost s₃)))
             × (counter s₄ ≡ just (counted (ghost s₄)))
             × (counter s₅ ≡ just (counted (ghost s₅)))
             × (counter s₆ ≡ just (counted (ghost s₆)))
counts-agree = refl , refl , refl , refl , refl , refl

------------------------------------------------------------------------
-- What each rule cost, read off the trace.

-- `dup` = py.incref: both counts up by one.
dup-costs-one : (counter s₁ ≡ just (counted 1)) × (counter s₂ ≡ just (counted 2))
dup-costs-one = refl , refl

-- `borrow` costs NOTHING. Neither count moves, and the machine is untouched --
-- which is what makes eliding a retain here correct rather than merely cheaper.
borrow-costs-nothing : mach s₃ ≡ mach s₂
borrow-costs-nothing = refl

-- ⭐ `move` costs nothing either, and this is the row where a pass most easily
-- gets it wrong: the SITE changed (`local 0 y` gave way to `local 0 w`) while
-- the count did not. A pass that emitted a release for the vanishing name
-- breaks the counter; one that emitted a retain for the new one breaks it the
-- other way.
move-changes-the-site : sites (mach s₄) ≡ (local 0 w , theObj) ∷ (local 0 x , theObj) ∷ []
move-changes-the-site = refl

move-keeps-both-counts : (counter s₄ ≡ counter s₃) × (ghost s₄ ≡ ghost s₃)
move-keeps-both-counts = refl , refl

-- The last `drop` takes the object to `finalizing`, not to `dead`, and leaves
-- the storage alone. Splitting those is what gives a finalizer somewhere to run.
last-drop-finalizes : lifeOf (mach s₆) theObj ≡ just finalizing
last-drop-finalizes = refl

-- ⭐ `ReachedZero`, read off the run rather than asserted of a literal. It is
-- the event a decref has to test for, and the point of tying it to `countAt` is
-- that `hit-zero` alone proves nothing about any program.
countAt : PState → RuntimeCount
countAt s with countOf (mach s) theObj
... | just c  = c
... | nothing = immortal

reaches-zero-at-the-end : ReachedZero (countAt s₆)
reaches-zero-at-the-end = hit-zero

-- and not one instruction earlier. Without this the predicate would be
-- satisfied by every state and the test would be no test.
not-zero-one-step-before : ¬ ReachedZero (countAt s₅)
not-zero-one-step-before ()

heap-untouched : heap (mach s₆) ≡ heap (mach s₀)
heap-untouched = refl

------------------------------------------------------------------------
-- Owned NAMES against owner SITES.
--
-- These are different numbers and the model keeps them apart. `borrow` is where
-- they come apart on purpose: after it, three names denote the object and only
-- two of them own it.

names-vs-sites : (ownedCount (env s₃) theObj ≡ 2) × (ghost s₃ ≡ 2)
names-vs-sites = refl , refl

-- and the borrowed name is genuinely there, just not counted.
borrowed-name-exists : entityOf (env s₃) z ≡ just theObj
borrowed-name-exists = refl

------------------------------------------------------------------------
-- ⭐ The dangling borrow the run produces.
--
-- `z` borrows from `y`; `move w y` removes `y`. From `s₄` onwards the borrow is
-- dangling, and the checker finds it. The environment it is found in was
-- produced by `step-move`, not written by hand -- which is the difference
-- between "the checker works on an example" and "the step relation can reach a
-- state the checker rejects".

borrow-is-fine-before-the-move : danglingAnchor (env s₃) z ≡ nothing
borrow-is-fine-before-the-move = refl

move-strands-the-borrow : danglingAnchor (env s₄) z ≡ just y
move-strands-the-borrow = refl

still-stranded-at-the-end : danglingAnchor (env s₆) z ≡ just y
still-stranded-at-the-end = refl

-- The step relation does NOT stop this. That is the finding, not an oversight
-- in the example: `step-move` has no premise about names borrowed from its
-- source, so a program can move out from under a live borrow and keep running.
-- Where the check has to go is what `drop-strands-its-borrows` in
-- Proof.Lython.Detect says -- at the point the anchor dies -- and `move` is a
-- second such point that the rule as written does not guard.

-- and it is a real `DanglingBorrow`, not just a report.
the-dangling-borrow : DanglingBorrow (env s₆) z
the-dangling-borrow = danglingAnchor-sound (env s₆) z y refl

------------------------------------------------------------------------
-- ⭐ Premature reclaim, and why the machine-level precondition is not enough.
--
-- At `s₆` the object is `finalizing` with NO owner site, which is exactly when
-- `Proof.RC.Ops.reclaim` hands the storage back. And `z` still denotes it.
--
-- The two are not in conflict by accident. A borrow is a NAME THAT HOLDS NO
-- SITE -- that is the whole content of "a borrow costs nothing" -- so the ghost
-- count reaching zero does not mean nothing refers to the object. The refcount
-- layer's precondition is about sites; the safety property is about names; and
-- this run is a state where they come apart.
--
-- This is the `PrematureReclaim` family, and it is the shape of the compiler's
-- open defects in that class: storage handed back while something still points
-- at it, with every count agreeing that it was safe.

reclaimedMach : Machine
reclaimedMach = machine (heap (mach s₆))
                        ((theObj , cell dead (counted 0) bk 0) ∷ [])
                        (sites (mach s₆))

-- The machine-level operation ACCEPTS this. Not a hypothetical: `reclaim`'s own
-- preconditions are met.
reclaim-is-licensed : reclaim (mach s₆) theObj ≡ ok (bk , reclaimedMach)
reclaim-is-licensed = refl

-- and yet a name still denotes the object.
z-still-names-it : StillNamed (env s₆) theObj
z-still-names-it = named-by z refl

the-premature-reclaim : PrematureReclaim (env s₆) reclaimedMach theObj
the-premature-reclaim = premature z-still-names-it refl

-- The checker sees it, and is silent one step earlier -- while the object is
-- `finalizing`, freeing has not happened and there is nothing to report.
reclaim-checker-fires : prematureReclaim? (env s₆) reclaimedMach theObj ≡ just z
reclaim-checker-fires = refl

reclaim-checker-silent-before : prematureReclaim? (env s₆) (mach s₆) theObj ≡ nothing
reclaim-checker-silent-before = refl

-- and it is not "report everything": with no name left, nothing is reported
-- even though the object is dead.
emptyEnv : Env
emptyEnv = []

reclaim-checker-ignores-unnamed : prematureReclaim? emptyEnv reclaimedMach theObj ≡ nothing
reclaim-checker-ignores-unnamed = refl

------------------------------------------------------------------------
-- ⭐ `Invalidity` inhabited, twice, on states this run reaches.
--
-- The top-level predicate had never been built. Both constructors that apply to
-- a sequential program are exercised here, and both witnesses come from the
-- same six-instruction block.

invalid-by-dangling-borrow : Invalidity (env s₆) (mach s₆)
invalid-by-dangling-borrow = dangling-borrow the-dangling-borrow

invalid-by-premature-reclaim : Invalidity (env s₆) reclaimedMach
invalid-by-premature-reclaim = premature-reclaim the-premature-reclaim

------------------------------------------------------------------------
-- ⭐ The invariant, carried along the run by the preservation theorem.
--
-- `WF s₀` is entirely vacuous -- the machine is empty, so every field has a
-- hypothesis nothing meets. `WF s₆` is not: the machine has an object, a
-- counter and a site map, and every field is a real constraint on them.
--
-- Nothing between the two is written down. The six states are related by
-- `step-preserves-WF` alone, which is what makes this a use of the theorem
-- rather than a restatement of the witnesses.

start-is-well-formed : WF s₀
start-is-well-formed = wfs
  (record
    { counted-exact      = λ p n cnt _ → ⊥-elim (no-just cnt)
    ; live-positive      = λ p lv _ → ⊥-elim (no-just lv)
    ; dead-unowned       = λ p dd → ⊥-elim (no-just dd)
    ; no-stale-owner     = λ s p ()
    ; owned-storage-live = λ p ()
    })
  (λ y p h → ⊥-elim (no-just h))
  where
    no-just : ∀ {A : Set} {v : A} → nothing ≡ just v → ⊥
    no-just ()

-- Over the ordinary reachability relation, not a special instruction-only one.
-- That distinction existed because terminators did not preserve the invariant;
-- with a block argument being a move, they do.
the-run : prog ⊢ s₀ —→* s₆
the-run = whole-block

-- ⭐ Six instruction steps later, the invariant still holds.
end-is-well-formed : WF s₆
end-is-well-formed = reachable-preserves-WF the-run start-is-well-formed

-- and it is not vacuous there: the object is in the table with a real counter,
-- so `counted-exact` has a hypothesis to meet.
end-has-a-real-object : countOf (mach s₆) theObj ≡ just (counted 0)
end-has-a-real-object = refl

-- ⭐ The window is a LEGITIMATE state, not an error state.
--
-- This is the half that keeps `no-dup-in-the-initialisation-window` from being
-- a statement that the window is broken. `WFRC` holds at `sWindow`: the storage
-- is live, the site is not stale, and every field about a counter is vacuous
-- because there is no cell. A model that made the window ill-formed would be
-- saying the compiler must not allocate before it initialises, which is not the
-- finding -- the finding is that it must not INCREF in between.
window-is-well-formed : WF sWindow
window-is-well-formed =
  reachable-preserves-WF (more (by-instr run-alloc) done) start-is-well-formed

-- The invariant at a state where it says something. At `s₂` the counter reads 2
-- and two sites hold the object; `counted-exact` there is the statement that
-- those are the same number, obtained from the theorem rather than by hand.
midpoint-is-well-formed : WF s₂
midpoint-is-well-formed =
  reachable-preserves-WF
    (more (by-instr run-alloc) (more (by-instr run-init) (more (by-instr run-dup) done)))
    start-is-well-formed

midpoint-counted-exact : 2 ≡ ghostRC (mach s₂) theObj
midpoint-counted-exact =
  counted-exact (rc midpoint-is-well-formed) theObj 2 refl refl
