{-# OPTIONS --safe #-}

-- ⭐ What the IR RECORDS, against what is TRUE.
--
-- The model has had exactly one notion of ownership: `Binding.mode`, which is
-- the truth by construction. The compiler has two, and the defect this module
-- exists for is them disagreeing.
--
-- A lowering pass cannot read the semantics. It reads ATTRIBUTES --
-- `ly.ownership.owned_local_object` and friends -- and decides from them. So
-- `boxRuntimeObject` (ABI/RuntimeABI.cpp) emitting the payload retain WITHOUT
-- marking the box owned was not a missing retain: the retain was there. It was
-- a missing RECORD, and the pass that later read the box as borrowed was
-- reading correctly from a wrong ledger.
--
-- The failure had a second layer that the model makes visible and prose did
-- not. `isOwnedIncoming` (Passes/Ownership.cpp) defaults to "not owned" when no
-- attribute is present, so an unrecorded value reads as a BORROW, and a borrow
-- crossing a block-argument edge needs its own reference -- so the pass tried
-- to synthesise a retain. The edge was a move. The retain was for nothing, and
-- the tree carried a silent `continue` for the cases where it could not even be
-- written.
--
-- `edgeRetain` below is that decision, and its TYPE is the finding: it takes a
-- `Maybe Mode` -- what the ledger says -- and no environment. A pass cannot
-- consult the truth. Every theorem here is therefore about the gap between two
-- things the compiler really does have, rather than about a mistake it could
-- have avoided by looking harder.

open import Proof.Memory.Element using (ElemSig)

module Proof.Program.Recorded (Sig : ElemSig) where

open ElemSig Sig

open import Data.Bool using (Bool; true; false; if_then_else_)
open import Data.Empty using (⊥; ⊥-elim)
open import Data.List using (List; []; _∷_)
open import Data.Maybe using (Maybe; just; nothing)
import Data.Maybe
open import Data.Nat using (ℕ; zero; suc)
open import Data.Product using (_×_; _,_; Σ; proj₁; proj₂)
open import Relation.Binary.PropositionalEquality
  using (_≡_; _≢_; refl; sym; trans; cong; subst)
open import Relation.Nullary using (¬_)

open import Proof.RC.Object using (ObjId; Life; live; RuntimeCount; counted;
  bumpUp; bumpDown)
open import Proof.RC.OwnerSite using (ThreadId)
open import Proof.RC.Machine Sig
open import Proof.RC.Properties Sig using (lookupObj-update-same)
open import Proof.RC.Invariant Sig using (WFRC; counted-exact)
open import Proof.Program.Syntax using (Var; BlockId; Function; Instr; dup; drop)
open import Proof.Program.Env
open import Proof.Program.Step Sig

------------------------------------------------------------------------
-- The ledger.
--
-- Same shape as an `Env`, deliberately: the point is that the two are separate
-- objects of the same kind which a compiler has to keep in step, not that one
-- is impoverished. `boxRuntimeObject` could have written the entry and did not.

Attrs : Set
Attrs = List (Var × Mode)

recordedMode : Attrs → Var → Maybe Mode
recordedMode []              _ = nothing
recordedMode ((y , md) ∷ as) x = if sameVar y x then just md else recordedMode as x

-- The two equations, carried explicitly, for the same reason `Proof.Program.Env`
-- carries its four: a `rewrite` on the boolean reduces the goal and leaves any
-- hypothesis mentioning the same `if` unreduced, so the two no longer meet.
recordedMode-cons-true : ∀ u md (as : Attrs) y → sameVar u y ≡ true →
                         recordedMode ((u , md) ∷ as) y ≡ just md
recordedMode-cons-true u md as y e rewrite e = refl

recordedMode-cons-false : ∀ u md (as : Attrs) y → sameVar u y ≡ false →
                          recordedMode ((u , md) ∷ as) y ≡ recordedMode as y
recordedMode-cons-false u md as y e rewrite e = refl

-- Every name the environment binds is recorded, with the mode it actually has.
--
-- Stated FROM the environment rather than from the ledger, because that is the
-- direction the compiler can get wrong by omission: a value nobody wrote an
-- attribute for is unrecorded, and unrecorded is where the defect lived. A
-- ledger with a spurious extra entry is a different (and unobserved) fault.
Faithful : Attrs → Env → Set
Faithful as es = ∀ x b → lookupVar es x ≡ just b → recordedMode as x ≡ just (mode b)

------------------------------------------------------------------------
-- The pass.
--
-- ⭐ `isOwnedIncoming`, as a function. It sees a `Maybe Mode` and nothing else.
--
-- `nothing` emitting a retain is not a modelling choice: it is what the
-- compiler does. Absence of the attribute is read as "not owned", so an
-- unrecorded value takes the borrow path.
edgeRetain : Maybe Mode → Var → Var → List Instr
edgeRetain (just owned)        p a = []
edgeRetain (just (borrowed _)) p a = dup p a ∷ []
edgeRetain nothing             p a = dup p a ∷ []

------------------------------------------------------------------------
-- What a faithful ledger buys.

-- ⭐ On a faithful ledger the pass emits nothing for an owned operand, which is
-- the correct answer: `Proof.Program.Preservation.block-arguments-are-free` says
-- a terminator touches no counter, so the number of runtime operations a block
-- argument requires is zero.
faithful-edge-is-free :
  ∀ (as : Attrs) (es : Env) (p a : Var) (o : ObjId) →
  Faithful as es → lookupVar es a ≡ just (bind o owned) →
  edgeRetain (recordedMode as a) p a ≡ []
faithful-edge-is-free as es p a o faith look
  rewrite faith a (bind o owned) look = refl

-- and it emits the retain for a genuine borrow, which is also correct: a borrow
-- holds no owner site, so the parameter it feeds would own nothing.
faithful-edge-retains-a-borrow :
  ∀ (as : Attrs) (es : Env) (p a v : Var) (o : ObjId) →
  Faithful as es → lookupVar es a ≡ just (bind o (borrowed v)) →
  edgeRetain (recordedMode as a) p a ≡ dup p a ∷ []
faithful-edge-retains-a-borrow as es p a v o faith look
  rewrite faith a (bind o (borrowed v)) look = refl

------------------------------------------------------------------------
-- ⭐ THE DEFECT, in both directions.

-- Direction one: OWNERSHIP TAKEN AND NOT RECORDED.
--
-- The box is owned; the ledger has no entry; the pass takes the borrow path and
-- emits a retain. This is `boxRuntimeObject` exactly.
unrecorded-ownership-emits-a-retain :
  ∀ (as : Attrs) (p a : Var) →
  recordedMode as a ≡ nothing →
  edgeRetain (recordedMode as a) p a ≡ dup p a ∷ []
unrecorded-ownership-emits-a-retain as p a silent rewrite silent = refl

-- and the retain is not free. It bumps the counter, while the faithful pass
-- leaves it alone -- so the two ledgers produce runs whose counters differ by
-- exactly one, with nothing downstream to release the difference.
--
-- Stated with the step ATTACHED rather than about `updateObj` directly: the
-- claim is about what running the emitted instruction does, and a lemma about
-- the table alone would hold of an instruction the relation never takes.
the-unrecorded-retain-bumps-the-counter :
  ∀ {f : Function} (t : ThreadId) (bid : BlockId) (rest : List Instr)
    (es : Env) (m : Machine) (p a : Var) (o : ObjId) (c : ObjCell) →
  lookupVar es a ≡ just (bind o owned) →
  lookupObj (objects m) o ≡ just c →
  life c ≡ live →
  Σ PState λ u → (f ⊢ pstate t bid (dup p a ∷ rest) es m —→ᵢ u)
               × (countOf (mach u) o ≡ just (bumpUp (count c)))
the-unrecorded-retain-bumps-the-counter t bid rest es m p a o c look tbl alive =
  _ , step-dup look tbl alive
    , cong (Data.Maybe.map count)
           (lookupObj-update-same (objects m) o (λ y → record y { count = bumpUp (count y) })
                                  c tbl)

-- Direction two: A BORROW RECORDED AS OWNED.
--
-- The pass emits nothing for the edge, so the parameter inherits the borrow and
-- owns no site -- and the release the pass will later place for it HAS NO STEP.
-- Not "is unsound", not "double-frees": the relation has no derivation, so a
-- compiler that emits it is emitting an operation the semantics does not have.
--
-- This is the direction that would have been introduced by "fixing" the first
-- one with a blanket `owned` attribute, and it is why the repair recorded the
-- ownership the boxing path actually takes rather than asserting ownership
-- everywhere the pass wanted it.
no-drop-of-a-borrow :
  ∀ {f t bid rest es m x o v} →
  lookupVar es x ≡ just (bind o (borrowed v)) →
  ¬ (Σ PState λ u → f ⊢ pstate t bid (drop x ∷ rest) es m —→ᵢ u)
no-drop-of-a-borrow look (_ , step-drop look′ _ _ _) with trans (sym look) look′
... | ()

-- ⛔ NOT stated here: "the pass cannot tell the two apart". It was, as
-- `recordedMode as a ≡ just md₁ → recordedMode as a ≡ just md₂ →
--  edgeRetain (just md₁) p a ≡ edgeRetain (just md₂) p a`, and that is `refl`
-- with extra steps -- the two hypotheses give `md₁ ≡ md₂` immediately, so the
-- lemma restates the congruence of a function rather than saying anything about
-- ledgers. The real content is that one ledger admits two truths, and it needs
-- the two environments exhibited; `Proof.Program.Run` does that on a state the
-- step relation produced.
--
-- The general half needs no lemma at all: it is `edgeRetain`'s TYPE. A function
-- of `Maybe Mode` cannot consult an `Env`, so no care inside the pass reaches
-- this defect and the obligation is on whoever writes the ledger.

------------------------------------------------------------------------
-- Faithfulness is not free either.
--
-- The obligation `boxRuntimeObject` failed is exactly this: after an operation
-- that takes ownership, extend the ledger. Written as a lemma so that "record
-- what you took" is an operation with a proof rather than a habit.

record-owned : Attrs → Var → ObjId → Attrs
record-owned as x o = (x , owned) ∷ as

recording-is-faithful :
  ∀ (as : Attrs) (es : Env) (x : Var) (o : ObjId) →
  Faithful as es →
  Faithful (record-owned as x o) (bindVar es x (bind o owned))
recording-is-faithful as es x o faith y b h = go (sameVar x y) refl
  where
    go : (bb : Bool) → sameVar x y ≡ bb →
         recordedMode (record-owned as x o) y ≡ just (mode b)
    just-inj : ∀ {A : Set} {u v : A} → just u ≡ just v → u ≡ v
    just-inj refl = refl

    go true e =
      trans (recordedMode-cons-true x owned as y e)
            (cong just (cong mode (just-inj
              (trans (sym (lookupVar-cons-true x (bind o owned) es y e)) h))))
    go false e =
      trans (recordedMode-cons-false x owned as y e)
            (faith y b (trans (sym (lookupVar-cons-false x (bind o owned) es y e)) h))

-- ⛔ and NOT recording is exactly what breaks it. The binding is owned and the
-- ledger is untouched, so a lookup that used to answer `nothing` still does.
--
-- The hypothesis is about the ledger BEFORE the binding, which is the point:
-- the ledger cannot become faithful by itself, so an operation that takes
-- ownership and returns without writing leaves the state unfaithful no matter
-- what the rest of the compiler does.
not-recording-breaks-faithfulness :
  ∀ (as : Attrs) (es : Env) (x : Var) (o : ObjId) →
  recordedMode as x ≡ nothing →
  ¬ Faithful as (bindVar es x (bind o owned))
not-recording-breaks-faithfulness as es x o silent faith
  with trans (sym silent)
             (faith x (bind o owned)
                    (lookupVar-cons-true x (bind o owned) es x (sameVar-refl x)))
... | ()

------------------------------------------------------------------------
-- ⭐ THE RUNTIME HALF OF `dup`, ALONE.
--
-- `step-dup` does three things at once: bump the counter, occupy an owner site,
-- bind a name. That atomicity is why a REDUNDANT `dup` is harmless here --
-- `Dup.preserves` and `instr-preserves-coherence` both go through -- and it is
-- why this model said an extra retain was SAFE while a measured, unbounded leak
-- in the compiler said otherwise.
--
-- The compiler can emit the counter bump alone: `Ly_IncRef` on a value is one
-- call, and nothing makes it arrive with a site and a name. Two such calls landed
-- on one value and one release came back, which is the leak. So the operation the
-- compiler actually has is SMALLER than any rule in this development, and the gap
-- between them is where the defect lived.
--
-- ⛔ Modelled as a FUNCTION on machines, not as a step rule, and the choice is
-- the point. A rule would have to be carved out of `step-preserves-WF`, because
-- it cannot preserve `WFRC` -- that being the whole content -- and weakening a
-- totality theorem to accommodate one unsound operation buys less than naming the
-- operation and proving exactly what it breaks. `Proof.Program.Step` stays the
-- relation of operations the semantics HAS; this is the one it does not.

private
  suc-inj : ∀ {a b : ℕ} → suc a ≡ suc b → a ≡ b
  suc-inj refl = refl

  suc≢self : ∀ (n : ℕ) → suc n ≢ n
  suc≢self zero    ()
  suc≢self (suc n) e = suc≢self n (suc-inj e)

  -- `countOf`/`lifeOf` answering `just` means the cell is there. Needed because
  -- the statements below are phrased over the counters a compiler can observe,
  -- not over the table.
  cellBehind : ∀ {A : Set} (x : Maybe ObjCell) (f : ObjCell → A) {y : A} →
               Data.Maybe.map f x ≡ just y → Σ ObjCell λ c → x ≡ just c
  cellBehind (just c) f _ = c , refl
  cellBehind nothing  f ()

  just-inj′ : ∀ {A : Set} {u v : A} → just u ≡ just v → u ≡ v
  just-inj′ refl = refl

-- A retain with no site and no name: the counter moves, the ghost state does not.
retainWithoutASite : Machine → ObjId → Machine
retainWithoutASite m o =
  machine (heap m)
          (updateObj (objects m) o (λ c → record c { count = bumpUp (count c) }))
          (sites m)

-- and its mirror, so the pair says "neither half may be emitted alone".
releaseWithoutVacating : Machine → ObjId → Machine
releaseWithoutVacating m o =
  machine (heap m)
          (updateObj (objects m) o (λ c → record c { count = bumpDown (count c) }))
          (sites m)

-- ⭐ A RETAIN THAT OCCUPIES NO SITE BREAKS THE COUNT INVARIANT.
--
-- This is the sentence the model was missing. `the-unrecorded-retain-bumps-the-
-- counter` above gets the arithmetic right -- the counter goes up by one -- and
-- the consequence wrong, because it models the emission as a real `step-dup`,
-- which brings a site along and so lands in a well-formed state. Here nothing
-- comes along, and the state is not well-formed at all.
retain-without-a-site-breaks-WFRC :
  ∀ (m : Machine) (o : ObjId) (n : ℕ) →
  countOf m o ≡ just (counted n) → lifeOf m o ≡ just live →
  WFRC m → ¬ WFRC (retainWithoutASite m o)
retain-without-a-site-breaks-WFRC m o n cnt lif wf broken =
  suc≢self n (trans after (sym prior))
  where
    up : ObjCell → ObjCell
    up c = record c { count = bumpUp (count c) }

    found : Σ ObjCell λ c → lookupObj (objects m) o ≡ just c
    found = cellBehind (lookupObj (objects m) o) count cnt

    tbl : lookupObj (objects m) o ≡ just (proj₁ found)
    tbl = proj₂ found

    counted-n : count (proj₁ found) ≡ counted n
    counted-n = just-inj′ (trans (sym (cong (Data.Maybe.map count) tbl)) cnt)

    live-cell : life (proj₁ found) ≡ live
    live-cell = just-inj′ (trans (sym (cong (Data.Maybe.map life) tbl)) lif)

    hit : lookupObj (objects (retainWithoutASite m o)) o ≡ just (up (proj₁ found))
    hit = lookupObj-update-same (objects m) o up (proj₁ found) tbl

    -- The counter now reads one more, and the ghost count is untouched, so
    -- `counted-exact` at this object says `suc n ≡ n`.
    after : suc n ≡ ghostRC m o
    after = counted-exact broken o (suc n)
              (trans (cong (Data.Maybe.map count) hit)
                     (cong (λ r → just (bumpUp r)) counted-n))
              (trans (cong (Data.Maybe.map life) hit) (cong just live-cell))

    prior : n ≡ ghostRC m o
    prior = counted-exact wf o n cnt lif

-- ⭐ And a release that vacates no site breaks it the other way.
--
-- Stated at `suc n` because `bumpDown (counted zero)` stays at zero: a decrement
-- of a counter already at zero changes nothing, so there is nothing to break, and
-- a statement quantified over every counter would be false for that one.
release-without-vacating-breaks-WFRC :
  ∀ (m : Machine) (o : ObjId) (n : ℕ) →
  countOf m o ≡ just (counted (suc n)) → lifeOf m o ≡ just live →
  WFRC m → ¬ WFRC (releaseWithoutVacating m o)
release-without-vacating-breaks-WFRC m o n cnt lif wf broken =
  suc≢self n (trans prior (sym after))
  where
    down : ObjCell → ObjCell
    down c = record c { count = bumpDown (count c) }

    found : Σ ObjCell λ c → lookupObj (objects m) o ≡ just c
    found = cellBehind (lookupObj (objects m) o) count cnt

    tbl : lookupObj (objects m) o ≡ just (proj₁ found)
    tbl = proj₂ found

    counted-n : count (proj₁ found) ≡ counted (suc n)
    counted-n = just-inj′ (trans (sym (cong (Data.Maybe.map count) tbl)) cnt)

    live-cell : life (proj₁ found) ≡ live
    live-cell = just-inj′ (trans (sym (cong (Data.Maybe.map life) tbl)) lif)

    hit : lookupObj (objects (releaseWithoutVacating m o)) o
            ≡ just (down (proj₁ found))
    hit = lookupObj-update-same (objects m) o down (proj₁ found) tbl

    after : n ≡ ghostRC m o
    after = counted-exact broken o n
              (trans (cong (Data.Maybe.map count) hit)
                     (cong (λ r → just (bumpDown r)) counted-n))
              (trans (cong (Data.Maybe.map life) hit) (cong just live-cell))

    prior : suc n ≡ ghostRC m o
    prior = counted-exact wf o (suc n) cnt lif

-- ⛔ What this does NOT say, and it matters for reading the compiler against it.
--
-- It does not say the compiler's extra retain is unsound in general. An IMMORTAL
-- object's counter is not a count -- `bumpUp immortal ≡ immortal` -- so a bare
-- retain on one changes nothing and breaks nothing, and `counted-exact` is
-- guarded on `counted n` precisely to exempt it. Both statements above therefore
-- take a `counted` premise, and neither reaches an immortal.
--
-- That exemption is not decidable in the compiler, because immortality there is a
-- runtime value (a refcount of INT64_MAX, fixed at creation) rather than a static
-- property. So the phase gate that enforces the site side
-- (`verifyOwnedTokenUniqueness`) is stricter than these theorems: it rejects a
-- second token on an immortal too. That is the right direction -- the IR is
-- malformed either way -- but the difference is real and is why the gate cannot
-- be justified by these lemmas alone.
