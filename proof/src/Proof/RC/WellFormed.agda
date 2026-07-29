{-# OPTIONS --safe #-}

-- Machines that SATISFY the invariant.
--
-- `WFRC` was a record nobody had built. That is not the same as "not yet
-- proved": a record nothing inhabits is a predicate that could be unsatisfiable,
-- and every theorem of the form `WFRC m → X` is then vacuously true. The
-- preservation lemmas in Proof.Program.Preservation are all of that form, so
-- until this module existed they carried no information about any machine.
--
-- What is built here is the whole reference-counting trace of Proof.RC.Trace --
-- allocate, incref, decref, decref-to-zero, reclaim -- with a witness at every
-- point. The invariant is therefore not merely satisfiable but MAINTAINED across
-- the five operations the layer defines.
--
-- Which fields are non-vacuous where is recorded at the bottom, because a
-- witness whose every field is vacuous is exactly the thing this module exists
-- to stop being mistaken for a result.

module Proof.RC.WellFormed where

open import Data.Bool using (Bool; true; false)
open import Data.Empty using (⊥; ⊥-elim)
open import Data.List using (List; []; _∷_)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc; _<_; s≤s; z≤n)
open import Data.Product using (_×_; _,_; Σ; proj₁; proj₂)
open import Data.Vec using (Vec; [])
open import Relation.Binary.PropositionalEquality
  using (_≡_; refl; sym; trans; cong; subst)
open import Relation.Nullary using (¬_)

open import Proof.Prelude using (Result; ok)
open import Proof.Memory.Heap using (Heap; Block; block; lookupBlock; generation;
  liveness; Storage; heapAlloc)
  renaming (live to blockLive)
open import Proof.Memory.Lython using (LythonSig)
open import Proof.RC.Object
open import Proof.RC.OwnerSite
open import Proof.RC.Machine LythonSig
open import Proof.RC.Invariant LythonSig
open import Proof.RC.Trace

private
  just-inj : ∀ {A : Set} {x y : A} → just x ≡ just y → x ≡ y
  just-inj refl = refl

  counted-inj : ∀ {a b : ℕ} → counted a ≡ counted b → a ≡ b
  counted-inj refl = refl

-- Every proof below is a case split on `sameObj theObj o` or `sameSite s t`.
-- With-abstraction does the rest: because those are BOOLEANS rather than
-- decision procedures, the occurrence buried inside `lookupObj` / `logicalRC` /
-- `strongAt` is literally the same term as the scrutinee, so abstracting it
-- reduces the hypotheses AND the goal at once. With `_≟-obj_` each of those
-- would sit under a different auxiliary function and none of this would compute
-- -- which is the reason recorded in Proof.RC.Machine for the choice.

------------------------------------------------------------------------
-- The heap under all five machines.
--
-- None of the reference-counting operations touches the heap, so one block
-- serves throughout. Obtained by evaluation rather than written out, so that a
-- change to `alloc` cannot leave a hand-written block silently disagreeing with
-- what allocation actually produces.

theBlock : Block
theBlock with lookupBlock (heap m₀) (objAllocation theObj)
... | just b  = b
... | nothing = block 0 0 0 0 [] blockLive heapAlloc

block-is-there : lookupBlock (heap m₀) (objAllocation theObj) ≡ just theBlock
block-is-there = refl

block-generation-matches : generation theBlock ≡ objGeneration theObj
block-generation-matches = refl

block-is-live : liveness theBlock ≡ blockLive
block-is-live = refl

private
  -- The two heap obligations, transported along `theObj ≡ o`. Both fields of
  -- WFRC that mention the heap need exactly this and nothing else, because
  -- `theObj` is the only object in play.
  storage-there : ∀ (o : ObjId) → theObj ≡ o →
                  Σ Block λ b → (lookupBlock (heap m₀) (objAllocation o) ≡ just b)
                              × (generation b ≡ objGeneration o)
  storage-there o refl = theBlock , refl , refl

  storage-live : ∀ (o : ObjId) → theObj ≡ o →
                 Σ Block λ b → (lookupBlock (heap m₀) (objAllocation o) ≡ just b)
                             × (liveness b ≡ blockLive)
  storage-live o refl = theBlock , refl , refl

------------------------------------------------------------------------
-- m₀: freshly allocated. One owner site, counter 1.

wf₀ : WFRC m₀
wf₀ = record
  { counted-exact      = ce
  ; live-positive      = lp
  ; dead-unowned       = du
  ; no-stale-owner     = nso
  ; owned-storage-live = osl
  }
  where
    ce : ∀ o n → countOf m₀ o ≡ just (counted n) → lifeOf m₀ o ≡ just live →
         n ≡ ghostRC m₀ o
    ce o n cnt _ with sameObj theObj o
    ... | true  = sym (counted-inj (just-inj cnt))
    ... | false with cnt
    ...   | ()

    lp : ∀ o → lifeOf m₀ o ≡ just live → IsCounted m₀ o → 0 < ghostRC m₀ o
    lp o lv _ with sameObj theObj o
    ... | true  = s≤s z≤n
    ... | false with lv
    ...   | ()

    -- Vacuous here: nothing in m₀ is dead. Discharged with a real hypothesis by
    -- `wfᵣ` below.
    du : ∀ o → lifeOf m₀ o ≡ just dead → ghostRC m₀ o ≡ 0
    du o dd with sameObj theObj o
    ... | true  with dd
    ...   | ()
    du o dd | false with dd
    ...   | ()

    -- Over `Holds`, walking a one-entry site map is one constructor pattern.
    -- The field moved from `strongAt` to membership so that `drop` could
    -- preserve it; the witnesses got shorter as a side effect.
    nso : ∀ s o → Holds (sites m₀) s o →
          Σ Block λ b → (lookupBlock (heap m₀) (objAllocation o) ≡ just b)
                      × (generation b ≡ objGeneration o)
    nso _ _ holds-here               = storage-there _ refl
    nso _ _ (holds-there ())

    osl : ∀ o → 0 < ghostRC m₀ o →
          Σ Block λ b → (lookupBlock (heap m₀) (objAllocation o) ≡ just b)
                      × (liveness b ≡ blockLive)
    osl o pos with sameObj theObj o in eq
    ... | true  = storage-live o (sameObj-sound theObj o eq)
    ... | false with pos
    ...   | ()

------------------------------------------------------------------------
-- m₁ = retain m₀ into site₂: two owner sites, counter 2.
--
-- This is the witness that matters most, because it is the one where the two
-- numbers could disagree and do not. `counted-exact` here is the statement that
-- py.incref moved BOTH the counter and the ghost count -- an implementation that
-- bumped only the counter fails this field and nothing else in the development
-- would notice.

wf₁ : WFRC m₁
wf₁ = record
  { counted-exact      = ce
  ; live-positive      = lp
  ; dead-unowned       = du
  ; no-stale-owner     = nso
  ; owned-storage-live = osl
  }
  where
    ce : ∀ o n → countOf m₁ o ≡ just (counted n) → lifeOf m₁ o ≡ just live →
         n ≡ ghostRC m₁ o
    ce o n cnt _ with sameObj theObj o
    ... | true  = sym (counted-inj (just-inj cnt))
    ... | false with cnt
    ...   | ()

    lp : ∀ o → lifeOf m₁ o ≡ just live → IsCounted m₁ o → 0 < ghostRC m₁ o
    lp o lv _ with sameObj theObj o
    ... | true  = s≤s z≤n
    ... | false with lv
    ...   | ()

    du : ∀ o → lifeOf m₁ o ≡ just dead → ghostRC m₁ o ≡ 0
    du o dd with sameObj theObj o
    ... | true  with dd
    ...   | ()
    du o dd | false with dd
    ...   | ()

    -- Two sites, so the walk is one level deeper. Both hold `theObj`, which is
    -- what makes retaining into a second site legal at all.
    nso : ∀ s o → Holds (sites m₁) s o →
          Σ Block λ b → (lookupBlock (heap m₁) (objAllocation o) ≡ just b)
                      × (generation b ≡ objGeneration o)
    nso _ _ holds-here                             = storage-there _ refl
    nso _ _ (holds-there holds-here)               = storage-there _ refl
    nso _ _ (holds-there (holds-there ()))

    osl : ∀ o → 0 < ghostRC m₁ o →
          Σ Block λ b → (lookupBlock (heap m₁) (objAllocation o) ≡ just b)
                      × (liveness b ≡ blockLive)
    osl o pos with sameObj theObj o in eq
    ... | true  = storage-live o (sameObj-sound theObj o eq)
    ... | false with pos
    ...   | ()

------------------------------------------------------------------------
-- m₂ = release m₁ from site₂: back to one site, counter 1.

wf₂ : WFRC m₂
wf₂ = record
  { counted-exact      = ce
  ; live-positive      = lp
  ; dead-unowned       = du
  ; no-stale-owner     = nso
  ; owned-storage-live = osl
  }
  where
    ce : ∀ o n → countOf m₂ o ≡ just (counted n) → lifeOf m₂ o ≡ just live →
         n ≡ ghostRC m₂ o
    ce o n cnt _ with sameObj theObj o
    ... | true  = sym (counted-inj (just-inj cnt))
    ... | false with cnt
    ...   | ()

    lp : ∀ o → lifeOf m₂ o ≡ just live → IsCounted m₂ o → 0 < ghostRC m₂ o
    lp o lv _ with sameObj theObj o
    ... | true  = s≤s z≤n
    ... | false with lv
    ...   | ()

    du : ∀ o → lifeOf m₂ o ≡ just dead → ghostRC m₂ o ≡ 0
    du o dd with sameObj theObj o
    ... | true  with dd
    ...   | ()
    du o dd | false with dd
    ...   | ()

    nso : ∀ s o → Holds (sites m₂) s o →
          Σ Block λ b → (lookupBlock (heap m₂) (objAllocation o) ≡ just b)
                      × (generation b ≡ objGeneration o)
    nso _ _ holds-here      = storage-there _ refl
    nso _ _ (holds-there ())

    osl : ∀ o → 0 < ghostRC m₂ o →
          Σ Block λ b → (lookupBlock (heap m₂) (objAllocation o) ≡ just b)
                      × (liveness b ≡ blockLive)
    osl o pos with sameObj theObj o in eq
    ... | true  = storage-live o (sameObj-sound theObj o eq)
    ... | false with pos
    ...   | ()

------------------------------------------------------------------------
-- m₃ = the last release: no sites, counter 0, `finalizing`.
--
-- ⚠ Every field of this witness is vacuous. `counted-exact` and `live-positive`
-- are guarded by `lifeOf ≡ just live` and the object is `finalizing`;
-- `dead-unowned` is guarded by `lifeOf ≡ just dead` and it is not dead either;
-- the site map is empty, so `no-stale-owner` and `owned-storage-live` have no
-- hypothesis to work from.
--
-- Included anyway, and labelled, because "the invariant survives the last
-- decref" is a claim the chain has to make -- and because a vacuous witness that
-- is not marked as one is how a proof layer starts overstating itself.

wf₃ : WFRC m₃
wf₃ = record
  { counted-exact      = ce
  ; live-positive      = lp
  ; dead-unowned       = λ o _ → refl
  ; no-stale-owner     = λ s o ()
  ; owned-storage-live = λ o ()
  }
  where
    ce : ∀ o n → countOf m₃ o ≡ just (counted n) → lifeOf m₃ o ≡ just live →
         n ≡ ghostRC m₃ o
    ce o n _ lv with sameObj theObj o
    ... | true  with lv
    ...   | ()
    ce o n _ lv | false with lv
    ...   | ()

    lp : ∀ o → lifeOf m₃ o ≡ just live → IsCounted m₃ o → 0 < ghostRC m₃ o
    lp o lv _ with sameObj theObj o
    ... | true  with lv
    ...   | ()
    lp o lv _ | false with lv
    ...   | ()

------------------------------------------------------------------------
-- mᵣ = after reclaim: dead, no sites.
--
-- This is the witness that discharges `dead-unowned` with a real hypothesis:
-- the object IS dead here, and the field says nothing holds it. That direction
-- is the one that turns a stale owner site into a violation of the invariant
-- rather than into a use-after-free at the next load.

mᵣ : Machine
mᵣ = machine (heap m₃)
             ((theObj , cell dead (counted 0) (proj₂ allocated) 0) ∷ [])
             (sites m₃)

-- and it really is the machine `reclaim` produces, not one written to suit.
reclaim-lands-here : reclaimed ≡ ok (proj₂ allocated , mᵣ)
reclaim-lands-here = refl

-- Non-vacuous: the hypothesis of `dead-unowned` is met at this object.
dead-object-is-really-dead : lifeOf mᵣ theObj ≡ just dead
dead-object-is-really-dead = refl

wfᵣ : WFRC mᵣ
wfᵣ = record
  { counted-exact      = ce
  ; live-positive      = lp
  ; dead-unowned       = λ o _ → refl
  ; no-stale-owner     = λ s o ()
  ; owned-storage-live = λ o ()
  }
  where
    ce : ∀ o n → countOf mᵣ o ≡ just (counted n) → lifeOf mᵣ o ≡ just live →
         n ≡ ghostRC mᵣ o
    ce o n _ lv with sameObj theObj o
    ... | true  with lv
    ...   | ()
    ce o n _ lv | false with lv
    ...   | ()

    lp : ∀ o → lifeOf mᵣ o ≡ just live → IsCounted mᵣ o → 0 < ghostRC mᵣ o
    lp o lv _ with sameObj theObj o
    ... | true  with lv
    ...   | ()
    lp o lv _ | false with lv
    ...   | ()

------------------------------------------------------------------------
-- Which field is discharged non-vacuously, and where.
--
--                       m₀   m₁   m₂   m₃   mᵣ
--   counted-exact        ●    ●    ●    ·    ·
--   live-positive        ●    ●    ●    ·    ·
--   dead-unowned         ·    ·    ·    ·    ●
--   no-stale-owner       ●    ●    ●    ·    ·
--   owned-storage-live   ●    ●    ●    ·    ·
--
-- Every field is met with a real hypothesis somewhere, which is what makes this
-- a demonstration that `WFRC` is SATISFIABLE rather than a demonstration that
-- its hypotheses can be avoided. A witness set in which some field were vacuous
-- everywhere would leave that field possibly-unsatisfiable, and every theorem
-- resting on it would be worth what it was worth before this module existed.
--
-- m₃ is entirely vacuous. That is a fact about the state, not a gap: after the
-- last decref the object is `finalizing` with no owners, and every field of the
-- invariant is guarded by a condition that state does not meet. The row is left
-- in because dropping it would make the chain look shorter than it is.
