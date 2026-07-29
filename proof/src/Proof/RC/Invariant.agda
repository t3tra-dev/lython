{-# OPTIONS --safe #-}

-- The invariant that makes the runtime counter mean something.
--
-- Without it, `count` is a number in a record and no placement of incref /
-- decref can be wrong. `counted-exact` is the statement that the counter
-- IMPLEMENTS the ghost site count -- and every other field here is a
-- consequence the compiler is allowed to rely on.

open import Proof.Memory.Element using (ElemSig)

module Proof.RC.Invariant (Sig : ElemSig) where

open ElemSig Sig

open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc; _<_; s≤s; z≤n)
open import Data.Product using (_×_; _,_; Σ)
open import Relation.Binary.PropositionalEquality using (_≡_; refl)
open import Relation.Nullary using (¬_)

open import Proof.Memory.Heap using (Heap; Liveness; lookupBlock; liveness;
  generation)
  renaming (live to blockLive; dead to blockDead)
open import Proof.RC.Object
open import Proof.RC.OwnerSite
open import Proof.RC.Machine Sig

record WFRC (m : Machine) : Set where
  field
    -- THE one. A live, counted object's runtime counter equals the number of
    -- owner sites holding it. Everything the compiler may assume about
    -- reference counts is downstream of this.
    counted-exact :
      ∀ o n → countOf m o ≡ just (counted n) → lifeOf m o ≡ just live →
      n ≡ ghostRC m o

    -- A live counted object is held by someone. This is what makes "the count
    -- reached zero" the same event as "nothing refers to it any more"; without
    -- it, an object could be live at zero and the next decref would underflow.
    live-positive :
      ∀ o → lifeOf m o ≡ just live → IsCounted m o → 0 < ghostRC m o

    -- Nothing holds a dead object. The converse of the above, and the one that
    -- turns a stale owner site into a violation HERE rather than into a
    -- use-after-free at the next load.
    dead-unowned :
      ∀ o → lifeOf m o ≡ just dead → ghostRC m o ≡ 0

    -- Every site holds an object whose generation still matches its block. This
    -- is where the refcount layer meets the memory layer: an owner site that
    -- survived a realloc is caught by this field and not by `resolve`, which is
    -- one step too late.
    --
    -- Stated over `Holds` -- MEMBERSHIP in the site map -- and not over
    -- `strongAt`. The first version used the lookup, and was then not preserved
    -- by `drop`: `vacate` removes the first entry at a site and exposes whatever
    -- was behind it, about which a property quantified over first-entries says
    -- nothing. Over membership the field is preserved, and it is also the
    -- property one wants: every reference the map records has live storage at a
    -- matching generation, not merely the ones a lookup happens to reach.
    no-stale-owner :
      ∀ s o → Holds (sites m) s o →
      Σ _ λ b → (lookupBlock (heap m) (objAllocation o) ≡ just b)
              × (generation b ≡ objGeneration o)

    -- Storage is still there while anyone holds the object. Stated as an
    -- implication from the ghost count to the heap, so that it constrains the
    -- HEAP -- the direction that makes it a memory-safety property rather than
    -- bookkeeping about the table.
    owned-storage-live :
      ∀ o → 0 < ghostRC m o →
      Σ _ λ b → (lookupBlock (heap m) (objAllocation o) ≡ just b)
              × (liveness b ≡ blockLive)

open WFRC public

-- What the invariant is FOR, spelled out so that it is not mistaken for
-- housekeeping:
--
--   counted-exact + live-positive  ⇒  a decref that reaches zero is the last
--                                     owner going away, so reclaiming then is
--                                     not premature.
--
--   dead-unowned + owned-storage-live
--                                  ⇒  no site can name freed storage, so a load
--                                     through a site's reference cannot be a
--                                     use-after-free.
--
--   no-stale-owner                 ⇒  and it cannot be a stale-generation
--                                     either, which is the realloc case.
--
-- Each is a property the compiler already assumes when it elides a retain. The
-- point of writing them down is that eliding a retain is then a step that has a
-- premise, instead of a judgement call.
