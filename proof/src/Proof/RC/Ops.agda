{-# OPTIONS --safe #-}

-- py.incref, py.decref, move, and borrow -- as transitions on the machine.
--
-- Each does TWO things at once, and that is the whole design: it changes the
-- runtime counter and it changes the ghost site map, in the same step. An
-- operation that changed one without the other is exactly what breaks
-- `counted-exact`, and this is the layer where the compiler's incref/decref
-- placement decisions land.
--
-- The table (from the note), and note that half the rows emit no code at all:
--
--   move x            owner site moves           no runtime operation
--   borrow x          owner sites unchanged      no runtime operation
--   dup x / share x   one more owner site        incref
--   drop x            one fewer owner site       decref
--
-- Reading nine increfs out of ten uses is what happens when the elaborator
-- works from occurrence counts instead of from this table.

open import Proof.Memory.Element using (ElemSig)

module Proof.RC.Ops (Sig : ElemSig) where

open ElemSig Sig

open import Data.List using (List; []; _∷_)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc)
open import Data.Product using (_×_; _,_; Σ; proj₁; proj₂)
open import Relation.Binary.PropositionalEquality using (_≡_; refl)
open import Relation.Nullary using (Dec; yes; no; ¬_)

open import Proof.Prelude using (Result; err; ok; _>>=ᴿ_; guardᴿ; maybeᴿ)
open import Proof.Memory.Fault using (MemoryFault; use-after-free;
  no-such-allocation)
open import Proof.RC.Object
open import Proof.RC.OwnerSite
open import Proof.Memory.Descriptor Sig using (Desc)
open import Proof.RC.Machine Sig

-- Faults specific to the reference-counting layer. Separate from MemoryFault
-- because they are broken at a different level: `refcount-underflow` is not a
-- bad address, it is a bookkeeping error that WOULD BECOME one.
data RCFault : Set where
  no-such-object       : RCFault
  -- incref through something that is not a live protected reference.
  retain-of-dead       : RCFault
  -- The destination site already holds something. Silently overwriting it would
  -- lose an owning reference and leak it, so it is refused.
  destination-occupied : RCFault
  -- decref of an object no site holds. This is the one that becomes a
  -- use-after-free one step later, and catching it HERE is the point.
  release-unowned      : RCFault
  -- The source site of a move or drop does not hold what the caller thinks.
  source-mismatch      : RCFault
  -- Reclaiming storage for an object that has not reached zero.
  not-finalizing       : RCFault
  -- Reclaiming storage while a site still holds the object. This is the
  -- condition that separates "the counter says zero" from "nothing owns it",
  -- and it is the one a biased or deferred counting scheme relaxes.
  still-owned          : RCFault

------------------------------------------------------------------------
-- The precondition every retain shares.
--
-- incref may NOT be performed from a raw pointer. It requires a reference that
-- is already protected -- an existing owning reference, or a borrow with a live
-- anchor. Without this the model would permit resurrecting an object whose
-- count has already reached zero, which is the exact shape of the
-- use-after-free that a naive "just bump the counter" incref produces.
record ProtectedRef (m : Machine) (o : ObjId) : Set where
  constructor protected-by
  field
    -- Some site already holds it, which is what makes the reference legitimate.
    anchor      : OwnerSite
    anchor-holds : strongAt (sites m) anchor ≡ just o
    -- and the object is not already being torn down
    is-live     : lifeOf m o ≡ just live

open ProtectedRef public

------------------------------------------------------------------------
-- py.incref
--
-- Counter up, and one more owner site. Both, or neither.

retain : (m : Machine) → (o : ObjId) → ProtectedRef m o → (dst : OwnerSite) →
         Result RCFault Machine
retain m o _ dst with strongAt (sites m) dst
... | just _  = err destination-occupied
... | nothing =
  ok (machine (heap m)
              (updateObj (objects m) o (λ c → record c { count = bumpUp (count c) }))
              (occupy (sites m) dst o))

------------------------------------------------------------------------
-- py.decref
--
-- Counter down, and one fewer owner site. When the counter reaches zero the
-- object enters `finalizing` -- NOT `dead`, and the storage is NOT freed here.
--
-- Splitting those is not ceremony. A finalizer runs arbitrary code, and during
-- that window the object must be neither usable nor freeable; collapsing the
-- two states makes one of those two wrong. Freeing the storage is a separate
-- step (`reclaim`) that requires the object to be in `finalizing` already.

release : (m : Machine) → (src : OwnerSite) → (o : ObjId) →
          Result RCFault Machine
release m src o with strongAt (sites m) src
... | nothing = err release-unowned
... | just p with p ≟-obj o
...   | no  _ = err source-mismatch
...   | yes _ = ok (machine (heap m) newObjects (vacate (sites m) src))
  where
    stepDown : ObjCell → ObjCell
    stepDown c with bumpDown (count c)
    ... | counted zero = record c { count = counted zero ; life = finalizing }
    ... | n            = record c { count = n }

    newObjects : ObjTable
    newObjects = updateObj (objects m) o stepDown

------------------------------------------------------------------------
-- move
--
-- The owner site changes hands. NO runtime operation: the count is unchanged
-- because the number of sites is unchanged. This is the row that a
-- refcount-inserting pass gets wrong by emitting an incref/decref pair, which
-- is correct but costs two atomic operations for nothing -- and, in this
-- compiler, is the shape whose *absence* produced a shipped over-release when
-- the source was read as dying at a use that executes N times.

moveRef : (m : Machine) → (src dst : OwnerSite) → (o : ObjId) →
          Result RCFault Machine
moveRef m src dst o with strongAt (sites m) src
... | nothing = err source-mismatch
... | just p with p ≟-obj o
...   | no _  = err source-mismatch
...   | yes _ with strongAt (sites m) dst
...     | just _  = err destination-occupied
...     | nothing = ok (machine (heap m) (objects m)
                                (occupy (vacate (sites m) src) dst o))

------------------------------------------------------------------------
-- borrow
--
-- Nothing happens. That is the entire content of the operation and the reason
-- it is worth having a name for: a borrow is the case where the correct number
-- of runtime operations is zero, and an elaborator that cannot express it emits
-- a retain/release pair for every read.
--
-- The obligation a borrow carries is not on the count but on LIFETIME: the
-- borrow may not outlive its anchor. That is `borrowed-anchored` in
-- Proof.RC.Invariant, and it is why RefMode's `borrowed` is region-indexed.

borrowRef : (m : Machine) → (o : ObjId) → ProtectedRef m o → Machine
borrowRef m _ _ = m

------------------------------------------------------------------------
-- reclaim
--
-- The storage handoff. Only from `finalizing`, and only when no site holds the
-- object -- the two conditions that together are what "the last reference went
-- away" means. The heap-level `dealloc` is deliberately NOT called here: this
-- module knows about counts, and the descriptor it would pass is the object's
-- `backing`, so the connection between the two layers is one obligation stated
-- in one place rather than an assumption spread over both.

-- Returns the canonical descriptor to hand to the heap's `dealloc`, and the
-- machine with the object marked dead. Two separate conditions, because they
-- can come apart: `finalizing` is what the COUNTER said, and `logicalRC ≡ 0` is
-- what the ghost state says. Requiring both is what makes a discrepancy between
-- them a refusal here rather than a free of something still referenced.
reclaim : (m : Machine) → (o : ObjId) → Result RCFault (Desc 1 × Machine)
reclaim m o with lookupObj (objects m) o
... | nothing = err no-such-object
... | just c  with life c
...   | live       = err not-finalizing
...   | dead       = err not-finalizing
...   | finalizing with logicalRC (sites m) o
...     | suc _ = err still-owned
...     | zero  =
          ok (backing c
             , machine (heap m)
                       (updateObj (objects m) o (λ x → record x { life = dead }))
                       (sites m))
