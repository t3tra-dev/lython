{-# OPTIONS --safe #-}

-- Boxed objects, their identity, and their runtime counter.
--
-- Object identity is (allocation, generation) -- the same identity the byte
-- heap uses, deliberately, because the refcount layer and the memory layer have
-- to agree about what "the same object" means or the invariant connecting them
-- says nothing.

module Proof.RC.Object where

open import Data.Nat using (ℕ; zero; suc; _≟_)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; cong)
open import Relation.Nullary using (Dec; yes; no; ¬_)

open import Proof.Memory.Heap using (AllocId; Generation)

record ObjId : Set where
  constructor obj
  field
    objAllocation : AllocId
    objGeneration : Generation

open ObjId public

_≟-obj_ : (o p : ObjId) → Dec (o ≡ p)
obj a g ≟-obj obj a' g' with a ≟ a' | g ≟ g'
... | yes refl | yes refl = yes refl
... | no ¬p    | _        = no λ { refl → ¬p refl }
... | _        | no ¬q    = no λ { refl → ¬q refl }

-- Three lifecycle states, not two.
--
-- `finalizing` is the window between "the count reached zero" and "the storage
-- is gone". It has to be a state of its own because a finalizer can run
-- arbitrary code, and during that window the object is neither safely usable
-- nor yet freed. Modelling it as `dead` would make a resurrection silently
-- legal; modelling it as `live` would make freeing it a double-free.
data Life : Set where
  live finalizing dead : Life

_≟-life_ : (l m : Life) → Dec (l ≡ m)
live       ≟-life live       = yes refl
finalizing ≟-life finalizing = yes refl
dead       ≟-life dead       = yes refl
live       ≟-life finalizing = no λ ()
live       ≟-life dead       = no λ ()
finalizing ≟-life live       = no λ ()
finalizing ≟-life dead       = no λ ()
dead       ≟-life live       = no λ ()
dead       ≟-life finalizing = no λ ()

-- An immortal object is not "counted at a very large number": its count is not
-- a number at all, and no decref may take it to zero. Lython has these -- the
-- small-int cache {0,1,2} is exactly this, and it is the axis on which one of
-- this compiler's shipped defects turned out to depend.
data RuntimeCount : Set where
  counted  : ℕ → RuntimeCount
  immortal : RuntimeCount

-- Incrementing an immortal is a no-op, not an error: the runtime really does
-- call incref on immortals and really does ignore it.
bumpUp : RuntimeCount → RuntimeCount
bumpUp (counted n) = counted (suc n)
bumpUp immortal    = immortal

-- Decrementing below zero is not representable: `counted zero` stays there. The
-- model does not need that case to be an error, because the INVARIANT rules it
-- out -- `live-positive` says a live counted object has a positive count, so
-- reaching a decref with zero is already a violation upstream.
bumpDown : RuntimeCount → RuntimeCount
bumpDown (counted zero)    = counted zero
bumpDown (counted (suc n)) = counted n
bumpDown immortal          = immortal

data ReachedZero : RuntimeCount → Set where
  hit-zero : ReachedZero (counted zero)

-- Immortals never reach zero, by construction rather than by convention. The
-- proof is `λ ()` -- there is no constructor of ReachedZero at `immortal` -- and
-- that is the point: it is a fact about the datatype, not a rule someone has to
-- remember to check at each decref site.
immortal-never-zero : ¬ ReachedZero immortal
immortal-never-zero ()
