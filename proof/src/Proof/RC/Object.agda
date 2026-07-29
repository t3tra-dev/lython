{-# OPTIONS --safe #-}

-- Boxed objects, their identity, and their runtime counter.
--
-- Object identity is (allocation, generation) -- the same identity the byte
-- heap uses, deliberately, because the refcount layer and the memory layer have
-- to agree about what "the same object" means or the invariant connecting them
-- says nothing.

module Proof.RC.Object where

open import Data.Bool using (Bool; true; false; _∧_)
open import Data.Nat using (ℕ; zero; suc; _≟_)
open import Data.Nat.Base using (_≡ᵇ_)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; cong; cong₂)
open import Relation.Nullary using (Dec; yes; no; ¬_)

open import Proof.Memory.Heap using (AllocId; Generation)

record ObjId : Set where
  constructor obj
  field
    objAllocation : AllocId
    objGeneration : Generation

open ObjId public

-- ONE definition of "the same object", used by the object table, the site map
-- and the program environment alike. There were three, and three definitions of
-- an equality are three things that can drift -- a count taken with one and
-- checked against another is counting different things.
--
-- Boolean rather than the decision procedure, for the reason recorded in
-- Proof.RC.Machine: `with p ≟-obj o` compiles to an auxiliary function, so the
-- occurrence inside an update is not the term a goal abstracts.
sameObj : ObjId → ObjId → Bool
sameObj (obj a g) (obj a' g') = (a ≡ᵇ a') ∧ (g ≡ᵇ g')

≡ᵇ-refl : ∀ (n : ℕ) → (n ≡ᵇ n) ≡ true
≡ᵇ-refl zero    = refl
≡ᵇ-refl (suc n) = ≡ᵇ-refl n

sameObj-refl : ∀ o → sameObj o o ≡ true
sameObj-refl (obj a g) rewrite ≡ᵇ-refl a | ≡ᵇ-refl g = refl

-- Top-level rather than local to `sameObj-sound`, because the program
-- environment's `sameVar` is the same boolean on the same type and needs the
-- same reflection. Two copies of it would be two things that can drift, which
-- is the reason `sameObj` itself was consolidated here.
≡ᵇ-sound : ∀ (m n : ℕ) → (m ≡ᵇ n) ≡ true → m ≡ n
≡ᵇ-sound zero    zero    _ = refl
≡ᵇ-sound (suc m) (suc n) e = cong suc (≡ᵇ-sound m n e)

≡ᵇ-sym : ∀ (m n : ℕ) → (m ≡ᵇ n) ≡ (n ≡ᵇ m)
≡ᵇ-sym zero    zero    = refl
≡ᵇ-sym zero    (suc n) = refl
≡ᵇ-sym (suc m) zero    = refl
≡ᵇ-sym (suc m) (suc n) = ≡ᵇ-sym m n

-- Reflecting the boolean back into a proposition, which is what the counting
-- lemmas need when they have to know the two objects really are the same.
sameObj-sound : ∀ o p → sameObj o p ≡ true → o ≡ p
sameObj-sound (obj a g) (obj a' g') eq = go a a' g g' eq
  where
    go : ∀ a a' g g' → ((a ≡ᵇ a') ∧ (g ≡ᵇ g')) ≡ true → obj a g ≡ obj a' g'
    go a a' g g' e with a ≡ᵇ a' | ≡ᵇ-sound a a'
    ... | true  | f with g ≡ᵇ g' | ≡ᵇ-sound g g'
    ...   | true  | h = cong₂ obj (f refl) (h refl)

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
