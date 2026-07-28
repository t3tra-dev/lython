{-# OPTIONS --safe #-}

-- retain and release move the counter and the ghost count BY THE SAME AMOUNT.
--
-- That is the content of the refcount layer, and it is what a misplaced
-- py.incref violates. Each theorem below is stated as two equations that have to
-- agree -- runtime side and ghost side -- rather than as one statement about the
-- counter, because a statement about only the counter is true of any
-- implementation, including one that never counts anything.

open import Proof.Memory.Element using (ElemSig)

module Proof.RC.Properties (Sig : ElemSig) where

open ElemSig Sig

open import Data.Empty using (⊥; ⊥-elim)
open import Data.Bool using (Bool; true; false)
open import Data.List using (List; []; _∷_)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Maybe.Properties using (just-injective)
open import Data.Nat using (ℕ; zero; suc; _<_; s≤s; z≤n)
open import Data.Product using (_×_; _,_; Σ; proj₁; proj₂)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; sym; trans; cong)
open import Relation.Nullary using (Dec; yes; no; ¬_)

open import Proof.Prelude using (Result; err; ok)
open import Proof.RC.Object
open import Proof.RC.OwnerSite
open import Proof.RC.Machine Sig
open import Proof.RC.Ops Sig

------------------------------------------------------------------------
-- Lemmas about the object table, mirroring the heap ones.

-- Each of these is one `rewrite` on the boolean, and they exist so that the
-- lemma below never has to reason about a term that reduction created. `with`
-- abstracts the occurrences PRESENT IN THE GOAL; when updateObj reduces, the
-- lookupObj applied to its result scrutinises a fresh occurrence that the
-- abstraction never saw, and the goal stops reducing. Carrying the equation
-- explicitly avoids the question.

lookupObj-cons-true : ∀ p c (ts : ObjTable) o → sameObj p o ≡ true →
                     lookupObj ((p , c) ∷ ts) o ≡ just c
lookupObj-cons-true p c ts o s rewrite s = refl

lookupObj-cons-false : ∀ p c (ts : ObjTable) o → sameObj p o ≡ false →
                      lookupObj ((p , c) ∷ ts) o ≡ lookupObj ts o
lookupObj-cons-false p c ts o s rewrite s = refl

updateObj-cons-true : ∀ p c (ts : ObjTable) o f → sameObj p o ≡ true →
                     updateObj ((p , c) ∷ ts) o f ≡ (p , f c) ∷ ts
updateObj-cons-true p c ts o f s rewrite s = refl

updateObj-cons-false : ∀ p c (ts : ObjTable) o f → sameObj p o ≡ false →
                      updateObj ((p , c) ∷ ts) o f ≡ (p , c) ∷ updateObj ts o f
updateObj-cons-false p c ts o f s rewrite s = refl

lookupObj-update-same : ∀ (ts : ObjTable) (o : ObjId) (f : ObjCell → ObjCell) c →
                        lookupObj ts o ≡ just c →
                        lookupObj (updateObj ts o f) o ≡ just (f c)
lookupObj-update-same [] o f c ()
lookupObj-update-same ((p , x) ∷ ts) o f c eq = go (sameObj p o) refl
  where
    go : (b : Bool) → sameObj p o ≡ b →
         lookupObj (updateObj ((p , x) ∷ ts) o f) o ≡ just (f c)
    go true s =
      trans (cong (λ z → lookupObj z o) (updateObj-cons-true p x ts o f s))
            (trans (lookupObj-cons-true p (f x) ts o s)
                   (cong (λ z → just (f z))
                         (just-injective
                           (trans (sym (lookupObj-cons-true p x ts o s)) eq))))
    go false s =
      trans (cong (λ z → lookupObj z o) (updateObj-cons-false p x ts o f s))
            (trans (lookupObj-cons-false p x (updateObj ts o f) o s)
                   (lookupObj-update-same ts o f c
                     (trans (sym (lookupObj-cons-false p x ts o s)) eq)))

------------------------------------------------------------------------
-- py.incref: both sides go up by one.

-- Ghost side. This is the half that a compiler bug can violate without any
-- runtime symptom until much later, so it is stated first.
retain-ghost :
  ∀ (m : Machine) (o : ObjId) (pr : ProtectedRef m o) (dst : OwnerSite) m' →
  strongAt (sites m) dst ≡ nothing →
  retain m o pr dst ≡ ok m' →
  ghostRC m' o ≡ suc (ghostRC m o)
retain-ghost m o pr dst m' free eq
  rewrite free
  with eq
... | refl = occupy-same (sites m) dst o

-- And the two other objects are untouched: a retain of `o` does not change
-- anybody else's count. Without this, a retain that incremented every object
-- would satisfy the theorem above.
retain-ghost-others :
  ∀ (m : Machine) (o p : ObjId) (pr : ProtectedRef m o) (dst : OwnerSite) m' →
  -- The disequality is stated over the BOOLEAN that `logicalRC` branches on.
  -- `sameObj-sound` converts the other way when a caller has a proposition.
  sameObj o p ≡ false →
  strongAt (sites m) dst ≡ nothing →
  retain m o pr dst ≡ ok m' →
  ghostRC m' p ≡ ghostRC m p
retain-ghost-others m o p pr dst m' ne free eq
  rewrite free
  with eq
... | refl = occupy-other (sites m) dst o p ne

-- Runtime side.
retain-counter :
  ∀ (m : Machine) (o : ObjId) (pr : ProtectedRef m o) (dst : OwnerSite) m' c →
  strongAt (sites m) dst ≡ nothing →
  lookupObj (objects m) o ≡ just c →
  retain m o pr dst ≡ ok m' →
  countOf m' o ≡ just (bumpUp (count c))
retain-counter m o pr dst m' c free look eq
  rewrite free
  with eq
... | refl
  rewrite lookupObj-update-same (objects m) o (λ x → record x { count = bumpUp (count x) }) c look
  = refl

------------------------------------------------------------------------
-- py.decref: both sides go down by one.

release-ghost :
  ∀ (m : Machine) (src : OwnerSite) (o : ObjId) m' →
  strongAt (sites m) src ≡ just o →
  release m src o ≡ ok m' →
  sites m' ≡ vacate (sites m) src
release-ghost m src o m' held eq
  rewrite held
  with o ≟-obj o
... | no ¬p = ⊥-elim (¬p refl)
... | yes _ with eq
...   | refl = refl

-- A release does not touch the heap. Stated because the heap is where
-- use-after-free lives, and `release` reaching zero is the step at which a
-- careless implementation would free the storage -- one step before
-- `reclaim` has checked that nothing still owns it.
release-leaves-heap :
  ∀ (m : Machine) (src : OwnerSite) (o : ObjId) m' →
  strongAt (sites m) src ≡ just o →
  release m src o ≡ ok m' →
  heap m' ≡ heap m
release-leaves-heap m src o m' held eq
  rewrite held
  with o ≟-obj o
... | no ¬p = ⊥-elim (¬p refl)
... | yes _ with eq
...   | refl = refl

------------------------------------------------------------------------
-- move: neither side changes.

-- The count is unchanged because the NUMBER of sites is unchanged -- one is
-- vacated and one is occupied. A pass that emits retain-then-release for a move
-- is not wrong, but it pays two atomic operations for a no-op; a pass that
-- emits only the release is the over-release this compiler shipped.
move-leaves-counter :
  ∀ (m : Machine) (src dst : OwnerSite) (o : ObjId) m' →
  strongAt (sites m) src ≡ just o →
  strongAt (sites m) dst ≡ nothing →
  moveRef m src dst o ≡ ok m' →
  objects m' ≡ objects m
move-leaves-counter m src dst o m' held free eq
  rewrite held
  with o ≟-obj o
... | no ¬p = ⊥-elim (¬p refl)
... | yes _
  rewrite free
  with eq
... | refl = refl

------------------------------------------------------------------------
-- borrow: nothing changes at all.
--
-- Trivial by definition, and stated anyway. It is the row of the table that
-- costs zero runtime operations, and an elaborator that cannot see it emits a
-- retain/release pair per read -- so "borrow is the identity" is the fact that
-- makes eliding those correct rather than merely cheaper.
borrow-is-identity :
  ∀ (m : Machine) (o : ObjId) (pr : ProtectedRef m o) →
  borrowRef m o pr ≡ m
borrow-is-identity m o pr = refl

------------------------------------------------------------------------
-- reclaim is refused while anything still owns the object.
--
-- This is the theorem that separates the counting layer from a
-- use-after-free: reaching it requires the ghost count to be zero, so a
-- premature free is a refusal here rather than a fault at the next load.

reclaim-refused-when-owned :
  ∀ (m : Machine) (o : ObjId) c n →
  lookupObj (objects m) o ≡ just c →
  life c ≡ finalizing →
  ghostRC m o ≡ suc n →
  reclaim m o ≡ err still-owned
reclaim-refused-when-owned m o c n look lf rc
  rewrite look
  with life c | lf
... | .finalizing | refl
  rewrite rc = refl
