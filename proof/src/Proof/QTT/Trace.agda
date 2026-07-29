{-# OPTIONS --safe #-}

-- The quantity layer, exercised.
--
-- `ErasedHasNoRuntimeOp` is the QTT layer's only judgment and nothing had ever
-- built one. A one-constructor datatype is easy to inhabit, which is exactly why
-- leaving it uninhabited was worth noticing: the module states the central claim
-- of the layer ("a quantity of 0 cannot be the source of a retain") and nothing
-- had ever checked that the claim's negative side holds -- that q0 is NOT
-- related to the three elaborations that do emit a runtime operation.
--
-- The positive alone is worth little. `data P : … where c : P q0 as-borrow` is
-- satisfied by a relation that also related everything else, and only the
-- refutations below pin it down.

module Proof.QTT.Trace where

open import Data.Empty using (⊥; ⊥-elim)
open import Data.Nat using (ℕ; zero; suc)
open import Data.Product using (_×_; _,_; Σ; proj₁; proj₂)
open import Relation.Binary.PropositionalEquality using (_≡_; _≢_; refl)
open import Relation.Nullary using (¬_)

open import Proof.QTT.Quantity

------------------------------------------------------------------------
-- ⭐ The judgment, inhabited.

erasure-is-a-borrow : ErasedHasNoRuntimeOp q0 as-borrow
erasure-is-a-borrow = erased-borrow

------------------------------------------------------------------------
-- And the three refutations that give it content.
--
-- These are what "erased has no runtime operation" MEANS. Without them the
-- judgment is consistent with q0 elaborating to an incref, which is the one
-- thing the layer exists to forbid.

erased-is-never-a-dup : ¬ ErasedHasNoRuntimeOp q0 as-dup
erased-is-never-a-dup ()

erased-is-never-a-drop : ¬ ErasedHasNoRuntimeOp q0 as-drop
erased-is-never-a-drop ()

-- A move emits no runtime operation either, so this one is NOT about cost: an
-- erased variable has no reference to hand on, so there is nothing to move.
erased-is-never-a-move : ¬ ErasedHasNoRuntimeOp q0 as-move
erased-is-never-a-move ()

-- The judgment does not extend to non-erased quantities. It is not the claim
-- that a borrow is free -- it is the claim that erasure forces one.
one-is-not-erased : ¬ ErasedHasNoRuntimeOp q1 as-borrow
one-is-not-erased ()

unrestricted-is-not-erased : ¬ ErasedHasNoRuntimeOp qω as-borrow
unrestricted-is-not-erased ()

------------------------------------------------------------------------
-- ⭐ The distinction the module's header is about.
--
--     a QUANTITY is how many times a variable is USED;
--     a REFERENCE COUNT is how many owning references EXIST AT ONCE.
--
-- Written as a theorem rather than as a comment: two bindings with the SAME
-- quantity and different modes. One owes a release and the other does not, so
-- an elaborator that read the refcount off the multiplicity would treat them
-- alike -- which is how "used ten times" becomes nine increfs.

quantity-does-not-determine-mode :
  Σ (Binding ℕ) λ b₁ → Σ (Binding ℕ) λ b₂ →
    (quantity b₁ ≡ quantity b₂) × (mode b₁ ≢ mode b₂)
quantity-does-not-determine-mode =
  binding q1 owned 0 , binding q1 (borrowed 0) 0 , refl , λ ()

-- and the other direction: the same MODE at different quantities. Ten borrowed
-- reads and one borrowed read are both zero refcount operations, so the count
-- does not determine the multiplicity either.
mode-does-not-determine-quantity :
  Σ (Binding ℕ) λ b₁ → Σ (Binding ℕ) λ b₂ →
    (mode b₁ ≡ mode b₂) × (quantity b₁ ≢ quantity b₂)
mode-does-not-determine-quantity =
  binding q1 (borrowed 0) 0 , binding qω (borrowed 0) 0 , refl , λ ()

------------------------------------------------------------------------
-- The semiring, checked.
--
-- `q0` is the additive unit and the multiplicative zero, and that second fact
-- is what makes erasure propagate: anything scaled by an erased quantity is
-- erased, so a field read off an erased value cannot become a retain either.

zero-is-additive-unit : (q0 +q q1 ≡ q1) × (q1 +q q0 ≡ q1)
                      × (q0 +q qω ≡ qω) × (qω +q q0 ≡ qω)
zero-is-additive-unit = refl , refl , refl , refl

-- Two uses of a linear variable are unrestricted, which is the step that makes
-- `q1` a real restriction rather than a label.
one-plus-one-is-omega : q1 +q q1 ≡ qω
one-plus-one-is-omega = refl

zero-absorbs : (q0 *q q1 ≡ q0) × (q1 *q q0 ≡ q0)
             × (q0 *q qω ≡ q0) × (qω *q q0 ≡ q0)
zero-absorbs = refl , refl , refl , refl

one-is-multiplicative-unit : (q1 *q q1 ≡ q1) × (q1 *q qω ≡ qω)
one-is-multiplicative-unit = refl , refl
