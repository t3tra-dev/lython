{-# OPTIONS --safe #-}

-- Quantitative type theory's multiplicities, and the reference modes they do
-- NOT determine.
--
-- The single most important thing in this file is a distinction, not a
-- definition:
--
--   a QUANTITY is how many times a variable is USED;
--   a REFERENCE COUNT is how many owning references EXIST AT ONCE.
--
-- They are not the same number and neither determines the other. `ω` says
-- "unrestricted use" and gives no reference count at all, and a variable used
-- ten times needs zero increfs if every use is a borrow. Identifying them is
-- how an elaborator ends up emitting nine increfs for ten reads.

module Proof.QTT.Quantity where

open import Data.Nat using (ℕ; zero; suc)
open import Relation.Binary.PropositionalEquality using (_≡_; refl)
open import Relation.Nullary using (Dec; yes; no)

-- The usual 0/1/ω semiring. `q0` is erased: it exists in types and proofs and
-- generates no runtime pointer, which is why it cannot be the source of a
-- retain.
data Quantity : Set where
  q0 q1 qω : Quantity

_+q_ : Quantity → Quantity → Quantity
q0 +q y  = y
x  +q q0 = x
_  +q _  = qω

_*q_ : Quantity → Quantity → Quantity
q0 *q _  = q0
_  *q q0 = q0
q1 *q y  = y
qω *q _  = qω

-- Multiplicity 1 does not tell you whether the reference owns anything: "read
-- it once through a borrow" and "consume it once by moving it" are both q1 and
-- have opposite effects on the refcount. That is why Binding carries a mode as
-- well as a quantity.
data RefMode : Set where
  owned    : RefMode
  -- Borrows are region-indexed so that a borrow cannot outlive its anchor. The
  -- region is abstract here; the anchoring obligation is stated in
  -- Proof.RC.Invariant as `borrowed-anchored`.
  borrowed : ℕ → RefMode
  weak     : RefMode

record Binding (ObjTy : Set) : Set where
  constructor binding
  field
    quantity : Quantity
    mode     : RefMode
    type     : ObjTy

open Binding public

-- What the elaborator has to decide, and the reason it cannot decide it from
-- the quantity alone:
--
--   read-only, does not escape ....... borrow      -- no runtime operation
--   last use, ownership handed on .... move        -- no runtime operation
--   needed in two places at once ..... dup/incref
--   owning reference no longer needed  drop/decref
--
-- Only the third and fourth touch the count. Deciding between them needs escape
-- analysis, liveness and control flow -- occurrence counting is not enough, and
-- this datatype is deliberately not a function of Quantity.
data Elaborated : Set where
  as-borrow as-move as-dup as-drop : Elaborated

-- q0 can never be elaborated to a runtime reference operation: it has no
-- runtime representation to retain or release. This is the one direction where
-- the quantity DOES settle the question.
data ErasedHasNoRuntimeOp : Quantity → Elaborated → Set where
  erased-borrow : ErasedHasNoRuntimeOp q0 as-borrow
