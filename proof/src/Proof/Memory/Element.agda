{-# OPTIONS --safe #-}

-- The element types a descriptor can be built over, bundled so the rest of the
-- kernel takes one parameter instead of four.
--
-- Parameterised rather than fixed to Lython's current set, because the width
-- and alignment facts the proofs use are the ONLY thing they need from an
-- element type. Fixing an enum here would make every later theorem look like it
-- depended on i64 when it does not.

module Proof.Memory.Element where

open import Data.Nat using (ℕ; _<_; zero; suc)
open import Relation.Binary.PropositionalEquality using (_≡_)
open import Relation.Nullary using (Dec)

record ElemSig : Set₁ where
  field
    ElemTy : Set

    width : ElemTy → ℕ
    align : ElemTy → ℕ

    -- A zero-width element would make every byte range empty, so every access
    -- would be trivially in bounds and the bounds theorem would hold vacuously.
    -- Requiring positivity here is what stops that.
    width-pos : ∀ τ → 0 < width τ

    -- Likewise: alignment 0 makes the divisibility check meaningless, since
    -- every offset is a multiple of 0 under the usual definition only when the
    -- offset is 0.
    align-pos : ∀ τ → 0 < align τ

    -- Descriptor equality is decided in the reinterpret_cast / view rules, so
    -- the element type has to be discrete.
    _≟-elem_ : (σ τ : ElemTy) → Dec (σ ≡ τ)

-- Memory spaces are opaque identifiers here. Step 1 of the plan is a single
-- space, but the CHECK exists from the start: adding the field later would mean
-- revisiting every operation, whereas a check that is always satisfied costs
-- one line and is already in the right place when GPU or DMA spaces arrive.
MemSpace : Set
MemSpace = ℕ
