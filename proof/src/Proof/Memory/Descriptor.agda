{-# OPTIONS --safe #-}

-- MemRef descriptors: a strided view onto one allocation.
--
-- A descriptor is NOT memory. It names an allocation, a generation, and an
-- affine map from logical indices to byte ranges inside that allocation. Two
-- descriptors may legitimately alias -- subview, transpose, reinterpret_cast
-- and memory_space_cast all produce one that shares its backing -- so aliasing
-- between descriptors is permitted here, and only aliasing WITHIN one
-- descriptor is forbidden (MLIR's memref type requires that too).

open import Proof.Memory.Element using (ElemSig; MemSpace)

module Proof.Memory.Descriptor (Sig : ElemSig) where

open ElemSig Sig

open import Data.Fin using (Fin; toℕ)
open import Data.Integer using (ℤ; +_; _*_; _+_; -[1+_])
open import Data.Nat using (ℕ)
  renaming (_+_ to _+ℕ_)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Vec using (Vec; []; _∷_)

open import Proof.Memory.Heap using (AllocId; Generation)
open import Proof.Memory.Index using (Ix; []; _∷_)

record Desc (rank : ℕ) : Set where
  constructor desc
  field
    allocation  : AllocId
    generation  : Generation

    -- Byte offset of this view's origin from the START OF THE ALLOCATION.
    -- Separate from `offset` because memref.view shifts by bytes while
    -- memref.subview shifts by elements, and collapsing them would make one of
    -- the two operations unrepresentable.
    alignedBase : ℕ

    -- Element-unit offset and strides, as in MLIR's strided layout.
    offset      : ℤ
    sizes       : Vec ℕ rank
    strides     : Vec ℤ rank

    elementType : ElemTy
    memorySpace : MemSpace

open Desc public

-- Strides are ℤ, not ℕ: a reversed or transposed view has negative strides, and
-- MLIR's own lowering distinguishes that case. Modelling them as ℕ would make
-- the model reject programs the compiler is required to accept.
dot : ∀ {r} {ns : Vec ℕ r} → Ix ns → Vec ℤ r → ℤ
dot []       []       = + 0
dot (i ∷ is) (s ∷ ss) = (+ toℕ i) * s + dot is ss

elementOffset : ∀ {r} (d : Desc r) → Ix (sizes d) → ℤ
elementOffset d i = offset d + dot i (strides d)

-- May be negative: a descriptor whose origin sits in the middle of the
-- allocation can address backwards. `resolve` is where that is checked; keeping
-- it in ℤ here means the check is a real one rather than a truncation that
-- silently produced a valid-looking offset.
byteStart : ∀ {r} (d : Desc r) → Ix (sizes d) → ℤ
byteStart d i = (+ alignedBase d) + (+ width (elementType d)) * elementOffset d i

elemWidth : ∀ {r} → Desc r → ℕ
elemWidth d = width (elementType d)

elemAlign : ∀ {r} → Desc r → ℕ
elemAlign d = align (elementType d)

-- Non-negativity by pattern matching rather than by an order proof: `+ n` and
-- `-[1+ n ]` already ARE the two cases, so this is the check, not a lemma.
nonNeg : ℤ → Maybe ℕ
nonNeg (+ n)      = just n
nonNeg -[1+ _ ]   = nothing
