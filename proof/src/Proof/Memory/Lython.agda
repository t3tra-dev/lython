{-# OPTIONS --safe #-}

-- The element signature Lython actually lowers to, so the parameterised modules
-- are INSTANTIATED rather than merely parameterised.
--
-- This project already has a name for why that matters:
-- `NonInstantiationIsNotConformance` in rfc/memory-safety-proof.md. A judgment
-- no contract instantiates is a judgment nothing can violate, and it reads as
-- discharged. The same is true of an Agda module parameter that is never
-- supplied -- every theorem below it holds vacuously for a signature that does
-- not exist.

module Proof.Memory.Lython where

open import Data.Nat using (ℕ; zero; suc; _<_; s≤s; z≤n)
open import Relation.Binary.PropositionalEquality using (_≡_; refl)
open import Relation.Nullary using (Dec; yes; no)

open import Proof.Memory.Element using (ElemSig)

-- The element types that appear in Lython's emitted memrefs. i8 is the one that
-- matters most: every boxed object is a memref<?xi8> allocation with the
-- refcount, the class id and the payload at fixed byte offsets inside it, and
-- the typed views onto those fields are exactly `memref.view`.
data LyTy : Set where
  i8 i32 i64 f64 ptr : LyTy

lyWidth : LyTy → ℕ
lyWidth i8  = 1
lyWidth i32 = 4
lyWidth i64 = 8
lyWidth f64 = 8
lyWidth ptr = 8

-- Natural alignment. Kept a separate function from width even though they agree
-- here, because they are different facts: a packed struct field has its type's
-- width and a weaker alignment, and collapsing them would make the
-- misaligned-access fault unreachable for exactly the layouts that need it.
lyAlign : LyTy → ℕ
lyAlign i8  = 1
lyAlign i32 = 4
lyAlign i64 = 8
lyAlign f64 = 8
lyAlign ptr = 8

lyWidth-pos : ∀ τ → 0 < lyWidth τ
lyWidth-pos i8  = s≤s z≤n
lyWidth-pos i32 = s≤s z≤n
lyWidth-pos i64 = s≤s z≤n
lyWidth-pos f64 = s≤s z≤n
lyWidth-pos ptr = s≤s z≤n

lyAlign-pos : ∀ τ → 0 < lyAlign τ
lyAlign-pos i8  = s≤s z≤n
lyAlign-pos i32 = s≤s z≤n
lyAlign-pos i64 = s≤s z≤n
lyAlign-pos f64 = s≤s z≤n
lyAlign-pos ptr = s≤s z≤n

_≟-ly_ : (σ τ : LyTy) → Dec (σ ≡ τ)
i8  ≟-ly i8  = yes refl
i32 ≟-ly i32 = yes refl
i64 ≟-ly i64 = yes refl
f64 ≟-ly f64 = yes refl
ptr ≟-ly ptr = yes refl
i8  ≟-ly i32 = no λ ()
i8  ≟-ly i64 = no λ ()
i8  ≟-ly f64 = no λ ()
i8  ≟-ly ptr = no λ ()
i32 ≟-ly i8  = no λ ()
i32 ≟-ly i64 = no λ ()
i32 ≟-ly f64 = no λ ()
i32 ≟-ly ptr = no λ ()
i64 ≟-ly i8  = no λ ()
i64 ≟-ly i32 = no λ ()
i64 ≟-ly f64 = no λ ()
i64 ≟-ly ptr = no λ ()
f64 ≟-ly i8  = no λ ()
f64 ≟-ly i32 = no λ ()
f64 ≟-ly i64 = no λ ()
f64 ≟-ly ptr = no λ ()
ptr ≟-ly i8  = no λ ()
ptr ≟-ly i32 = no λ ()
ptr ≟-ly i64 = no λ ()
ptr ≟-ly f64 = no λ ()

LythonSig : ElemSig
LythonSig = record
  { ElemTy    = LyTy
  ; width     = lyWidth
  ; align     = lyAlign
  ; width-pos = lyWidth-pos
  ; align-pos = lyAlign-pos
  ; _≟-elem_  = _≟-ly_
  }
