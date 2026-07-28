{-# OPTIONS --safe #-}

-- An element signature together with a designated WORD type.
--
-- The object layer needs one element type whose width is the word size, because
-- every field of a box is a word index. Bundling it here rather than fixing
-- `i64` keeps the object design independent of any particular backend's element
-- enum -- which is the point, since this layer is a redesign and not a
-- transcription of what the compiler currently emits.

module Proof.Object.WordSig where

open import Data.Nat using (ℕ)
open import Relation.Binary.PropositionalEquality using (_≡_)

open import Proof.Memory.Element using (ElemSig)
open import Proof.Object.Word using (WordBytes)

record WordSig : Set₁ where
  field
    sig : ElemSig

  open ElemSig sig public

  field
    word : ElemTy

    -- The two equations the accessors need. `word-width` is what makes a
    -- `load` at a word index return exactly a word; `word-align` is what makes
    -- every word index naturally aligned, so no field access can be a
    -- misaligned-access fault.
    word-width : width word ≡ WordBytes
    word-align : align word ≡ WordBytes
