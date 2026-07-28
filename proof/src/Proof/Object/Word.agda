{-# OPTIONS --safe #-}

-- Machine words as byte sequences, with a round trip.
--
-- This module exists because "the object is one lane" is a claim about BYTES.
-- If the refcount were a ℕ stored in a ghost field beside the allocation, the
-- one-lane property would be true by construction and would say nothing. Here
-- the refcount is genuinely eight bytes inside the box's own allocation, and
-- reading it back is `load` at a word index -- so `decode (encode n) ≡ n` is the
-- lemma the whole design rests on.
--
-- Little-endian, base 256, least significant byte first.

module Proof.Object.Word where

open import Data.Fin using (Fin; toℕ; fromℕ<)
open import Data.Nat
open import Data.Nat.DivMod using (_%_; _/_; m%n<n; m≡m%n+[m/n]*n)
open import Data.Nat.Properties
open import Data.Vec using (Vec; []; _∷_)
open import Relation.Binary.PropositionalEquality
  using (_≡_; refl; sym; trans; cong; cong₂; module ≡-Reasoning)

open import Proof.Memory.Byte using (Byte)

-- Word width in bytes. Eight, because the box is a memref of i64 and a word
-- index into it is what a header field access will be.
WordBytes : ℕ
WordBytes = 8

------------------------------------------------------------------------
-- Encoding.

encodeAt : (k : ℕ) → ℕ → Vec Byte k
encodeAt zero    _ = []
encodeAt (suc k) n = fromℕ< (m%n<n n 256) ∷ encodeAt k (n / 256)

decode : ∀ {k} → Vec Byte k → ℕ
decode []       = 0
decode (b ∷ bs) = toℕ b + 256 * decode bs

encode : ℕ → Vec Byte WordBytes
encode = encodeAt WordBytes

------------------------------------------------------------------------
-- The round trip, and the bound it needs.
--
-- Unbounded it is FALSE: eight bytes hold 2^64 values and a larger ℕ comes back
-- reduced. So the statement carries its precondition, and every caller has to
-- supply it -- which is the honest form, since a real refcount that overflowed
-- its word would be a genuine defect and not a modelling artefact.

toℕ-fromℕ< : ∀ {m n} (lt : m < n) → toℕ (fromℕ< lt) ≡ m
toℕ-fromℕ< {zero}  {suc n} (s≤s z≤n) = refl
toℕ-fromℕ< {suc m} {suc n} (s≤s lt)  = cong suc (toℕ-fromℕ< lt)

Capacity : ℕ → ℕ
Capacity zero    = 1
Capacity (suc k) = 256 * Capacity k

decode-encodeAt : ∀ (k n : ℕ) → n < Capacity k → decode (encodeAt k n) ≡ n
decode-encodeAt zero zero _ = refl
decode-encodeAt zero (suc n) (s≤s ())
decode-encodeAt (suc k) n lt = begin
    toℕ (fromℕ< (m%n<n n 256)) + 256 * decode (encodeAt k (n / 256))
  ≡⟨ cong (_+ 256 * decode (encodeAt k (n / 256))) (toℕ-fromℕ< (m%n<n n 256)) ⟩
    n % 256 + 256 * decode (encodeAt k (n / 256))
  ≡⟨ cong (λ z → n % 256 + 256 * z) (decode-encodeAt k (n / 256) quotient-bound) ⟩
    n % 256 + 256 * (n / 256)
  ≡⟨ cong (n % 256 +_) (*-comm 256 (n / 256)) ⟩
    n % 256 + (n / 256) * 256
  ≡⟨ sym (m≡m%n+[m/n]*n n 256) ⟩
    n
  ∎
  where
    open ≡-Reasoning
    quotient-bound : n / 256 < Capacity k
    quotient-bound = *-cancelˡ-< 256 (n / 256) (Capacity k) helper
      where
        helper : 256 * (n / 256) < 256 * Capacity k
        helper = ≤-trans (s≤s (m≤n+m (256 * (n / 256)) (n % 256))) lt-shifted
          where
            lt-shifted : suc (n % 256 + 256 * (n / 256)) ≤ 256 * Capacity k
            lt-shifted = subst (λ z → suc z ≤ 256 * Capacity k)
                               (trans (m≡m%n+[m/n]*n n 256)
                                      (cong (n % 256 +_) (*-comm (n / 256) 256)))
                               lt
              where open import Relation.Binary.PropositionalEquality using (subst)

decode-encode : ∀ (n : ℕ) → n < Capacity WordBytes → decode (encode n) ≡ n
decode-encode = decode-encodeAt WordBytes
