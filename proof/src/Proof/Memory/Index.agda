{-# OPTIONS --safe #-}

-- Multi-dimensional indices that are in bounds BY CONSTRUCTION.
--
-- The note proposes `Ix sizes = (k : Fin rank) → Fin (lookup sizes k)`. That is
-- correct but it makes an index a FUNCTION, and then `i ≡ j` -- which the
-- layout-injectivity field of a well-formed descriptor is stated in terms of --
-- needs function extensionality, which `--safe` Agda does not give for free.
--
-- So the index is a first-order inductive family instead. It carries exactly
-- the same information, `i ≡ j` is ordinary propositional equality, and it has
-- decidable equality. The functional reading is recovered by `lookupIx`.

module Proof.Memory.Index where

open import Data.Fin using (Fin; zero; suc; toℕ)
open import Data.Nat using (ℕ)
open import Data.Vec using (Vec; []; _∷_; lookup)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; cong)

infixr 5 _∷_

data Ix : ∀ {r} → Vec ℕ r → Set where
  []  : Ix []
  _∷_ : ∀ {r n} {ns : Vec ℕ r} → Fin n → Ix ns → Ix (n ∷ ns)

-- The functional view. Everything the note writes as `i k` is `lookupIx i k`.
lookupIx : ∀ {r} {sizes : Vec ℕ r} → Ix sizes → (k : Fin r) → Fin (lookup sizes k)
lookupIx (x ∷ _)  zero    = x
lookupIx (_ ∷ xs) (suc k) = lookupIx xs k

-- Componentwise equality implies equality. This is the lemma that function
-- extensionality would have been needed for, and here it is provable.
ix-ext : ∀ {r} {sizes : Vec ℕ r} (i j : Ix sizes) →
         (∀ k → lookupIx i k ≡ lookupIx j k) → i ≡ j
ix-ext []       []       _ = refl
ix-ext (x ∷ xs) (y ∷ ys) f
  with f zero | ix-ext xs ys (λ k → f (suc k))
... | refl | refl = refl
