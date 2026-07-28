{-# OPTIONS --safe #-}

-- Shared vocabulary. Deliberately thin: everything else in the kernel is
-- domain-specific, and a large prelude is where a model starts drifting from
-- the thing it is supposed to be modelling.

module Proof.Prelude where

open import Level using (Level; _⊔_)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc; _+_; _≤_; _<_; z≤n; s≤s)
open import Data.Product using (_×_; _,_)
open import Relation.Binary.PropositionalEquality using (_≡_; refl)
open import Relation.Nullary using (Dec; yes; no)

private
  variable
    a b : Level

-- Why an explicit Result and not Maybe: every failure in this development has a
-- NAME (Proof.Memory.Fault), and the point of the model is that MLIR's
-- "undefined behaviour" becomes a specific fault rather than a missing value.
-- `nothing` would erase exactly the information the theorems are about.
data Result (E : Set a) (A : Set b) : Set (a ⊔ b) where
  err : E → Result E A
  ok  : A → Result E A

-- `IsOk r` is the proposition that r succeeded, carrying the payload. Stating
-- postconditions against this rather than against a boolean keeps the value
-- available to the proof that needs it.
data IsOk {E : Set a} {A : Set b} : Result E A → Set (a ⊔ b) where
  is-ok : (x : A) → IsOk (ok x)

okValue : ∀ {E : Set a} {A : Set b} {r : Result E A} → IsOk r → A
okValue (is-ok x) = x

-- Sequencing, so that a check chain reads top-to-bottom as the list of
-- conditions it is. Written out with nested `with` instead, an eight-condition
-- resolve becomes eight levels of indentation and it stops being reviewable
-- whether every condition is actually tested.
infixl 1 _>>=ᴿ_
_>>=ᴿ_ : ∀ {E : Set a} {A B : Set b} → Result E A → (A → Result E B) → Result E B
err e >>=ᴿ _ = err e
ok  x >>=ᴿ f = f x

-- Each of these turns a *decision* into a *named fault*. The fault argument is
-- mandatory: there is no combinator here that fails anonymously, because an
-- anonymous failure is exactly the "undefined behaviour" this model exists to
-- replace with something specific.
guardᴿ : ∀ {E : Set a} {A : Set b} → Dec A → E → Result E A
guardᴿ (yes p) _ = ok p
guardᴿ (no  _) e = err e

maybeᴿ : ∀ {E : Set a} {A : Set b} → Maybe A → E → Result E A
maybeᴿ (just x) _ = ok x
maybeᴿ nothing  e = err e

-- A half-open byte interval [start, start + len).
record ByteRange : Set where
  constructor range
  field
    start : ℕ
    len   : ℕ

open ByteRange public

-- `Within r n` says the whole range lies inside a block of n bytes. Stated as
-- one inequality on the END rather than as a family of per-byte facts, so that
-- containment is a single arithmetic obligation.
Within : ByteRange → ℕ → Set
Within r n = (start r + len r) ≤ n

InRange : ByteRange → ℕ → Set
InRange r i = (start r ≤ i) × (i < start r + len r)

-- Two ranges overlap when some byte belongs to both. Phrased constructively --
-- an explicit witness index -- because the race predicate later has to point AT
-- the shared byte, not merely know that one exists.
record Overlaps (r s : ByteRange) : Set where
  constructor overlap-at
  field
    byte      : ℕ
    in-first  : InRange r byte
    in-second : InRange s byte

open Overlaps public
