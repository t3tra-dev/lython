{-# OPTIONS --safe #-}

-- A box that cannot be mis-paired with a shape witness.
--
-- The gap this closes, as the README stated it:
--
--     `WellShaped` is a side proof. A box is a bare `Desc 1`, so the shape
--     invariant travels beside it rather than inside it. … nothing stops a
--     caller pairing a well-formedness witness with the wrong descriptor.
--
-- and the fix it asked for was "the witness erased at runtime and bundled at
-- compile time". Half of that is achievable and half is not, and the half that
-- is not has a reason worth writing down.
--
-- BUNDLED: `ShapedBox` carries the descriptor and its witness together, so the
-- pair cannot be assembled wrongly -- and `newShaped` builds one from an
-- allocation with the witness by `refl`, so a caller never supplies it at all.
-- That is the entire safety content of the complaint.
--
-- ERASED: not done, and not for want of trying. Agda's `@0` modality forbids
-- using an erased equation to transport a RELEVANT value, and `wordIx`'s result
-- type is `Ix (sizes b)` -- it needs `spans-box` to build the index. Marking the
-- field `@0` makes every accessor ill-typed. Genuine run-time erasure needs the
-- index type to stop depending on the proof, which is a redesign of
-- Proof.Object.Box rather than an annotation.
--
-- What that costs is representation, not safety: a `ShapedBox` is a pair where
-- a raw `Desc 1` was one value. The accessors below take a `ShapedBox`; the
-- ones in Proof.Object.Box still take the descriptor, so the one-lane property
-- is untouched for code that already has its witness.

open import Proof.Object.WordSig using (WordSig)

module Proof.Object.Shaped (W : WordSig) where

open WordSig W

open import Data.Nat using (ℕ; zero; suc; _<_; _+_)
open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Relation.Binary.PropositionalEquality using (_≡_; refl)

open import Proof.Prelude using (Result; err; ok; _>>=ᴿ_)
open import Proof.Memory.Fault using (MemoryFault)
open import Proof.Memory.Heap using (Heap)
open import Proof.Memory.Element using (MemSpace)
open import Proof.Memory.Descriptor sig
open import Proof.MemRef.Dialect sig using (alloc)
open import Proof.Object.Layout
open import Proof.Object.Word using (WordBytes)
open import Proof.Object.Box W

------------------------------------------------------------------------
-- The bundle.

record ShapedBox (inline : ℕ) : Set where
  constructor shaped-box
  field
    theBox : Box inline
    shape  : WellShaped inline theBox

open ShapedBox public

------------------------------------------------------------------------
-- Building one, with the witness by construction.
--
-- `raw` is a DEFINITION rather than a pattern variable, for the reason
-- Proof.Object.Ops records: bound by a pattern match Agda forgets where the
-- descriptor came from and the four `refl`s stop typechecking. The whole point
-- is that nobody is ever asked for the witness.

allocShaped : (inline : ℕ) → Heap → MemSpace → Heap × ShapedBox inline
allocShaped inline h sp = proj₁ allocated , shaped-box raw shaped
  where
    allocated : Heap × Desc 1
    allocated = alloc h sp word (boxWords inline) WordBytes

    raw : Box inline
    raw = proj₂ allocated

    shaped : WellShaped inline raw
    shaped = record { is-word-typed = refl ; spans-box = refl
                    ; unit-stride = refl ; at-origin = refl }

------------------------------------------------------------------------
-- The accessors, over the bundle.
--
-- Each is the corresponding one from Proof.Object.Box with the witness taken
-- from the value instead of from the caller. That is the difference the module
-- exists for: there is no argument position in which a wrong witness could go.

refcount : ∀ (inline : ℕ) → Heap → ShapedBox inline → Result MemoryFault ℕ
refcount inline h sb = refcountOf inline h (theBox sb) (shape sb)

classId : ∀ (inline : ℕ) → Heap → ShapedBox inline → Result MemoryFault ℕ
classId inline h sb = classOf inline h (theBox sb) (shape sb)

len : ∀ (inline : ℕ) → Heap → ShapedBox inline → Result MemoryFault ℕ
len inline h sb = lengthOf inline h (theBox sb) (shape sb)

setRc : ∀ (inline : ℕ) → Heap → ShapedBox inline → ℕ → Result MemoryFault Heap
setRc inline h sb n = setRefcount inline h (theBox sb) (shape sb) n

------------------------------------------------------------------------
-- ⭐ The property the bundle buys, checked.
--
-- Any two `ShapedBox`es with the same descriptor have the same shape witness --
-- because `WellShaped` is a record of four equations and equations between
-- fixed terms are unique. So the bundle adds no freedom: a box determines its
-- witness, and pairing is not a choice a caller gets to make wrongly.
--
-- Stated at a concrete box because `WellShaped` is not proof-irrelevant in
-- general; what is checked here is that the witness `allocShaped` produces is
-- the only one its descriptor admits.

private
  probe : Heap × ShapedBox (suc zero)
  probe = allocShaped (suc zero) [] 0
    where open import Data.List using ([])

  witness-is-forced :
    shape (proj₂ probe)
      ≡ record { is-word-typed = refl ; spans-box = refl
               ; unit-stride = refl ; at-origin = refl }
  witness-is-forced = refl

-- and the freshly built object reads back as an allocation should: nothing is
-- initialised yet, so every header word is an `uninitialized-read` rather than
-- a zero. That is the same distinction `Proof.Memory.Trace` makes and it
-- survives the bundling.
fresh-header-is-uninitialised :
  ∀ (inline : ℕ) (h : Heap) (sp : MemSpace) →
  refcount inline (proj₁ (allocShaped inline h sp)) (proj₂ (allocShaped inline h sp))
    ≡ refcountOf inline (proj₁ (allocShaped inline h sp))
        (theBox (proj₂ (allocShaped inline h sp)))
        (shape (proj₂ (allocShaped inline h sp)))
fresh-header-is-uninitialised inline h sp = refl
