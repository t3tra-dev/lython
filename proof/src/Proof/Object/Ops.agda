{-# OPTIONS --safe #-}

-- alloc, free, move, retain, release, and resize -- all through the one lane.
--
-- The design being checked here is the note's §11 split:
--
--     stable box                       resizable buffer
--     ┌──────────────────┐             ┌────────────────────┐
--     │ rc class len cap │──alloc,gen──▶│ elements …         │
--     └──────────────────┘             └────────────────────┘
--
-- The box never moves. Resizing reallocates the BUFFER and rewrites two words
-- inside the box, so every alias of the object -- every copy of the one
-- descriptor -- sees the new buffer immediately. `memref.realloc` applied to the
-- object itself could not do that: it invalidates aliases it cannot reach.
--
-- That is the whole argument for the split, and `resize-preserves-references`
-- in Proof.Object.Coherence is it as a theorem.

open import Proof.Object.WordSig using (WordSig)

module Proof.Object.Ops (W : WordSig) where

open WordSig W

open import Data.Fin using (Fin; toℕ)
open import Data.Integer using (ℤ; +_)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc; _+_; _<_; s≤s; z≤n; _≟_)
open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Data.Vec using (Vec; []; _∷_)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; sym; subst)
open import Relation.Nullary using (Dec; yes; no)

open import Proof.Prelude using (Result; err; ok; _>>=ᴿ_; guardᴿ)
open import Proof.Memory.Byte using (Byte)
open import Proof.Memory.Element using (MemSpace)
open import Proof.Memory.Fault using (MemoryFault; invalid-free; use-after-free)
open import Proof.Memory.Heap using (Heap; AllocId; Generation; freshId)
open import Proof.Memory.Index using (Ix; []; _∷_)
open import Proof.Memory.Descriptor sig
open import Proof.MemRef.Dialect sig using (alloc; dealloc; load; store)
open import Proof.MemRef.Realloc sig using (realloc)
open import Proof.Object.Layout
open import Proof.Object.Word using (WordBytes)
open import Proof.Object.Box W

------------------------------------------------------------------------
-- Faults of the object layer.

data ObjFault : Set where
  -- Freeing an object that something still refers to. Caught HERE rather than
  -- becoming a use-after-free at the next load, which is the entire reason the
  -- refcount is in the box.
  refcount-not-zero : ObjFault
  -- The header said one thing and the heap another.
  malformed-box     : ObjFault
  memory            : MemoryFault → ObjFault

lift : ∀ {A : Set} → Result MemoryFault A → Result ObjFault A
lift (err e) = err (memory e)
lift (ok x)  = ok x

------------------------------------------------------------------------
-- alloc
--
-- One allocation, one descriptor, header initialised. The refcount starts at 1
-- because the caller holds it: starting at 0 would mean the object is
-- reclaimable the instant it exists, and every constructor would have to race
-- to retain it.

newObject : (inline : ℕ) → Heap → MemSpace → (classId : ℕ) →
            Result ObjFault (Heap × Box inline)
newObject inline h sp classId =
  lift (setRefcount inline h₁ b shaped 1)       >>=ᴿ λ h₂ →
  lift (setClass    inline h₂ b shaped classId) >>=ᴿ λ h₃ →
  lift (setLength   inline h₃ b shaped 0)       >>=ᴿ λ h₄ →
  lift (setCapacity inline h₄ b shaped 0)       >>=ᴿ λ h₅ →
  lift (setBuffer   inline h₅ b shaped 0 0)     >>=ᴿ λ h₆ →
  ok (h₆ , b)
  where
    -- `b` is a DEFINITION rather than a pattern variable, so it still unfolds to
    -- the descriptor alloc built and `shaped` can be four `refl`s. Bound by a
    -- pattern match instead, Agda forgets where it came from and the shape
    -- witness becomes unprovable -- the object layer would then have to take the
    -- invariant on trust from its own constructor.
    allocated : Heap × Desc 1
    allocated = alloc h sp word (boxWords inline) WordBytes

    h₁ : Heap
    h₁ = proj₁ allocated

    b : Box inline
    b = proj₂ allocated

    shaped : WellShaped inline b
    shaped = record { is-word-typed = refl ; spans-box = refl
                    ; unit-stride = refl ; at-origin = refl }

------------------------------------------------------------------------
-- retain / release
--
-- Both are read-modify-write ON THE BOX'S OWN FIRST WORD. No side table, no
-- separate refcount object -- which is what makes them work without knowing the
-- object's class, exactly as CPython's Py_INCREF does.

retain : ∀ (inline : ℕ) → Heap → (b : Box inline) → WellShaped inline b →
         Result ObjFault Heap
retain inline h b ws =
  lift (refcountOf inline h b ws) >>=ᴿ λ n →
  lift (setRefcount inline h b ws (suc n))

release : ∀ (inline : ℕ) → Heap → (b : Box inline) → WellShaped inline b →
          Result ObjFault Heap
release inline h b ws =
  lift (refcountOf inline h b ws) >>=ᴿ λ n →
  lift (setRefcount inline h b ws (pred′ n))
  where
    -- Saturating, because the model must not make `0 - 1` a large number: an
    -- underflowed refcount is an object that never dies, and it would present
    -- as a leak with no other symptom.
    pred′ : ℕ → ℕ
    pred′ zero    = zero
    pred′ (suc k) = k

------------------------------------------------------------------------
-- move
--
-- Copying the descriptor. It is deliberately the identity on the heap: the
-- whole content of "a move costs nothing" is that no byte and no count changes,
-- and a `move` that touched the refcount would be a dup.

moveObject : ∀ (inline : ℕ) → Box inline → Box inline
moveObject inline b = b

------------------------------------------------------------------------
-- free
--
-- Refuses unless the refcount word reads zero. Note the ORDER: the count is
-- read through the lane *before* the allocation is released, because after
-- `dealloc` the same read is a use-after-free -- the model would then be unable
-- to state its own precondition.

freeObject : ∀ (inline : ℕ) → Heap → (b : Box inline) → WellShaped inline b →
             Result ObjFault Heap
freeObject inline h b ws =
  lift (refcountOf inline h b ws) >>=ᴿ λ n →
  guardᴿ (n ≟ 0) refcount-not-zero >>=ᴿ λ _ →
  lift (dealloc h b)

------------------------------------------------------------------------
-- resize
--
-- The operation the split exists for. It reallocates the BUFFER and rewrites
-- words 4 and 5 of the box. The box's own allocation is untouched, so:
--
--   * every existing reference to the object stays valid, and
--   * every one of them reads the NEW buffer, because they all read word 4.
--
-- Compare with resizing the object itself: that invalidates every alias, and
-- there is no way to reach them. This is the design decision the note argues
-- for, and here it is a difference in what is provable rather than a preference.

resizeBuffer : ∀ (inline : ℕ) → Heap → (b : Box inline) → WellShaped inline b →
               (bufferDesc : Desc 1) → (newCount : ℕ) →
               Result ObjFault (Heap × Desc 1)
resizeBuffer inline h b ws buf newCount =
  lift (realloc h buf newCount) >>=ᴿ λ hd →
  lift (setBuffer inline (proj₁ hd) b ws
                  (allocation (proj₂ hd)) (generation (proj₂ hd))) >>=ᴿ λ h' →
  lift (setCapacity inline h' b ws newCount) >>=ᴿ λ h'' →
  ok (h'' , proj₂ hd)
