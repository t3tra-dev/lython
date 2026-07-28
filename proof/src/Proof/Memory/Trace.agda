{-# OPTIONS --safe #-}

-- A concrete trace, checked by computation.
--
-- Every proof in Proof.Memory.Properties is conditional: "if the lookup
-- succeeds and the generation matches and the block is live, then ...". A model
-- in which no heap ever satisfies those hypotheses would satisfy all of them and
-- describe nothing. These are unconditional equations about actual heaps, and
-- each `refl` is the typechecker running the model and agreeing.
--
-- They are also the answer to "does this thing distinguish the faults, or does
-- it just have constructors named after them": `use-after-free` and
-- `double-free` below are produced, not asserted.

module Proof.Memory.Trace where

open import Data.Fin using (Fin; zero; suc)
open import Data.Integer using (ℤ; +_)
open import Data.List using (List; [])
open import Data.Nat using (ℕ)
open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Data.Vec using (Vec; []; _∷_)
open import Relation.Binary.PropositionalEquality using (_≡_; refl)

open import Proof.Prelude using (Result; err; ok)
open import Proof.Memory.Byte using (Byte; StoredByte; uninit; init)
open import Proof.Memory.Fault
open import Proof.Memory.Heap using (Heap; AllocId)
open import Proof.Memory.Index using (Ix; []; _∷_)
open import Proof.Memory.Lython using (LythonSig; LyTy; i8; i64)

open import Proof.Memory.Descriptor LythonSig
open import Proof.Memory.Resolve    LythonSig
open import Proof.Memory.Ops        LythonSig

------------------------------------------------------------------------
-- A four-byte allocation in memory space 0.

h₀ : Heap
h₀ = []

allocated : Heap × Desc 1
allocated = allocate h₀ 0 i8 4 1

h₁ : Heap
h₁ = proj₁ allocated

d : Desc 1
d = proj₂ allocated

-- Index 2 of 4.
i₂ : Ix (sizes d)
i₂ = suc (suc zero) ∷ []

byte₇ : Byte
byte₇ = suc (suc (suc (suc (suc (suc (suc zero))))))

------------------------------------------------------------------------
-- 1. Reading fresh storage is a fault, not a value.

fresh-read : load h₁ d i₂ ≡ err uninitialized-read
fresh-read = refl

------------------------------------------------------------------------
-- 2. Store, then load, returns what was stored.
--
-- This is the direction that a model can fail SILENTLY: a heap whose store is a
-- no-op satisfies every safety theorem in this development, because it never
-- produces a wrong value -- it produces no value at all. So it has to be
-- checked separately from the faults.

h₂ : Heap
h₂ with store h₁ d i₂ (byte₇ ∷ [])
... | ok h  = h
... | err _ = h₁

round-trip : load h₂ d i₂ ≡ ok (byte₇ ∷ [])
round-trip = refl

-- And the neighbouring element is untouched: the write went to byte 2, not to
-- the whole block. Without this, a store that filled the allocation would pass
-- the round-trip test above.
neighbour-untouched : load h₂ d (suc zero ∷ []) ≡ err uninitialized-read
neighbour-untouched = refl

------------------------------------------------------------------------
-- 3. Free, then use.

h₃ : Heap
h₃ with dealloc h₂ d
... | ok h  = h
... | err _ = h₂

freeing-works : dealloc h₂ d ≡ ok h₃
freeing-works = refl

use-after-free-caught : load h₃ d i₂ ≡ err use-after-free
use-after-free-caught = refl

-- The same descriptor, freed a second time. `double-free`, and specifically not
-- `use-after-free` -- the model tells the two apart.
double-free-caught : dealloc h₃ d ≡ err double-free
double-free-caught = refl

------------------------------------------------------------------------
-- 4. Bounds.

out-of-bounds-is-unreachable-by-typing : Ix (sizes d) → ℕ
out-of-bounds-is-unreachable-by-typing _ = 0

-- There is no `Fin 4` beyond 3, so an out-of-range index cannot be BUILT. That
-- is the "in bounds by construction" half. The `out-of-bounds` fault is for the
-- other half: a descriptor whose own footprint leaves its allocation, which
-- typing cannot rule out because the sizes are runtime data.
over-long : Desc 1
over-long = desc (allocation d) (generation d) 0 (+ 0) (8 ∷ []) ((+ 1) ∷ []) i8 0

descriptor-overruns-allocation : resolve h₂ over-long (suc (suc (suc (suc zero))) ∷ [])
                                   ≡ err out-of-bounds
descriptor-overruns-allocation = refl

------------------------------------------------------------------------
-- 5. Alignment, and why `align` is not `width`.
--
-- This needs its own, larger allocation, and the reason is worth recording: an
-- i64 view at byte 1 of the FOUR-byte block above is out of bounds before it is
-- misaligned, because `resolve` checks containment first. The first version of
-- this section asserted `misaligned-access` there and the typechecker rejected
-- it with `out-of-bounds != misaligned-access`. The order of the checks in
-- `resolve` is observable, and a concrete trace is what makes it so.

wide : Heap × Desc 1
wide = allocate h₂ 0 i8 16 8

h₄ : Heap
h₄ = proj₁ wide

-- In bounds (1 + 8 ≤ 16) and still misaligned, so the alignment check is the
-- one that fires. A model with `align = λ _ → 1` would accept this.
misaligned : Desc 1
misaligned = desc (allocation (proj₂ wide)) (generation (proj₂ wide))
                  1 (+ 0) (1 ∷ []) ((+ 1) ∷ []) i64 0

misalignment-caught : resolve h₄ misaligned (zero ∷ []) ≡ err misaligned-access
misalignment-caught = refl

-- The same view at byte 8 is aligned, and resolves. Without this the lemma
-- above would also hold for a `resolve` that rejected every i64.
aligned-view : Desc 1
aligned-view = desc (allocation (proj₂ wide)) (generation (proj₂ wide))
                    8 (+ 0) (1 ∷ []) ((+ 1) ∷ []) i64 0

alignment-is-not-a-blanket-refusal :
  load h₄ aligned-view (zero ∷ []) ≡ err uninitialized-read
alignment-is-not-a-blanket-refusal = refl

------------------------------------------------------------------------
-- 6. A view into a space the block does not live in.

elsewhere : Desc 1
elsewhere = memorySpaceCast d 1

wrong-space-caught : resolve h₂ elsewhere i₂ ≡ err invalid-memory-space
wrong-space-caught = refl

------------------------------------------------------------------------
-- 7. A subview cannot free its parent.

sub : Desc 1
sub = subview d (+ 1) (2 ∷ []) ((+ 1) ∷ [])

subview-cannot-free : dealloc h₂ sub ≡ err invalid-free
subview-cannot-free = refl

-- but it can still read through it, at an index of its own
subview-reads : load h₂ sub (suc zero ∷ []) ≡ ok (byte₇ ∷ [])
subview-reads = refl
