{-# OPTIONS --safe #-}

-- Turning (heap, descriptor, index) into a byte range, or into a named fault.
--
-- Every condition MLIR leaves as a precondition is a check here, and each one
-- has its own fault constructor. That is the whole design: `memref.load` with
-- an out-of-range index is undefined behaviour in MLIR, and undefined behaviour
-- modelled as "anything may happen" makes the safety theorem unprovable, while
-- undefined behaviour modelled as a *specific* fault makes it a statement about
-- reachability.

open import Proof.Memory.Element using (ElemSig; MemSpace)

module Proof.Memory.Resolve (Sig : ElemSig) where

open ElemSig Sig

open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; _+_; _≤?_; _≟_)
open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Data.Nat.Divisibility using (_∣_; _∣?_)
open import Data.Vec using (Vec)
open import Relation.Binary.PropositionalEquality using (_≡_; refl)
open import Relation.Nullary using (Dec; yes; no)

open import Proof.Prelude using (Result; err; ok; ByteRange; range; _>>=ᴿ_; guardᴿ; maybeᴿ)
open import Proof.Memory.Fault using (MemoryFault; out-of-bounds; use-after-free;
  stale-generation; misaligned-access; invalid-memory-space; no-such-allocation)
open import Proof.Memory.Heap using (Heap; Block; block; AllocId; Generation;
  Liveness; live; dead; lookupBlock; generation; space; sizeBytes; liveness)
open import Proof.Memory.Index using (Ix)
open import Proof.Memory.Descriptor Sig

-- Liveness is two constructors, so its decision is one line rather than an
-- imported instance.
live? : (l : Liveness) → Dec (l ≡ live)
live? live = yes refl
live? dead = no λ ()

-- Returns the block as well as the range. `load` and `store` both need it, and
-- having them re-look-up the allocation would mean the bytes they touch are
-- only *probably* the ones the checks were performed against -- a second lookup
-- is a second chance to disagree.
resolveIn : Heap → ∀ {r} (d : Desc r) → Ix (sizes d) →
            Result MemoryFault (Block × ByteRange)
resolveIn h d i =
  -- 1. the allocation exists at all
  maybeᴿ (lookupBlock h (allocation d)) no-such-allocation >>=ᴿ λ b →
  -- 2. the generation matches. Separate from (3) on purpose: a live block whose
  --    generation differs is a DIFFERENT allocation that happens to reuse the
  --    id, which is not the same failure as touching a freed one.
  guardᴿ (generation d ≟ generation b) stale-generation >>=ᴿ λ _ →
  -- 3. the block has not been freed
  guardᴿ (live? (liveness b)) use-after-free >>=ᴿ λ _ →
  -- 4. the memory space is the one the block was allocated in
  guardᴿ (memorySpace d ≟ space b) invalid-memory-space >>=ᴿ λ _ →
  -- 5. the computed origin is not before the start of the allocation. This is
  --    where a negative stride that walks off the front is caught.
  maybeᴿ (nonNeg (byteStart d i)) out-of-bounds >>=ᴿ λ s →
  -- 6. the whole element, not just its first byte, is inside the allocation
  guardᴿ ((s + elemWidth d) ≤? sizeBytes b) out-of-bounds >>=ᴿ λ _ →
  -- 7. the access is aligned for its element type
  guardᴿ (elemAlign d ∣? s) misaligned-access >>=ᴿ λ _ →
  ok (b , range s (elemWidth d))

resolve : Heap → ∀ {r} (d : Desc r) → Ix (sizes d) → Result MemoryFault ByteRange
resolve h d i = resolveIn h d i >>=ᴿ λ br → ok (proj₂ br)
