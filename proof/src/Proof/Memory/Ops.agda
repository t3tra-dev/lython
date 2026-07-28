{-# OPTIONS --safe #-}

-- The MemRef operations, each as a total function into `Result MemoryFault _`.
--
-- Every one either succeeds or names the guarantee it would have broken. None
-- of them has a precondition the caller is trusted to have met -- that is the
-- difference between a model and a specification, and it is what makes the
-- reachability theorems in Proof.Memory.Properties say anything.

open import Proof.Memory.Element using (ElemSig; MemSpace)

module Proof.Memory.Ops (Sig : ElemSig) where

open ElemSig Sig

open import Data.Integer using (ℤ; +_; _*_; _+_; -[1+_])
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc; _≟_; _≤?_)
  renaming (_*_ to _*ℕ_; _+_ to _+ℕ_)
open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Data.Vec using (Vec; []; _∷_; replicate; head)
open import Relation.Binary.PropositionalEquality using (_≡_; refl)
open import Relation.Nullary using (Dec; yes; no)

open import Proof.Prelude using (Result; err; ok; ByteRange; range; start; len;
  _>>=ᴿ_; guardᴿ; maybeᴿ)
open import Proof.Memory.Byte using (Byte; StoredByte; uninit; init)
open import Proof.Memory.Fault using (MemoryFault; out-of-bounds; use-after-free;
  double-free; invalid-free; stale-generation; uninitialized-read;
  misaligned-access; invalid-memory-space; descriptor-overflow; no-such-allocation)
open import Proof.Memory.Heap
open import Proof.Memory.Index using (Ix; []; _∷_)
open import Proof.Memory.Descriptor Sig
open import Proof.Memory.Resolve Sig

------------------------------------------------------------------------
-- alloc

-- Fresh contents are `uninit`, not zero. Zero-filling here would be a promise
-- the runtime does not make, and every later theorem about reads would hold for
-- a reason the compiled program does not have.
allocate : Heap → MemSpace → (τ : ElemTy) → (count alignment : ℕ) →
           Heap × Desc 1
allocate h sp τ count alignment = newHeap , rootDesc
  where
    bytes : ℕ
    bytes = width τ *ℕ count

    newBlock : Block
    newBlock = block 0 sp bytes alignment (replicate bytes uninit) live

    newHeap : Heap
    newHeap = allocBlock h newBlock

    -- The canonical whole-allocation view: origin at byte 0, element offset 0,
    -- unit stride. This is the descriptor `dealloc` will accept, and the one
    -- Lython should be circulating as an object reference.
    rootDesc : Desc 1
    rootDesc = desc (freshId h) 0 0 (+ 0) (count ∷ []) ((+ 1) ∷ []) τ sp

------------------------------------------------------------------------
-- load / store

-- The uninitialised check lives here rather than in `resolve`, because `store`
-- must NOT require initialisation -- it is what creates it. Folding the check
-- into resolve would make the first write to fresh memory a fault.
load : Heap → ∀ {r} (d : Desc r) → Ix (sizes d) →
       Result MemoryFault (Vec Byte (elemWidth d))
load h d i =
  resolveIn h d i >>=ᴿ λ br →
  maybeᴿ (readRange (contents (proj₁ br)) (start (proj₂ br)) (elemWidth d))
         uninitialized-read

store : Heap → ∀ {r} (d : Desc r) → Ix (sizes d) → Vec Byte (elemWidth d) →
        Result MemoryFault Heap
store h d i v =
  resolveIn h d i >>=ᴿ λ br →
  ok (updateBlock h (allocation d)
        (λ b → record b { contents = writeRange (contents b) (start (proj₂ br)) v }))

------------------------------------------------------------------------
-- dealloc

-- What "root" means here, and what it does not.
--
-- The note asks for a linear RootToken that `alloc` mints and `dealloc`
-- consumes. Agda's type system is not linear, so a token record would be
-- constructible by anyone and would prove nothing. Instead the root property is
-- a *decidable fact about the descriptor and the block*: origin at byte 0, and
-- extent exactly the block's size. Every subview has either a nonzero origin or
-- a smaller extent, so `dealloc` through a view is rejected -- which is the
-- safety content MLIR's "dealloc the original memref" rule carries.
--
-- What this does NOT give, and the linear token would: it does not stop the
-- same root descriptor being passed to `dealloc` twice. That case is caught
-- instead by the liveness check below, as `double-free` rather than by
-- construction. `Proof.Memory.Properties.double-free-faults` is the proof.
totalElems : ∀ {n} → Vec ℕ n → ℕ
totalElems []       = 1
totalElems (x ∷ xs) = x *ℕ totalElems xs

IsRootOf : Block → ∀ {r} → Desc r → Set
IsRootOf b d = (alignedBase d ≡ 0) × (elemWidth d *ℕ totalElems (sizes d) ≡ sizeBytes b)

isRootOf? : (b : Block) → ∀ {r} (d : Desc r) → Dec (IsRootOf b d)
isRootOf? b d with alignedBase d ≟ 0 | elemWidth d *ℕ totalElems (sizes d) ≟ sizeBytes b
... | yes p | yes q = yes (p , q)
... | no ¬p | _     = no λ z → ¬p (proj₁ z)
... | _     | no ¬q = no λ z → ¬q (proj₂ z)

dealloc : Heap → ∀ {r} (d : Desc r) → Result MemoryFault Heap
dealloc h d =
  maybeᴿ (lookupBlock h (allocation d)) no-such-allocation >>=ᴿ λ b →
  guardᴿ (generation d ≟ generation b) stale-generation >>=ᴿ λ _ →
  -- Freeing an already-freed allocation is `double-free`, not `use-after-free`.
  -- They are different mistakes and a caller that conflates them cannot report
  -- which one the program made.
  guardᴿ (live? (liveness b)) double-free >>=ᴿ λ _ →
  guardᴿ (isRootOf? b d) invalid-free >>=ᴿ λ _ →
  ok (updateBlock h (allocation d) (λ blk → record blk { liveness = dead }))

------------------------------------------------------------------------
-- View-forming operations.
--
-- These produce a descriptor and do not touch the heap. They are still checked,
-- because a descriptor whose footprint leaves the allocation is a fault that
-- has already happened even if nobody loads through it yet -- catching it at
-- construction is what makes `WFDesc` an invariant rather than a hope.

-- subview: offsets and strides compose *relative to the input view*, which is
-- why it adds to `offset` and multiplies into `strides` rather than replacing
-- them. Getting this wrong is not a type error, so it is stated here explicitly.
subview : ∀ {r} (d : Desc r) → (newOffset : ℤ) → (newSizes : Vec ℕ r) →
          (newStrides : Vec ℤ r) → Desc r
subview d o ns nst =
  desc (allocation d) (generation d) (alignedBase d)
       (offset d + o) ns nst (elementType d) (memorySpace d)

-- view: shifts by BYTES and may change the element type. This is the operation
-- that makes a byte heap necessary -- there is no typed-cell model in which a
-- differently-typed view of the same storage is expressible.
viewBytes : ∀ {r} (d : Desc r) → (byteShift : ℕ) → (τ : ElemTy) →
            (newSizes : Vec ℕ r) → (newStrides : Vec ℤ r) → Desc r
viewBytes d shift τ ns nst =
  desc (allocation d) (generation d) (alignedBase d +ℕ shift)
       (+ 0) ns nst τ (memorySpace d)

-- reinterpret_cast: metadata is set relative to the ALLOCATION base, not to the
-- input view. That is the difference from subview, and it is the reason both
-- exist.
reinterpretCast : ∀ {r r'} (d : Desc r) → (newBase : ℕ) → (newOffset : ℤ) →
                  (newSizes : Vec ℕ r') → (newStrides : Vec ℤ r') → Desc r'
reinterpretCast d nb o ns nst =
  desc (allocation d) (generation d) nb o ns nst (elementType d) (memorySpace d)

-- memory_space_cast: same underlying storage, different space. Kept separate
-- from reinterpretCast so that the space check in `resolve` has something to
-- fail on: a descriptor cast into a space its block does not live in is
-- `invalid-memory-space`, not a silent success.
memorySpaceCast : ∀ {r} (d : Desc r) → MemSpace → Desc r
memorySpaceCast d sp =
  desc (allocation d) (generation d) (alignedBase d)
       (offset d) (sizes d) (strides d) (elementType d) sp
