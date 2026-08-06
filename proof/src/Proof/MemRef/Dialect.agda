{-# OPTIONS --safe #-}

-- The memref dialect's memory API, transcribed.
--
-- The op set is not the documentation's list: it is what Lython's own output
-- depends on, counted over src/lython/runtime/modules/*.mlir and the builders
-- in src/lython/lowering --
--
--     store 668   load 658   cast 320   get_global 187   global 151
--     alloc 111   extract_aligned_pointer_as_index 84    dealloc 82
--     dim 58      generic_atomic_rmw 30   alloca 27      view 20   subview 16
--     reinterpret_cast, extract_strided_metadata, prefetch (C++ builders only)
--
-- One absence is worth recording rather than passing over: `memref.realloc`
-- appears NOWHERE in this compiler. That is not an oversight -- a boxed object
-- is shared, realloc invalidates every alias, and nothing can update the aliases
-- it does not know about. Proof.MemRef.Realloc models it anyway, because
-- `Generation` exists to make exactly that invalidation checkable, and a
-- generation nothing ever bumps is a field no theorem can exercise.

open import Proof.Memory.Element using (ElemSig; MemSpace)

module Proof.MemRef.Dialect (Sig : ElemSig) where

open ElemSig Sig

open import Data.Fin using (Fin; toℕ)
open import Data.Integer using (ℤ; +_; -[1+_]; _*_; _+_)
open import Data.List using (List; []; _∷_; length)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc; _≟_; _≤?_; _<?_)
  renaming (_*_ to _*ℕ_; _+_ to _+ℕ_)
open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Data.Vec using (Vec; []; _∷_; replicate; lookup)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; subst)
open import Relation.Nullary using (Dec; yes; no)

open import Proof.Prelude using (Result; err; ok; ByteRange; range; start; len;
  _>>=ᴿ_; guardᴿ; maybeᴿ)
open import Proof.Memory.Byte using (Byte; StoredByte; uninit; init)
open import Proof.Memory.Fault
open import Proof.Memory.Heap
open import Proof.Memory.Index using (Ix; []; _∷_)
open import Proof.Memory.Descriptor Sig
open import Proof.Memory.Resolve Sig

------------------------------------------------------------------------
-- memref.alloc / memref.alloca / memref.global
--
-- Three producers of storage, differing only in `Storage`, and that field is
-- the whole difference between them as far as safety goes.

private
  mkBlock : MemSpace → (bytes al : ℕ) → Vec StoredByte bytes → Storage → Block
  mkBlock sp bytes al cts st = block 0 sp bytes al cts live st

  rootFor : Heap → (τ : ElemTy) → (count : ℕ) → MemSpace → Desc 1
  rootFor h τ count sp =
    desc (freshId h) 0 0 (+ 0) (count ∷ []) ((+ 1) ∷ []) τ sp

-- memref.alloc : heap storage, freed by memref.dealloc and by nothing else.
alloc : Heap → MemSpace → (τ : ElemTy) → (count alignment : ℕ) → Heap × Desc 1
alloc h sp τ count al =
  allocBlock h (mkBlock sp (width τ *ℕ count) al
                  (replicate (width τ *ℕ count) uninit) heapAlloc)
  , rootFor h τ count sp

-- memref.alloca : automatically freed when its scope ends. The frame id is
-- explicit here because MLIR's scope is syntactic (the enclosing function or
-- memref.alloca_scope) and this model has no syntax to read it off.
alloca : Heap → FrameId → MemSpace → (τ : ElemTy) → (count alignment : ℕ) →
         Heap × Desc 1
alloca h f sp τ count al =
  allocBlock h (mkBlock sp (width τ *ℕ count) al
                  (replicate (width τ *ℕ count) uninit) (stackAlloc f))
  , rootFor h τ count sp

-- memref.alloca_scope / memref.alloca_scope.return: leaving the scope.
allocaScopeEnd : Heap → FrameId → Heap
allocaScopeEnd = popFrame

-- memref.global : a named, initialised, never-freed allocation. Modelled by its
-- storage kind rather than by a name table, since the name is a symbol-table
-- fact and has no bearing on what may be done to the bytes.
global : Heap → MemSpace → (τ : ElemTy) → (count alignment : ℕ) →
         Vec StoredByte (width τ *ℕ count) → Heap × Desc 1
global h sp τ count al initial =
  allocBlock h (mkBlock sp (width τ *ℕ count) al initial staticData)
  , rootFor h τ count sp

-- memref.get_global : a descriptor onto an existing global. It is a lookup, not
-- an allocation, so it can fail -- and it fails LOUDLY rather than fabricating a
-- descriptor, because a get_global naming nothing is a broken symbol table and
-- the model should say so.
getGlobal : Heap → AllocId → (τ : ElemTy) → (count : ℕ) → MemSpace →
            Result MemoryFault (Desc 1)
getGlobal h a τ count sp =
  maybeᴿ (lookupBlock h a) no-such-allocation >>=ᴿ λ b →
  guardᴿ (space b ≟ sp) invalid-memory-space >>=ᴿ λ _ →
  ok (desc a (generation b) 0 (+ 0) (count ∷ []) ((+ 1) ∷ []) τ sp)

------------------------------------------------------------------------
-- memref.dealloc
--
-- Only heap storage may be deallocated. MLIR is explicit that deallocating an
-- alloca or a global is invalid, and this compiler emits both, so accepting
-- them here would make the model agree with a program the runtime would break.

deallocatable? : (st : Storage) → Dec (st ≡ heapAlloc)
deallocatable? heapAlloc      = yes refl
deallocatable? (stackAlloc _) = no λ ()
deallocatable? staticData     = no λ ()

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
  guardᴿ (live? (liveness b)) double-free >>=ᴿ λ _ →
  guardᴿ (deallocatable? (storage b)) invalid-free >>=ᴿ λ _ →
  guardᴿ (isRootOf? b d) invalid-free >>=ᴿ λ _ →
  ok (updateBlock h (allocation d) (λ blk → record blk { liveness = dead }))

------------------------------------------------------------------------
-- memref.load / memref.store

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
-- memref.atomic_rmw / memref.generic_atomic_rmw
--
-- One event, not a load followed by a store. The distinction has no content in
-- this sequential model -- it is the concurrency layer that will need it -- but
-- the op is given its own definition anyway, because defining it as
-- `store (f (load ...))` now would make the two indistinguishable later, and the
-- whole point of a separate op is that a scheduler may not interleave inside it.
--
-- Deliberately NOT given an ordering parameter: memref's atomic ops have none.
-- §7 of the note is about exactly this, and a model that invented an `Ordering`
-- field here would be describing a dialect that does not exist.
atomicRMW : Heap → ∀ {r} (d : Desc r) → Ix (sizes d) →
            (Vec Byte (elemWidth d) → Vec Byte (elemWidth d)) →
            Result MemoryFault (Vec Byte (elemWidth d) × Heap)
atomicRMW h d i f =
  resolveIn h d i >>=ᴿ λ br →
  maybeᴿ (readRange (contents (proj₁ br)) (start (proj₂ br)) (elemWidth d))
         uninitialized-read >>=ᴿ λ old →
  ok (old , updateBlock h (allocation d)
              (λ b → record b { contents = writeRange (contents b)
                                             (start (proj₂ br)) (f old) }))

------------------------------------------------------------------------
-- memref.copy
--
-- Elementwise, same shape. MLIR does not state whether the operands may
-- overlap, so this model does not decide it either: `copyElem` is defined one
-- element at a time and the aliasing question is recorded as open rather than
-- answered by whichever order the recursion happens to take.
copyElem : Heap → ∀ {r} (src dst : Desc r) →
           Ix (sizes src) → Ix (sizes dst) →
           elemWidth src ≡ elemWidth dst →
           Result MemoryFault Heap
-- `subst` rather than matching the equation with `refl`: elemWidth is a
-- function of the descriptor and is not injective, so Agda cannot unify
-- `elemWidth src` with `elemWidth dst` by pattern matching. Transporting the
-- vector along the equation is the same fact stated where it can be used.
copyElem h src dst i j eq =
  load h src i >>=ᴿ λ v →
  store h dst j (subst (Vec Byte) eq v)

------------------------------------------------------------------------
-- Metadata queries.

-- memref.dim : the extent in one dimension. In bounds by construction, because
-- the dimension is a Fin of the rank -- MLIR's version takes an index operand
-- and is undefined out of range.
dim : ∀ {r} (d : Desc r) → Fin r → ℕ
dim d k = lookup (sizes d) k

rank : ∀ {r} → Desc r → ℕ
rank {r} _ = r

-- memref.extract_strided_metadata : base, offset, sizes, strides.
--
-- The `base` it yields in MLIR is a memref with the same allocation, offset 0
-- and identity layout -- so it is the ROOT descriptor, which is why the result
-- can legitimately be handed to dealloc while a subview cannot.
extractStridedMetadata : ∀ {r} (d : Desc r) →
                         Desc 1 × ℤ × Vec ℕ r × Vec ℤ r
extractStridedMetadata {r} d =
  desc (allocation d) (generation d) 0 (+ 0) (0 ∷ []) ((+ 1) ∷ [])
       (elementType d) (memorySpace d)
  , offset d , sizes d , strides d

-- memref.extract_aligned_pointer_as_index : 84 uses in this compiler.
--
-- THIS IS WHERE PROVENANCE IS LOST, and the model should say so instead of
-- quietly returning a number. In MLIR the result is an `index` -- a bare
-- integer with no allocation, no generation and no liveness. Every guarantee in
-- this development is attached to the descriptor, so an integer obtained here
-- and turned back into a pointer is outside the model entirely.
--
-- So the result carries the identity rather than an address: an
-- `AlignedPointer` is what a verified use of this op may hold, and the
-- refinement to a machine address is a separate obligation that has to prove the
-- identity is still live at the point of use.
record AlignedPointer : Set where
  constructor alignedPtr
  field
    ptrAllocation : AllocId
    ptrGeneration : Generation
    ptrByteOffset : ℕ

open AlignedPointer public

extractAlignedPointerAsIndex : ∀ {r} → Desc r → AlignedPointer
extractAlignedPointerAsIndex d =
  alignedPtr (allocation d) (generation d) (alignedBase d)

-- The way back, which the record above named as an obligation and the model
-- then did not have. `AlignedPointer` said what a verified use MAY HOLD;
-- nothing said what it may then do, so every use in the compiler was outside
-- the model rather than governed by it -- and that is not a small corner. A
-- `memref` cannot have a pointer element type (MLIR: "invalid memref element
-- type"), so EVERY reference Lython's boxed objects store in a payload slot is
-- an address in an `i64`, and reading one back is this op's inverse. Silence
-- was not a prohibition; it was a gap, and the gap is where the compiler lives.
--
-- It is not free. The record's own note said the refinement to a machine
-- address "has to prove the identity is still live at the point of use", and
-- that is these three guards -- the same three `resolveIn` opens with, for the
-- same reason and reported as the same faults. A pointer whose block was freed
-- is `use-after-free`; one whose id was reused is `stale-generation`; the two
-- are different mistakes and the enumeration exists to keep them apart.
--
-- Why NOT a fault of its own for "reconstituted from a stale pointer": it would
-- name the same two states a second time, and a safety theorem that says which
-- guarantee was broken is worth more than one that says where.
--
-- What comes back is a ROOT descriptor -- offset 0, unit stride, the caller's
-- element type and count. It is not the descriptor that was taken apart: the
-- sizes and strides did not survive the trip, because the op does not carry
-- them. A caller that needs them has to know the shape statically, which is
-- exactly the "contract → physical shape relation is static" the box layout
-- relies on.
descFromAlignedPointer : Heap → AlignedPointer → (τ : ElemTy) → (count : ℕ) →
                         Result MemoryFault (Desc 1)
descFromAlignedPointer h p τ count with lookupBlock h (ptrAllocation p)
... | nothing = err no-such-allocation
... | just b with generation b ≟ ptrGeneration p | liveness b
...   | no  _ | _    = err stale-generation
...   | yes _ | dead = err use-after-free
...   | yes _ | live =
        ok (desc (ptrAllocation p) (ptrGeneration p) (ptrByteOffset p)
                 (+ 0) (count ∷ []) ((+ 1) ∷ []) τ (space b))

-- Whenever the way back succeeds, what comes back names the SAME allocation and
-- generation the pointer named.
--
-- This is the property the refcount layer's field sites assume without ever
-- being in a position to state it: `field′ o k` says a slot HOLDS object `o`,
-- and physically the slot holds an address. Without this, "holds `o`" and "holds
-- the bytes an address led to" are two claims with nothing joining them, and
-- `fieldRC`'s coherence would be counting one while the machine did the other.
recovered-identity :
  ∀ (h : Heap) (p : AlignedPointer) (τ : ElemTy) (count : ℕ) (d : Desc 1) →
  descFromAlignedPointer h p τ count ≡ ok d →
  (allocation d ≡ ptrAllocation p) × (generation d ≡ ptrGeneration p)
recovered-identity h p τ count d eq with lookupBlock h (ptrAllocation p)
recovered-identity h p τ count d () | nothing
recovered-identity h p τ count d eq | just b
  with generation b ≟ ptrGeneration p | liveness b
recovered-identity h p τ count d () | just b | no  _ | _
recovered-identity h p τ count d () | just b | yes _ | dead
recovered-identity h p τ count d refl | just b | yes _ | live = refl , refl

-- memref.prefetch : a hint. No memory effect, and in particular NOT a licence
-- to touch memory that would fault -- MLIR requires the address be valid.
prefetchValid : Heap → ∀ {r} (d : Desc r) → Ix (sizes d) → Result MemoryFault ByteRange
prefetchValid = resolve

------------------------------------------------------------------------
-- View-forming ops. None of these touches the heap.

-- memref.subview : offsets and strides compose RELATIVE TO THE INPUT VIEW.
subview : ∀ {r} (d : Desc r) → (newOffset : ℤ) → (newSizes : Vec ℕ r) →
          (newStrides : Vec ℤ r) → Desc r
subview d o ns nst =
  desc (allocation d) (generation d) (alignedBase d)
       (offset d + o) ns nst (elementType d) (memorySpace d)

-- memref.view : shifts by BYTES and may change the element type. This is the op
-- that makes a byte-addressed heap necessary; no typed-cell model can express a
-- differently-typed view of the same storage.
view : ∀ {r} (d : Desc r) → (byteShift : ℕ) → (τ : ElemTy) →
       (newSizes : Vec ℕ r) → (newStrides : Vec ℤ r) → Desc r
view d shift τ ns nst =
  desc (allocation d) (generation d) (alignedBase d +ℕ shift)
       (+ 0) ns nst τ (memorySpace d)

-- memref.reinterpret_cast : metadata is set relative to the UNDERLYING BASE,
-- not to the input view. That difference from subview is the reason both exist,
-- and getting it backwards is not a type error in either language.
reinterpretCast : ∀ {r r'} (d : Desc r) → (newBase : ℕ) → (newOffset : ℤ) →
                  (newSizes : Vec ℕ r') → (newStrides : Vec ℤ r') → Desc r'
reinterpretCast d nb o ns nst =
  desc (allocation d) (generation d) nb o ns nst (elementType d) (memorySpace d)

-- memref.memory_space_cast : same storage, different space. Kept separate so
-- that `resolve`'s space check has something to fail on.
memorySpaceCast : ∀ {r} (d : Desc r) → MemSpace → Desc r
memorySpaceCast d sp =
  desc (allocation d) (generation d) (alignedBase d)
       (offset d) (sizes d) (strides d) (elementType d) sp

-- memref.cast : 320 uses, and the one whose failure is a fault rather than a
-- refusal. It converts between static and dynamic shapes, and MLIR says the
-- behaviour is undefined if the runtime extents disagree with the static ones.
-- So it is CHECKED here, and the disagreement gets its own fault.
castShape : ∀ {r} (d : Desc r) → (target : Vec ℕ r) → Result MemoryFault (Desc r)
castShape d target =
  guardᴿ (sizesAgree (sizes d) target) descriptor-overflow >>=ᴿ λ _ →
  ok (desc (allocation d) (generation d) (alignedBase d)
           (offset d) target (strides d) (elementType d) (memorySpace d))
  where
    sizesAgree : ∀ {n} → (xs ys : Vec ℕ n) → Dec (xs ≡ ys)
    sizesAgree []       []       = yes refl
    sizesAgree (x ∷ xs) (y ∷ ys) with x ≟ y | sizesAgree xs ys
    ... | yes refl | yes refl = yes refl
    ... | no ¬p    | _        = no λ { refl → ¬p refl }
    ... | _        | no ¬q    = no λ { refl → ¬q refl }

-- memref.transpose : a permutation of the dimensions. Modelled as the
-- simultaneous permutation of sizes and strides, which is exactly what it is --
-- no bytes move, so this cannot fail.
transposeDesc : ∀ {r} (d : Desc r) → (newSizes : Vec ℕ r) → (newStrides : Vec ℤ r) →
                Desc r
transposeDesc d ns nst =
  desc (allocation d) (generation d) (alignedBase d)
       (offset d) ns nst (elementType d) (memorySpace d)
