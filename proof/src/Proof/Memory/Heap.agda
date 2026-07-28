{-# OPTIONS --safe #-}

-- The byte heap, keyed by allocation IDENTITY rather than by address.
--
-- `Memory = Address → Byte` cannot express use-after-free at all: once a block
-- is freed and the allocator hands the same address back, a stale pointer and a
-- fresh one are the same value, so no predicate over that heap can separate
-- them. Identity is therefore (allocation, generation, offset), and physical
-- addresses are left to lowering.

module Proof.Memory.Heap where

open import Data.Fin using (Fin)
open import Data.List using (List; []; _∷_; length; _++_; [_])
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc; _+_; _<_; s≤s; z≤n)
open import Data.Vec using (Vec; []; _∷_)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; cong)
open import Relation.Nullary using (Dec; yes; no)

open import Proof.Memory.Byte using (Byte; StoredByte; uninit; init)
open import Proof.Memory.Element using (MemSpace)

AllocId Generation : Set
AllocId    = ℕ
Generation = ℕ

-- Two states, not a Bool, so that a fault case reads as the state it is about.
-- `finalizing` arrives with the refcount layer; it has no meaning for raw bytes.
data Liveness : Set where
  live dead : Liveness

record Block : Set where
  constructor block
  field
    generation : Generation
    space      : MemSpace
    sizeBytes  : ℕ
    alignment  : ℕ
    contents   : Vec StoredByte sizeBytes
    liveness   : Liveness

open Block public

-- A freed block keeps its size and its generation. That is deliberate: the
-- model must still be able to say "this descriptor points into allocation 3,
-- generation 0, which is dead" -- deleting the entry instead would turn every
-- use-after-free into no-such-allocation and lose the distinction the fault
-- enumeration exists to make.
Heap : Set
Heap = List Block

lookupBlock : Heap → AllocId → Maybe Block
lookupBlock []       _       = nothing
lookupBlock (b ∷ _)  zero    = just b
lookupBlock (_ ∷ bs) (suc n) = lookupBlock bs n

-- Appending is what makes freshness free: the new id is the old length, and
-- every existing id keeps its block (`lookup-fresh-old` below). A model that
-- reused ids would need the generation counter to do that work instead.
allocBlock : Heap → Block → Heap
allocBlock h b = h ++ [ b ]

freshId : Heap → AllocId
freshId = length

updateBlock : Heap → AllocId → (Block → Block) → Heap
updateBlock []       _       _ = []
updateBlock (b ∷ bs) zero    f = f b ∷ bs
updateBlock (b ∷ bs) (suc n) f = b ∷ updateBlock bs n f

------------------------------------------------------------------------
-- Byte-level access inside one block's contents.

getStored : ∀ {n} → Vec StoredByte n → ℕ → Maybe StoredByte
getStored []       _       = nothing
getStored (x ∷ _)  zero    = just x
getStored (_ ∷ xs) (suc i) = getStored xs i

setStored : ∀ {n} → Vec StoredByte n → ℕ → StoredByte → Vec StoredByte n
setStored []       _       _ = []
setStored (_ ∷ xs) zero    v = v ∷ xs
setStored (x ∷ xs) (suc i) v = x ∷ setStored xs i v

-- Reading fails on the FIRST uninitialised byte rather than substituting a
-- default. A default would make the model's answer depend on the choice of
-- default, and every theorem about reads would then be a theorem about that
-- choice instead of about the program.
readRange : ∀ {n} → Vec StoredByte n → ℕ → (len : ℕ) → Maybe (Vec Byte len)
readRange v s zero    = just []
readRange v s (suc l) with getStored v s
... | nothing        = nothing
... | just uninit    = nothing
... | just (init b)  with readRange v (suc s) l
...   | nothing = nothing
...   | just bs = just (b ∷ bs)

writeRange : ∀ {n l} → Vec StoredByte n → ℕ → Vec Byte l → Vec StoredByte n
writeRange v _ []       = v
writeRange v s (b ∷ bs) = writeRange (setStored v s (init b)) (suc s) bs

------------------------------------------------------------------------
-- The two facts that make `allocBlock` a legitimate notion of "fresh".

lookup-fresh-old : ∀ (h : Heap) (b : Block) (i : AllocId) →
                   i < length h → lookupBlock (allocBlock h b) i ≡ lookupBlock h i
lookup-fresh-old []       b i       ()
lookup-fresh-old (x ∷ xs) b zero    _        = refl
lookup-fresh-old (x ∷ xs) b (suc i) (s≤s lt) = lookup-fresh-old xs b i lt

lookup-fresh-new : ∀ (h : Heap) (b : Block) →
                   lookupBlock (allocBlock h b) (freshId h) ≡ just b
lookup-fresh-new []       b = refl
lookup-fresh-new (x ∷ xs) b = lookup-fresh-new xs b
