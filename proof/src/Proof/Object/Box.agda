{-# OPTIONS --safe #-}

-- A boxed object: ONE descriptor, and every field an index into it.
--
-- The accessors below all go through the same `load`/`store` on the same
-- `Desc 1`. There is no second descriptor anywhere in this module, and that is
-- the design claim -- `Proof.Object.Coherence` turns it into a theorem by
-- showing every field access resolves inside the box's own allocation.

open import Proof.Object.WordSig using (WordSig)

module Proof.Object.Box (W : WordSig) where

open WordSig W

open import Data.Fin using (Fin; fromℕ<; toℕ)
open import Data.Integer using (ℤ; +_)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc; _+_; _<_; s≤s; z≤n)
open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Data.Vec using (Vec; []; _∷_)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; sym; subst)

open import Proof.Prelude using (Result; err; ok; _>>=ᴿ_)
open import Proof.Memory.Byte using (Byte)
open import Proof.Memory.Element using (MemSpace)
open import Proof.Memory.Fault using (MemoryFault)
open import Proof.Memory.Heap using (Heap)
open import Proof.Memory.Index using (Ix; []; _∷_)
open import Proof.Memory.Descriptor sig
open import Proof.MemRef.Dialect sig using (load; store)
open import Proof.Object.Layout
open import Proof.Object.Word using (WordBytes; encode; decode)

------------------------------------------------------------------------
-- A box is a rank-1 descriptor of words. Nothing else.

Box : ℕ → Set
Box inline = Desc 1

-- The shape a well-formed box has. Stated as a predicate rather than baked into
-- a record, so that `Box` really is `Desc 1` -- a box that were a *pair* of a
-- descriptor and a proof would be two values again, which is what one-laning is
-- supposed to remove.
-- `inline` is EXPLICIT. `Box inline` reduces to `Desc 1`, so a box does not
-- carry its own payload width in its type -- which is precisely the one-lane
-- property -- and an implicit parameter here would be unsolvable at every call
-- site. Making it explicit is the honest consequence of the design.
record WellShaped (inline : ℕ) (b : Box inline) : Set where
  field
    is-word-typed : elementType b ≡ word
    spans-box     : sizes b ≡ boxWords inline ∷ []
    unit-stride   : strides b ≡ (+ 1) ∷ []
    at-origin     : offset b ≡ + 0

open WellShaped public

------------------------------------------------------------------------
-- Word indices into the box.
--
-- `wordIx` is the whole of the addressing story: a field is a NUMBER, and the
-- object reference is unchanged by which field you are looking at.

wordIx : ∀ (inline : ℕ) (b : Box inline) → WellShaped inline b →
         (k : ℕ) → k < boxWords inline → Ix (sizes b)
wordIx inline b ws k lt
  rewrite spans-box ws = fromℕ< lt ∷ []

------------------------------------------------------------------------
-- Reading and writing a word.
--
-- `elemWidth b` is `width word`, which WordSig fixes at WordBytes, so the
-- vector a load returns is exactly a word. The `subst` carries that equation;
-- without WordSig's `word-width` field there would be nothing to carry and the
-- accessors could not be typed at all.

loadWord : ∀ (inline : ℕ) (h : Heap) (b : Box inline) (ws : WellShaped inline b) →
           (k : ℕ) → (lt : k < boxWords inline) → Result MemoryFault ℕ
loadWord inline h b ws k lt =
  load h b (wordIx inline b ws k lt) >>=ᴿ λ v → ok (decode v)

-- Hoisted rather than local: `Box inline` reduces to `Desc 1`, so `inline` is
-- not recoverable from the type of a box -- which is exactly the point of
-- one-laning, and also why a `where`-bound helper here leaves it unsolved.
-- Takes the element-type equation directly rather than a WellShaped: `inline`
-- plays no part in it, and asking for the record would make the payload width
-- of the box a premise of a fact about its element type.
widthIsWord : (b : Desc 1) → elementType b ≡ word → elemWidth b ≡ WordBytes
widthIsWord b eq rewrite eq = word-width

storeWord : ∀ (inline : ℕ) (h : Heap) (b : Box inline) (ws : WellShaped inline b) →
            (k : ℕ) → (lt : k < boxWords inline) → ℕ → Result MemoryFault Heap
storeWord inline h b ws k lt n =
  store h b (wordIx inline b ws k lt)
        (subst (Vec Byte) (sym (widthIsWord b (is-word-typed ws))) (encode n))

------------------------------------------------------------------------
-- The header accessors.
--
-- Each is `loadWord` at a constant. Note what is NOT here: no view, no
-- reinterpret_cast, no second descriptor. Reading an object's class id is the
-- same operation as reading its third payload word, at a different number.

refcountOf : ∀ (inline : ℕ) → Heap → (b : Box inline) → WellShaped inline b →
             Result MemoryFault ℕ
refcountOf inline h b ws = loadWord inline h b ws rcWord (rc-fits inline)

classOf : ∀ (inline : ℕ) → Heap → (b : Box inline) → WellShaped inline b →
          Result MemoryFault ℕ
classOf inline h b ws = loadWord inline h b ws classWord (class-fits inline)

lengthOf : ∀ (inline : ℕ) → Heap → (b : Box inline) → WellShaped inline b →
           Result MemoryFault ℕ
lengthOf inline h b ws = loadWord inline h b ws lengthWord (length-fits inline)

capacityOf : ∀ (inline : ℕ) → Heap → (b : Box inline) → WellShaped inline b →
             Result MemoryFault ℕ
capacityOf inline h b ws = loadWord inline h b ws capacityWord (capacity-fits inline)

bufferAllocOf : ∀ (inline : ℕ) → Heap → (b : Box inline) → WellShaped inline b →
                Result MemoryFault ℕ
bufferAllocOf inline h b ws = loadWord inline h b ws bufAllocWord (bufAlloc-fits inline)

bufferGenOf : ∀ (inline : ℕ) → Heap → (b : Box inline) → WellShaped inline b →
              Result MemoryFault ℕ
bufferGenOf inline h b ws = loadWord inline h b ws bufGenWord (bufGen-fits inline)

setRefcount : ∀ (inline : ℕ) → Heap → (b : Box inline) → WellShaped inline b → ℕ →
              Result MemoryFault Heap
setRefcount inline h b ws n = storeWord inline h b ws rcWord (rc-fits inline) n

setClass : ∀ (inline : ℕ) → Heap → (b : Box inline) → WellShaped inline b → ℕ →
           Result MemoryFault Heap
setClass inline h b ws n = storeWord inline h b ws classWord (class-fits inline) n

setLength : ∀ (inline : ℕ) → Heap → (b : Box inline) → WellShaped inline b → ℕ →
            Result MemoryFault Heap
setLength inline h b ws n = storeWord inline h b ws lengthWord (length-fits inline) n

setCapacity : ∀ (inline : ℕ) → Heap → (b : Box inline) → WellShaped inline b → ℕ →
              Result MemoryFault Heap
setCapacity inline h b ws n = storeWord inline h b ws capacityWord (capacity-fits inline) n

setBuffer : ∀ (inline : ℕ) → Heap → (b : Box inline) → WellShaped inline b →
            (allocId gen : ℕ) → Result MemoryFault Heap
setBuffer inline h b ws a g =
  storeWord inline h b ws bufAllocWord (bufAlloc-fits inline) a >>=ᴿ λ h' →
  storeWord inline h' b ws bufGenWord (bufGen-fits inline) g

-- Payload access, at the same lane, past the header.
loadPayload : ∀ (inline : ℕ) → Heap → (b : Box inline) → WellShaped inline b →
              Fin inline → Result MemoryFault ℕ
loadPayload inline h b ws i =
  loadWord inline h b ws (HeaderWords + toℕ i) (payload-fits inline i)

storePayload : ∀ (inline : ℕ) → Heap → (b : Box inline) → WellShaped inline b →
               Fin inline → ℕ → Result MemoryFault Heap
storePayload inline h b ws i n =
  storeWord inline h b ws (HeaderWords + toℕ i) (payload-fits inline i) n
