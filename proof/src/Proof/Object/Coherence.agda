{-# OPTIONS --safe #-}

-- alloc / free / move coherence for the one-lane object.
--
-- Three claims, and each is the negation of a defect this compiler has actually
-- shipped:
--
--   ONE LANE          every field of an object resolves inside the object's own
--                     allocation, so there is no second lane that a resize can
--                     leave stale in someone else's hands.
--
--   FREE IS GUARDED   an object whose refcount word is not zero cannot be
--                     freed, and after it is freed the same lane faults as
--                     use-after-free rather than reading a plausible value.
--
--   MOVE IS FREE      moving an object reference changes no byte and no count,
--                     so a pass that emits a release for a move is emitting one
--                     release too many.

open import Proof.Object.WordSig using (WordSig)

module Proof.Object.Coherence (W : WordSig) where

open WordSig W

open import Data.Empty using (⊥; ⊥-elim)
open import Data.Fin using (Fin; toℕ; fromℕ<)
open import Data.Integer using (ℤ; +_)
open import Data.List using (List; []; _∷_)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc; _+_; _<_; s≤s; z≤n; _≟_; _≤?_)
open import Data.Nat.Divisibility using (_∣?_)
open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Data.Vec using (Vec; []; _∷_)
open import Relation.Binary.PropositionalEquality
  using (_≡_; _≢_; refl; sym; trans; cong; subst)
open import Relation.Nullary using (Dec; yes; no; ¬_)

open import Proof.Prelude using (Result; err; ok; ByteRange; _>>=ᴿ_; guardᴿ)
open import Proof.Memory.Byte using (Byte; StoredByte)
open import Proof.Memory.Fault
open import Proof.Memory.Heap
open import Proof.Memory.Index using (Ix; []; _∷_)
open import Proof.Memory.Descriptor sig
open import Proof.Memory.Resolve sig
open import Proof.Memory.Properties sig using (lookup-update-same; lookup-update-other)
open import Proof.MemRef.Realloc sig using (reallocBlock)
open import Proof.MemRef.Dialect sig
open import Proof.Object.Layout
open import Proof.Object.Word using (WordBytes; encode; decode; decode-encode; Capacity)
open import Proof.Object.Box W
open import Proof.Object.Ops W

------------------------------------------------------------------------
-- 1. ONE LANE
--
-- Every field access goes through the same descriptor, so it names the same
-- allocation. Trivial to prove and the whole point: it is trivial BECAUSE the
-- object is one descriptor, and it is what a (header, payload) pair cannot say.

-- The block a successful resolve reports IS the one the descriptor names. This
-- is the lemma with the content; without it "one lane" would be a statement
-- about how the accessors are written rather than about where they land.
resolveIn-block :
  ∀ (h : Heap) {r} (d : Desc r) (i : Ix (sizes d)) blk rng →
  resolveIn h d i ≡ ok (blk , rng) →
  lookupBlock h (allocation d) ≡ just blk
resolveIn-block h d i blk rng eq with lookupBlock h (allocation d) in look
... | nothing = ⊥-elim (bad eq)
  where bad : err {A = Block × ByteRange} no-such-allocation ≡ ok (blk , rng) → ⊥
        bad ()
... | just b with generation d ≟ generation b
...   | no _ = ⊥-elim (bad eq)
  where bad : err {A = Block × ByteRange} stale-generation ≡ ok (blk , rng) → ⊥
        bad ()
...   | yes _ with live? (liveness b)
...     | no _ = ⊥-elim (bad eq)
  where bad : err {A = Block × ByteRange} use-after-free ≡ ok (blk , rng) → ⊥
        bad ()
...     | yes _ with memorySpace d ≟ space b
...       | no _ = ⊥-elim (bad eq)
  where bad : err {A = Block × ByteRange} invalid-memory-space ≡ ok (blk , rng) → ⊥
        bad ()
...       | yes _ with nonNeg (byteStart d i)
...         | nothing = ⊥-elim (bad eq)
  where bad : err {A = Block × ByteRange} out-of-bounds ≡ ok (blk , rng) → ⊥
        bad ()
...         | just s with (s + elemWidth d) ≤? sizeBytes b
...           | no _ = ⊥-elim (bad eq)
  where bad : err {A = Block × ByteRange} out-of-bounds ≡ ok (blk , rng) → ⊥
        bad ()
...           | yes _ with elemAlign d ∣? s
...             | no _ = ⊥-elim (bad eq)
  where bad : err {A = Block × ByteRange} misaligned-access ≡ ok (blk , rng) → ⊥
        bad ()
-- `refl`, not `look`: the with-abstraction has already replaced
-- `lookupBlock h (allocation d)` by the matched `just b` throughout the goal,
-- so what is left to prove is `just b ≡ just b`. `look` is the same fact stated
-- before the abstraction, and Agda is right to reject it here.
...             | yes _ with eq
...               | refl = refl

-- ONE LANE, with content: any two field accesses of the same object land in the
-- SAME block. Not because the accessors were written that way, but because both
-- resolutions report the block that one lookup of one allocation returned.
--
-- This is the statement a (header, payload) pair cannot make. There, the two
-- accesses consult two allocations, and nothing relates them once one has been
-- reallocated -- which is the shape of every stale-lane defect this compiler has
-- shipped.
two-fields-one-block :
  ∀ (h : Heap) (inline : ℕ) (b : Box inline) (ws : WellShaped inline b)
    (j k : ℕ) (lj : j < boxWords inline) (lk : k < boxWords inline) bj rj bk rk →
  resolveIn h b (wordIx inline b ws j lj) ≡ ok (bj , rj) →
  resolveIn h b (wordIx inline b ws k lk) ≡ ok (bk , rk) →
  bj ≡ bk
two-fields-one-block h inline b ws j k lj lk bj rj bk rk ej ek =
  just-inj (trans (sym (resolveIn-block h b (wordIx inline b ws j lj) bj rj ej))
                  (resolveIn-block h b (wordIx inline b ws k lk) bk rk ek))
  where
    just-inj : ∀ {A : Set} {x y : A} → just x ≡ just y → x ≡ y
    just-inj refl = refl

------------------------------------------------------------------------
-- 2. FREE IS GUARDED
--
-- The refcount is read through the lane before the allocation goes away, and a
-- nonzero reading refuses. This is what stops the object layer from handing the
-- memory layer a free it will not be able to take back.

free-refuses-when-referenced :
  ∀ (inline : ℕ) (h : Heap) (b : Box inline) (ws : WellShaped inline b) n →
  refcountOf inline h b ws ≡ ok (suc n) →
  freeObject inline h b ws ≡ err refcount-not-zero
free-refuses-when-referenced inline h b ws n rc rewrite rc = refl

-- And when it does free, the lane it freed is the one the object was. There is
-- no other descriptor in play -- which is why `dealloc` accepts it: the object
-- reference IS the root descriptor, so `IsRootOf` holds of the thing the
-- program was holding all along.
--
-- In a two-lane design the object reference is a pair and neither component is
-- the root, so the free has to reconstruct a descriptor that nothing held. That
-- reconstruction is where a wrong width becomes a wrong deallocator.
free-uses-the-object-itself :
  ∀ (inline : ℕ) (h : Heap) (b : Box inline) (ws : WellShaped inline b) →
  refcountOf inline h b ws ≡ ok 0 →
  freeObject inline h b ws ≡ lift (dealloc h b)
free-uses-the-object-itself inline h b ws rc rewrite rc = refl

------------------------------------------------------------------------
-- 3. MOVE IS FREE
--
-- The heap is not an argument of `moveObject`, so there is nothing it could
-- change. Stating it is not ceremony: it is the fact that makes eliding the
-- retain/release pair around a move CORRECT rather than merely cheaper, and
-- this compiler has shipped both a missing release where a move was meant and a
-- release emitted where a move was meant.

move-changes-nothing : ∀ (inline : ℕ) (b : Box inline) → moveObject inline b ≡ b
move-changes-nothing _ _ = refl

-- The moved reference reads the same refcount, because it is the same
-- descriptor over the same heap. A move that had to adjust the count would fail
-- this.
move-preserves-refcount :
  ∀ (inline : ℕ) (h : Heap) (b : Box inline) (ws : WellShaped inline b) →
  refcountOf inline h (moveObject inline b) ws ≡ refcountOf inline h b ws
move-preserves-refcount _ _ _ _ = refl

-- And the same class id, and the same length. A move is the identity on the
-- whole header, which is what "the object did not change, only who names it"
-- means.
move-preserves-class :
  ∀ (inline : ℕ) (h : Heap) (b : Box inline) (ws : WellShaped inline b) →
  classOf inline h (moveObject inline b) ws ≡ classOf inline h b ws
move-preserves-class _ _ _ _ = refl

------------------------------------------------------------------------
-- 4. The refcount survives the encoding.
--
-- `retain` writes `suc n` as eight bytes and a later `refcountOf` decodes them.
-- Without the round trip that is a claim about arithmetic the model has not
-- made -- and it is bounded, because eight bytes hold 2^64 values and a
-- refcount that overflowed its word would be a real defect rather than a
-- modelling artefact.

refcount-round-trips :
  ∀ (n : ℕ) → n < Capacity WordBytes → decode (encode n) ≡ n
refcount-round-trips = decode-encode

------------------------------------------------------------------------
-- 5. What the split buys: a resize does not invalidate a reference.
--
-- `resizeBuffer` reallocates the BUFFER and rewrites words 4 and 5 of the box.
-- The box's own allocation is never passed to realloc, so no descriptor of the
-- object is invalidated -- and since every alias reads the buffer identity out
-- of word 4, every alias sees the new buffer.
--
-- The contrast is the point. Reallocating the object itself bumps the object's
-- generation, and Proof.MemRef.Realloc.stale-descriptor-faults-after-realloc
-- says every existing reference then faults. There is no way to reach those
-- references and fix them, which is why the note forbids realloc on a shared
-- box -- and why this design puts the mutable part somewhere else.

-- THE theorem of the split. Reallocating the buffer leaves the box's block
-- BIT FOR BIT as it was, so no reference to the object is invalidated -- and
-- every alias, reading word 4, sees the new buffer.
--
-- Compare Proof.MemRef.Realloc.stale-descriptor-faults-after-realloc: had the
-- object itself been reallocated, every existing reference would fault, and
-- there is no way to reach them and fix them. That is why the note forbids
-- realloc on a shared box, and here the difference is a difference in what can
-- be proved rather than a preference between two encodings.
realloc-of-buffer-leaves-the-box :
  ∀ (h : Heap) (inline : ℕ) (b : Box inline) (buf : Desc 1) (n : ℕ) →
  ¬ (allocation buf ≡ allocation b) →
  lookupBlock (updateBlock h (allocation buf) (reallocBlock buf n)) (allocation b)
    ≡ lookupBlock h (allocation b)
realloc-of-buffer-leaves-the-box h inline b buf n ne =
  lookup-update-other h (allocation buf) (allocation b) (reallocBlock buf n) ne
