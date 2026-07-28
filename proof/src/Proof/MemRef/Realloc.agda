{-# OPTIONS --safe #-}

-- memref.realloc, and the invalidation it forces.
--
-- MLIR: "The buffer for the input memref is deallocated; the result memref
-- points to a new buffer. Any references to the old memref are invalid."
--
-- That last sentence is the entire reason `Generation` is a field. Without it,
-- a stale descriptor after a realloc names a live block of the right size and
-- every check passes -- the model would accept precisely the program MLIR calls
-- undefined. With it, the old descriptor carries the old generation and
-- `resolve` answers `stale-generation`, which is a DIFFERENT fault from
-- use-after-free on purpose: the storage is live, it is simply not this
-- descriptor's storage any more.
--
-- Modelling choice, stated because it is a choice: MLIR permits realloc to
-- return the SAME buffer when the new size is no larger. This model always
-- invalidates. That is the conservative direction -- it rejects programs a
-- particular runtime might survive and never accepts one that would break --
-- and a verified compiler must not depend on which case its allocator took.

open import Proof.Memory.Element using (ElemSig; MemSpace)

module Proof.MemRef.Realloc (Sig : ElemSig) where

open ElemSig Sig

open import Data.Empty using (⊥; ⊥-elim)
open import Data.Integer using (ℤ; +_)
open import Data.List using (List; length)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc; _≟_; _⊓_)
  renaming (_*_ to _*ℕ_; _+_ to _+ℕ_)
open import Data.Nat.Properties using (1+n≢n)
open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Data.Vec using (Vec; []; _∷_; replicate)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; cong; sym; trans)
open import Relation.Nullary using (Dec; yes; no; ¬_)

open import Proof.Prelude using (Result; err; ok; _>>=ᴿ_; guardᴿ; maybeᴿ)
open import Proof.Memory.Byte using (Byte; StoredByte; uninit; init)
open import Proof.Memory.Fault
open import Proof.Memory.Heap
open import Proof.Memory.Index using (Ix)
open import Proof.Memory.Properties Sig using (lookup-update-same)
open import Proof.Memory.Descriptor Sig
open import Proof.Memory.Resolve Sig
open import Proof.MemRef.Dialect Sig

------------------------------------------------------------------------
-- Carrying the old contents across.
--
-- Copies a prefix, and leaves the tail `uninit`. Deliberately not zero-filling:
-- realloc makes no such promise, and a model that filled the tail would let a
-- program read a value the runtime never wrote -- the same reason fresh blocks
-- are uninitialised.
--
-- Recursion is on the byte COUNT, downwards, so termination is structural and
-- no fuel parameter can be miscounted.

copyPrefix : ∀ {m n} → Vec StoredByte m → Vec StoredByte n → ℕ → Vec StoredByte n
copyPrefix old acc zero    = acc
copyPrefix old acc (suc k) with getStored old k
... | nothing = copyPrefix old acc k
... | just sb = copyPrefix old (setStored acc k sb) k

------------------------------------------------------------------------
-- memref.realloc
--
-- The block and descriptor it produces are top-level definitions rather than a
-- `where` block, so that the theorems below can NAME the resulting heap. Stated
-- against an opaque `realloc h d n ≡ ok (h' , d')`, every lemma about h' would
-- have to re-derive the operation first.

reallocBytes : ∀ {r} → Desc r → ℕ → ℕ
reallocBytes d n = width (elementType d) *ℕ n

reallocBlock : ∀ {r} → Desc r → ℕ → Block → Block
reallocBlock d n b =
  block (suc (generation b)) (space b) (reallocBytes d n) (alignment b)
        (copyPrefix (contents b) (replicate (reallocBytes d n) uninit)
                    (sizeBytes b ⊓ reallocBytes d n))
        live heapAlloc

reallocDesc : ∀ {r} → Desc r → ℕ → Block → Desc 1
reallocDesc d n b =
  desc (allocation d) (suc (generation b)) 0 (+ 0)
       (n ∷ []) ((+ 1) ∷ []) (elementType d) (memorySpace d)

-- The result names the SAME allocation id at a NEW generation. A fresh id would
-- make a stale descriptor fail as `no-such-allocation` -- or worse, silently
-- name a different live block later. Keeping the id is what makes the fault
-- specifically `stale-generation`, and therefore attributable to realloc.
realloc : Heap → ∀ {r} (d : Desc r) → (newCount : ℕ) →
          Result MemoryFault (Heap × Desc 1)
realloc h d newCount =
  maybeᴿ (lookupBlock h (allocation d)) no-such-allocation >>=ᴿ λ b →
  guardᴿ (generation d ≟ generation b) stale-generation >>=ᴿ λ _ →
  guardᴿ (live? (liveness b)) use-after-free >>=ᴿ λ _ →
  -- Same rule as dealloc, for the same reason: realloc frees the old buffer, so
  -- it may only be applied to something that may be freed. An alloca or a
  -- global reaching here is `invalid-free`, not a resize.
  guardᴿ (deallocatable? (storage b)) invalid-free >>=ᴿ λ _ →
  guardᴿ (isRootOf? b d) invalid-free >>=ᴿ λ _ →
  ok (updateBlock h (allocation d) (reallocBlock d newCount) , reallocDesc d newCount b)

------------------------------------------------------------------------
-- What realloc buys, and it is the whole reason Generation is a field.

realloc-succeeds :
  ∀ (h : Heap) {r} (d : Desc r) b n →
  lookupBlock h (allocation d) ≡ just b →
  generation d ≡ generation b →
  liveness b ≡ live →
  storage b ≡ heapAlloc →
  IsRootOf b d →
  realloc h d n ≡ ok (updateBlock h (allocation d) (reallocBlock d n)
                     , reallocDesc d n b)
realloc-succeeds h d b n look gen liv st root
  rewrite look
  with generation d ≟ generation b
... | no ¬p = ⊥-elim (¬p gen)
... | yes _
  with liveness b | liv
... | .live | refl
  with storage b | st
... | .heapAlloc | refl
  with isRootOf? b d
... | no ¬r = ⊥-elim (¬r root)
... | yes _ = refl

-- THE theorem. After a realloc, the descriptor that was live a moment ago names
-- a live block of a plausible size -- and is still refused, because its
-- generation is one behind. Every other check in `resolve` passes here; this is
-- the only one that catches it.
--
-- It is also why `stale-generation` is not folded into `use-after-free`: the
-- storage IS live. Reporting use-after-free would send a reader looking for a
-- missing dealloc that does not exist.
stale-descriptor-faults-after-realloc :
  ∀ (h : Heap) {r} (d : Desc r) (i : Ix (sizes d)) b n →
  lookupBlock h (allocation d) ≡ just b →
  generation d ≡ generation b →
  resolveIn (updateBlock h (allocation d) (reallocBlock d n)) d i
    ≡ err stale-generation
stale-descriptor-faults-after-realloc h d i b n look gen
  rewrite lookup-update-same h (allocation d) (reallocBlock d n) b look
  with generation d ≟ suc (generation b)
... | yes p = ⊥-elim (1+n≢n (sym (trans (sym gen) p)))
... | no  _ = refl

-- And the descriptor realloc HANDED BACK is accepted: the theorem above is not
-- "realloc breaks everything". Without this, a realloc that invalidated its own
-- result would satisfy the previous lemma too.
fresh-descriptor-generation-matches :
  ∀ {r} (d : Desc r) n b →
  generation (reallocDesc d n b) ≡ generation (reallocBlock d n b)
fresh-descriptor-generation-matches d n b = refl
