{-# OPTIONS --safe #-}

-- What the model actually buys.
--
-- A memory model that never rejects anything is consistent and worthless. These
-- are the theorems that say this one SEPARATES the failures it names: that a
-- freed allocation is refused as use-after-free and not as something else, that
-- freeing twice is refused as double-free, that fresh storage does not read back
-- as a value. Each is a statement a compiler bug would have to violate.
--
-- The postconditions are written FORWARD -- from the hypotheses that make an
-- operation succeed to the heap it produces -- rather than backward from an
-- opaque `op h ≡ ok h'`. Backward, the resulting heap is a variable the proof
-- cannot look inside, and every lemma about it needs the operation re-derived.

open import Proof.Memory.Element using (ElemSig; MemSpace)

module Proof.Memory.Properties (Sig : ElemSig) where

open ElemSig Sig

open import Data.Empty using (⊥; ⊥-elim)
open import Data.Integer using (ℤ; +_)
-- Heap is a List, so its constructors have to be the ones in scope unqualified.
-- Vec's are deliberately not imported here: with both, every `[]` in a Heap
-- pattern is ambiguous, and the error points at the pattern rather than at the
-- import that caused it.
open import Data.List using (List; []; _∷_; length)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc; _<_; s≤s; z≤n; _≟_; _≤?_)
  renaming (_*_ to _*ℕ_; _+_ to _+ℕ_)
open import Data.Nat.Divisibility using (_∣?_)
open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Data.Sum using (_⊎_; inj₁; inj₂)
open import Data.Vec using (Vec; replicate)
open import Relation.Binary.PropositionalEquality
  using (_≡_; refl; sym; trans; cong; cong₂)
open import Relation.Nullary using (Dec; yes; no; ¬_)

open import Proof.Prelude using (Result; err; ok; ByteRange; range; start; len;
  _>>=ᴿ_; guardᴿ; maybeᴿ)
open import Proof.Memory.Byte using (Byte; StoredByte; uninit; init)
open import Proof.Memory.Fault
open import Proof.Memory.Heap
open import Proof.Memory.Index using (Ix)
open import Proof.Memory.Descriptor Sig
open import Proof.Memory.Resolve Sig
-- The operations come from the dialect transcription. Proof.Memory.Ops used to
-- define its own alloc/load/store/dealloc; two definitions of the same
-- operation are two things that can drift apart, and the one the theorems are
-- about would not have been the one the model exports.
open import Proof.MemRef.Dialect Sig

private
  -- The heap transformation `dealloc` performs, named so theorems can talk
  -- about the result without re-deriving it.
  killed : Heap → AllocId → Heap
  killed h a = updateBlock h a (λ blk → record blk { liveness = dead })

------------------------------------------------------------------------
-- Lemmas about updating one block.

lookup-update-same : ∀ (h : Heap) (a : AllocId) (f : Block → Block) (b : Block) →
                     lookupBlock h a ≡ just b →
                     lookupBlock (updateBlock h a f) a ≡ just (f b)
lookup-update-same []       a       f b ()
lookup-update-same (x ∷ xs) zero    f b refl = refl
lookup-update-same (x ∷ xs) (suc n) f b eq   = lookup-update-same xs n f b eq

lookup-update-other : ∀ (h : Heap) (a a' : AllocId) (f : Block → Block) →
                      ¬ (a ≡ a') →
                      lookupBlock (updateBlock h a f) a' ≡ lookupBlock h a'
lookup-update-other []       _       _       _ _  = refl
lookup-update-other (x ∷ xs) zero    zero    f ne = ⊥-elim (ne refl)
lookup-update-other (x ∷ xs) zero    (suc m) f _  = refl
lookup-update-other (x ∷ xs) (suc n) zero    f _  = refl
lookup-update-other (x ∷ xs) (suc n) (suc m) f ne =
  lookup-update-other xs n m f (λ eq → ne (cong suc eq))

------------------------------------------------------------------------
-- alloc is fresh: it disturbs nothing that already existed.
--
-- This is what lets a descriptor obtained before an allocation still be used
-- after it. Without it, every operation would have to be re-verified against
-- the whole heap after any other allocation anywhere.

alloc-preserves-old : ∀ (h : Heap) sp τ count al (a : AllocId) → a < length h →
                      lookupBlock (proj₁ (alloc h sp τ count al)) a ≡ lookupBlock h a
alloc-preserves-old h sp τ count al a lt = lookup-fresh-old h _ a lt

-- `heapAlloc` is part of the statement, not incidental. It is what makes the
-- block deallocatable, and an `alloc` that produced stack storage would satisfy
-- every other theorem here while making its own result impossible to free.
alloc-new-is-live : ∀ (h : Heap) sp τ count al →
                    lookupBlock (proj₁ (alloc h sp τ count al)) (freshId h)
                      ≡ just (block 0 sp (width τ *ℕ count) al
                                     (replicate (width τ *ℕ count) uninit)
                                     live heapAlloc)
alloc-new-is-live h sp τ count al = lookup-fresh-new h _

-- And alloca produces stack storage, which dealloc must refuse. Stated next to
-- the previous one because together they are the reason `Storage` exists.
alloca-is-stack : ∀ (h : Heap) f sp τ count al →
                  lookupBlock (proj₁ (alloca h f sp τ count al)) (freshId h)
                    ≡ just (block 0 sp (width τ *ℕ count) al
                                   (replicate (width τ *ℕ count) uninit)
                                   live (stackAlloc f))
alloca-is-stack h f sp τ count al = lookup-fresh-new h _

-- The descriptor alloc hands back names the allocation alloc just made. Stated
-- because it is the only link between the two halves of the return value, and a
-- model where they disagreed would still satisfy every other theorem here.
alloc-desc-names-block : ∀ (h : Heap) sp τ count al →
                         allocation (proj₂ (alloc h sp τ count al)) ≡ freshId h
alloc-desc-names-block h sp τ count al = refl

------------------------------------------------------------------------
-- Fresh storage does not read back as a value.

getStored-replicate : ∀ (n s : ℕ) →
                      (getStored (replicate n uninit) s ≡ nothing)
                    ⊎ (getStored (replicate n uninit) s ≡ just uninit)
getStored-replicate zero    _       = inj₁ refl
getStored-replicate (suc n) zero    = inj₂ refl
getStored-replicate (suc n) (suc s) = getStored-replicate n s

-- Positive length is essential: a zero-width read succeeds vacuously, which is
-- why ElemSig demands `width-pos`. Without it this lemma would be false and the
-- uninitialized-read fault unreachable.
readRange-all-uninit : ∀ (n s w : ℕ) →
                       readRange (replicate n uninit) s (suc w) ≡ nothing
readRange-all-uninit n s w with getStored-replicate n s
... | inj₁ eq rewrite eq = refl
... | inj₂ eq rewrite eq = refl

-- Written with an explicit trans/cong rather than `rewrite`, because the goal
-- mentions `load` and only unfolds to `resolveIn` by conversion -- `rewrite`
-- matches syntactically and would not fire.
load-uninit-block :
  ∀ (h : Heap) {r} (d : Desc r) (i : Ix (sizes d)) b br →
  resolveIn h d i ≡ ok (b , br) →
  readRange (contents b) (start br) (elemWidth d) ≡ nothing →
  load h d i ≡ err uninitialized-read
load-uninit-block h d i b br res rr =
  trans (cong (_>>=ᴿ (λ x → maybeᴿ (readRange (contents (proj₁ x)) (start (proj₂ x))
                                              (elemWidth d)) uninitialized-read)) res)
        (cong (λ z → maybeᴿ z uninitialized-read) rr)

------------------------------------------------------------------------
-- dealloc, and the two mistakes it must tell apart.

dealloc-succeeds :
  ∀ (h : Heap) {r} (d : Desc r) b →
  lookupBlock h (allocation d) ≡ just b →
  generation d ≡ generation b →
  liveness b ≡ live →
  storage b ≡ heapAlloc →
  IsRootOf b d →
  dealloc h d ≡ ok (killed h (allocation d))
dealloc-succeeds h d b look gen liv st root
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

-- An alloca cannot be deallocated, and the fault is `invalid-free`. MLIR says
-- so, and this compiler emits 27 allocas -- a model that accepted this would
-- agree with a program the runtime would break, which is the direction that
-- matters.
alloca-cannot-be-deallocated :
  ∀ (h : Heap) {r} (d : Desc r) b f →
  lookupBlock h (allocation d) ≡ just b →
  generation d ≡ generation b →
  liveness b ≡ live →
  storage b ≡ stackAlloc f →
  dealloc h d ≡ err invalid-free
alloca-cannot-be-deallocated h d b f look gen liv st
  rewrite look
  with generation d ≟ generation b
... | no ¬p = ⊥-elim (¬p gen)
... | yes _
  with liveness b | liv
... | .live | refl
  with storage b | st
... | .(stackAlloc f) | refl = refl

-- Same for a global. Lython emits 151 of them.
global-cannot-be-deallocated :
  ∀ (h : Heap) {r} (d : Desc r) b →
  lookupBlock h (allocation d) ≡ just b →
  generation d ≡ generation b →
  liveness b ≡ live →
  storage b ≡ staticData →
  dealloc h d ≡ err invalid-free
global-cannot-be-deallocated h d b look gen liv st
  rewrite look
  with generation d ≟ generation b
... | no ¬p = ⊥-elim (¬p gen)
... | yes _
  with liveness b | liv
... | .live | refl
  with storage b | st
... | .staticData | refl = refl

-- Freeing what is already freed is `double-free`. Not `use-after-free`: they are
-- different mistakes, and a model that returned one for the other could not be
-- used to attribute a compiler bug to the pass that caused it.
double-free-is-caught :
  ∀ (h : Heap) {r} (d : Desc r) b →
  lookupBlock h (allocation d) ≡ just b →
  generation d ≡ generation b →
  dealloc (killed h (allocation d)) d ≡ err double-free
double-free-is-caught h d b look gen
  rewrite lookup-update-same h (allocation d) (λ blk → record blk { liveness = dead }) b look
  with generation d ≟ generation b
... | no ¬p = ⊥-elim (¬p gen)
... | yes _ = refl

-- And reading it afterwards is `use-after-free`, for ANY index and any rank:
-- the check that fails is on the block, so no index can dodge it.
use-after-free-is-caught :
  ∀ (h : Heap) {r} (d : Desc r) (i : Ix (sizes d)) b →
  lookupBlock h (allocation d) ≡ just b →
  generation d ≡ generation b →
  resolveIn (killed h (allocation d)) d i ≡ err use-after-free
use-after-free-is-caught h d i b look gen
  rewrite lookup-update-same h (allocation d) (λ blk → record blk { liveness = dead }) b look
  with generation d ≟ generation b
... | no ¬p = ⊥-elim (¬p gen)
... | yes _ = refl

-- A descriptor that is not the allocation's root cannot free it, whatever else
-- is true of it. This is MLIR's "dealloc the memref that alloc returned" rule,
-- and it is what stops a subview from freeing its parent.
non-root-cannot-free :
  ∀ (h : Heap) {r} (d : Desc r) b →
  lookupBlock h (allocation d) ≡ just b →
  generation d ≡ generation b →
  liveness b ≡ live →
  storage b ≡ heapAlloc →
  ¬ IsRootOf b d →
  dealloc h d ≡ err invalid-free
non-root-cannot-free h d b look gen liv st ¬root
  rewrite look
  with generation d ≟ generation b
... | no ¬p = ⊥-elim (¬p gen)
... | yes _
  with liveness b | liv
... | .live | refl
  with storage b | st
... | .heapAlloc | refl
  with isRootOf? b d
... | yes r = ⊥-elim (¬root r)
... | no  _ = refl
