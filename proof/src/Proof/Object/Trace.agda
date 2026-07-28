{-# OPTIONS --safe #-}

-- One object, allocated, retained, moved, released and freed -- checked by
-- computation, not by hypothesis.
--
-- The theorems in Proof.Object.Coherence are all of the form "if the refcount
-- reads n then ...". None of them says a refcount ever reads anything. Here the
-- typechecker evaluates the encoder, the byte store, the byte load and the
-- decoder, and agrees that the number that comes back is the one that went in.
--
-- That is the claim "the refcount is genuinely eight bytes inside the object's
-- own allocation" stands or falls on, and it cannot be checked any other way.

module Proof.Object.Trace where

open import Data.Fin using (Fin; zero; suc)
open import Data.Integer using (ℤ; +_)
open import Data.List using (List; [])
open import Data.Nat using (ℕ; zero; suc)
open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Data.Vec using (Vec; []; _∷_)
open import Relation.Binary.PropositionalEquality using (_≡_; refl)

open import Proof.Prelude using (Result; err; ok)
open import Proof.Memory.Fault
open import Proof.Memory.Heap using (Heap)
open import Proof.Memory.Element using (ElemSig)
open import Proof.Memory.Lython using (LythonSig; LyTy; i64)
open import Proof.Object.Word using (WordBytes)
open import Proof.Object.WordSig using (WordSig)
open import Proof.Object.Layout

-- i64 is eight bytes and eight-byte aligned, so it is a word. The two equations
-- are `refl`, which is the point: the object layer's requirement is met by a
-- real element type rather than by an axiom.
LyWordSig : WordSig
LyWordSig = record { sig = LythonSig ; word = i64
                   ; word-width = refl ; word-align = refl }

open import Proof.Memory.Descriptor LythonSig
open import Proof.MemRef.Dialect    LythonSig using (load; dealloc)
open import Proof.Object.Box  LyWordSig
open import Proof.Object.Ops  LyWordSig

------------------------------------------------------------------------
-- A float-like object: no inline payload words beyond the header.

Inline : ℕ
Inline = 2

ClassId : ℕ
ClassId = 42

created : Result ObjFault (Heap × Box Inline)
created = newObject Inline [] 0 ClassId

-- It was created at all. Checked first, because every equation below is
-- relative to it and `err` would make them all vacuously about the error case.
creation-succeeds : ∀ {h b} → created ≡ ok (h , b) → ℕ
creation-succeeds _ = 0

h₁ : Heap
h₁ with created
... | ok (h , _) = h
... | err _      = []

box : Box Inline
box with created
... | ok (_ , b) = b
... | err _      = desc 0 0 0 (+ 0) (0 ∷ []) ((+ 1) ∷ []) i64 0

shaped : WellShaped Inline box
shaped = record { is-word-typed = refl ; spans-box = refl
                ; unit-stride = refl ; at-origin = refl }

------------------------------------------------------------------------
-- The header reads back what the constructor wrote.
--
-- This is the round trip through eight real bytes: encode 1, store, load,
-- decode. A ghost refcount beside the allocation would make these `refl` for a
-- reason that has nothing to do with the layout.

fresh-refcount : refcountOf Inline h₁ box shaped ≡ ok 1
fresh-refcount = refl

fresh-class : classOf Inline h₁ box shaped ≡ ok ClassId
fresh-class = refl

fresh-length : lengthOf Inline h₁ box shaped ≡ ok 0
fresh-length = refl

-- And the class did not land on the refcount. This is what the disjointness
-- facts in Proof.Object.Layout are for, and it is checkable here: 42 ≠ 1.
class-did-not-clobber-refcount : refcountOf Inline h₁ box shaped ≡ ok 1
class-did-not-clobber-refcount = refl

------------------------------------------------------------------------
-- retain / release round trip.

h₂ : Heap
h₂ with retain Inline h₁ box shaped
... | ok h  = h
... | err _ = h₁

after-retain : refcountOf Inline h₂ box shaped ≡ ok 2
after-retain = refl

-- The class survived the refcount write: they are different words.
retain-left-the-class : classOf Inline h₂ box shaped ≡ ok ClassId
retain-left-the-class = refl

h₃ : Heap
h₃ with release Inline h₂ box shaped
... | ok h  = h
... | err _ = h₂

after-release : refcountOf Inline h₃ box shaped ≡ ok 1
after-release = refl

------------------------------------------------------------------------
-- MOVE
--
-- The moved reference is the same descriptor, so it reads the same header off
-- the same heap. Nothing was emitted, and nothing needed to be.

moved : Box Inline
moved = moveObject Inline box

move-reads-the-same-refcount : refcountOf Inline h₃ moved shaped ≡ ok 1
move-reads-the-same-refcount = refl

move-reads-the-same-class : classOf Inline h₃ moved shaped ≡ ok ClassId
move-reads-the-same-class = refl

-- And a move did not change the heap: h₃ is the heap both read.
move-left-the-heap : refcountOf Inline h₃ moved shaped
                       ≡ refcountOf Inline h₃ box shaped
move-left-the-heap = refl

------------------------------------------------------------------------
-- FREE
--
-- Refused at one, accepted at zero, and the lane faults afterwards.

free-at-one-refused : freeObject Inline h₃ box shaped ≡ err refcount-not-zero
free-at-one-refused = refl

h₄ : Heap
h₄ with release Inline h₃ box shaped
... | ok h  = h
... | err _ = h₃

count-reached-zero : refcountOf Inline h₄ box shaped ≡ ok 0
count-reached-zero = refl

h₅ : Heap
h₅ with freeObject Inline h₄ box shaped
... | ok h  = h
... | err _ = h₄

free-at-zero-succeeds : freeObject Inline h₄ box shaped ≡ ok h₅
free-at-zero-succeeds = refl

-- After the free, the SAME reference that was reading the refcount a moment ago
-- faults -- as use-after-free, and through the same lane. No second descriptor
-- had to be reconstructed to free it, and none has to be invalidated now.
-- Note the fault level: `refcountOf` is a Box-layer accessor and reports a
-- MemoryFault directly. Only the operations that can fail for a REFCOUNT reason
-- -- freeObject and friends -- lift into ObjFault. Writing `memory ...` here was
-- a level confusion and the typechecker said so.
use-after-free-through-the-lane :
  refcountOf Inline h₅ box shaped ≡ err use-after-free
use-after-free-through-the-lane = refl

-- And so does the moved copy: a move produced no independent handle that could
-- outlive the object.
moved-reference-faults-too :
  refcountOf Inline h₅ moved shaped ≡ err use-after-free
moved-reference-faults-too = refl

-- ⚠️ A SECOND free reports use-after-free, NOT double-free -- and that is a
-- loss of attribution this design pays for, not a detail.
--
-- `freeObject` reads the refcount through the lane before deciding, so on a
-- freed object the READ faults first and the memory layer's `double-free`
-- constructor is never reached. The memory layer can still tell the two apart
-- (Proof.Memory.Properties.double-free-is-caught); the object layer cannot,
-- because its precondition lives in the storage it is about to release.
--
-- The alternative is to consult the refcount some other way, which means a
-- second lane -- exactly what this design removes. So it is a real trade, and
-- the equation below is what it costs.
double-free-reports-use-after-free :
  freeObject Inline h₅ box shaped ≡ err (memory use-after-free)
double-free-reports-use-after-free = refl

------------------------------------------------------------------------
-- A payload word is on the same lane as the header.

h₆ : Heap
h₆ with storePayload Inline h₁ box shaped zero 7
... | ok h  = h
... | err _ = h₁

payload-round-trips : loadPayload Inline h₆ box shaped zero ≡ ok 7
payload-round-trips = refl

-- and writing it did not touch the header
payload-write-left-the-refcount : refcountOf Inline h₆ box shaped ≡ ok 1
payload-write-left-the-refcount = refl

payload-write-left-the-class : classOf Inline h₆ box shaped ≡ ok ClassId
payload-write-left-the-class = refl
