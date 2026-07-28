{-# OPTIONS --safe #-}

-- The one-lane object layout.
--
-- This is a REDESIGN, not a transcription of what Lython emits today. The
-- design rule it follows is one sentence:
--
--     an object reference is ONE descriptor, and every field of the object is
--     an index into it.
--
-- No (header, payload) pair, no side lane, no second SSA value travelling
-- alongside. That is what "one lane" means, and it is checkable: the theorems
-- in Proof.Object.Coherence say that every field access resolves inside the
-- same allocation as the reference itself, so there is no lane that can go
-- stale independently of the handle. The multi-lane failures this compiler has
-- shipped -- a holder keeping an `items` lane that a growth already freed --
-- are not expressible here, because there is nothing to keep.
--
--     ┌──────────────────── memref<N x i64> ────────────────────┐
--     │ rc │ class │ length │ capacity │ buf.alloc │ buf.gen │ … │
--     └─────────────────────────────────────────────────────────┘
--       0     1        2         3           4          5      6…
--
-- Every object has the same six-word header, exactly as CPython gives every
-- object the same PyObject header, and for the same reason: incref, decref and
-- type dispatch must work without knowing the class.

module Proof.Object.Layout where

open import Data.Fin using (Fin; zero; suc; fromℕ<; toℕ)
open import Data.Nat using (ℕ; zero; suc; _+_; _<_; s≤s; z≤n; _≤_)
open import Data.Nat.Properties using (m≤m+n; ≤-refl; ≤-trans; ≤-reflexive; <⇒≱; +-suc; +-comm)
open import Data.Empty using (⊥; ⊥-elim)
open import Relation.Binary.PropositionalEquality using (_≡_; _≢_; refl; cong; sym; subst)
open import Relation.Nullary using (¬_)

------------------------------------------------------------------------
-- Header word indices.
--
-- `length` and `capacity` are BOTH here, and they are different facts:
-- length is semantic (how many elements the object has) and capacity is
-- representational (how many the buffer can hold). Conflating them is how a
-- container reports a size that its storage does not support.

rcWord classWord lengthWord capacityWord bufAllocWord bufGenWord : ℕ
rcWord       = 0
classWord    = 1
lengthWord   = 2
capacityWord = 3
bufAllocWord = 4
bufGenWord   = 5

HeaderWords : ℕ
HeaderWords = 6

-- Total words in a box with `inline` payload words stored in-place.
--
-- Fixed-size objects (a float, a small int) put their payload inline and leave
-- the buffer words zero. Variable-size ones (str, list, dict) keep the payload
-- in a SEPARATE allocation named by words 4 and 5 -- the note's "stable box,
-- resizable buffer" split. The box itself never moves, which is what lets an
-- object be resized while shared: every alias holds the same box descriptor and
-- therefore sees the updated buffer words. `memref.realloc` on the object
-- itself could not do that, because it cannot reach the aliases.
boxWords : ℕ → ℕ
boxWords inline = HeaderWords + inline

-- Every header index is inside every box, whatever its payload. Proved rather
-- than assumed, because it is what makes the accessors total.
header-fits : ∀ (inline k : ℕ) → k < HeaderWords → k < boxWords inline
header-fits inline k lt = ≤-trans lt (m≤m+n HeaderWords inline)

rc-fits       : ∀ inline → rcWord       < boxWords inline
class-fits    : ∀ inline → classWord    < boxWords inline
length-fits   : ∀ inline → lengthWord   < boxWords inline
capacity-fits : ∀ inline → capacityWord < boxWords inline
bufAlloc-fits : ∀ inline → bufAllocWord < boxWords inline
bufGen-fits   : ∀ inline → bufGenWord   < boxWords inline
rc-fits       inline = header-fits inline rcWord       (s≤s z≤n)
class-fits    inline = header-fits inline classWord    (s≤s (s≤s z≤n))
length-fits   inline = header-fits inline lengthWord   (s≤s (s≤s (s≤s z≤n)))
capacity-fits inline = header-fits inline capacityWord (s≤s (s≤s (s≤s (s≤s z≤n))))
bufAlloc-fits inline = header-fits inline bufAllocWord (s≤s (s≤s (s≤s (s≤s (s≤s z≤n)))))
bufGen-fits   inline = header-fits inline bufGenWord
                         (s≤s (s≤s (s≤s (s≤s (s≤s (s≤s z≤n))))))

-- A payload word is at HeaderWords + i, and it is inside the box because the
-- box is exactly that long. This is the fact that keeps payload access on the
-- same lane as the header -- an object whose payload lived elsewhere would
-- need a second descriptor here, and that is the design being rejected.
payload-fits : ∀ (inline : ℕ) (i : Fin inline) → HeaderWords + toℕ i < boxWords inline
payload-fits inline i = go HeaderWords inline i
  where
    go : ∀ (h n : ℕ) (j : Fin n) → h + toℕ j < h + n
    go zero    (suc n) zero    = s≤s z≤n
    go zero    (suc n) (suc j) = s≤s (go zero n j)
    go (suc h) n       j       = s≤s (go h n j)

------------------------------------------------------------------------
-- The header words are pairwise distinct.
--
-- Stated because the accessors would otherwise be free to alias: a layout that
-- put the refcount and the class id at the same index would satisfy every
-- containment fact above and still be wrong, and incref would rewrite the type.

-- Distinctness is by `λ ()` on ℕ literals: the indices are 0..5 and Agda
-- discharges each disequality by constructor clash. Cheap, and the point is
-- that they are stated at all -- an accessor set is only as good as the fact
-- that its indices differ.
rc≢class      : rcWord       ≢ classWord
rc≢length     : rcWord       ≢ lengthWord
rc≢capacity   : rcWord       ≢ capacityWord
rc≢bufAlloc   : rcWord       ≢ bufAllocWord
rc≢bufGen     : rcWord       ≢ bufGenWord
class≢length  : classWord    ≢ lengthWord
length≢cap    : lengthWord   ≢ capacityWord
bufAlloc≢gen  : bufAllocWord ≢ bufGenWord
rc≢class      = λ ()
rc≢length     = λ ()
rc≢capacity   = λ ()
rc≢bufAlloc   = λ ()
rc≢bufGen     = λ ()
class≢length  = λ ()
length≢cap    = λ ()
bufAlloc≢gen  = λ ()

-- And no header word is a payload word: the payload starts where the header
-- ends. This is what stops a payload write from clobbering the refcount, which
-- is the single most consequential aliasing question in the layout.
header≢payload : ∀ (k : ℕ) → k < HeaderWords → ∀ (i : ℕ) → k ≢ HeaderWords + i
header≢payload k lt i eq = <⇒≱ shifted (m≤m+n HeaderWords i)
  where
    -- If a header index WERE a payload index, the header would have to start
    -- after itself.
    shifted : HeaderWords + i < HeaderWords
    shifted = subst (_< HeaderWords) eq lt
