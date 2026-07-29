{-# OPTIONS --safe #-}

-- A concrete reference-counting trace, checked by computation.
--
-- The theorems in Proof.RC.Properties are conditional, and Proof.RC.Invariant
-- is a record nobody has yet been required to build. So this module does the
-- one thing neither of those does: it runs the operations on an actual machine
-- and checks that the runtime counter and the ghost count move together.
--
-- If they ever disagree, every `refl` below stops typechecking.

module Proof.RC.Trace where

open import Data.List using (List; []; _∷_)
open import Data.Integer using (ℤ; +_)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc)
open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Data.Vec using (Vec; []; _∷_)
open import Relation.Binary.PropositionalEquality using (_≡_; refl)

open import Proof.Prelude using (Result; err; ok)
open import Proof.Memory.Heap using (Heap)
open import Proof.Memory.Lython using (LythonSig; i8)
open import Proof.Memory.Descriptor LythonSig using (Desc; desc)
open import Proof.MemRef.Dialect LythonSig using (alloc)

open import Proof.RC.Object
open import Proof.RC.OwnerSite
open import Proof.RC.Machine LythonSig
open import Proof.RC.Ops     LythonSig

------------------------------------------------------------------------
-- One object, allocated, and held by one local slot.

allocated : Heap × Desc 1
allocated = alloc [] 0 i8 8 8

theObj : ObjId
theObj = obj 0 0

-- Two sites in thread 0: slot 1 already holds the object, slot 2 is free.
site₁ site₂ : OwnerSite
site₁ = local 0 1
site₂ = local 0 2

m₀ : Machine
m₀ = machine (proj₁ allocated)
             ((theObj , cell live (counted 1) (proj₂ allocated) 0) ∷ [])
             ((site₁ , theObj) ∷ [])

-- The starting point IS consistent: counter 1, one owner site. Checked rather
-- than assumed, because every equation below is relative to it.
start-consistent : countOf m₀ theObj ≡ just (counted 1)
start-consistent = refl

start-ghost : ghostRC m₀ theObj ≡ 1
start-ghost = refl

------------------------------------------------------------------------
-- py.incref

protectedRef : ProtectedRef m₀ theObj
protectedRef = protected-by site₁ refl refl

m₁ : Machine
m₁ with retain m₀ theObj protectedRef site₂
... | ok m  = m
... | err _ = m₀

-- Both sides moved, and by the same amount. Separately stated: a retain that
-- bumped only the counter satisfies the first and not the second, and that is
-- precisely the bug this layer exists to rule out.
incref-counter : countOf m₁ theObj ≡ just (counted 2)
incref-counter = refl

incref-ghost : ghostRC m₁ theObj ≡ 2
incref-ghost = refl

-- Retaining into an occupied site is refused. Overwriting would drop the
-- reference that site already held -- a leak, silently.
incref-into-occupied : retain m₀ theObj protectedRef site₁ ≡ err destination-occupied
incref-into-occupied = refl

------------------------------------------------------------------------
-- py.decref

m₂ : Machine
m₂ with release m₁ site₂ theObj
... | ok m  = m
... | err _ = m₁

decref-counter : countOf m₂ theObj ≡ just (counted 1)
decref-counter = refl

decref-ghost : ghostRC m₂ theObj ≡ 1
decref-ghost = refl

-- The object is still live: one site still holds it.
decref-keeps-live : lifeOf m₂ theObj ≡ just live
decref-keeps-live = refl

-- Releasing a site that holds nothing is refused rather than silently ignored.
-- An ignored decref is how a counter drifts below its true value, and the
-- resulting use-after-free surfaces arbitrarily far away.
decref-unowned : release m₂ site₂ theObj ≡ err release-unowned
decref-unowned = refl

------------------------------------------------------------------------
-- The last reference.

m₃ : Machine
m₃ with release m₂ site₁ theObj
... | ok m  = m
... | err _ = m₂

last-decref-ghost : ghostRC m₃ theObj ≡ 0
last-decref-ghost = refl

-- Zero, and `finalizing` -- NOT dead, and the storage is still there. Splitting
-- those two states is what gives a finalizer somewhere to run.
last-decref-finalizes : lifeOf m₃ theObj ≡ just finalizing
last-decref-finalizes = refl

last-decref-leaves-heap : heap m₃ ≡ heap m₀
last-decref-leaves-heap = refl

------------------------------------------------------------------------
-- reclaim

reclaimed : Result RCFault (Desc 1 × Machine)
reclaimed = reclaim m₃ theObj

-- It succeeds, and hands back the CANONICAL descriptor -- the one alloc
-- returned, which is the only one memref.dealloc accepts. Handing back a view
-- here would make the free invalid at the next layer down.
reclaim-yields-root : reclaimed ≡ ok (proj₂ allocated
                                     , machine (heap m₃)
                                               ((theObj , cell dead (counted 0)
                                                               (proj₂ allocated) 0) ∷ [])
                                               (sites m₃))
reclaim-yields-root = refl

-- And it is refused one step earlier, while a site still holds the object.
-- This is the theorem that keeps a premature free from becoming a
-- use-after-free at the memory layer.
reclaim-too-early : reclaim m₂ theObj ≡ err not-finalizing
reclaim-too-early = refl

------------------------------------------------------------------------
-- move and borrow cost nothing.

moved : Result RCFault Machine
moved = moveRef m₀ site₁ site₂ theObj

-- The counter is untouched, and so is the count. A pass that emitted an
-- incref/decref pair here would be correct and would pay two atomic operations
-- for a no-op; one that emitted only the decref is the over-release shape.
move-keeps-counter : moved ≡ ok (machine (heap m₀) (objects m₀)
                                         ((site₂ , theObj) ∷ []))
move-keeps-counter = refl

borrow-costs-nothing : borrowRef m₀ theObj protectedRef ≡ m₀
borrow-costs-nothing = refl

------------------------------------------------------------------------
-- Immortals.
--
-- Lython's small-int cache is exactly this, and one of its shipped defects
-- turned on the {0,1,2} boundary. An immortal's counter does not move in either
-- direction, so it can never reach zero and can never be reclaimed.

immortalObj : ObjId
immortalObj = obj 1 0

mᵢ : Machine
mᵢ = machine (proj₁ allocated)
             ((immortalObj , cell live immortal (proj₂ allocated) 0) ∷ [])
             ((site₁ , immortalObj) ∷ [])

immortal-incref-is-noop :
  countOf mᵢ immortalObj ≡ just immortal
immortal-incref-is-noop = refl

immortal-decref-keeps-live :
  (release mᵢ site₁ immortalObj) ≡
    ok (machine (heap mᵢ)
                ((immortalObj , cell live immortal (proj₂ allocated) 0) ∷ [])
                [])
immortal-decref-keeps-live = refl

-- And this is the interesting one: after that release NO site holds it, so the
-- ghost count is 0 while the object is still `live`. That combination is
-- exactly what `WFRC.live-positive` forbids -- for COUNTED objects. Immortals
-- are excluded from it by `IsCounted`, and this equation is why that exclusion
-- has to be there rather than being an oversight.
immortal-can-have-zero-owners :
  ghostRC (machine (heap mᵢ)
                   ((immortalObj , cell live immortal (proj₂ allocated) 0) ∷ [])
                   []) immortalObj ≡ 0
immortal-can-have-zero-owners = refl
