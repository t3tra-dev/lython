{-# OPTIONS --safe #-}

-- Each invalidity, exhibited on a concrete state.
--
-- The predicates in Proof.Lython.Invalid are records. A record nobody builds is
-- a predicate nothing satisfies, and a checker for it would be trivially sound.
-- These are inhabitants -- and, for the immortal case, a proof that the
-- predicate is NOT inhabited, which is the direction that saves atomics.

module Proof.Lython.Trace where

open import Data.Bool using (Bool; true; false)
open import Data.Empty using (⊥; ⊥-elim)
open import Data.List using (List; []; _∷_)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc)
open import Data.Product using (_×_; _,_; Σ; proj₁; proj₂)
open import Data.Vec using ([]; _∷_)
open import Data.Integer using (+_)
open import Relation.Binary.PropositionalEquality using (_≡_; _≢_; refl; sym; trans)
open import Relation.Nullary using (¬_)

open import Proof.Prelude using (range)
open import Proof.Memory.Heap using (Heap)
open import Proof.Memory.Lython using (LythonSig; i64)
open import Proof.Memory.Descriptor LythonSig using (Desc; desc)
open import Proof.RC.Object
open import Proof.RC.OwnerSite using (OwnerSite; local; global)
open import Proof.RC.Machine LythonSig
open import Proof.Program.Syntax using (Var)
open import Proof.Program.Env
open import Proof.Object.Word using (WordBytes)
open import Proof.Concurrent.Event using (Event; event; access; rmw; plain)
open import Proof.Lython.Invalid LythonSig
open import Proof.Lython.Detect  LythonSig

------------------------------------------------------------------------
-- Two objects: one counted, one immortal.

counted-obj immortal-obj : ObjId
counted-obj  = obj 0 0
immortal-obj = obj 1 0

d0 : Desc 1
d0 = desc 0 0 0 (+ 0) (8 ∷ []) ((+ 1) ∷ []) i64 0

-- Held by thread 0's slot 1 and thread 1's slot 1 -- two threads.
sharedSites : List (OwnerSite × ObjId)
sharedSites = (local 0 1 , counted-obj)
            ∷ (local 1 1 , counted-obj)
            ∷ (local 0 2 , immortal-obj)
            ∷ (local 1 2 , immortal-obj)
            ∷ []

m : Machine
m = machine []
            ((counted-obj  , cell live (counted 3) d0)
             ∷ (immortal-obj , cell live immortal   d0)
             ∷ [])
            sharedSites

------------------------------------------------------------------------
-- 1. Sharing across threads is inhabited.

sharing : SharedAcrossThreads m counted-obj
sharing = shared-by (local 0 1) (local 1 1) refl refl
                    (λ ()) not-one-thread
  where
    not-one-thread : ¬ (Σ ℕ λ t →
                         (ownerThread (local 0 1) ≡ just t)
                       × (ownerThread (local 1 1) ≡ just t))
    -- p : just 0 ≡ just t and q : just 1 ≡ just t, so composing them the other
    -- way round gives just 0 ≡ just 1.
    not-one-thread (t , (p , q)) with trans p (sym q)
    ... | ()

------------------------------------------------------------------------
-- 2. A plain refcount RMW on the shared counted object IS a race.

plainIncref : Event
plainIncref = event 1 (access rmw plain) (just (rcFootprint counted-obj))

the-race : RefcountRace m counted-obj plainIncref
the-race = rc-race (sharing , (3 , refl)) refl refl

------------------------------------------------------------------------
-- 3. ⭐ The same event on the IMMORTAL object is NOT a race.
--
-- This is the direction worth having. `{0,1,2}` in this runtime are immortal,
-- every thread that touches a small integer shares them, and this says no
-- synchronisation is owed -- so the atomics a conservative implementation would
-- emit there are provably unnecessary rather than merely believed to be.

plainIncrefImmortal : Event
plainIncrefImmortal = event 1 (access rmw plain) (just (rcFootprint immortal-obj))

immortals-are-free : ¬ RefcountRace m immortal-obj plainIncrefImmortal
immortals-are-free = immortal-rc-update-is-not-a-race m immortal-obj
                       plainIncrefImmortal refl

------------------------------------------------------------------------
-- 4. A dangling borrow, and the checker finding it.

anchorVar borrowVar : Var
anchorVar = 10
borrowVar = 11

-- `borrowVar` borrows from `anchorVar`, and `anchorVar` was dropped.
strandedEnv : Env
strandedEnv = (borrowVar , bind counted-obj (borrowed anchorVar)) ∷ []

checker-finds-it : danglingAnchor strandedEnv borrowVar ≡ just anchorVar
checker-finds-it = refl

it-really-is-one : DanglingBorrow strandedEnv borrowVar
it-really-is-one = danglingAnchor-sound strandedEnv borrowVar anchorVar refl

-- And with the anchor still bound, the checker is silent. Without this the
-- checker could be `λ _ _ → just 0` and still pass the test above.
healthyEnv : Env
healthyEnv = (borrowVar , bind counted-obj (borrowed anchorVar))
           ∷ (anchorVar , bind counted-obj owned)
           ∷ []

checker-is-silent-when-healthy : danglingAnchor healthyEnv borrowVar ≡ nothing
checker-is-silent-when-healthy = refl

-- An owned name is never reported. The third direction, and the one that stops
-- the checker being "report everything".
checker-ignores-owned : danglingAnchor healthyEnv anchorVar ≡ nothing
checker-ignores-owned = refl

------------------------------------------------------------------------
-- 5. Iterator invalidation.

iterOverContainer : Iterator
iterOverContainer = iter counted-obj 3

still-valid : lengthAtStart iterOverContainer ≡ 3
still-valid = refl

grew : IteratorInvalidated iterOverContainer 4
grew = invalidated λ ()

-- and it is NOT invalidated by a length that did not change -- the guard has to
-- distinguish the two or every iteration would raise.
unchanged-is-not-invalidation : ¬ IteratorInvalidated iterOverContainer 3
unchanged-is-not-invalidation inv = changed inv refl
