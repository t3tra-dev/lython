{-# OPTIONS --safe #-}

-- Each invalidity, exhibited on a concrete state.
--
-- The predicates in Proof.Lython.Invalid are records. A record nobody builds is
-- a predicate nothing satisfies, and a checker for it would be trivially sound.
-- These are inhabitants -- and, for the immortal case, a proof that the
-- predicate is NOT inhabited, which is the direction that saves atomics.
--
-- The machine below is WELL-FORMED: `wf` at the bottom is a `WFRC` witness for
-- it. The first version of this module was not, and that mattered. It gave
-- `counted-obj` a counter of 3 while only two owner sites held it, and it gave
-- the whole machine an empty heap while both objects claimed allocations in it.
-- Every equation still typechecked, because none of them looked at the
-- invariant -- so the entire invalidity layer was demonstrated on a state the
-- refcount layer forbids. A race exhibited on an impossible machine is not
-- evidence that the race is reachable.

module Proof.Lython.Trace where

open import Data.Bool using (Bool; true; false)
open import Data.Empty using (⊥; ⊥-elim)
open import Data.List using (List; []; _∷_)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc; _<_; s≤s; z≤n)
open import Data.Product using (_×_; _,_; Σ; proj₁; proj₂)
open import Data.Vec using (Vec; []; _∷_)
open import Data.Integer using (+_)
open import Relation.Binary.PropositionalEquality using (_≡_; _≢_; refl; sym; trans)
open import Relation.Nullary using (¬_)

open import Proof.Prelude using (range)
open import Proof.Memory.Heap using (Heap; Block; block; lookupBlock; generation;
  liveness; heapAlloc)
  renaming (live to blockLive; dead to blockDead)
open import Proof.Memory.Lython using (LythonSig; i64)
open import Proof.Memory.Descriptor LythonSig using (Desc; desc)
open import Proof.MemRef.Dialect LythonSig using (alloc)
open import Proof.RC.Object
open import Proof.RC.OwnerSite using (OwnerSite; local; global; SiteMap;
  strongAt; logicalRC; sameSite; Holds; holds-here; holds-there)
open import Proof.RC.Machine LythonSig
open import Proof.RC.Invariant LythonSig
open import Proof.Program.Syntax using (Var)
open import Proof.Program.Env
open import Proof.Object.Word using (WordBytes)
open import Proof.Concurrent.Event using (Event; event; access; rmw; plain; rcFootprint)
open import Proof.Lython.Invalid LythonSig
open import Proof.Lython.Detect  LythonSig

------------------------------------------------------------------------
-- Two objects: one counted, one immortal.

counted-obj immortal-obj : ObjId
counted-obj  = obj 0 0
immortal-obj = obj 1 0

-- A real heap, with a block for each. Written as two `alloc`s rather than as
-- two hand-built blocks so that the generations and liveness are whatever
-- allocation actually produces -- `no-stale-owner` compares against them, and a
-- hand-written block could satisfy the field while disagreeing with `alloc`.
theHeap : Heap
theHeap = proj₁ (alloc (proj₁ (alloc [] 0 i64 1 8)) 0 i64 1 8)

d0 d1 : Desc 1
d0 = desc 0 0 0 (+ 0) (8 ∷ []) ((+ 1) ∷ []) i64 0
d1 = desc 1 0 0 (+ 0) (8 ∷ []) ((+ 1) ∷ []) i64 0

-- Held by thread 0's slot 1 and thread 1's slot 1 -- two threads.
sharedSites : SiteMap
sharedSites = (local 0 1 , counted-obj)
            ∷ (local 1 1 , counted-obj)
            ∷ (local 0 2 , immortal-obj)
            ∷ (local 1 2 , immortal-obj)
            ∷ []

-- Counter 2, and exactly two owner sites. The two numbers are the same number,
-- which is what `wf` below has to prove and what the first version of this
-- module got wrong.
m : Machine
m = machine theHeap
            ((counted-obj  , cell live (counted 2) d0 0)
             ∷ (immortal-obj , cell live immortal   d1 0)
             ∷ [])
            sharedSites

------------------------------------------------------------------------
-- The machine is well-formed.
--
-- Placed FIRST, before anything is exhibited on it, because everything below is
-- a claim about a reachable state and a state the invariant rejects is not one.

private
  just-inj : ∀ {A : Set} {x y : A} → just x ≡ just y → x ≡ y
  just-inj refl = refl

  counted-inj : ∀ {a b : ℕ} → counted a ≡ counted b → a ≡ b
  counted-inj refl = refl

  -- The two objects are distinct, so no `o` can be both. Needed because the
  -- table and the site map are each walked by a boolean test, and the
  -- (true, true) corner of the case matrix has to be closed rather than assumed
  -- away.
  not-both : ∀ (o : ObjId) → sameObj counted-obj o ≡ true →
             sameObj immortal-obj o ≡ true → ⊥
  not-both o e₀ e₁ with trans (sameObj-sound counted-obj o e₀)
                             (sym (sameObj-sound immortal-obj o e₁))
  ... | ()

blk₀ blk₁ : Block
blk₀ with lookupBlock theHeap 0
... | just b  = b
... | nothing = block 0 0 0 0 [] blockLive heapAlloc
blk₁ with lookupBlock theHeap 1
... | just b  = b
... | nothing = block 0 0 0 0 [] blockLive heapAlloc

private
  storage₀ : ∀ (o : ObjId) → counted-obj ≡ o →
             Σ Block λ b → (lookupBlock (heap m) (objAllocation o) ≡ just b)
                         × (generation b ≡ objGeneration o)
  storage₀ o refl = blk₀ , refl , refl

  storage₁ : ∀ (o : ObjId) → immortal-obj ≡ o →
             Σ Block λ b → (lookupBlock (heap m) (objAllocation o) ≡ just b)
                         × (generation b ≡ objGeneration o)
  storage₁ o refl = blk₁ , refl , refl

  live₀ : ∀ (o : ObjId) → counted-obj ≡ o →
          Σ Block λ b → (lookupBlock (heap m) (objAllocation o) ≡ just b)
                      × (liveness b ≡ blockLive)
  live₀ o refl = blk₀ , refl , refl

  live₁ : ∀ (o : ObjId) → immortal-obj ≡ o →
          Σ Block λ b → (lookupBlock (heap m) (objAllocation o) ≡ just b)
                      × (liveness b ≡ blockLive)
  live₁ o refl = blk₁ , refl , refl

wf : WFRC m
wf = record
  { counted-exact      = ce
  ; live-positive      = lp
  ; dead-unowned       = du
  ; no-stale-owner     = nso
  ; owned-storage-live = osl
  }
  where
    -- ⭐ The field the first version of this module violated. In the
    -- (true, false) corner the hypothesis reduces to `counted 2 ≡ counted n`
    -- and the goal to `n ≡ 2`; with the old counter of 3 it would have reduced
    -- to `counted 3 ≡ counted n` against a goal of `n ≡ 2` and no proof exists.
    ce : ∀ o n → countOf m o ≡ just (counted n) → lifeOf m o ≡ just live →
         n ≡ ghostRC m o
    ce o n cnt _ with sameObj counted-obj o in e₀ | sameObj immortal-obj o in e₁
    ... | true  | true  = ⊥-elim (not-both o e₀ e₁)
    ... | true  | false = sym (counted-inj (just-inj cnt))
    ... | false | true  with cnt
    ...   | ()
    ce o n cnt _ | false | false with cnt
    ...   | ()

    lp : ∀ o → lifeOf m o ≡ just live → IsCounted m o → 0 < ghostRC m o
    lp o lv ct with sameObj counted-obj o in e₀ | sameObj immortal-obj o in e₁
    ... | true  | true  = ⊥-elim (not-both o e₀ e₁)
    ... | true  | false = s≤s z≤n
    -- The immortal corner is where the field would be FALSE if `IsCounted` were
    -- not a premise: the object is live and two sites hold it, but its counter
    -- is not a number, and `live-positive` is a statement about numbers.
    ... | false | true  with ct
    ...   | (_ , ())
    lp o lv ct | false | false with lv
    ...   | ()

    du : ∀ o → lifeOf m o ≡ just dead → ghostRC m o ≡ 0
    du o dd with sameObj counted-obj o | sameObj immortal-obj o
    ... | true  | _     with dd
    ...   | ()
    du o dd | false | true  with dd
    ...   | ()
    du o dd | false | false with dd
    ...   | ()

    -- Over `Holds`, the walk is four constructor patterns instead of four
    -- boolean tests -- and it is the shape that made the field preservable in
    -- the first place.
    nso : ∀ s o → Holds (sites m) s o →
          Σ Block λ b → (lookupBlock (heap m) (objAllocation o) ≡ just b)
                      × (generation b ≡ objGeneration o)
    nso _ _ holds-here                                           = storage₀ _ refl
    nso _ _ (holds-there holds-here)                             = storage₀ _ refl
    nso _ _ (holds-there (holds-there holds-here))               = storage₁ _ refl
    nso _ _ (holds-there (holds-there (holds-there holds-here))) = storage₁ _ refl
    nso _ _ (holds-there (holds-there (holds-there (holds-there ()))))

    osl : ∀ o → 0 < ghostRC m o →
          Σ Block λ b → (lookupBlock (heap m) (objAllocation o) ≡ just b)
                      × (liveness b ≡ blockLive)
    osl o pos with sameObj counted-obj o in e₀ | sameObj immortal-obj o in e₁
    ... | true  | true  = ⊥-elim (not-both o e₀ e₁)
    ... | true  | false = live₀ o (sameObj-sound counted-obj o e₀)
    ... | false | true  = live₁ o (sameObj-sound immortal-obj o e₁)
    ... | false | false with pos
    ...   | ()

------------------------------------------------------------------------
-- 1. Sharing across threads is inhabited.

sharing : SharedAcrossThreads m counted-obj
sharing = shared-by (local 0 1) (local 1 1) holds-here (holds-there holds-here)
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
--
-- The `2` is the counter of a well-formed machine, so this is a race that a
-- reachable state exhibits rather than one produced by an inconsistent one.

plainIncref : Event
plainIncref = event 1 (access rmw plain) (just (rcFootprint counted-obj))

the-race : RefcountRace m counted-obj plainIncref
the-race = rc-race (sharing , (2 , refl)) refl refl

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
