{-# OPTIONS --safe #-}

-- The situations Lython's semantics forbid, as predicates that can be checked.
--
-- Deliberately NOT a permission algebra. A fractional-permission PCM answers
-- "may this thread touch these bytes" in general; what is needed here is
-- narrower and more specific -- the handful of things THIS language calls
-- invalid, each stated so that a program can be shown not to reach it.
--
-- The four below are chosen because each is a real rule of this implementation,
-- not a general safety principle:
--
--   1. a refcount update on an object two threads can reach must be atomic
--      -- this is the rule the GIL used to enforce, and the one PEP 703's
--         biased counting replaces. The model already indexes owner sites by
--         ThreadId, so "two threads can reach it" is decidable from the site
--         map rather than requiring an escape analysis.
--
--   2. a container may not change length while an iterator over it is live
--      -- CPython raises RuntimeError; this compiler emits a mutation guard.
--         It is a SEMANTIC rule, not a memory-safety one, and it stays true in
--         a one-lane world where the representational hazard is already gone.
--
--   3. a borrow may not outlive its anchor
--      -- the only obligation a borrow carries, and the reason Mode.borrowed
--         records the name it came from.
--
--   4. an object must not be reclaimed while any name still denotes it
--      -- the program-level form of what `reclaim` checks at the machine level.

open import Proof.Memory.Element using (ElemSig)

module Proof.Lython.Invalid (Sig : ElemSig) where

open ElemSig Sig

open import Data.Bool using (Bool; true; false)
open import Data.Empty using (⊥; ⊥-elim)
open import Data.List using (List; []; _∷_)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc; _≟_)
open import Data.Product using (_×_; _,_; Σ; proj₁; proj₂)
open import Relation.Binary.PropositionalEquality using (_≡_; _≢_; refl; sym; trans)
open import Relation.Nullary using (Dec; yes; no; ¬_)

open import Proof.RC.Object using (ObjId; obj; objAllocation; objGeneration;
  Life; live; finalizing; dead;
  RuntimeCount; counted; immortal)
open import Proof.RC.OwnerSite using (OwnerSite; local; field′; global; queue; temp;
  ThreadId; SiteMap; strongAt; logicalRC)
open import Proof.RC.Machine Sig
open import Proof.Program.Syntax using (Var)
open import Proof.Program.Env
open import Proof.Prelude using (ByteRange; range)
open import Proof.Memory.Heap using (AllocId; Generation)
open import Proof.Object.Word using (WordBytes)
open import Proof.Concurrent.Event using (Event; kind; footprint; modeOf;
  AccessMode; reads; writes; rmw; Atomicity; plain; atomic)

------------------------------------------------------------------------
-- 1. Sharing, and when a refcount update has to be atomic.
--
-- An owner site knows which thread it belongs to. `local` and `temp` are
-- thread-indexed; `field′`, `global` and `queue` are not, because a reference
-- parked in a field, a global or a message queue is reachable from ANY thread
-- that can reach the holder. Treating those as thread-local is the mistake that
-- makes a biased refcounting scheme unsound.

ownerThread : OwnerSite → Maybe ThreadId
ownerThread (local t _) = just t
ownerThread (temp  t _) = just t
ownerThread (field′ _ _) = nothing
ownerThread (global _)   = nothing
ownerThread (queue _ _)  = nothing

-- Two sites hold the object and they belong to different threads -- or one of
-- them belongs to no thread at all, which is the escaped case.
record SharedAcrossThreads (m : Machine) (o : ObjId) : Set where
  constructor shared-by
  field
    site₁ site₂ : OwnerSite
    holds₁ : strongAt (sites m) site₁ ≡ just o
    holds₂ : strongAt (sites m) site₂ ≡ just o
    -- Different sites, and not both owned by the same thread. Stated as a
    -- disjunction so the escaped case (`nothing`, a field or a global) is a
    -- witness in its own right rather than an afterthought.
    distinct : site₁ ≢ site₂
    not-same-thread : ¬ (Σ ThreadId λ t →
                          (ownerThread site₁ ≡ just t) × (ownerThread site₂ ≡ just t))

open SharedAcrossThreads public

-- When a refcount update has to be atomic: the object is reachable from more
-- than one thread AND its counter is a number that changes. Both conjuncts are
-- load-bearing, and the second is why the small-int cache is safe to share
-- without synchronisation.
NeedsAtomicRC : Machine → ObjId → Set
NeedsAtomicRC m o = SharedAcrossThreads m o × (Σ ℕ λ n → countOf m o ≡ just (counted n))

-- ⭐ An immortal never needs one. A theorem rather than a convention: there is
-- no `n` with `just immortal ≡ just (counted n)`, so the second conjunct cannot
-- be met however widely the object is shared.
--
-- This is the Lython-specific payoff. `{0,1,2}` are immortal in this runtime,
-- they are shared by every thread that touches a small integer, and no
-- synchronisation is owed for them.
immortal-needs-no-atomic :
  ∀ (m : Machine) (o : ObjId) →
  countOf m o ≡ just immortal → ¬ NeedsAtomicRC m o
immortal-needs-no-atomic m o imm (_ , (n , cnt)) with trans (sym imm) cnt
... | ()

-- The byte range a refcount update touches: word 0 of the object's own
-- allocation, which is where the one-lane layout puts it.
rcFootprint : ObjId → AllocId × Generation × ByteRange
rcFootprint o = objAllocation o , objGeneration o , range 0 WordBytes

-- THE invalidity: a PLAIN read-modify-write on the refcount word of an object
-- that needs an atomic one.
--
-- Every conjunct is a real proposition. An earlier draft had two of them as
-- `⊤`, which would have made a race derivable for any object at all -- the same
-- vacuity that three lemmas in the object layer and two conjuncts of `Conflict`
-- were rewritten for.
record RefcountRace (m : Machine) (o : ObjId) (e : Event) : Set where
  constructor rc-race
  field
    needs-atomic : NeedsAtomicRC m o
    -- The event really is a read-modify-write, and really is non-atomic.
    is-plain-rmw : modeOf (kind e) ≡ just (rmw , plain)
    -- and it really is this object's refcount word
    targets-rc   : footprint e ≡ just (rcFootprint o)

open RefcountRace public

-- The same event on an immortal is not a race, by the theorem above. Stated
-- because "we made the counter atomic everywhere" and "we proved where it does
-- not have to be" are different results, and only the second is worth the
-- atomics it saves.
immortal-rc-update-is-not-a-race :
  ∀ (m : Machine) (o : ObjId) (e : Event) →
  countOf m o ≡ just immortal → ¬ RefcountRace m o e
immortal-rc-update-is-not-a-race m o e imm r =
  immortal-needs-no-atomic m o imm (needs-atomic r)

------------------------------------------------------------------------
-- 2. Iterator invalidation.
--
-- An iterator over a container records the container and the length it saw. It
-- is invalidated when the container's length changes -- which is CPython's rule
-- and the one this compiler already guards for dicts.
--
-- Note what this does NOT depend on: the one-lane representation removes the
-- REPRESENTATIONAL hazard (there is no separate items lane to go stale), and
-- this rule survives it untouched, because it is about the container's SIZE
-- being observed, not about where the bytes live. Conflating the two is why
-- "we made it one lane" is not an answer to "does mutation during iteration
-- still raise".

record Iterator : Set where
  constructor iter
  field
    container    : ObjId
    lengthAtStart : ℕ

open Iterator public

record IteratorInvalidated (it : Iterator) (currentLength : ℕ) : Set where
  constructor invalidated
  field
    changed : lengthAtStart it ≢ currentLength

open IteratorInvalidated public

-- Decidable, because the guard the compiler emits has to decide it at runtime.
iteratorValid? : (it : Iterator) → (n : ℕ) → Dec (lengthAtStart it ≡ n)
iteratorValid? it n = lengthAtStart it ≟ n

------------------------------------------------------------------------
-- 3. A dangling borrow.
--
-- `x` is bound as a borrow of `a`, and `a` is no longer bound. This is the
-- whole obligation a borrow carries: it costs no runtime operation, so the only
-- way it can be wrong is by outliving what it points into.

record DanglingBorrow (es : Env) (x : Var) : Set where
  constructor dangling
  field
    anchor      : Var
    is-borrow   : Σ ObjId λ o → lookupVar es x ≡ just (bind o (borrowed anchor))
    anchor-gone : lookupVar es anchor ≡ nothing

open DanglingBorrow public

-- And the negative form, which is what a program is asked to satisfy.
NoDanglingBorrows : Env → Set
NoDanglingBorrows es = ∀ x → ¬ DanglingBorrow es x

------------------------------------------------------------------------
-- 4. Premature reclaim.
--
-- The program-level form of the machine-level check. A name still denotes the
-- object, so freeing its storage would make that name a use-after-free at the
-- next read.

record StillNamed (es : Env) (o : ObjId) : Set where
  constructor named-by
  field
    name      : Var
    holds-it  : entityOf es name ≡ just o

open StillNamed public

record PrematureReclaim (es : Env) (m : Machine) (o : ObjId) : Set where
  constructor premature
  field
    still-named : StillNamed es o
    being-freed : lifeOf m o ≡ just dead

open PrematureReclaim public

------------------------------------------------------------------------
-- The whole list, as one predicate.
--
-- A state is INVALID when any of them holds. Enumerating them rather than
-- folding them into one condition keeps the attribution: a checker that
-- answered "invalid" could not say which rule, and this project has repeatedly
-- found that being able to name which guarantee broke is what makes a defect
-- attributable to the pass that caused it.

data Invalidity (es : Env) (m : Machine) : Set where
  refcount-race     : ∀ {o e} → RefcountRace m o e → Invalidity es m
  dangling-borrow   : ∀ {x} → DanglingBorrow es x → Invalidity es m
  premature-reclaim : ∀ {o} → PrematureReclaim es m o → Invalidity es m

Valid : Env → Machine → Set
Valid es m = ¬ Invalidity es m
