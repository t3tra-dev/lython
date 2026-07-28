{-# OPTIONS --safe #-}

-- Threads, a scheduler, and happens-before.
--
-- Sequentially consistent: the scheduler picks one runnable thread and it takes
-- one step. The note is explicit that this is where to start, and that weak
-- memory is a later refinement -- a model that began with release/acquire event
-- graphs would make every ownership rule harder to state for no gain on any
-- defect this compiler has.
--
-- The sequential relation in Proof.Program.Step is the one-thread case of this
-- one, which is why it was worth defining the vocabulary before it is needed.

open import Proof.Memory.Element using (ElemSig)

module Proof.Concurrent.Machine (Sig : ElemSig) where

open ElemSig Sig

open import Data.Bool using (Bool; true; false; if_then_else_)
open import Data.List using (List; []; _∷_; length; _++_)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc; _≟_)
open import Data.Nat.Base using (_≡ᵇ_)
open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; sym; trans)
open import Relation.Nullary using (Dec; yes; no; ¬_)

open import Proof.RC.Machine Sig using (Machine)
open import Proof.Program.Syntax using (Function; Var; BlockId; Instr)
open import Proof.Program.Env using (Env)
open import Proof.Program.Step Sig using (PState; pstate; _⊢_—→_; env; mach)
open import Proof.Concurrent.Event

------------------------------------------------------------------------
-- The thread pool.
--
-- Each thread has its own program position and environment -- names are
-- thread-local, which is why `OwnerSite.local` is indexed by a ThreadId -- and
-- they SHARE the machine. That split is the whole reason concurrency is hard:
-- the env is private and the heap is not.

-- `mkThread`, not `thread`: Event already exports a `thread` field, and a clash
-- would be resolved by import order rather than by intent -- the same care the
-- step relation needed for `term`.
record Thread : Set where
  constructor mkThread
  field
    tid   : ThreadId
    pos   : BlockId
    todo  : List Instr
    tenv  : Env

open Thread public

ThreadPool : Set
ThreadPool = List Thread

record CMachine : Set where
  constructor cmachine
  field
    pool   : ThreadPool
    shared : Machine
    -- The events that have happened, most recent first. Kept because a race is
    -- a property of a PAIR of events, so a relation over states alone cannot
    -- express it.
    history : List Event

open CMachine public

------------------------------------------------------------------------
-- Scheduling.

replaceThread : ThreadPool → ThreadId → Thread → ThreadPool
replaceThread []       _ _  = []
replaceThread (t ∷ ts) i t' =
  if tid t ≡ᵇ i then t' ∷ ts else t ∷ replaceThread ts i t'

-- The scheduler is a NONDETERMINISTIC choice, not a function. Safety has to
-- hold for every schedule, so a model that computed the next thread would prove
-- safety of one schedule -- which is the mistake a test suite makes and a proof
-- must not.
data Scheduled (p : ThreadPool) : Thread → Set where
  here  : ∀ {t ts} → p ≡ t ∷ ts → Scheduled p t
  later : ∀ {t u ts} → p ≡ u ∷ ts → Scheduled ts t → Scheduled p t

------------------------------------------------------------------------
-- The concurrent step.
--
-- One scheduled thread takes one sequential step, and the event it performed is
-- appended to the history. `spawn` and `join` are separate rules because they
-- change the POOL, which no sequential rule can do.

data _⊢_⇒_ (f : Function) : CMachine → CMachine → Set where

  -- A thread runs. Its private state changes, the shared machine changes, and
  -- the event is recorded.
  sched-step :
    ∀ {p sh hs t s' e} →
    Scheduled p t →
    f ⊢ pstate (pos t) (todo t) (tenv t) sh —→ s' →
    mkThread (tid t) (PState.current s') (PState.pending s') (env s') ≡ t →
    f ⊢ cmachine p sh hs ⇒ cmachine p (mach s') (e ∷ hs)

  -- Spawning adds a thread with its OWN environment. The parent's names are not
  -- the child's: whatever the child is to own has to be passed, which is where
  -- the permission split will go when the algebra arrives.
  sched-spawn :
    ∀ {p sh hs parent child e} →
    Scheduled p parent →
    kind e ≡ spawn (tid child) →
    f ⊢ cmachine p sh hs ⇒ cmachine (child ∷ p) sh (e ∷ hs)

  -- Joining removes it. The child's owned names have to have gone somewhere
  -- before this, and that obligation is `join-collects-permissions` below --
  -- stated, not proved.
  sched-join :
    ∀ {p sh hs joiner childId e} →
    Scheduled p joiner →
    kind e ≡ join childId →
    f ⊢ cmachine p sh hs ⇒ cmachine p sh (e ∷ hs)

data _⊢_⇒*_ (f : Function) : CMachine → CMachine → Set where
  cdone : ∀ {c} → f ⊢ c ⇒* c
  cmore : ∀ {c d e} → f ⊢ c ⇒ d → f ⊢ d ⇒* e → f ⊢ c ⇒* e

------------------------------------------------------------------------
-- Happens-before, and the race predicate.
--
-- Program order within a thread, plus the synchronisation edges. Defined over
-- the HISTORY rather than over the states, because that is what a race is about.

-- `e₁` is earlier than `e₂` in a history written most-recent-first.
data Earlier : List Event → Event → Event → Set where
  adjacent : ∀ {hs e₁ e₂} → Earlier (e₂ ∷ e₁ ∷ hs) e₁ e₂
  skip     : ∀ {hs e₁ e₂ e} → Earlier hs e₁ e₂ → Earlier (e ∷ hs) e₁ e₂

-- Same thread ⇒ ordered. This is program order and it is the only ordering this
-- module establishes; the synchronisation edges (release/acquire pairs,
-- spawn/join) need the permission algebra to say what they transfer, and that
-- is deliberately not yet here.
data HappensBefore (hs : List Event) : Event → Event → Set where
  program-order :
    ∀ {e₁ e₂} → thread e₁ ≡ thread e₂ → Earlier hs e₁ e₂ → HappensBefore hs e₁ e₂
  -- spawn happens-before everything the spawned thread does
  spawn-edge :
    ∀ {e₁ e₂ t} → kind e₁ ≡ spawn t → thread e₂ ≡ t → Earlier hs e₁ e₂ →
    HappensBefore hs e₁ e₂
  -- everything the joined thread did happens-before the join
  join-edge :
    ∀ {e₁ e₂ t} → thread e₁ ≡ t → kind e₂ ≡ join t → Earlier hs e₁ e₂ →
    HappensBefore hs e₁ e₂
  hb-trans :
    ∀ {e₁ e₂ e₃} → HappensBefore hs e₁ e₂ → HappensBefore hs e₂ e₃ →
    HappensBefore hs e₁ e₃

-- THE definition. A race is a conflict that nothing orders -- in either
-- direction, which is why both are required: an analysis that checked only one
-- would clear a program by taking the events in the convenient order.
record Race (hs : List Event) (e₁ e₂ : Event) : Set where
  constructor race
  field
    conflicting : Conflict e₁ e₂
    unordered₁  : ¬ HappensBefore hs e₁ e₂
    unordered₂  : ¬ HappensBefore hs e₂ e₁

open Race public

RaceFree : List Event → Set
RaceFree hs = ∀ e₁ e₂ → ¬ Race hs e₁ e₂

------------------------------------------------------------------------
-- What is NOT here, stated so the absence is not read as a result.
--
-- There is no permission algebra, so there is no theorem that a well-formed
-- program is race-free. The obligations that theorem would discharge are:
--
--   read      needs a positive read share
--   write     needs the whole exclusive permission
--   free      needs the whole permission and the root token
--   spawn     splits the parent's permission into the child's
--   join      collects the child's back
--   release   deposits a resource into the lock invariant
--   acquire   takes one out
--
-- They are a partial commutative monoid -- the Views/Iris shape -- and adding
-- one is a self-contained piece of work. `RaceFree` above is the statement it
-- would prove; nothing currently proves it, and `Race` being definable is not
-- the same as any program being shown free of one.
