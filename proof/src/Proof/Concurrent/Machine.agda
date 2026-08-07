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
open import Data.Maybe using (Maybe; just; nothing; maybe′)
open import Data.Nat using (ℕ; zero; suc; _≟_)
open import Data.Nat.Base using (_≡ᵇ_)
open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; sym; trans)
open import Relation.Nullary using (Dec; yes; no; ¬_)

open import Proof.RC.Machine Sig using (Machine)
open import Proof.Program.Syntax using (Function; Var; BlockId; Instr)
open import Proof.Program.Env using (Env)
open import Proof.Program.Step Sig
  using (PState; pstate; _⊢_—→_; current; pending; env; mach)
open import Proof.Concurrent.Event
open import Proof.RC.Object using (ObjId)
open import Proof.Program.Env using (entityOf; lookupVar; mode; entity; isOwned)
open import Proof.Program.Syntax
  using (Var; Instr; alloc; init; move; dup; drop; borrow; getField;
         setField; callOut)

------------------------------------------------------------------------
-- ⭐ The event an instruction performs.
--
-- The first version let the scheduler record ANY event. A history is what
-- `Race` quantifies over, so an arbitrary history makes a race a statement
-- about nothing: a `borrow` could record a write, and two events of one thread
-- could be attributed to two. Deriving the event from the instruction is what
-- makes the history a record of the program.
--
-- The table is the ownership table read as memory traffic:
--
--   alloc   allocates, and touches no existing object
--   init    writes the header of an object no other thread can name yet
--   dup     a read-modify-write of the refcount word     -- py.incref
--   drop    the same                                     -- py.decref
--   move    NOTHING. This is the row where the correct number of runtime
--           operations is zero, and it has to be `nothing` here or the model
--           would claim traffic that the compiler is right not to emit.
--   borrow  nothing, for the same reason
--
-- `getField` / `setField` are `nothing` too, and that is a LIMITATION rather
-- than a claim: their footprint is a payload word, which needs the object's
-- layout, and the layout lives in Proof.Object.Layout over a different element
-- signature. Field traffic is therefore invisible to the race predicate here.

-- What the compiler emits for an object's refcount updates.
--
-- `nothing` means NO memory operation at all, which is the right answer for an
-- immortal: `bumpUp immortal ≡ immortal`, so there is nothing to write and a
-- compiler that emitted a plain rmw there would be creating a race for a value
-- that never changes. `just plain` and `just atomic` are the two real
-- emissions.
--
-- A parameter rather than a constant, because it is the only decision in the
-- whole emission that can be wrong -- `Proof.Concurrent.RaceFree` is the
-- theorem about choosing it correctly.
Policy : Set
Policy = ObjId → Maybe Atomicity

-- A refcount update, if the policy asks for one at all. Only for an OWNED
-- name: `step-dup` and `step-drop` both require it, and an event emitted for a
-- borrow would be traffic the step relation says does not happen.
rcEventFor : ThreadId → Policy → Env → Var → Maybe Event
rcEventFor t pol es v =
  maybe′ (λ b → if isOwned (mode b)
                  then maybe′ (λ a → just (event t (access rmw a)
                                                 (just (rcFootprint (entity b)))))
                              nothing (pol (entity b))
                  else nothing)
         nothing (lookupVar es v)

-- A payload access. The footprint is one word, at `HeaderWords + k` -- which is
-- never word 0, so it can never be mistaken for refcount traffic.
fieldEventFor : ThreadId → AccessMode → Env → Var → FieldId → Maybe Event
fieldEventFor t md es v k =
  maybe′ (λ o → just (event t (access md plain) (just (fieldFootprint o k))))
         nothing (entityOf es v)

instrEvent : ThreadId → Policy → Instr → Env → Maybe Event
instrEvent t pol (alloc x c)      es = just (event t allocate nothing)
-- ⛔ A LIMITATION, recorded rather than asserted away. `init` really does write
-- word 0, so the honest event is a plain write at `rcFootprint`. Emitting one
-- would oblige `RaceFree` to show that no other thread can name a freshly
-- allocated object -- a cross-thread freshness theorem this layer does not
-- have. Folding it into `allocate` claims less than the truth (an inert event
-- where there is traffic) rather than more, which is the direction that cannot
-- make a racy program look race-free: `Race` needs BOTH sides to be accesses,
-- so an event that is not one is never half of a race here.
instrEvent t pol (init x)         es = just (event t allocate nothing)
instrEvent t pol (dup dst src)    es = rcEventFor t pol es src
instrEvent t pol (drop x)         es = rcEventFor t pol es x
instrEvent t pol (move _ _)       es = nothing
instrEvent t pol (borrow _ _)     es = nothing
instrEvent t pol (getField _ src k) es = fieldEventFor t reads  es src k
instrEvent t pol (setField dst k _) es = fieldEventFor t writes es dst k
-- `callOut` is a move: one site vacated, one occupied, no byte touched and no
-- counter changed. `move` is `nothing` for the same reason.
instrEvent t pol (callOut _ _)    es = nothing

-- The event of the instruction a thread is about to run. A thread at a
-- terminator has none.
eventFor : ThreadId → Policy → List Instr → Env → Maybe Event
eventFor t pol []      es = nothing
eventFor t pol (i ∷ _) es = instrEvent t pol i es

-- Recording it. Operations that perform no memory traffic append nothing --
-- which is what makes "a move is free" true of the history and not only of the
-- counter.
record? : Maybe Event → List Event → List Event
record? nothing  hs = hs
record? (just e) hs = e ∷ hs

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

data _⊢[_]_⇒_ (f : Function) (pol : Policy) : CMachine → CMachine → Set where

  -- A thread runs. Its private state changes, the shared machine changes, and
  -- the event is recorded.
  --
  -- Two things here were wrong in the first version and were found by asking
  -- whether the relation had ever been inhabited -- it had not, and it could
  -- not have been.
  --
  -- (1) The pool was carried through UNCHANGED and the rule instead demanded
  --     `mkThread (tid t) (current s') (pending s') (env s') ≡ t`: that the
  --     thread be identical after the step. Every real step consumes an
  --     instruction, so `pending` shrinks and the equation is unsatisfiable.
  --     A thread that cannot advance is not a scheduler rule, and the whole
  --     concurrent layer was uninhabitable through it.
  --
  -- (2) The event `e` was a free variable of the rule, so the history was
  --     unconstrained -- a single-threaded program could record events
  --     attributed to other threads and `Conflict.different-threads` became
  --     satisfiable inside it. It is now DERIVED by `eventFor` from the
  --     instruction being run, so `e` is not a parameter of the rule at all.
  sched-step :
    ∀ {p sh hs t s'} →
    Scheduled p t →
    -- ⭐ The sequential state is now created ON THE SCHEDULED THREAD. Before
    -- `PState` carried a thread id, `siteOf` named a fixed one, and any
    -- instruction that touched the site map would have attributed thread 1's
    -- owner sites to thread 0 -- so the concurrent layer could only schedule
    -- instructions that leave the machine alone. It can now schedule anything.
    f ⊢ pstate (tid t) (pos t) (todo t) (tenv t) sh —→ s' →
    f ⊢[ pol ] cmachine p sh hs
      ⇒ cmachine (replaceThread p (tid t)
                    (mkThread (tid t) (current s') (pending s') (env s')))
                 (mach s')
                 (record? (eventFor (tid t) pol (todo t) (tenv t)) hs)

  -- Spawning adds a thread with its OWN environment. The parent's names are not
  -- the child's: whatever the child is to own has to be passed, which is where
  -- the permission split will go when the algebra arrives.
  sched-spawn :
    ∀ {p sh hs parent child e} →
    Scheduled p parent →
    kind e ≡ spawn (tid child) →
    thread e ≡ tid parent →
    f ⊢[ pol ] cmachine p sh hs ⇒ cmachine (child ∷ p) sh (e ∷ hs)

  -- Joining removes it. The child's owned names have to have gone somewhere
  -- before this, and that obligation is `join-collects-permissions` below --
  -- stated, not proved.
  sched-join :
    ∀ {p sh hs joiner childId e} →
    Scheduled p joiner →
    kind e ≡ join childId →
    thread e ≡ tid joiner →
    f ⊢[ pol ] cmachine p sh hs ⇒ cmachine p sh (e ∷ hs)

data _⊢[_]_⇒*_ (f : Function) (pol : Policy) : CMachine → CMachine → Set where
  cdone : ∀ {c} → f ⊢[ pol ] c ⇒* c
  cmore : ∀ {c d e} → f ⊢[ pol ] c ⇒ d → f ⊢[ pol ] d ⇒* e → f ⊢[ pol ] c ⇒* e

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
