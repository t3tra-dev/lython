{-# OPTIONS --safe #-}

-- The concurrent layer, taking steps.
--
-- Proof.Concurrent.Machine defines a scheduler, a step relation, happens-before
-- and a race predicate in 190 lines, and NOTHING had ever been derived in any of
-- them: `Scheduled`, `Conflict`, `Race`, `Earlier` were uninhabited and
-- `HappensBefore` appeared only under a negation. A relation with no derivation
-- cannot be told apart from an unsatisfiable one, and `¬ P` is free when `P` is
-- empty -- so neither the positive nor the negative results meant anything.
--
-- Asking for an inhabitant found the reason: `sched-step` was uninhabitable.
-- See the comment on that rule for what was wrong with it.
--
-- Everything below is on `Proof.Lython.Trace.m`, which carries a `WFRC` witness.
-- A race exhibited on a machine the refcount invariant rejects would not be
-- evidence that the race is reachable.

module Proof.Concurrent.Trace where

open import Data.Bool using (Bool; true; false)
open import Data.Empty using (⊥; ⊥-elim)
open import Data.List using (List; []; _∷_)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc; _<_; _≤_; s≤s; z≤n)
open import Data.Product using (_×_; _,_; Σ; proj₁; proj₂)
open import Data.Sum using (_⊎_; inj₁; inj₂)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; sym; trans)
open import Relation.Nullary using (¬_)

open import Proof.Prelude using (ByteRange; range; Overlaps; overlap-at; InRange)
open import Proof.Memory.Lython using (LythonSig)
open import Proof.RC.Object using (ObjId; obj)
open import Proof.RC.Object using (counted; immortal; bumpUp)
open import Proof.RC.OwnerSite using (OwnerSite; occupy; logicalRC; local)
open import Proof.RC.Machine LythonSig
open import Proof.Program.Syntax
open import Proof.Program.Env
open import Proof.Program.Step LythonSig
open import Proof.Concurrent.Event
open import Proof.Concurrent.Machine LythonSig
open import Proof.Concurrent.Event using (rcFootprint)
open import Proof.Lython.Trace using (m; counted-obj; wf)
open import Proof.Program.Preservation LythonSig using (WF; wfs; backed)
open import Proof.Concurrent.RaceFree LythonSig
open import Proof.Lython.Detect LythonSig using (sharedPair)

------------------------------------------------------------------------
-- A program both threads are inside.
--
-- Both threads run `dup c a` -- a py.incref of the SAME object. That is now
-- expressible for the second thread as well: `siteOf` is thread-indexed, so
-- thread 1's dup occupies `local 1 c` and not thread 0's slot. Before, the
-- sequential rules named a fixed thread and the concurrent layer could only
-- schedule instructions that leave the machine alone.

a c : Var
a = 1
c = 5

bb0 : BlockId
bb0 = 0

prog : Function
prog = function (block bb0 [] (dup c a ∷ []) unwind ∷ []) bb0

------------------------------------------------------------------------
-- Two threads, each with its OWN environment and the SHARED machine.
--
-- Each names the shared object; the names are thread-local and the object is
-- not, which is the entire reason concurrency is hard here.

env₀ env₁ : Env
env₀ = (a , bind counted-obj owned) ∷ []
env₁ = (a , bind counted-obj owned) ∷ []

t₀ t₁ : Thread
t₀ = mkThread 0 bb0 (dup c a ∷ []) env₀
t₁ = mkThread 1 bb0 (dup c a ∷ []) env₁

thePool : ThreadPool
thePool = t₀ ∷ t₁ ∷ []

-- ⭐ `Scheduled` inhabited, in both positions. The scheduler is a RELATION, so
-- both of these hold at once and neither is "the" next thread -- which is the
-- property that makes a safety theorem about this relation a theorem about
-- every schedule.
picks-first : Scheduled thePool t₀
picks-first = here refl

picks-second : Scheduled thePool t₁
picks-second = later refl (here refl)

------------------------------------------------------------------------
-- ⭐ The events, DERIVED.
--
-- Not written down and handed to the rule: `eventFor` reads them off the
-- instruction and the environment it runs in. An earlier version of
-- `sched-step` took the event as a free variable, and a history that a rule
-- does not constrain makes every statement about races vacuous.

incref₀ incref₁ : Event
incref₀ = event 0 (access rmw plain) (just (rcFootprint counted-obj))
incref₁ = event 1 (access rmw plain) (just (rcFootprint counted-obj))

-- The naive policy: every refcount update plain. This is the emission a
-- compiler with no sharing analysis produces, and the race below is what it
-- costs. `Proof.Concurrent.RaceFree` is the theorem about the other policy.
naive : Policy
naive _ = just plain

thread0-records-an-incref : eventFor 0 naive (todo t₀) (tenv t₀) ≡ just incref₀
thread0-records-an-incref = refl

thread1-records-an-incref : eventFor 1 naive (todo t₁) (tenv t₁) ≡ just incref₁
thread1-records-an-incref = refl

-- and a `move` or a `borrow` records nothing at all -- the row of the table
-- where the correct number of runtime operations is zero, now visible in the
-- history rather than only in the counter.
move-records-nothing : eventFor 0 naive (move c a ∷ []) env₀ ≡ nothing
move-records-nothing = refl

borrow-records-nothing : eventFor 0 naive (borrow c a ∷ []) env₀ ≡ nothing
borrow-records-nothing = refl

------------------------------------------------------------------------
-- Two concurrent steps.

m₀ m₁ m₂ : Machine
m₀ = m
m₁ = machine (heap m₀)
             (updateObj (objects m₀) counted-obj
                        (λ y → record y { count = bumpUp (count y) }))
             (occupy (sites m₀) (local 0 c) counted-obj)
m₂ = machine (heap m₁)
             (updateObj (objects m₁) counted-obj
                        (λ y → record y { count = bumpUp (count y) }))
             (occupy (sites m₁) (local 1 c) counted-obj)

t₀′ t₁′ : Thread
t₀′ = mkThread 0 bb0 [] (bindVar env₀ c (bind counted-obj owned))
t₁′ = mkThread 1 bb0 [] (bindVar env₁ c (bind counted-obj owned))

c₀ c₁ c₂ : CMachine
c₀ = cmachine thePool         m₀ []
c₁ = cmachine (t₀′ ∷ t₁ ∷ []) m₁ (incref₀ ∷ [])
c₂ = cmachine (t₀′ ∷ t₁′ ∷ []) m₂ (incref₁ ∷ incref₀ ∷ [])

-- ⭐ The first derivation of the concurrent step relation. The pool advances,
-- the SHARED machine changes, and the history gains the event the instruction
-- actually performs.
thread0-runs : prog ⊢[ naive ] c₀ ⇒ c₁
thread0-runs = sched-step picks-first (by-instr (step-dup refl refl refl))

thread1-runs : prog ⊢[ naive ] c₁ ⇒ c₂
thread1-runs = sched-step (later refl (here refl))
                          (by-instr (step-dup refl refl refl))

both-run : prog ⊢[ naive ] c₀ ⇒* c₂
both-run = cmore thread0-runs (cmore thread1-runs cdone)

-- Two increfs happened, and the counter says so. This is the interleaving in
-- which nothing went wrong -- the scheduler serialised them.
counter-after-both : countOf m₂ counted-obj ≡ just (counted 4)
counter-after-both = refl

ghost-after-both : ghostRC m₂ counted-obj ≡ 4
ghost-after-both = refl

------------------------------------------------------------------------
-- A race.
--
-- Two threads, a plain read-modify-write each, on the same refcount word --
-- and both events came out of `eventFor`, so this is a race between operations
-- the program performs rather than between two records someone wrote down.

-- ⭐ And this is the history the run above produced, not one written to suit.
raceHistory : List Event
raceHistory = history c₂

history-is-the-run's : raceHistory ≡ incref₁ ∷ incref₀ ∷ []
history-is-the-run's = refl

-- ⭐ `Conflict` inhabited. Every conjunct is discharged with a real witness:
-- the threads differ, the footprints are the same allocation AND the same
-- generation, the overlap names byte 0, one side writes, and both are plain.
-- An earlier draft of this record had two of those conjuncts as `⊤`; that draft
-- would have made this term typecheck for any two events at all, and nothing
-- would have detected it, because nothing ever built one.
the-conflict : Conflict incref₀ incref₁
the-conflict = conflict
  (λ ())
  (rmw , plain) (rmw , plain) refl refl
  (rcFootprint counted-obj) (rcFootprint counted-obj) refl refl
  refl refl
  (overlap-at 0 (z≤n , s≤s z≤n) (z≤n , s≤s z≤n))
  (inj₁ refl)
  (refl , refl)

-- Nothing orders them. Proving this needs the shape of `Earlier` in a
-- two-event history, because `hb-trans` would otherwise let a chain through an
-- arbitrary middle event.
private
  earlier-shape : ∀ {e f} → Earlier raceHistory e f → (e ≡ incref₀) × (f ≡ incref₁)
  earlier-shape adjacent      = refl , refl
  earlier-shape (skip (skip ()))

  -- Every base rule of happens-before needs an `Earlier`, and in this history
  -- that pins the pair. Each rule then fails its own side condition: the two
  -- events are on different threads (so not program order), neither is a spawn
  -- and neither is a join. `hb-trans` recurses into its left premise, which is
  -- structurally smaller, so the argument terminates.
  no-order : ∀ {e f} → HappensBefore raceHistory e f → ⊥
  no-order (program-order po ea) with earlier-shape ea
  ... | refl , refl with po
  ...   | ()
  no-order (spawn-edge k _ ea) with earlier-shape ea
  ... | refl , refl with k
  ...   | ()
  no-order (join-edge _ k ea) with earlier-shape ea
  ... | refl , refl with k
  ...   | ()
  no-order (hb-trans p _) = no-order p

-- ⭐ `Race` inhabited.
the-race : Race raceHistory incref₀ incref₁
the-race = race the-conflict no-order no-order

------------------------------------------------------------------------
-- Three things that are NOT races.
--
-- Without these the definitions above are consistent with `Conflict` holding of
-- every pair, and "we can derive a race" would be worth nothing.

-- 1. Atomics. Same threads, same word, same read-modify-write -- and no
-- conflict, because `both-plain` cannot be met. This is how sharing is done
-- legally, and it is the property the compiler is buying when it emits an
-- atomic refcount update.
atomicIncref₀ atomicIncref₁ : Event
atomicIncref₀ = event 0 (access rmw atomic) (just (rcFootprint counted-obj))
atomicIncref₁ = event 1 (access rmw atomic) (just (rcFootprint counted-obj))

atomics-do-not-conflict : ¬ Conflict atomicIncref₀ atomicIncref₁
atomics-do-not-conflict
  (conflict _ a₁ _ ia₁ _ _ _ _ _ _ _ _ _ (bp₁ , _)) with ia₁
... | refl with bp₁
...   | ()

-- 2. Two reads. Same threads, same word, both plain -- and no conflict, because
-- neither side writes.
read₀ read₁ : Event
read₀ = event 0 (access reads plain) (just (rcFootprint counted-obj))
read₁ = event 1 (access reads plain) (just (rcFootprint counted-obj))

reads-do-not-conflict : ¬ Conflict read₀ read₁
reads-do-not-conflict
  (conflict _ a₁ a₂ ia₁ ia₂ _ _ _ _ _ _ _ ow _) with ia₁ | ia₂
... | refl | refl with ow
...   | inj₁ ()
...   | inj₂ ()

-- 3. One thread with itself. A thread cannot race with itself however it
-- accesses the word, and this is the conjunct an unconstrained history would
-- have made satisfiable inside a single-threaded program -- which is why
-- `sched-step` now requires the event to be attributed to the thread that took
-- the step.
sameThreadWrite : Event
sameThreadWrite = event 0 (access writes plain) (just (rcFootprint counted-obj))

one-thread-does-not-race-itself : ¬ Conflict incref₀ sameThreadWrite
one-thread-does-not-race-itself c = different-threads c refl

------------------------------------------------------------------------
-- Happens-before, inhabited.
--
-- The census found `HappensBefore` used only under a negation, and `¬ P` is
-- free when `P` is empty. This is the positive direction: two accesses by the
-- SAME thread, adjacent in the history, are ordered by program order.

seqHistory : List Event
seqHistory = sameThreadWrite ∷ read₀ ∷ []

-- `Earlier` inhabited.
read-comes-first : Earlier seqHistory read₀ sameThreadWrite
read-comes-first = adjacent

-- ⭐ `HappensBefore` inhabited.
program-orders-them : HappensBefore seqHistory read₀ sameThreadWrite
program-orders-them = program-order refl read-comes-first

-- and being ordered is what stops them being a race -- via the ORDER, not via
-- the conflict, so the negative result uses the positive one.
ordered-is-not-a-race : ¬ Race seqHistory read₀ sameThreadWrite
ordered-is-not-a-race r = unordered₁ r program-orders-them

------------------------------------------------------------------------
-- What this does NOT establish, so the absence is not read as a result.
--
-- 1. The EVENT is still chosen by the derivation. `sched-step` now requires
--    `thread e ≡ tid t`, which is enough to stop a single-threaded program
--    fabricating a race, but nothing ties the event's kind or footprint to the
--    instruction that was executed. A `borrow` can record a write. Closing that
--    means giving each instruction rule its event, and it is the next thing this
--    layer needs.
--
-- 2. `siteOf x = local 0 x` in Proof.Program.Step hardcodes thread 0. The
--    sequential step relation therefore cannot name a second thread's owner
--    sites, so no instruction that touches the site map can be run on `t₁`.
--    That is why the steps above are `borrow`. `OwnerSite.local` is
--    thread-indexed precisely so this would be expressible, and the program
--    layer does not yet use the index.
--
-- 3. There is still no race-FREEDOM theorem. `RaceFree` is defined and nothing
--    proves it of any program; that needs the permission algebra, which is
--    deliberately absent. What this module changes is that `Race` is now known
--    to be inhabited, so `RaceFree` is a real obligation rather than a
--    vacuously satisfiable one.

------------------------------------------------------------------------
-- ⭐ Two policies that DO discharge the obligation, on the same program.
--
-- The race above is what the naive emission costs. These are the two ways out,
-- and the model says both work: make the update atomic, or do not make it at
-- all. The second is the right answer for an immortal, whose counter cannot
-- change -- so the atomics a conservative implementation emits on `{0,1,2}` are
-- not merely believed to be unnecessary.

atomicPolicy elidingPolicy : Policy
atomicPolicy  _ = just atomic
elidingPolicy _ = nothing

atomic-follows : FollowsTheChecker m atomicPolicy
atomic-follows o s u _ ()

eliding-follows : FollowsTheChecker m elidingPolicy
eliding-follows o s u _ ()

-- and the naive one does NOT, on this machine: `counted-obj` is shared.
counted-obj-is-shared : Σ OwnerSite λ s → Σ OwnerSite λ u → sharedPair m counted-obj ≡ just (s , u)
counted-obj-is-shared = local 0 1 , local 1 1 , refl

naive-does-not-follow : ¬ FollowsTheChecker m naive
naive-does-not-follow follows =
  follows counted-obj (local 0 1) (local 1 1) refl refl

------------------------------------------------------------------------
-- ⭐ Under the atomic policy, the two increfs no longer conflict.
--
-- The same program, the same schedule, the same two threads -- and the pair the
-- naive policy made a race is now not even a conflict.

atomicIncref₀′ atomicIncref₁′ : Event
atomicIncref₀′ = event 0 (access rmw atomic) (just (rcFootprint counted-obj))
atomicIncref₁′ = event 1 (access rmw atomic) (just (rcFootprint counted-obj))

atomic-emits-atomically :
  eventFor 0 atomicPolicy (todo t₀) (tenv t₀) ≡ just atomicIncref₀′
atomic-emits-atomically = refl

-- and eliding emits nothing at all.
eliding-emits-nothing : eventFor 0 elidingPolicy (todo t₀) (tenv t₀) ≡ nothing
eliding-emits-nothing = refl

emitted₀ : Emitted atomicPolicy atomicIncref₀′
emitted₀ = rc 0 counted-obj atomic refl refl

emitted₁ : Emitted atomicPolicy atomicIncref₁′
emitted₁ = rc 1 counted-obj atomic refl refl

-- ⭐ No race, on any history, under a conforming policy.
no-race-under-the-atomic-policy :
  PayloadSeparated → AllShared m →
  ∀ hs → ¬ Race hs atomicIncref₀′ atomicIncref₁′
no-race-under-the-atomic-policy sep shared hs =
  history-is-race-free m atomicPolicy atomic-follows sep shared emitted₀ emitted₁ hs

------------------------------------------------------------------------
-- ⭐ And two threads writing DIFFERENT fields never conflict, whatever the
-- policy -- which is the statement a permission algebra would have been built
-- to make.

writeField₀ writeField₁ : Event
writeField₀ = event 0 (access writes plain) (just (fieldFootprint counted-obj 0))
writeField₁ = event 1 (access writes plain) (just (fieldFootprint counted-obj 1))

different-fields-are-fine : ¬ Conflict writeField₀ writeField₁
different-fields-are-fine =
  different-fields-do-not-conflict counted-obj 0 1 (λ ()) writeField₀ writeField₁ refl refl

-- and a field write never collides with a refcount update.
field-vs-refcount : ¬ Conflict incref₀ writeField₁
field-vs-refcount =
  fields-never-touch-the-refcount counted-obj 1 incref₀ writeField₁ refl refl
