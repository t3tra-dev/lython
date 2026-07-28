{-# OPTIONS --safe #-}

-- Events, threads, and what a data race is.
--
-- Modelled now although no defect this compiler has shipped is concurrent,
-- because the shape of the answer constrains the sequential layer: if threads
-- are added later by bolting a pool onto a step relation that assumed one
-- thread, every ownership rule has to be revisited. Defining the event
-- vocabulary first means the sequential rules are already the one-thread case
-- of the concurrent ones.

module Proof.Concurrent.Event where

open import Data.Bool using (Bool; true; false; _∧_; not)
open import Data.List using (List; []; _∷_)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc; _≟_; _<_)
open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Data.Sum using (_⊎_; inj₁; inj₂)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; sym)
open import Relation.Nullary using (Dec; yes; no; ¬_)

open import Proof.Prelude using (ByteRange; range; start; len; Overlaps; InRange)
open import Proof.Memory.Heap using (AllocId; Generation)

ThreadId : Set
ThreadId = ℕ

-- Whether an access is atomic. A race needs BOTH sides non-atomic: two atomic
-- accesses to the same location are not a race, and mixing one of each is a
-- different fault from mixing two plain ones -- which is why the fault
-- enumeration has to separate them.
data Atomicity : Set where
  plain  : Atomicity
  atomic : Atomicity

data AccessMode : Set where
  reads  : AccessMode
  writes : AccessMode
  -- A read-modify-write is ONE event, not a read followed by a write. The
  -- distinction has no content in a sequential model and is the whole point in
  -- a concurrent one: no schedule may interleave inside it.
  rmw    : AccessMode

data EventKind : Set where
  access   : AccessMode → Atomicity → EventKind
  allocate : EventKind
  free     : EventKind
  spawn    : ThreadId → EventKind
  join     : ThreadId → EventKind
  acquire  : ℕ → EventKind
  release  : ℕ → EventKind
  fence    : EventKind

record Event : Set where
  constructor event
  field
    thread     : ThreadId
    kind       : EventKind
    -- `nothing` for events that touch no memory: spawn, join, fence.
    footprint  : Maybe (AllocId × Generation × ByteRange)

open Event public

------------------------------------------------------------------------
-- Conflict.
--
-- The definition is the note's, spelled out so that each conjunct can be
-- pointed at. Dropping any one of them changes which programs are legal:
--
--   different threads          -- a thread cannot race with itself
--   same allocation+generation -- reusing an address is not the same location
--   overlapping bytes          -- with a WITNESS, so the shared byte is nameable
--   at least one write         -- two reads do not conflict
--   both non-atomic            -- atomics are how you legally share

isWrite : AccessMode → Bool
isWrite reads  = false
isWrite writes = true
isWrite rmw    = true

-- The access mode and atomicity of an event, when it has them. Events that
-- touch no memory answer `nothing`, and that is what stops a `spawn` from being
-- one half of a race.
modeOf : EventKind → Maybe (AccessMode × Atomicity)
modeOf (access m a) = just (m , a)
modeOf allocate     = nothing
modeOf free         = nothing
modeOf (spawn _)    = nothing
modeOf (join _)     = nothing
modeOf (acquire _)  = nothing
modeOf (release _)  = nothing
modeOf fence        = nothing

record Conflict (e₁ e₂ : Event) : Set where
  constructor conflict
  field
    different-threads : ¬ (thread e₁ ≡ thread e₂)

    acc₁ acc₂ : AccessMode × Atomicity
    is-access₁ : modeOf (kind e₁) ≡ just acc₁
    is-access₂ : modeOf (kind e₂) ≡ just acc₂

    loc₁ loc₂ : AllocId × Generation × ByteRange
    at₁ : footprint e₁ ≡ just loc₁
    at₂ : footprint e₂ ≡ just loc₂

    same-allocation : proj₁ loc₁ ≡ proj₁ loc₂
    same-generation : proj₁ (proj₂ loc₁) ≡ proj₁ (proj₂ loc₂)

    -- Constructive, so whatever needs to point at the shared byte can.
    overlapping : Overlaps (proj₂ (proj₂ loc₁)) (proj₂ (proj₂ loc₂))

    -- At least one side writes. Two reads do not conflict, and stating this as
    -- a disjunction rather than as a boolean keeps the WHICH available.
    one-writes : (isWrite (proj₁ acc₁) ≡ true) ⊎ (isWrite (proj₁ acc₂) ≡ true)

    -- Both sides non-atomic. Two atomics are how sharing is done legally, and a
    -- mixed pair is a DIFFERENT fault -- which is why this is a conjunction of
    -- two equations rather than one predicate over the pair.
    both-plain : (proj₂ acc₁ ≡ plain) × (proj₂ acc₂ ≡ plain)

open Conflict public

-- A conflict is symmetric, as it must be: a race is a property of a PAIR, and a
-- definition that held in one order only would let an analysis clear a program
-- by considering the events in the convenient order.
conflict-sym : ∀ {e₁ e₂} → Conflict e₁ e₂ → Conflict e₂ e₁
conflict-sym c = conflict
  (λ eq → different-threads c (sym eq))
  (acc₂ c) (acc₁ c) (is-access₂ c) (is-access₁ c)
  (loc₂ c) (loc₁ c) (at₂ c) (at₁ c)
  (sym (same-allocation c)) (sym (same-generation c))
  (overlaps-sym (overlapping c))
  (swap⊎ (one-writes c))
  (proj₂ (both-plain c) , proj₁ (both-plain c))
  where
    overlaps-sym : ∀ {r s} → Overlaps r s → Overlaps s r
    overlaps-sym o = record { byte = Overlaps.byte o
                            ; in-first = Overlaps.in-second o
                            ; in-second = Overlaps.in-first o }
    swap⊎ : ∀ {A B : Set} → A ⊎ B → B ⊎ A
    swap⊎ (inj₁ a) = inj₂ a
    swap⊎ (inj₂ b) = inj₁ b
