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
open import Data.Nat using (ℕ; zero; suc; _≟_; _<_; _≤_; _+_; _*_; s≤s; z≤n)
open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Data.Sum using (_⊎_; inj₁; inj₂)
open import Relation.Binary.PropositionalEquality using (_≡_; _≢_; refl; sym; trans; cong; subst)
open import Relation.Nullary using (Dec; yes; no; ¬_)

open import Proof.Prelude using (ByteRange; range; start; len; Overlaps; InRange;
  byte; in-first; in-second)
open import Data.Nat.Properties using (<-cmp)
open import Relation.Binary.Definitions using (Tri; tri<; tri≈; tri>)

open import Proof.Memory.Heap using (AllocId; Generation)
open import Proof.RC.Object using (ObjId; objAllocation; objGeneration)
open import Proof.Object.Word using (WordBytes)
open import Proof.Object.Layout using (HeaderWords)

FieldId : Set
FieldId = ℕ

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

------------------------------------------------------------------------
-- ⭐ Footprints, and what replaces a permission algebra.
--
-- Every access this IR performs is ONE WORD of ONE OBJECT'S OWN ALLOCATION.
-- That is the one-lane layout's doing: the refcount is word 0, the header runs
-- to word `HeaderWords`, and field `k` is word `HeaderWords + k` -- there is no
-- second lane and no view, so there is nothing else to point at.
--
-- A fractional-permission PCM exists to answer "may this thread touch these
-- bytes" when the bytes can be carved up arbitrarily. Here they cannot: two
-- accesses overlap exactly when they name the same word of the same object,
-- and `different-words-do-not-conflict` below is that fact. The algebra a
-- general model would need is replaced by one arithmetic lemma.

wordFootprint : ObjId → ℕ → AllocId × Generation × ByteRange
wordFootprint o i = objAllocation o , objGeneration o , range (i * WordBytes) WordBytes

-- The refcount is word 0. Defined HERE rather than in Proof.Lython.Invalid,
-- where it started, because the scheduler derives an instruction's event and
-- needs the same footprint -- two definitions of one address are two things
-- that can drift, and a race predicate comparing different footprints would
-- clear a program by looking at the wrong bytes.
rcFootprint : ObjId → AllocId × Generation × ByteRange
rcFootprint o = wordFootprint o 0

-- Field `k` is word `HeaderWords + k`. Never word 0, because `HeaderWords` is
-- positive -- so payload traffic can never be mistaken for refcount traffic,
-- which is what makes the refcount obligation separable from the rest.
fieldFootprint : ObjId → FieldId → AllocId × Generation × ByteRange
fieldFootprint o k = wordFootprint o (HeaderWords + k)

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

------------------------------------------------------------------------
-- ⭐ Word-aligned blocks of equal width are disjoint unless they are the same
-- block.
--
-- This is the whole of the "permission algebra" this language needs. In a model
-- where accesses can carve bytes arbitrarily one needs a partial commutative
-- monoid to say which splits are compatible; here every access is one whole
-- word at a word-aligned offset, so compatibility is `i ≢ j` and the proof is
-- arithmetic.

private
  -- `i < j` puts the whole of block `i` below the start of block `j`.
  below : ∀ (w i j : ℕ) → suc i ≤ j → i * w + w ≤ j * w
  below w i j lt = ≤-trans (≤-reflexive step) (*-monoˡ-≤ w lt)
    where
      open import Data.Nat.Properties using (≤-trans; ≤-reflexive; *-monoˡ-≤; +-comm)
      step : i * w + w ≡ suc i * w
      step = +-comm (i * w) w

  one-sided : ∀ (w i j : ℕ) → suc i ≤ j →
              ¬ Overlaps (range (i * w) w) (range (j * w) w)
  -- The witness byte is below block j's start and inside it, so it is below
  -- itself.
  -- b < i*w + w ≤ j*w ≤ b, so b < b.
  one-sided w i j lt ov =
    <-irrefl refl (<-≤-trans (<-≤-trans (proj₂ (in-first ov)) le)
                             (proj₁ (in-second ov)))
    where
      open import Data.Nat.Properties using (<-irrefl; <-≤-trans)
      le : i * w + w ≤ j * w
      le = below w i j lt

aligned-blocks-disjoint :
  ∀ (w i j : ℕ) → i ≢ j → ¬ Overlaps (range (i * w) w) (range (j * w) w)
aligned-blocks-disjoint w i j ne ov with <-cmp i j
... | tri< lt _ _ = one-sided w i j lt ov
... | tri≈ _ eq _ = ne eq
... | tri> _ _ gt = one-sided w j i gt (overlaps-sym ov)
  where
    overlaps-sym : ∀ {r s} → Overlaps r s → Overlaps s r
    overlaps-sym o = record { byte = byte o
                            ; in-first = in-second o
                            ; in-second = in-first o }

-- ⭐ Two accesses to different words of one object cannot conflict.
--
-- `Conflict` demands a shared byte and there is none, so no `Race` can be
-- assembled from them however they are ordered. This is what says two threads
-- writing different fields of one object are fine -- the statement a
-- permission algebra would have been built to make.
different-words-do-not-conflict :
  ∀ (o : ObjId) (i j : ℕ) → i ≢ j → ∀ (e₁ e₂ : Event) →
  footprint e₁ ≡ just (wordFootprint o i) →
  footprint e₂ ≡ just (wordFootprint o j) →
  ¬ Conflict e₁ e₂
different-words-do-not-conflict o i j ne e₁ e₂ f₁ f₂ c =
  aligned-blocks-disjoint WordBytes i j ne shared
  where
    just-inj : ∀ {A : Set} {u v : A} → just u ≡ just v → u ≡ v
    just-inj refl = refl
    l₁ : loc₁ c ≡ wordFootprint o i
    l₁ = just-inj (trans (sym (at₁ c)) f₁)
    l₂ : loc₂ c ≡ wordFootprint o j
    l₂ = just-inj (trans (sym (at₂ c)) f₂)
    shared : Overlaps (range (i * WordBytes) WordBytes) (range (j * WordBytes) WordBytes)
    shared = subst₂ (λ a b → Overlaps (proj₂ (proj₂ a)) (proj₂ (proj₂ b)))
                    l₁ l₂ (overlapping c)
      where open import Relation.Binary.PropositionalEquality using (subst₂)

-- ⭐ A field access never touches the refcount word. `HeaderWords` is positive,
-- so `HeaderWords + k` is never 0 -- which is why the refcount obligation can
-- be discharged on its own, independently of whatever the payload is doing.
fields-never-touch-the-refcount :
  ∀ (o : ObjId) (k : FieldId) (e₁ e₂ : Event) →
  footprint e₁ ≡ just (rcFootprint o) →
  footprint e₂ ≡ just (fieldFootprint o k) →
  ¬ Conflict e₁ e₂
fields-never-touch-the-refcount o k = different-words-do-not-conflict o 0 (HeaderWords + k) λ ()

-- and two DIFFERENT fields never touch each other.
different-fields-do-not-conflict :
  ∀ (o : ObjId) (k l : FieldId) → k ≢ l → ∀ (e₁ e₂ : Event) →
  footprint e₁ ≡ just (fieldFootprint o k) →
  footprint e₂ ≡ just (fieldFootprint o l) →
  ¬ Conflict e₁ e₂
different-fields-do-not-conflict o k l ne =
  different-words-do-not-conflict o (HeaderWords + k) (HeaderWords + l)
    (λ eq → ne (+-cancelˡ-≡ HeaderWords k l eq))
  where open import Data.Nat.Properties using (+-cancelˡ-≡)

-- ⭐ And two accesses to different OBJECTS cannot conflict, whatever the words.
--
-- `Conflict` requires the same allocation AND the same generation, and an
-- `ObjId` is exactly that pair -- so agreeing on both IS being the same object.
-- That is the payoff of identity being provenance rather than an address: with
-- addresses, two objects could share one and this theorem would be false.
different-objects-do-not-conflict :
  ∀ (o p : ObjId) (i j : ℕ) → o ≢ p → ∀ (e₁ e₂ : Event) →
  footprint e₁ ≡ just (wordFootprint o i) →
  footprint e₂ ≡ just (wordFootprint p j) →
  ¬ Conflict e₁ e₂
different-objects-do-not-conflict o p i j ne e₁ e₂ f₁ f₂ c = ne same
  where
    just-inj : ∀ {A : Set} {u v : A} → just u ≡ just v → u ≡ v
    just-inj refl = refl
    l₁ : loc₁ c ≡ wordFootprint o i
    l₁ = just-inj (trans (sym (at₁ c)) f₁)
    l₂ : loc₂ c ≡ wordFootprint p j
    l₂ = just-inj (trans (sym (at₂ c)) f₂)
    open import Relation.Binary.PropositionalEquality using (subst₂; cong₂)
    sameAlloc : objAllocation o ≡ objAllocation p
    sameAlloc = subst₂ (λ a b → proj₁ a ≡ proj₁ b) l₁ l₂ (same-allocation c)
    sameGen : objGeneration o ≡ objGeneration p
    sameGen = subst₂ (λ a b → proj₁ (proj₂ a) ≡ proj₁ (proj₂ b)) l₁ l₂ (same-generation c)
    same : o ≡ p
    same = cong₂ (λ a g → record { objAllocation = a ; objGeneration = g })
                 sameAlloc sameGen
