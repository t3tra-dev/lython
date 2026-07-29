{-# OPTIONS --safe #-}

-- ⭐ Race freedom, for whole histories, without a permission algebra.
--
-- `RaceFree` was defined and nothing proved it of anything, and the module that
-- defined it said so: "a predicate being definable is not the same as any
-- program being shown free of it". This closes that.
--
-- A fractional-permission PCM exists to answer "may this thread touch these
-- bytes" when the bytes can be carved up arbitrarily. In this IR they cannot:
-- every access is ONE WORD of one object's own allocation, because the one-lane
-- layout leaves nothing else to point at. So two accesses overlap exactly when
-- they name the same word of the same object, and the algebra collapses into
-- `Proof.Concurrent.Event.aligned-blocks-disjoint` -- one arithmetic lemma.
--
-- What is left is two obligations, and they belong to different people:
--
--   REFCOUNT traffic is the compiler's. It is emitted at every dup and drop,
--   the program never asked for it, and a plain update on a shared object is a
--   race the source could not have avoided. `FollowsTheChecker` is that
--   obligation and `sharedPair` decides it.
--
--   PAYLOAD traffic is the program's. Two threads writing one field of one
--   object is a data race in the source, exactly as in CPython without the GIL,
--   and no lowering can fix it. `PayloadSeparated` is that condition, stated as
--   a hypothesis rather than proved, because it is not the compiler's to
--   discharge.

open import Proof.Memory.Element using (ElemSig)

module Proof.Concurrent.RaceFree (Sig : ElemSig) where

open ElemSig Sig

open import Data.Bool using (Bool; true; false; if_then_else_)
open import Data.Empty using (⊥; ⊥-elim)
open import Data.List using (List; []; _∷_)
open import Data.Maybe using (Maybe; just; nothing; maybe′)
open import Data.Nat using (ℕ; zero; suc; _+_; _≟_)
open import Data.Product using (_×_; _,_; Σ; proj₁; proj₂)
open import Relation.Binary.PropositionalEquality
  using (_≡_; _≢_; refl; sym; trans; cong; subst)
open import Relation.Nullary using (¬_; yes; no)

open import Proof.Object.Layout using (HeaderWords)
open import Proof.RC.Object using (ObjId; obj; _≟-obj_; RuntimeCount; counted; immortal)
open import Proof.RC.OwnerSite using (OwnerSite)
open import Proof.RC.Machine Sig
open import Proof.Concurrent.Event
open import Proof.Concurrent.Machine Sig
open import Proof.Program.Syntax using (Var; Instr; alloc; init; move; dup; drop;
  borrow; getField; setField)
open import Proof.Program.Env
open import Proof.Lython.Invalid Sig
open import Proof.Lython.Detect Sig using (sharedPair)

private
  just-inj : ∀ {A : Set} {u v : A} → just u ≡ just v → u ≡ v
  just-inj refl = refl

  atomic≢plain : atomic ≢ plain
  atomic≢plain ()

------------------------------------------------------------------------
-- The compiler's obligation.
--
-- Stated over `sharedPair` rather than over `needsAtomic?`, and the difference
-- is the immortals. `needsAtomic?` is gated on the counter being a NUMBER,
-- which is right for `RefcountRace` -- an immortal's counter never moves, so
-- updating it is not a race in the Lython sense. `Conflict` does not know that:
-- two plain read-modify-writes of one word from two threads conflict whatever
-- value is written. So the EMISSION obligation covers every shared object, and
-- the way to discharge it for an immortal is to emit nothing -- which is what
-- the runtime does anyway, and what `Policy` returning `nothing` means.

FollowsTheChecker : Machine → Policy → Set
FollowsTheChecker m pol =
  ∀ o s u → sharedPair m o ≡ just (s , u) → pol o ≢ just plain

------------------------------------------------------------------------
-- What an event from this IR can be.

data Emitted (pol : Policy) : Event → Set where
  -- No access mode at all: `allocate`, and the scheduler's spawn / join.
  inert : ∀ {e} → modeOf (kind e) ≡ nothing → Emitted pol e
  -- A refcount update the policy asked for.
  rc    : ∀ {e} t o a → pol o ≡ just a →
          e ≡ event t (access rmw a) (just (rcFootprint o)) → Emitted pol e
  -- A payload access.
  fld   : ∀ {e} t md o k →
          e ≡ event t (access md plain) (just (fieldFootprint o k)) → Emitted pol e

private
  no-just : ∀ {A : Set} {w : A} → nothing ≡ just w → ⊥
  no-just ()

  rc-shape : ∀ t (pol : Policy) (es : Env) v e →
             rcEventFor t pol es v ≡ just e → Emitted pol e
  rc-shape t pol es v e eq = go (lookupVar es v) refl
    where
      go : (r : Maybe Binding) → lookupVar es v ≡ r → Emitted pol e
      go nothing lb = ⊥-elim (no-just (trans (sym miss) eq))
        where miss : rcEventFor t pol es v ≡ nothing
              miss rewrite lb = refl
      go (just b) lb = own (isOwned (mode b)) refl
        where
          unowned : isOwned (mode b) ≡ false → rcEventFor t pol es v ≡ nothing
          unowned io rewrite lb | io = refl

          elided : isOwned (mode b) ≡ true → pol (entity b) ≡ nothing →
                   rcEventFor t pol es v ≡ nothing
          elided io pa rewrite lb | io | pa = refl

          issued : ∀ a → isOwned (mode b) ≡ true → pol (entity b) ≡ just a →
                   rcEventFor t pol es v
                     ≡ just (event t (access rmw a) (just (rcFootprint (entity b))))
          issued a io pa rewrite lb | io | pa = refl

          own : (bb : Bool) → isOwned (mode b) ≡ bb → Emitted pol e
          own false io = ⊥-elim (no-just (trans (sym (unowned io)) eq))
          own true  io = pick (pol (entity b)) refl
            where
              pick : (r : Maybe Atomicity) → pol (entity b) ≡ r → Emitted pol e
              pick nothing  pa = ⊥-elim (no-just (trans (sym (elided io pa)) eq))
              pick (just a) pa =
                rc t (entity b) a pa (sym (just-inj (trans (sym (issued a io pa)) eq)))

  fld-shape : ∀ t md (pol : Policy) (es : Env) v k e →
              fieldEventFor t md es v k ≡ just e → Emitted pol e
  fld-shape t md pol es v k e eq = go (entityOf es v) refl
    where
      go : (r : Maybe ObjId) → entityOf es v ≡ r → Emitted pol e
      go nothing ev = ⊥-elim (no-just (trans (sym miss) eq))
        where miss : fieldEventFor t md es v k ≡ nothing
              miss rewrite ev = refl
      go (just o) ev = fld t md o k (sym (just-inj (trans (sym hit) eq)))
        where hit : fieldEventFor t md es v k
                      ≡ just (event t (access md plain) (just (fieldFootprint o k)))
              hit rewrite ev = refl

-- ⭐ Every event an instruction produces is one of the three. `move` and
-- `borrow` produce none at all, which is "a move is free" read off the history.
instrEvent-shape :
  ∀ t (pol : Policy) (i : Instr) (es : Env) e →
  instrEvent t pol i es ≡ just e → Emitted pol e
instrEvent-shape t pol (alloc x c) es e eq with just-inj eq
... | refl = inert refl
instrEvent-shape t pol (init x)   es e eq with just-inj eq
... | refl = inert refl
instrEvent-shape t pol (dup dst src)      es e eq = rc-shape t pol es src e eq
instrEvent-shape t pol (drop v)           es e eq = rc-shape t pol es v e eq
instrEvent-shape t pol (move _ _)         es e ()
instrEvent-shape t pol (borrow _ _)       es e ()
instrEvent-shape t pol (getField _ src k) es e eq = fld-shape t reads  pol es src k e eq
instrEvent-shape t pol (setField dst k _) es e eq = fld-shape t writes pol es dst k e eq

eventFor-shape :
  ∀ t (pol : Policy) (is : List Instr) (es : Env) e →
  eventFor t pol is es ≡ just e → Emitted pol e
eventFor-shape t pol []      es e ()
eventFor-shape t pol (i ∷ _) es e eq = instrEvent-shape t pol i es e eq

------------------------------------------------------------------------
-- Provenance: every event in a reachable history is one of the three.

data FromProgram (pol : Policy) : List Event → Set where
  fp-nil  : FromProgram pol []
  fp-cons : ∀ {e hs} → Emitted pol e → FromProgram pol hs → FromProgram pol (e ∷ hs)

private
  record-preserves :
    ∀ (pol : Policy) (me : Maybe Event) (hs : List Event) →
    (∀ e → me ≡ just e → Emitted pol e) →
    FromProgram pol hs → FromProgram pol (record? me hs)
  record-preserves pol nothing  hs _  fp = fp
  record-preserves pol (just e) hs sh fp = fp-cons (sh e refl) fp

-- Spawn and join record an event the rule does not derive, so their inertness
-- is a premise. Both really are inert -- `modeOf (spawn t)` and `modeOf
-- (join t)` are `nothing` -- and the rules carry `kind e ≡ spawn _`, so a
-- caller can discharge it from the derivation it already has.
step-keeps-provenance :
  ∀ {f pol c d} → f ⊢[ pol ] c ⇒ d →
  (∀ e → modeOf (kind e) ≡ nothing) →
  FromProgram pol (history c) → FromProgram pol (history d)
step-keeps-provenance {pol = pol} (sched-step {t = t} sch st) _ fp =
  record-preserves pol (eventFor (tid t) pol (todo t) (tenv t)) _
    (λ e eq → eventFor-shape (tid t) pol (todo t) (tenv t) e eq) fp
step-keeps-provenance (sched-spawn {e = e} _ _ _) inertness fp =
  fp-cons (inert (inertness e)) fp
step-keeps-provenance (sched-join {e = e} _ _ _) inertness fp =
  fp-cons (inert (inertness e)) fp

run-keeps-provenance :
  ∀ {f pol c d} → f ⊢[ pol ] c ⇒* d →
  (∀ e → modeOf (kind e) ≡ nothing) →
  FromProgram pol (history c) → FromProgram pol (history d)
run-keeps-provenance cdone          _  fp = fp
run-keeps-provenance (cmore s rest) ie fp =
  run-keeps-provenance rest ie (step-keeps-provenance s ie fp)

------------------------------------------------------------------------
-- The program's obligation.

PayloadSeparated : Set
PayloadSeparated =
  ∀ t₁ t₂ md₁ md₂ o k → t₁ ≢ t₂ →
  ¬ Conflict (event t₁ (access md₁ plain) (just (fieldFootprint o k)))
             (event t₂ (access md₂ plain) (just (fieldFootprint o k)))

-- The objects under refcount traffic really are shared. A fact about the
-- machine rather than about the program, and the reason it is a premise is that
-- `Emitted` records what the policy chose without recording where the object
-- lives. `Proof.Lython.Decide` settles the other direction: a state where
-- nothing is shared, and plain updates are safe everywhere.
AllShared : Machine → Set
AllShared m = ∀ o → Σ OwnerSite λ s → Σ OwnerSite λ u → sharedPair m o ≡ just (s , u)

------------------------------------------------------------------------
-- ⭐ No two emitted events conflict.

emitted-do-not-conflict :
  ∀ (m : Machine) (pol : Policy) → FollowsTheChecker m pol → PayloadSeparated →
  AllShared m →
  ∀ {e₁ e₂} → Emitted pol e₁ → Emitted pol e₂ → ¬ Conflict e₁ e₂

-- An inert event has no access mode, so `Conflict` cannot even start.
emitted-do-not-conflict m pol _ _ _ (inert i) _ c
  with trans (sym i) (is-access₁ c)
... | ()
emitted-do-not-conflict m pol _ _ _ _ (inert i) c
  with trans (sym i) (is-access₂ c)
... | ()

-- Two refcount updates. The same word, so the only defence is atomicity -- and
-- the policy supplies it wherever the object is shared.
emitted-do-not-conflict m pol follows _ shared
                        (rc t₁ o₁ a₁ p₁ refl) (rc t₂ o₂ a₂ p₂ refl) c =
  atomic≢plain (trans (sym a₁-is-atomic) a₁-is-plain)
  where
    rep : Σ OwnerSite λ s → Σ OwnerSite λ u → sharedPair m o₁ ≡ just (s , u)
    rep = shared o₁

    not-plain : pol o₁ ≢ just plain
    not-plain = follows o₁ (proj₁ rep) (proj₁ (proj₂ rep)) (proj₂ (proj₂ rep))

    a₁-is-plain : a₁ ≡ plain
    a₁-is-plain = trans (cong proj₂ (just-inj (is-access₁ c))) (proj₁ (both-plain c))

    -- `a₁` is a pattern variable of the parent clause, so it cannot be
    -- `with`ed here; the case split goes through a helper instead.
    decide : ∀ (a : Atomicity) → pol o₁ ≡ just a → a ≡ atomic
    decide atomic _ = refl
    decide plain  q = ⊥-elim (not-plain q)

    a₁-is-atomic : a₁ ≡ atomic
    a₁-is-atomic = decide a₁ p₁

-- A refcount update and a payload access: word 0 against word `HeaderWords + k`,
-- and `HeaderWords` is positive, so they cannot overlap at all.
emitted-do-not-conflict m _ _ _ _ (rc t₁ o₁ a₁ _ refl) (fld t₂ md o₂ k refl) c
  with o₁ ≟-obj o₂
... | yes refl = different-words-do-not-conflict o₁ 0 (HeaderWords + k) (λ ()) _ _
                   refl refl c
... | no  ne   = different-objects-do-not-conflict o₁ o₂ 0 (HeaderWords + k) ne _ _
                   refl refl c

emitted-do-not-conflict m _ _ _ _ (fld t₁ md o₁ k refl) (rc t₂ o₂ a₂ _ refl) c
  with o₁ ≟-obj o₂
... | yes refl = different-words-do-not-conflict o₁ (HeaderWords + k) 0 (λ ()) _ _
                   refl refl c
... | no  ne   = different-objects-do-not-conflict o₁ o₂ (HeaderWords + k) 0 ne _ _
                   refl refl c

-- Two payload accesses. Different objects or different fields cannot overlap;
-- the same field of the same object from two threads is the program's race.
emitted-do-not-conflict m _ _ sep _ (fld t₁ md₁ o₁ k refl) (fld t₂ md₂ o₂ l refl) c
  with o₁ ≟-obj o₂
... | no ne = different-objects-do-not-conflict o₁ o₂ (HeaderWords + k)
                (HeaderWords + l) ne _ _ refl refl c
... | yes refl with k ≟ l
...   | no ne    = different-fields-do-not-conflict o₁ k l ne _ _ refl refl c
...   | yes refl = sep t₁ t₂ md₁ md₂ o₁ k (different-threads c) c

------------------------------------------------------------------------
-- ⭐ Race freedom for a whole history.
--
-- A race is a conflict that nothing orders, and no two events the program emits
-- conflict at all. Whatever happens-before does or does not order, there is
-- nothing to order.

history-is-race-free :
  ∀ (m : Machine) (pol : Policy) → FollowsTheChecker m pol → PayloadSeparated →
  AllShared m →
  ∀ {e₁ e₂} → Emitted pol e₁ → Emitted pol e₂ → ∀ hs → ¬ Race hs e₁ e₂
history-is-race-free m pol follows sep shared em₁ em₂ hs r =
  emitted-do-not-conflict m pol follows sep shared em₁ em₂ (conflicting r)

------------------------------------------------------------------------
-- ⭐ The immortal saving, which survives all of this.
--
-- `{0,1,2}` are immortal in this runtime and every thread that touches a small
-- integer shares them. The policy discharges its obligation for them by
-- emitting NOTHING -- `pol o ≡ nothing` is not `just plain` -- so neither the
-- atomic a conservative implementation would emit nor the plain update a naive
-- one would emit is necessary.

eliding-satisfies-the-obligation :
  ∀ (m : Machine) (pol : Policy) (o : ObjId) → pol o ≡ nothing →
  ∀ s u → sharedPair m o ≡ just (s , u) → pol o ≢ just plain
eliding-satisfies-the-obligation m pol o none s u _ eq with trans (sym none) eq
... | ()

-- and no refcount race is derivable for an immortal whatever the policy says,
-- which is the Lython-level statement rather than the `Conflict`-level one.
immortals-race-free :
  ∀ (m : Machine) (o : ObjId) (e : Event) →
  countOf m o ≡ just immortal → ¬ RefcountRace m o e
immortals-race-free = immortal-rc-update-is-not-a-race

------------------------------------------------------------------------
-- What is a hypothesis, and why.
--
-- `PayloadSeparated` is the SOURCE program's condition. Two threads writing one
-- field of one object race in Lython exactly as they do in CPython without the
-- GIL, and no lowering can prevent it. What a lowering CAN do is not add
-- traffic of its own, and that is `FollowsTheChecker` -- which is decided, by
-- `sharedPair`, from the site map.
--
-- That split is the result. A permission algebra would have been built to say
-- which byte-splits are compatible; here the answer is "same word of the same
-- object", the compiler's share of the obligation is decidable, and the rest is
-- not the compiler's to discharge.
