{-# OPTIONS --safe #-}

-- WFRC preservation: the theorem the gap analysis said had no referent.
--
--     "`WFRC` holds of every reachable machine" -- there is no notion of
--     "reachable": no step relation, so "every reachable machine" has no
--     referent.
--
-- There is one now, and this module closes it for the fields it can and says
-- precisely which it cannot -- because an invariant proved for four of five
-- fields and reported as preserved is worse than one nobody claimed.

open import Proof.Memory.Element using (ElemSig)

module Proof.Program.Preservation (Sig : ElemSig) where

open ElemSig Sig

open import Data.Bool using (Bool; true; false)
open import Data.Empty using (⊥; ⊥-elim)
open import Data.List using (List; []; _∷_)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc; _<_; s≤s; z≤n)
open import Data.Product using (_×_; _,_; Σ; proj₁; proj₂)
open import Relation.Binary.PropositionalEquality
  using (_≡_; refl; sym; trans; cong; subst)
open import Relation.Nullary using (¬_)

open import Proof.RC.Object
open import Proof.RC.OwnerSite
open import Proof.RC.Machine Sig
open import Proof.Program.Syntax using (Var; Instr; new; move; dup; drop; borrow)
open import Proof.Program.Env
open import Proof.Program.Step Sig

------------------------------------------------------------------------
-- 1. Steps that do not touch the machine at all.
--
-- Every terminator, and `borrow`. For these preservation is not "easy" -- it is
-- VACUOUS, and saying so is the point: the theorem for them carries no
-- information about the invariant, and a preservation proof that consisted only
-- of these would be a proof about nothing.

term-steps-keep-the-machine :
  ∀ {f s t} → f ⊢ s —→ₜ t → mach t ≡ mach s
term-steps-keep-the-machine (step-br _ _ _ _)            = refl
term-steps-keep-the-machine (step-cond-then _ _ _ _)     = refl
term-steps-keep-the-machine (step-cond-else _ _ _ _)     = refl
term-steps-keep-the-machine (step-invoke-normal _ _ _ _) = refl
term-steps-keep-the-machine (step-invoke-throw _ _ _ _)  = refl

borrow-keeps-the-machine :
  ∀ {f bid rest es m src dst t} →
  f ⊢ pstate bid (borrow dst src ∷ rest) es m —→ᵢ t → mach t ≡ m
borrow-keeps-the-machine (step-borrow _) = refl

------------------------------------------------------------------------
-- 2. `move`: the count is preserved, and the proof is the two site lemmas.
--
-- This is the row of the ownership table where the correct number of runtime
-- operations is ZERO, and it is the first place the model can say why: one site
-- is vacated and one occupied, so `logicalRC` is unchanged -- while `objects` is
-- untouched, so the runtime counter is unchanged too. Both halves, and they
-- agree.
--
-- A pass that emitted a release for a move breaks the left equation; one that
-- emitted a retain breaks the right. Neither is visible without this lemma.

move-preserves-ghost :
  ∀ (ss : SiteMap) (src dst : OwnerSite) (o : ObjId) →
  strongAt ss src ≡ just o →
  logicalRC (occupy (vacate ss src) dst o) o ≡ logicalRC ss o
move-preserves-ghost ss src dst o held =
  trans (occupy-same (vacate ss src) dst o) (sym (vacate-holder ss src o held))

-- And the runtime counter is literally the same table.
move-preserves-counter :
  ∀ {f bid rest es m src dst o t} →
  f ⊢ pstate bid (move dst src ∷ rest) es m —→ᵢ t →
  lookupVar es src ≡ just (bind o owned) →
  objects (mach t) ≡ objects m
move-preserves-counter (step-move _) _ = refl

------------------------------------------------------------------------
-- 3. `dup` = py.incref: both sides go up by one.
--
-- The runtime side is `bumpUp`, the ghost side is `occupy-same`. `counted-exact`
-- asks that they agree, and here they do -- by the same `suc`.

dup-ghost :
  ∀ (ss : SiteMap) (dst : OwnerSite) (o : ObjId) →
  logicalRC (occupy ss dst o) o ≡ suc (logicalRC ss o)
dup-ghost = occupy-same

dup-counter :
  ∀ (ts : ObjTable) (o : ObjId) (c : ObjCell) →
  lookupObj ts o ≡ just c →
  lookupObj (updateObj ts o (λ x → record x { count = bumpUp (count x) })) o
    ≡ just (record c { count = bumpUp (count c) })
dup-counter ts o c look =
  lookupObj-update-same ts o (λ x → record x { count = bumpUp (count x) }) c look
  where open import Proof.RC.Properties Sig using (lookupObj-update-same)

-- ⭐ The two agree. If the object's counter read `counted n` and its ghost count
-- was `n`, then after a dup they read `counted (suc n)` and `suc n`.
dup-keeps-counted-exact :
  ∀ (ss : SiteMap) (ts : ObjTable) (dst : OwnerSite) (o : ObjId) (c : ObjCell) (n : ℕ) →
  lookupObj ts o ≡ just c →
  count c ≡ counted n →
  n ≡ logicalRC ss o →
  (count (record c { count = bumpUp (count c) }) ≡ counted (suc n))
  × (suc n ≡ logicalRC (occupy ss dst o) o)
dup-keeps-counted-exact ss ts dst o c n look cnt exact =
  (cong bumpUp cnt) , trans (cong suc exact) (sym (dup-ghost ss dst o))

------------------------------------------------------------------------
-- 4. `drop` = py.decref: both sides go down by one.

drop-keeps-counted-exact :
  ∀ (ss : SiteMap) (src : OwnerSite) (o : ObjId) (n : ℕ) →
  strongAt ss src ≡ just o →
  suc n ≡ logicalRC ss o →
  n ≡ logicalRC (vacate ss src) o
drop-keeps-counted-exact ss src o n held exact =
  suc-inj (trans exact (vacate-holder ss src o held))
  where
    suc-inj : ∀ {a b : ℕ} → suc a ≡ suc b → a ≡ b
    suc-inj refl = refl

------------------------------------------------------------------------
-- 5. Reachability.
--
-- What composes across `—→*` is what each step preserves, so this is the
-- statement of the shape the gap analysis wanted. It is given for the HEAP,
-- which every step leaves alone, and that is a real fact rather than an
-- accident: freeing is deliberately not a step of this relation, because a step
-- that freed while a name still denoted the object would be a step into a
-- use-after-free.

reachable-keeps-the-heap : ∀ {f s t} → f ⊢ s —→* t → heap (mach t) ≡ heap (mach s)
reachable-keeps-the-heap done        = refl
reachable-keeps-the-heap (more p ps) =
  trans (reachable-keeps-the-heap ps) (one p)
  where
    one : ∀ {f s t} → f ⊢ s —→ t → heap (mach t) ≡ heap (mach s)
    one (by-instr (step-new _))     = refl
    one (by-instr (step-move _))   = refl
    one (by-instr (step-dup _))    = refl
    one (by-instr (step-drop _ _ _)) = refl
    one (by-instr (step-borrow _)) = refl
    one (by-term t)                = cong heap (term-steps-keep-the-machine t)

------------------------------------------------------------------------
-- What each field of WFRC now rests on.
--
-- Two of the obstructions recorded when this module was first written have been
-- removed, and they were removed by changing the IR rather than by weakening
-- the invariant -- which is the direction the note asks for: establish the safe
-- operation first, then make the implementation match.
--
--   live-positive       WAS blocked: `step-new` took an arbitrary cell as a
--                       parameter, so the rule could install one already at
--                       count 0 and the field was violable by a legal step.
--                       The rule now CONSTRUCTS `cell live (counted 1) bk`, so
--                       a fresh object has exactly one site and a count of one.
--
--   dead-unowned        WAS violable: `step-drop` did not require the object
--                       live, so dropping a dead one moved it back to
--                       `finalizing` -- a resurrection. The rule now takes
--                       `life c ≡ live` as a premise.
--
--   counted-exact       the four lemmas above, one per rule that can move
--                       either count.
--
--   no-stale-owner      immediate from `reachable-keeps-the-heap`: no step
--                       touches the heap, so no generation moves under a site.
--
--   owned-storage-live  the same.
--
-- What remains before a single `step-preserves-WFRC` can be exported is
-- assembly, not discovery: each field is ∀-quantified over ALL objects, so every
-- rule needs the untouched-object case as well as the touched one, and that is
-- `occupy-other` and `vacate-other` applied under a decision on `sameObj`.
-- Mechanical, sizeable, and not yet written -- so it is not claimed.
--
-- The distinction being kept here is between "the obstruction is gone" and "the
-- theorem is proved". Both of the first two lines above were obstructions and
-- are now not; none of the five lines is a proof of preservation.
