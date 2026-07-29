{-# OPTIONS --safe #-}

-- Ownership tracking across a program, and the theorem this whole layer was
-- built for.
--
-- The gap analysis found that the shipped SIGSEGV's root --
--
--     one allocation, TWO SSA NAMES, and the ownership machinery treats that as
--     two entities
--
-- had no sentence in an allocation-level model. `br-creates-an-alias` below is
-- that sentence, and `drop-does-not-invalidate-only-one-name` is why treating
-- the two names as two entities is unsound.

open import Proof.Memory.Element using (ElemSig)

module Proof.Program.Ownership (Sig : ElemSig) where

open ElemSig Sig

open import Data.Bool using (Bool; true; false)
open import Data.Empty using (⊥; ⊥-elim)
open import Data.List using (List; []; _∷_)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc)
open import Data.Nat.Base using (_≡ᵇ_)
open import Data.Product using (_×_; _,_; Σ; proj₁; proj₂)
open import Relation.Binary.PropositionalEquality
  using (_≡_; refl; sym; trans; cong; subst)
open import Relation.Nullary using (Dec; yes; no; ¬_)

open import Proof.RC.Object using (ObjId; objAllocation; objGeneration;
  sameObj; sameObj-refl;
  Life; live; finalizing; dead;
  RuntimeCount; counted; immortal; bumpUp)
open import Proof.RC.OwnerSite using (OwnerSite; ThreadId; SiteMap; occupy; vacate;
  logicalRC; occupy-same; occupy-other)
open import Proof.RC.Machine Sig
open import Proof.Program.Syntax
open import Proof.Program.Env
open import Proof.Program.Step Sig

------------------------------------------------------------------------
-- Binding facts, in the shape the theorems need.

lookup-bind-same : ∀ (es : Env) (x : Var) (b : Binding) →
                   lookupVar (bindVar es x b) x ≡ just b
lookup-bind-same es x b rewrite sameVar-refl x = refl

entity-bind-same : ∀ (es : Env) (x : Var) (o : ObjId) (md : Mode) →
                   entityOf (bindVar es x (bind o md)) x ≡ just o
entity-bind-same es x o md rewrite sameVar-refl x = refl

------------------------------------------------------------------------
-- 1. A branch MOVES. It does not create a second name.
--
-- This is the decision, and it changed what this section says. `bindParams`
-- bound the successor's parameter and left the operand bound, so one entity had
-- two owned names and one owner site -- the shape of the shipped SIGSEGV.
--
-- `moveOne` unbinds the operand and relocates its site. What is left is one
-- name, one site, and a counter nobody touched.

moveOne-binds-the-parameter :
  ∀ (t : ThreadId) (es : Env) (ss : SiteMap) (p a : Var) (b : Binding) →
  lookupVar es a ≡ just b →
  lookupVar (unbindVar es a) a ≡ nothing →
  moveOne t (es , ss) p a
    ≡ just (bindVar (unbindVar es a) p b , relocate t ss a p b)
moveOne-binds-the-parameter t es ss p a b look gone rewrite look | gone = refl

-- ⭐ And the operand's name is GONE. Stated because it is the whole difference:
-- a pass that still sees `a` after the branch places a release for it, and that
-- release is the over-release.
moveOne-unbinds-the-operand :
  ∀ (t : ThreadId) (es : Env) (p a : Var) (b : Binding) →
  sameVar p a ≡ false →
  lookupVar (unbindVar es a) a ≡ nothing →
  lookupVar (bindVar (unbindVar es a) p b) a ≡ nothing
moveOne-unbinds-the-operand t es p a b ne gone
  rewrite lookupVar-cons-false p b (unbindVar es a) a ne = gone

-- A shadowed operand has NO step: `moveOne` refuses rather than moving a name
-- that survives its own unbind.
moveOne-refuses-shadowing :
  ∀ (t : ThreadId) (es : Env) (ss : SiteMap) (p a : Var) (b c : Binding) →
  lookupVar es a ≡ just b →
  lookupVar (unbindVar es a) a ≡ just c →
  moveOne t (es , ss) p a ≡ nothing
moveOne-refuses-shadowing t es ss p a b c look still rewrite look | still = refl

------------------------------------------------------------------------
-- 2. Where an alias comes from now.
--
-- Not from a branch. `dup` is the only instruction that gives one entity two
-- owned names, and it PAYS for it: a second owner site and a counter that went
-- up. That is the difference between an alias the model licenses and the one it
-- used to admit for free.

two-names-one-object :
  ∀ (es : Env) (x y : Var) (o : ObjId) →
  entityOf es x ≡ just o → entityOf es y ≡ just o →
  Aliases es x y
two-names-one-object es x y o px py = aliased o px py

owned-names-are-counted :
  ∀ (es : Env) (x : Var) (o : ObjId) →
  ownedCount (bindVar es x (bind o owned)) o ≡ suc (ownedCount es o)
owned-names-are-counted es x o rewrite sameObj-refl o = refl

-- A borrow does NOT add to the count. This is the row of the table where the
-- correct number of runtime operations is zero, now stated over the program
-- rather than over an isolated operation.
borrowed-names-are-not-counted :
  ∀ (es : Env) (x anchor : Var) (o : ObjId) →
  ownedCount (bindVar es x (bind o (borrowed anchor))) o ≡ ownedCount es o
borrowed-names-are-not-counted es x anchor o = refl

------------------------------------------------------------------------
-- 3. What each step does to the ghost count.
--
-- These are the rules read back as arithmetic. `move` is the interesting one:
-- it is the only rule that changes the site map without changing the count, and
-- a pass that emitted a release for it would be emitting one release too many.

step-move-preserves-objects :
  ∀ {f t bid rest es m src dst o s'} →
  f ⊢ pstate t bid (move dst src ∷ rest) es m —→ᵢ s' →
  lookupVar es src ≡ just (bind o owned) →
  objects (mach s') ≡ objects m
step-move-preserves-objects (step-move _ _) _ = refl

step-borrow-preserves-everything :
  ∀ {f t bid rest es m src dst s'} →
  f ⊢ pstate t bid (borrow dst src ∷ rest) es m —→ᵢ s' →
  mach s' ≡ m
step-borrow-preserves-everything (step-borrow _) = refl

------------------------------------------------------------------------
-- 4. Reachability preserves the heap's identity structure.
--
-- No step in this relation deallocates. That is a fact about the IR rather than
-- an omission: freeing is a separate operation with its own precondition
-- (`reclaim` requires `finalizing` AND zero owner sites), and putting it in the
-- step relation would let a program free while a name still holds the object.
--
-- Stated as: every reachable state has the same heap as the one it started
-- from. It is what makes "the storage is still there" available to every later
-- theorem without re-deriving it at each step.

steps-preserve-heap : ∀ {f s t} → f ⊢ s —→ t → heap (mach t) ≡ heap (mach s)
steps-preserve-heap (by-instr (step-new _ _ _ _ _)) = refl
steps-preserve-heap (by-instr (step-move _ _))      = refl
steps-preserve-heap (by-instr (step-dup _ _ _))     = refl
steps-preserve-heap (by-instr (step-drop _ _ _ _))  = refl
steps-preserve-heap (by-instr (step-borrow _))      = refl
steps-preserve-heap (by-term (step-br _ _ _ _))            = refl
steps-preserve-heap (by-term (step-cond-then _ _ _ _))     = refl
steps-preserve-heap (by-term (step-cond-else _ _ _ _))     = refl
steps-preserve-heap (by-term (step-invoke-normal _ _ _ _)) = refl
steps-preserve-heap (by-term (step-invoke-throw _ _ _ _))  = refl

reachable-preserves-heap : ∀ {f s t} → f ⊢ s —→* t → heap (mach t) ≡ heap (mach s)
reachable-preserves-heap done        = refl
reachable-preserves-heap (more p ps) =
  trans (reachable-preserves-heap ps) (steps-preserve-heap p)

------------------------------------------------------------------------
-- 5. The unwind edge is reachable.
--
-- Families A and B were releases on an edge no INPUT reaches. The model does
-- not know about inputs -- both successors of an `invoke` are steps -- and that
-- is the right choice for a safety proof: a release placed on an edge the model
-- can take is a release that has to be correct there.
--
-- So this is not a theorem about a particular program. It is the statement that
-- the edge EXISTS as a step, which is precisely what was missing.

unwind-edge-is-a-step :
  ∀ (f : Function) t bid es m x l a pad pa cur nxt es' ss' →
  findBlock f bid ≡ cur ∷ [] →
  term cur ≡ invoke x l a pad pa →
  findBlock f pad ≡ nxt ∷ [] →
  moveArgs t (env-and-sites es m) (params nxt) pa ≡ just (es' , ss') →
  f ⊢ pstate t bid [] es m —→ pstate t pad (body nxt) es' (afterArgs m ss')
unwind-edge-is-a-step f t bid es m x l a pad pa cur nxt es' ss' fb tm fp bp =
  by-term (step-invoke-throw fb tm fp bp)
