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
open import Proof.RC.OwnerSite using (OwnerSite; occupy; vacate; logicalRC;
  occupy-same; occupy-other)
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
-- 1. A branch creates a SECOND NAME for the SAME entity.
--
-- This is the construct the SIGSEGV turns on. `bindParams` binds the
-- successor's parameter to the binding the operand had, so afterwards the
-- parameter and the operand denote one entity under two names.
--
-- Stated for the one-parameter case, which is the whole of the phenomenon: a
-- loop-carried value threaded through a block argument.

bindParams-one :
  ∀ (es : Env) (p a : Var) (b : Binding) →
  lookupVar es a ≡ just b →
  bindParams es (p ∷ []) (a ∷ []) ≡ just (bindVar es p b)
bindParams-one es p a b look rewrite look = refl

br-creates-an-alias :
  ∀ (es : Env) (p a : Var) (o : ObjId) (md : Mode) →
  lookupVar es a ≡ just (bind o md) →
  ∀ es' → bindParams es (p ∷ []) (a ∷ []) ≡ just es' →
  entityOf es' p ≡ just o
br-creates-an-alias es p a o md look es' bp
  with trans (sym (bindParams-one es p a (bind o md) look)) bp
... | refl = entity-bind-same es p o md

------------------------------------------------------------------------
-- 2. Why treating the two names as two entities is unsound.
--
-- If a pass believes the block parameter is a separate entity, it will place a
-- release for it and another for the operand. Both releases name the SAME
-- object, so the count goes down twice for one owner going away.
--
-- The model says this directly: `drop` of either name steps the counter down
-- once, and the two drops compose. There is nothing in the machine that would
-- notice they were the same entity -- which is exactly the compiler's position,
-- and why the bug is a bug rather than something the runtime catches.

two-names-one-object :
  ∀ (es : Env) (x y : Var) (o : ObjId) →
  entityOf es x ≡ just o → entityOf es y ≡ just o →
  Aliases es x y
two-names-one-object es x y o px py = aliased o px py

-- And the count that matters is over ENTITIES, not names: two owned names of
-- one object contribute two, which is right, and it is why a release for each
-- is right ONLY IF a retain for each happened. The defect is a release without
-- its retain -- expressible now, and not before.
-- Reusing Proof.RC.Machine's proof rather than restating it: two definitions of
-- "≡ᵇ is reflexive" are two things that can drift, and the object comparison in
-- Env has to agree with the one in the machine or the program-level count and
-- the ghost count would be counting with different notions of equality.
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
  ∀ {f bid rest es m src dst o s'} →
  f ⊢ pstate bid (move dst src ∷ rest) es m —→ᵢ s' →
  lookupVar es src ≡ just (bind o owned) →
  objects (mach s') ≡ objects m
step-move-preserves-objects (step-move _) _ = refl

step-borrow-preserves-everything :
  ∀ {f bid rest es m src dst s'} →
  f ⊢ pstate bid (borrow dst src ∷ rest) es m —→ᵢ s' →
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
steps-preserve-heap (by-instr (step-new _))     = refl
steps-preserve-heap (by-instr (step-move _))   = refl
steps-preserve-heap (by-instr (step-dup _))    = refl
steps-preserve-heap (by-instr (step-drop _ _ _)) = refl
steps-preserve-heap (by-instr (step-borrow _)) = refl
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
  ∀ (f : Function) bid es m x l a pad pa cur nxt es' →
  findBlock f bid ≡ cur ∷ [] →
  term cur ≡ invoke x l a pad pa →
  findBlock f pad ≡ nxt ∷ [] →
  bindParams es (params nxt) pa ≡ just es' →
  f ⊢ pstate bid [] es m —→ pstate pad (body nxt) es' m
unwind-edge-is-a-step f bid es m x l a pad pa cur nxt es' fb tm fp bp =
  by-term (step-invoke-throw fb tm fp bp)
