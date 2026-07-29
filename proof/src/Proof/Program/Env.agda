{-# OPTIONS --safe #-}

-- The environment: names to entities, with an ownership mode.
--
-- This is the module the gap analysis said was missing. `Aliases env x y` --
-- two names, one entity -- is the sentence an allocation-level model cannot
-- write, and every theorem about the SIGSEGV class is downstream of it.

module Proof.Program.Env where

open import Data.Bool using (Bool; true; false; if_then_else_; _∧_)
open import Data.List using (List; []; _∷_; length)
open import Data.Maybe using (Maybe; just; nothing)
import Data.Maybe
open import Data.Nat using (ℕ; zero; suc; _≟_)
open import Data.Nat.Base using (_≡ᵇ_)
open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; sym; trans; cong)
open import Relation.Nullary using (Dec; yes; no; ¬_)

open import Proof.RC.Object using (ObjId; obj; objAllocation; objGeneration;
  sameObj; sameObj-refl; sameObj-sound; ≡ᵇ-sound)
open import Proof.Program.Syntax using (Var)

-- How a name holds its entity.
--
-- The distinction is not decoration: an owned name owes a release and a
-- borrowed one does not, and the same name at the same multiplicity can be
-- either. A model with only "x refers to o" cannot tell a leak from a
-- correctly-elided retain.
-- `borrowed` carries its ANCHOR: the name it was borrowed from. Without it the
-- only obligation a borrow has -- that it does not outlive what it borrows from
-- -- is unstatable, and `RefMode.borrowed` being region-indexed in
-- Proof.QTT.Quantity would be decoration. With it, a dangling borrow is a
-- checkable property of the environment.
data Mode : Set where
  owned    : Mode
  borrowed : Var → Mode

record Binding : Set where
  constructor bind
  field
    entity : ObjId
    mode   : Mode

open Binding public

Env : Set
Env = List (Var × Binding)

-- Boolean rather than the decision procedure, for the reason recorded in
-- Proof.RC.Machine: `with x ≟ y` compiles to an auxiliary function and the
-- occurrence inside an update is then not the term a goal abstracts.
sameVar : Var → Var → Bool
sameVar = _≡ᵇ_

sameVar-refl : ∀ x → sameVar x x ≡ true
sameVar-refl zero    = refl
sameVar-refl (suc n) = sameVar-refl n

-- Reflection, shared with `sameObj-sound` rather than reproved: `sameVar` IS
-- `_≡ᵇ_` on ℕ, and a second proof of the same fact is a second thing to keep in
-- step.
sameVar-sound : ∀ x y → sameVar x y ≡ true → x ≡ y
sameVar-sound = ≡ᵇ-sound

lookupVar : Env → Var → Maybe Binding
lookupVar []             _ = nothing
lookupVar ((y , b) ∷ es) x = if sameVar y x then just b else lookupVar es x

-- Binding shadows. `unbind` removes the first, so a `move` that unbinds its
-- source really does remove the name the source had.
bindVar : Env → Var → Binding → Env
bindVar es x b = (x , b) ∷ es

unbindVar : Env → Var → Env
unbindVar []             _ = []
unbindVar ((y , b) ∷ es) x = if sameVar y x then es else (y , b) ∷ unbindVar es x

-- The four equations, carried explicitly.
--
-- Reduction inside `lookupVar (unbindVar es x) y` creates a scrutinee that no
-- with-abstraction ever saw, so a proof that case splits at the top gets stuck
-- one layer down. Carrying the equation avoids the question -- the same device
-- and the same reason as `lookupObj-cons-true` and its siblings.
lookupVar-cons-true : ∀ u b (es : Env) y → sameVar u y ≡ true →
                      lookupVar ((u , b) ∷ es) y ≡ just b
lookupVar-cons-true u b es y e rewrite e = refl

lookupVar-cons-false : ∀ u b (es : Env) y → sameVar u y ≡ false →
                       lookupVar ((u , b) ∷ es) y ≡ lookupVar es y
lookupVar-cons-false u b es y e rewrite e = refl

unbindVar-cons-true : ∀ u b (es : Env) x → sameVar u x ≡ true →
                      unbindVar ((u , b) ∷ es) x ≡ es
unbindVar-cons-true u b es x e rewrite e = refl

unbindVar-cons-false : ∀ u b (es : Env) x → sameVar u x ≡ false →
                       unbindVar ((u , b) ∷ es) x ≡ (u , b) ∷ unbindVar es x
unbindVar-cons-false u b es x e rewrite e = refl

-- Written with `map` rather than with a `with`, and the difference is not
-- cosmetic: a `with` compiles to an auxiliary function, so a caller that case
-- splits on `lookupVar es x` does NOT thereby reduce an `entityOf es x` sitting
-- in a hypothesis, and inverting "this name denotes o" back to "this name is
-- bound" becomes unprovable. Same reason `sameObj` is a boolean.
entityOf : Env → Var → Maybe ObjId
entityOf es x = Data.Maybe.map entity (lookupVar es x)

------------------------------------------------------------------------
-- THE predicate.
--
-- Two names, one entity. This is what a block argument creates, what a `dup`
-- creates, and what the ownership machinery in the compiler failed to see when
-- it treated a loop-carried cell's two names as two entities.

record Aliases (es : Env) (x y : Var) : Set where
  constructor aliased
  field
    shared      : ObjId
    x-holds-it  : entityOf es x ≡ just shared
    y-holds-it  : entityOf es y ≡ just shared

open Aliases public

-- Aliasing is symmetric and reflexive-where-bound, as it must be: if it were
-- not, "these two names denote the same thing" would depend on the order they
-- were written, and a pass could be right about one direction and wrong about
-- the other.
aliases-sym : ∀ {es x y} → Aliases es x y → Aliases es y x
aliases-sym (aliased o px py) = aliased o py px

aliases-refl : ∀ {es x o} → entityOf es x ≡ just o → Aliases es x x
aliases-refl {o = o} p = aliased o p p

aliases-trans : ∀ {es x y z} → Aliases es x y → Aliases es y z → Aliases es x z
aliases-trans {es} {x} {y} {z} (aliased o px py) (aliased o' py' pz) =
  aliased o px (subst (λ w → entityOf es z ≡ just w) (just-inj (trans (sym py') py)) pz)
  where
    open import Relation.Binary.PropositionalEquality using (subst)
    just-inj : ∀ {A : Set} {a b : A} → just a ≡ just b → a ≡ b
    just-inj refl = refl

------------------------------------------------------------------------
-- Binding a block's parameters.
--
-- `bindParams` is where the second name appears. The parameters and the
-- operands are DIFFERENT names, and after this they denote the same entities --
-- which is the fact `br-creates-aliases` in Proof.Program.Step turns into a
-- theorem.
--
-- Length mismatch yields `nothing`: a branch supplying the wrong number of
-- operands is a malformed program, and silently zipping the shorter list would
-- make the model accept it.

bindParams : Env → List Var → List Var → Maybe Env
bindParams es []       []       = just es
bindParams es (p ∷ ps) (a ∷ as) with lookupVar es a
... | nothing = nothing
... | just b  with bindParams es ps as
...   | nothing  = nothing
...   | just es' = just (bindVar es' p b)
bindParams _  []       (_ ∷ _)  = nothing
bindParams _  (_ ∷ _)  []       = nothing

------------------------------------------------------------------------
-- Counting owned names of an entity.
--
-- This is the program-level counterpart of Proof.RC.OwnerSite.logicalRC, and
-- the two have to agree -- that agreement is what connects a program to the
-- refcount invariant. Borrowed bindings are NOT counted, which is the whole
-- content of "a borrow costs nothing".

isOwned : Mode → Bool
isOwned owned        = true
isOwned (borrowed _) = false

-- One boolean scrutinee, not two nested `with`s, and for the third time in this
-- development the reason is that a `with` becomes an auxiliary function: through
-- one, `sameObj (entity b) o` inside this count is not the same term as the one
-- inside `logicalRC`, and no single case split reduces both. Comparing the two
-- counts is the entire point of having them, so they have to branch on the same
-- thing. `isOwned (mode b) ∧ …` reduces to `sameObj (entity b) o` exactly when
-- the binding is owned, which is what makes that comparison possible.
ownedCount : Env → ObjId → ℕ
ownedCount []             _ = 0
ownedCount ((_ , b) ∷ es) o =
  if isOwned (mode b) ∧ sameObj (entity b) o
    then suc (ownedCount es o)
    else ownedCount es o
