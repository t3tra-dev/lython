{-# OPTIONS --safe #-}

-- Deciding the invalidities, and what a step can and cannot introduce.
--
-- A predicate nobody can evaluate is a specification, not a check. These are
-- the procedures a pass would actually run, with soundness proofs -- which is
-- the part that stops "we added a checker" from being the whole claim.

open import Proof.Memory.Element using (ElemSig)

module Proof.Lython.Detect (Sig : ElemSig) where

open ElemSig Sig

open import Data.Bool using (Bool; true; false)
open import Data.Empty using (⊥; ⊥-elim)
open import Data.List using (List; []; _∷_)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc; _≟_)
open import Data.Product using (_×_; _,_; Σ; proj₁; proj₂)
open import Relation.Binary.PropositionalEquality
  using (_≡_; _≢_; refl; sym; trans; cong)
open import Relation.Nullary using (Dec; yes; no; ¬_)

open import Proof.RC.Object using (ObjId; Life; live; finalizing; dead;
  RuntimeCount; counted; immortal)
open import Proof.RC.Machine Sig
open import Proof.Program.Syntax using (Var; Instr; new; move; dup; drop; borrow)
open import Proof.Program.Env
open import Proof.Program.Step Sig
open import Proof.Lython.Invalid Sig

------------------------------------------------------------------------
-- Env facts the checker needs.

lookup-bind-here : ∀ (es : Env) (x : Var) (b : Binding) →
                   lookupVar (bindVar es x b) x ≡ just b
lookup-bind-here es x b rewrite sameVar-refl x = refl

-- Distinctness is needed because binding SHADOWS: `borrow x x` would rebind the
-- anchor to the borrow itself, and then the anchor is trivially present. That
-- case is not a dangling borrow but it is not a sensible program either, and
-- the hypothesis is where the model says so out loud.
lookup-bind-there : ∀ (es : Env) (x y : Var) (b : Binding) →
                    sameVar x y ≡ false →
                    lookupVar (bindVar es x b) y ≡ lookupVar es y
lookup-bind-there es x y b ne rewrite ne = refl

------------------------------------------------------------------------
-- The dangling-borrow check.
--
-- Computable, and it needs no analysis: the anchor is recorded in the binding,
-- so the whole question is a second lookup. That is the payoff of putting the
-- anchor in `Mode.borrowed` rather than leaving borrows anonymous.

danglingAnchor : Env → Var → Maybe Var
danglingAnchor es x with lookupVar es x
... | nothing = nothing
... | just b with mode b
...   | owned      = nothing
...   | borrowed a with lookupVar es a
...     | nothing = just a
...     | just _  = nothing

-- Soundness: what the checker reports really is a dangling borrow.
--
-- The reverse direction -- completeness -- is not proved here, and the
-- difference matters: a sound checker never cries wolf, and this one may still
-- miss cases the specification calls dangling. It does not, but that is not
-- established below, and reporting it as if it were is the failure this project
-- has recorded repeatedly.
danglingAnchor-sound :
  ∀ (es : Env) (x a : Var) → danglingAnchor es x ≡ just a → DanglingBorrow es x
danglingAnchor-sound es x a rep with lookupVar es x in lx
... | nothing = ⊥-elim (bad rep)
  where bad : nothing ≡ just a → ⊥
        bad ()
... | just b with mode b in mb
...   | owned = ⊥-elim (bad rep)
  where bad : nothing ≡ just a → ⊥
        bad ()
...   | borrowed a' with lookupVar es a' in la
...     | just _  = ⊥-elim (bad rep)
  where bad : nothing ≡ just a → ⊥
        bad ()
...     | nothing = dangling a' (entity b , is-b) la
  where
    -- Binding is a record, so `b` is definitionally `bind (entity b) (mode b)`
    -- -- eta. That is what lets the equation about `mode b` be transported into
    -- an equation about the whole binding without destructing it.
    is-b : lookupVar es x ≡ just (bind (entity b) (borrowed a'))
    is-b = trans lx (cong (λ md → just (bind (entity b) md)) mb)

------------------------------------------------------------------------
-- What a `borrow` step introduces, and what it does not.
--
-- The rule's premise is that the anchor is bound, so a FRESH borrow is never
-- dangling. Every dangling borrow therefore arises later, when the anchor goes
-- away -- which is the shape a checker should look for, and it means the check
-- belongs at the point the anchor dies rather than at the borrow.

fresh-borrow-is-not-dangling :
  ∀ (es : Env) (src dst : Var) (o : ObjId) (md : Mode) →
  sameVar dst src ≡ false →
  lookupVar es src ≡ just (bind o md) →
  danglingAnchor (bindVar es dst (bind o (borrowed src))) dst ≡ nothing
fresh-borrow-is-not-dangling es src dst o md ne look
  rewrite lookup-bind-here es dst (bind o (borrowed src))
        | lookup-bind-there es dst src (bind o (borrowed src)) ne
        | look = refl

-- And the step really does bind it that way. Without this the lemma above would
-- be about an environment the relation never produces.
-- Stated with the object taken FROM the step's own premise rather than supplied
-- separately: given independently they are two different `o`s and the equation
-- is not the one the rule proves.
borrow-step-binds-the-anchor :
  ∀ {f bid rest es m src dst o md s'} →
  (st : f ⊢ pstate bid (borrow dst src ∷ rest) es m —→ᵢ s') →
  lookupVar es src ≡ just (bind o md) →
  Σ ObjId λ o' → env s' ≡ bindVar es dst (bind o' (borrowed src))
borrow-step-binds-the-anchor {o = o} (step-borrow {o = o'} _) _ = o' , refl

------------------------------------------------------------------------
-- Dropping an anchor is what creates the danger.
--
-- `drop x` unbinds `x`. Any borrow anchored at `x` is dangling from that point
-- on. This is the theorem that says WHERE the check has to run: not at the
-- borrow, at the drop.

drop-strands-its-borrows :
  ∀ (es : Env) (x y : Var) (o : ObjId) →
  lookupVar es y ≡ just (bind o (borrowed x)) →
  sameVar y x ≡ false →
  lookupVar (unbindVar es x) x ≡ nothing →
  lookupVar (unbindVar es x) y ≡ just (bind o (borrowed x)) →
  DanglingBorrow (unbindVar es x) y
drop-strands-its-borrows es x y o look ne gone still =
  dangling x (o , still) gone

-- The hypothesis `lookupVar (unbindVar es x) x ≡ nothing` is not free: `bindVar`
-- shadows, so an environment with two bindings for `x` keeps one after
-- `unbindVar`. Requiring it rather than assuming it is the difference between a
-- theorem about this model and a theorem about a model where names are unique.
