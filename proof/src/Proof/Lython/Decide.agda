{-# OPTIONS --safe #-}

-- ⭐ `Valid`, decided by a FINITE check.
--
-- `Valid es m = ¬ Invalidity es m` was stated and no program had ever been
-- shown to satisfy it, because three of the four invalidities had no procedure.
-- They all have one now, and this module ties them together.
--
-- The first version of this module quantified its obligations over `Var` and
-- `ObjId`, both of which are ℕ -- so discharging it meant proving a statement
-- about every natural number rather than running a check. That was the last
-- thing standing between `Valid` and something a pass could actually establish.
--
-- The lists are read off the state: the names the environment binds, the
-- objects its bindings denote, and the objects the site map holds. Each
-- membership lemma in Proof.Lython.Detect says an invalidity can only be ABOUT
-- something in them, which is what makes checking the lists equivalent to
-- checking everything.

open import Proof.Memory.Element using (ElemSig)

module Proof.Lython.Decide (Sig : ElemSig) where

open ElemSig Sig

open import Data.Bool using (Bool; true; false)
open import Data.Empty using (⊥; ⊥-elim)
open import Data.List using (List; []; _∷_; _++_)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc; _<_)
open import Data.Product using (_×_; _,_; Σ; proj₁; proj₂)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; sym; trans)
open import Relation.Nullary using (¬_)

open import Proof.RC.Object using (ObjId)
open import Proof.RC.OwnerSite using (OwnerSite; holds-positive)
open import Proof.RC.Machine Sig
open import Proof.Concurrent.Event using (Event)
open import Proof.Program.Syntax using (Var)
open import Proof.Program.Env
open import Proof.Lython.Invalid Sig
open import Proof.Lython.Detect Sig

------------------------------------------------------------------------
-- The three finite domains.
--
-- `objectsInPlay` is the concatenation because the two invalidities look in
-- different places: a premature reclaim is about an object a NAME still
-- denotes, a leak and a race are about an object a SITE still holds. An object
-- can be in one list and not the other -- indeed a leak is exactly that -- so
-- neither alone would do.

objectsInPlay : Env → Machine → List ObjId
objectsInPlay es m = objectsNamed es ++ objectsOwned (sites m)

private
  ∈ᵒ-left : ∀ {o : ObjId} (xs ys : List ObjId) → o ∈ᵒ xs → o ∈ᵒ (xs ++ ys)
  ∈ᵒ-left []       ys ()
  ∈ᵒ-left (_ ∷ xs) ys o-head     = o-head
  ∈ᵒ-left (_ ∷ xs) ys (o-tail m) = o-tail (∈ᵒ-left xs ys m)

  ∈ᵒ-right : ∀ {o : ObjId} (xs ys : List ObjId) → o ∈ᵒ ys → o ∈ᵒ (xs ++ ys)
  ∈ᵒ-right []       ys m = m
  ∈ᵒ-right (_ ∷ xs) ys m = o-tail (∈ᵒ-right xs ys m)

------------------------------------------------------------------------
-- The obligation, as four finite checks.

record AllChecksSilent (es : Env) (m : Machine) : Set where
  constructor checks-silent
  field
    no-dangling  : Every (λ x → danglingAnchor es x ≡ nothing) (names es)
    no-premature : Every (λ o → prematureReclaim? es m o ≡ nothing) (objectsInPlay es m)
    no-leak      : Every (λ o → leaked? es m o ≡ false) (objectsInPlay es m)
    -- ⭐ The race conjunct is over the OBJECT, not over an event.
    --
    -- `needsAtomic?` decides whether an object is one whose refcount updates
    -- must be atomic; whether a particular emitted operation IS atomic is a
    -- property of the instruction the compiler is about to write, which it
    -- knows by construction. Splitting it here is the honest division: the
    -- analysis is decided, the emission is `FollowsTheChecker` in
    -- Proof.Concurrent.RaceFree.
    no-shared    : Every (λ o → needsAtomic? m o ≡ nothing) (objectsInPlay es m)

open AllChecksSilent public

------------------------------------------------------------------------
-- ⭐ Silence on the finite lists means valid.

silence-means-valid : ∀ (es : Env) (m : Machine) →
                      AllChecksSilent es m → Valid es m

silence-means-valid es m ok (dangling-borrow {x} d) =
  silence-means-safe es x (every-at-var (no-dangling ok) listed) d
  where
    -- A dangling borrow is about a name the environment binds, so it is one of
    -- the keys the check walked.
    listed : x ∈ᵥ names es
    listed = name-is-listed es x _ (proj₂ (is-borrow d))

silence-means-valid es m ok (premature-reclaim {o} pr) =
  reclaim-is-safe-when-silent es m o (every-at-obj (no-premature ok) listed) pr
  where
    -- A premature reclaim is about an object a name still denotes.
    listed : o ∈ᵒ objectsInPlay es m
    listed = ∈ᵒ-left (objectsNamed es) (objectsOwned (sites m))
               (named-is-listed es (name (still-named pr)) o (holds-it (still-named pr)))

silence-means-valid es m ok (leak {o} lk) =
  no-leak-when-silent es m o (every-at-obj (no-leak ok) listed) lk
  where
    -- A leak is about an object a SITE still holds -- and, by definition, one
    -- no name denotes. That is why the two lists are concatenated.
    listed : o ∈ᵒ objectsInPlay es m
    listed = ∈ᵒ-right (objectsNamed es) (objectsOwned (sites m))
               (owned-is-listed (sites m) o (still-owned lk))

silence-means-valid es m ok (refcount-race {o} {e} r) = clash
  where
    -- A race is about an object two owner sites hold.
    listed : o ∈ᵒ objectsInPlay es m
    listed = ∈ᵒ-right (objectsNamed es) (objectsOwned (sites m))
               (holds-is-listed (sites m) (site₁ (proj₁ (needs-atomic r))) o
                 (holds₁ (proj₁ (needs-atomic r))))

    clash : ⊥
    clash with needsAtomic?-complete m o (needs-atomic r)
    ... | (s , u , rep) with trans (sym (every-at-obj (no-shared ok) listed)) rep
    ...   | ()

------------------------------------------------------------------------
-- What this settles.
--
-- On a state whose four checks are silent over the lists the state itself
-- provides, no `Invalidity` is derivable. Every quantifier is finite and every
-- procedure computes, so this is a property a pass can establish by running
-- something rather than by proving something.
--
-- What it does not settle is the emission side: `no-shared` says no object
-- needs an atomic refcount update in this state, which is what makes a plain
-- update safe. The other direction -- a state where objects DO need them, and
-- a policy that supplies them -- is `Proof.Concurrent.RaceFree`.
