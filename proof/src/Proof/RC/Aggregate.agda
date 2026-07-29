{-# OPTIONS --safe #-}

-- Aggregates, and the multiplicity the gap analysis said the model lacked.
--
-- The finding was:
--
--     `aggregate(parent, path)` is not a judgment, and the two open leak
--     families need a token COUNT rather than a token NAME.
--
-- Both halves are answered here, and the first one turns out to have been
-- available all along: `OwnerSite.field′` is indexed by an object AND a field
-- id, so "field k of p holds c" is `Holds ss (field′ p k) c` and a path is a
-- chain of them. What was missing is the arithmetic, and that is what the
-- second half is about.
--
-- The multiplicity is in `logicalRC` already, because it COUNTS a list: an
-- object held in three fields of one parent contributes three. A model with a
-- set of owners rather than a list would have collapsed those to one, and a
-- release of one field would then have looked like a release of all three --
-- which is exactly the over-release shape.

open import Proof.Memory.Element using (ElemSig)

module Proof.RC.Aggregate (Sig : ElemSig) where

open ElemSig Sig

open import Data.Bool using (Bool; true; false)
open import Data.Empty using (⊥; ⊥-elim)
open import Data.List using (List; []; _∷_; foldl)
open import Data.Maybe using (Maybe; just; nothing; maybe′)
open import Data.Nat using (ℕ; zero; suc; _<_; s≤s; z≤n)
open import Data.Product using (_×_; _,_; Σ; proj₁; proj₂)
open import Relation.Binary.PropositionalEquality
  using (_≡_; _≢_; refl; sym; trans; cong)
open import Relation.Nullary using (¬_)

open import Proof.RC.Object
open import Proof.RC.OwnerSite
open import Proof.RC.Machine Sig
open import Proof.RC.Invariant Sig
open import Proof.Memory.Descriptor Sig using (Desc)

------------------------------------------------------------------------
-- The judgment.

-- `c` is held by field `k` of `p`. One step of a path.
Field : SiteMap → ObjId → FieldId → ObjId → Set
Field ss p k c = Holds ss (field′ p k) c

-- A path of field ids from a parent to a descendant. The empty path is the
-- object itself, which is what makes `Aggregate ss p [] p` hold and is the base
-- case an aggregate release needs.
data Aggregate (ss : SiteMap) : ObjId → List FieldId → ObjId → Set where
  here  : ∀ {p} → Aggregate ss p [] p
  step  : ∀ {p k ks c d} → Field ss p k c → Aggregate ss c ks d →
          Aggregate ss p (k ∷ ks) d

-- ⭐ Every object on a path is held by a site, so its count is positive. This
-- is what makes an aggregate's members un-reclaimable while the parent holds
-- them -- and its contrapositive is the leak: reclaim the parent without
-- vacating its fields and the members keep a count nobody will ever lower.
aggregate-members-are-owned :
  ∀ (ss : SiteMap) (p : ObjId) (k : FieldId) (ks : List FieldId) (d : ObjId) →
  Aggregate ss p (k ∷ ks) d → Σ ObjId λ c → 0 < logicalRC ss c
aggregate-members-are-owned ss p k ks d (step {c = c} f _) =
  c , holds-positive ss (field′ p k) c f

------------------------------------------------------------------------
-- Releasing an aggregate.
--
-- Vacating the parent's field sites, one at a time. Written as a fold so that
-- the count arithmetic below is about a LIST of releases rather than about one,
-- which is where the multiplicity shows up.

releaseFields : SiteMap → ObjId → List FieldId → SiteMap
releaseFields ss p []       = ss
releaseFields ss p (k ∷ ks) = releaseFields (vacate ss (field′ p k)) p ks

-- ⭐ Each field release drops the child's count by exactly one -- not to zero.
--
-- This is the token COUNT the gap analysis asked for. A container holding one
-- object in two of its fields owns TWO references to it, and releasing one
-- field must leave one. An implementation that released "the object" once per
-- distinct child under-releases; one that treated the two as unrelated
-- entities over-releases.
one-field-release-drops-one :
  ∀ (ss : SiteMap) (p : ObjId) (k : FieldId) (c : ObjId) →
  strongAt ss (field′ p k) ≡ just c →
  logicalRC ss c ≡ suc (logicalRC (vacate ss (field′ p k)) c)
one-field-release-drops-one ss p k c held = vacate-holder ss (field′ p k) c held

-- and it leaves every OTHER object alone, which is the half that stops an
-- aggregate release from becoming a sweep.
one-field-release-spares-others :
  ∀ (ss : SiteMap) (p : ObjId) (k : FieldId) (c q : ObjId) →
  strongAt ss (field′ p k) ≡ just c → sameObj c q ≡ false →
  logicalRC (vacate ss (field′ p k)) q ≡ logicalRC ss q
one-field-release-spares-others ss p k c q held ne =
  vacate-holder-other ss (field′ p k) c q held ne

------------------------------------------------------------------------
-- ⭐ Multiplicity, on a concrete aggregate.
--
-- One parent, one child, held in TWO of the parent's fields. The count is two,
-- and each release takes it down by one. Checked by computation, because the
-- whole question is arithmetic and a conditional theorem about it would be
-- satisfied by a model that never had two.

private
  parent child : ObjId
  parent = obj 0 0
  child  = obj 1 0

  -- The child sits in fields 0 and 1 of the parent, and the parent itself is
  -- held by a local.
  twoFields : SiteMap
  twoFields = (local 0 0 , parent)
            ∷ (field′ parent 0 , child)
            ∷ (field′ parent 1 , child)
            ∷ []

  child-is-owned-twice : logicalRC twoFields child ≡ 2
  child-is-owned-twice = refl

  parent-is-owned-once : logicalRC twoFields parent ≡ 1
  parent-is-owned-once = refl

  -- Releasing ONE field leaves one reference. A pass that read "the child" as a
  -- single token would release it once and leak; one that read the two fields
  -- as two entities would release twice and over-release.
  after-one : SiteMap
  after-one = vacate twoFields (field′ parent 0)

  one-release-leaves-one : logicalRC after-one child ≡ 1
  one-release-leaves-one = refl

  after-both : SiteMap
  after-both = vacate after-one (field′ parent 1)

  both-releases-reach-zero : logicalRC after-both child ≡ 0
  both-releases-reach-zero = refl

  -- and the fold does the same thing, which is what makes `releaseFields` the
  -- operation rather than a description of one.
  fold-agrees : releaseFields twoFields parent (0 ∷ 1 ∷ []) ≡ after-both
  fold-agrees = refl

  aggregate-release-frees-the-child :
    logicalRC (releaseFields twoFields parent (0 ∷ 1 ∷ [])) child ≡ 0
  aggregate-release-frees-the-child = refl

  -- The parent is untouched by its own aggregate release. Obvious, and worth
  -- an equation: a fold that vacated by object rather than by SITE would have
  -- taken the parent down too.
  parent-survives : logicalRC (releaseFields twoFields parent (0 ∷ 1 ∷ [])) parent ≡ 1
  parent-survives = refl

  -- The judgment is inhabited on this state, at both fields.
  child-in-field-0 : Field twoFields parent 0 child
  child-in-field-0 = holds-there holds-here

  child-in-field-1 : Field twoFields parent 1 child
  child-in-field-1 = holds-there (holds-there holds-here)

  reachable-from-parent : Aggregate twoFields parent (0 ∷ []) child
  reachable-from-parent = step child-in-field-0 here

------------------------------------------------------------------------
-- ⭐ The leak an omitted aggregate release produces.
--
-- Release the parent's own site and stop there. The parent's count reaches
-- zero -- so it is reclaimable -- while the child is still held by two sites
-- that name a parent nobody will ever visit again. Nothing lowers those, and
-- the child is leaked.
--
-- This is the shape of the two unattributed leak families, and it is the first
-- state in the development that exhibits it with a COUNT rather than a name.

private
  parent-released : SiteMap
  parent-released = vacate twoFields (local 0 0)

  parent-is-reclaimable : logicalRC parent-released parent ≡ 0
  parent-is-reclaimable = refl

  -- and the child still has two owners.
  child-still-owned-twice : logicalRC parent-released child ≡ 2
  child-still-owned-twice = refl

  -- ⭐ The two sites that hold it name a parent with no owners left. Every
  -- reference to them has gone, so the count can never come down.
  orphaned : Field parent-released parent 0 child
  orphaned = holds-here

-- Stated without the concrete state, because it is the general fact: an object
-- reachable only through a parent whose own count has reached zero is owned by
-- sites nothing will ever vacate.
--
-- `Orphaned ss p c` is the model's name for what the compiler's missing
-- `aggregate_release` produces.
record Orphaned (ss : SiteMap) (p c : ObjId) : Set where
  constructor orphan
  field
    -- The parent is finished ...
    parent-unowned : logicalRC ss p ≡ 0
    -- ... and the child is still held from one of its fields.
    held-by-parent : Σ FieldId λ k → Field ss p k c

open Orphaned public

orphan-count-is-positive :
  ∀ (ss : SiteMap) (p c : ObjId) → Orphaned ss p c → 0 < logicalRC ss c
orphan-count-is-positive ss p c o =
  holds-positive ss (field′ p (proj₁ (held-by-parent o))) c (proj₂ (held-by-parent o))

-- and the concrete state above is one.
private
  the-orphan : Orphaned parent-released parent child
  the-orphan = orphan refl (0 , orphaned)

------------------------------------------------------------------------
-- ⭐ `aggregate_release`, as an operation.
--
-- `releaseFields` took the field list as an argument, so "release every field"
-- was not expressible: nothing in the model said WHICH fields an object had.
-- `ObjCell` now carries its class's arity, so they can be enumerated and the
-- operation is derivable rather than described.

fieldIds : ℕ → List FieldId
fieldIds zero    = []
fieldIds (suc n) = n ∷ fieldIds n

-- Every field of the object, released. This is the compiler's
-- `aggregate_release` and it takes no list from anyone.
aggregateRelease : Machine → ObjId → SiteMap
aggregateRelease m o =
  maybe′ (λ c → releaseFields (sites m) o (fieldIds (arity c))) (sites m)
         (lookupObj (objects m) o)

-- and the same on the whole machine.
releaseAggregate : Machine → ObjId → Machine
releaseAggregate m o = machine (heap m) (objects m) (aggregateRelease m o)

-- The backing descriptor plays no part in any of this, so the probes are
-- parameterised over an arbitrary one rather than instantiating the element
-- signature -- which this module does not fix.
private
  module Probe (d : Desc 1) where
    -- Two fields, released by enumeration rather than by a supplied list.
    parentCell : ObjCell
    parentCell = cell live (counted 1) d 2

    aggMachine : Machine
    aggMachine = machine [] ((parent , parentCell) ∷ []) twoFields

    enumerates-both : fieldIds (arity parentCell) ≡ 1 ∷ 0 ∷ []
    enumerates-both = refl

    -- ⭐ The child's count reaches zero, and nobody supplied the field list.
    aggregate-release-is-complete :
      logicalRC (aggregateRelease aggMachine parent) child ≡ 0
    aggregate-release-is-complete = refl

    -- and the parent is untouched by its own aggregate release.
    parent-untouched :
      logicalRC (aggregateRelease aggMachine parent) parent ≡ 1
    parent-untouched = refl

    -- An object with no fields releases nothing, which is what stops
    -- `aggregateRelease` from being a sweep.
    leafCell : ObjCell
    leafCell = cell live (counted 1) d 0

    leafMachine : Machine
    leafMachine = machine [] ((parent , leafCell) ∷ []) twoFields

    leaf-releases-nothing : aggregateRelease leafMachine parent ≡ twoFields
    leaf-releases-nothing = refl

------------------------------------------------------------------------
-- What is still not here.
--
-- `Aggregate` is a path through the SITE MAP, so it describes what the machine
-- currently holds rather than what the class declares. An object whose field is
-- unset has no `field′` entry and no `Aggregate` step through it -- which is
-- right for release (there is nothing to release) and would be wrong for a
-- reachability analysis that needed to know the field exists. Distinguishing
-- "unset" from "absent" needs the field slots in the cell, not just their
-- count.
