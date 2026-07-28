{-# OPTIONS --safe #-}

-- The reference-counting machine: a byte heap, an object table, and the ghost
-- site map that the object table's counters are supposed to implement.

open import Proof.Memory.Element using (ElemSig)

module Proof.RC.Machine (Sig : ElemSig) where

open ElemSig Sig

open import Data.Bool using (Bool; true; false; _∧_; if_then_else_)
open import Data.List using (List; []; _∷_)
open import Data.Maybe using (Maybe; just; nothing)
import Data.Maybe
open import Data.Nat using (ℕ; zero; suc; _≟_)
open import Data.Nat.Base using (_≡ᵇ_)
open import Data.Product using (_×_; _,_; Σ; proj₁; proj₂)
open import Relation.Binary.PropositionalEquality using (_≡_; refl)
open import Relation.Nullary using (Dec; yes; no; ¬_)

open import Proof.Memory.Heap using (Heap; AllocId; Generation; lookupBlock)
open import Proof.Memory.Descriptor Sig using (Desc)
open import Proof.RC.Object
open import Proof.RC.OwnerSite

-- The per-object runtime state. `backing` is the descriptor for the whole
-- allocation -- the canonical one, the same one `dealloc` accepts -- because an
-- object reference that circulated as a view could not be freed when its count
-- reached zero.
record ObjCell : Set where
  constructor cell
  field
    life    : Life
    count   : RuntimeCount
    backing : Desc 1

open ObjCell public

ObjTable : Set
ObjTable = List (ObjId × ObjCell)

-- A BOOLEAN test rather than the decision procedure, and the reason is
-- mechanical rather than stylistic: `with p ≟-obj o` compiles to an auxiliary
-- function, so the decision inside `updateObj` is not the same syntactic term as
-- the one in the goal, and `with` cannot abstract both at once. Proving
-- `lookupObj (updateObj ts o f) o ≡ just (f c)` then becomes impossible without
-- inspect-idiom scaffolding. With `if`, both occurrences are the same term.
-- `sameObj`, `≡ᵇ-refl` and `sameObj-refl` now live in Proof.RC.Object, so the
-- site map and the program environment use the same one.

lookupObj : ObjTable → ObjId → Maybe ObjCell
lookupObj []             _ = nothing
lookupObj ((p , c) ∷ ts) o = if sameObj p o then just c else lookupObj ts o

updateObj : ObjTable → ObjId → (ObjCell → ObjCell) → ObjTable
updateObj []             _ _ = []
updateObj ((p , c) ∷ ts) o f =
  if sameObj p o then (p , f c) ∷ ts else (p , c) ∷ updateObj ts o f

record Machine : Set where
  constructor machine
  field
    heap    : Heap
    objects : ObjTable
    -- Ghost state. It has no runtime representation and is not compiled; it is
    -- the thing `count` is a claim about.
    sites   : SiteMap

open Machine public

------------------------------------------------------------------------
-- Reading the machine.

lifeOf : Machine → ObjId → Maybe Life
lifeOf m o = Data.Maybe.map life (lookupObj (objects m) o)

countOf : Machine → ObjId → Maybe RuntimeCount
countOf m o = Data.Maybe.map count (lookupObj (objects m) o)

-- The ghost count. Note it does not consult the object table at all: that is
-- the point -- it is the number the counter has to match, computed
-- independently of the counter.
ghostRC : Machine → ObjId → ℕ
ghostRC m o = logicalRC (sites m) o

IsCounted : Machine → ObjId → Set
IsCounted m o = Σ ℕ λ n → countOf m o ≡ just (counted n)
