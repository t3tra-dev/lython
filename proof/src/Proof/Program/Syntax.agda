{-# OPTIONS --safe #-}

-- The linear resource IR: a program, with names and control flow.
--
-- This layer exists because of one finding. The model below it is about
-- ALLOCATIONS; the compiler's defects are about SSA VALUES and where operations
-- sit relative to control flow. The shipped SIGSEGV's root is
--
--     one allocation, TWO SSA NAMES, and the ownership machinery treats that as
--     two entities
--
-- and no sentence in an allocation-level model can say that, because a name
-- distinct from an allocation is not something it has. So names are primitive
-- here, and `Env` -- not the heap -- is what maps them to entities.
--
-- Block arguments are in the IR for the same reason and are not optional
-- decoration: they are exactly how a loop-carried value acquires a second name.

module Proof.Program.Syntax where

open import Data.List using (List; []; _∷_; length)
open import Data.Nat using (ℕ; zero; suc; _≟_)
open import Data.Product using (_×_; _,_)
open import Relation.Binary.PropositionalEquality using (_≡_; refl)
open import Relation.Nullary using (Dec; yes; no)

open import Proof.RC.Object using (ObjId)

-- An SSA name. Deliberately NOT an ObjId: the entire content of this layer is
-- that the two are different, and that a program can hold several names for one
-- entity.
Var BlockId FieldId ClassId : Set
Var     = ℕ
BlockId = ℕ
FieldId = ℕ
ClassId = ℕ

_≟-var_ : (x y : Var) → Dec (x ≡ y)
_≟-var_ = _≟_

------------------------------------------------------------------------
-- Instructions.
--
-- The set is the note's §4 linear resource IR. Half of these emit no runtime
-- operation -- `move` and `borrow` are bookkeeping -- and that is the point:
-- an IR in which a move is indistinguishable from a dup cannot be used to say
-- that eliding a retain was correct.

data Instr : Set where
  -- Allocate a fresh object and bind it to a name, owned, refcount 1.
  new     : Var → ClassId → Instr
  -- Transfer ownership from one name to another. The source name is gone
  -- afterwards; no runtime operation.
  move    : Var → Var → Instr
  -- A second owning reference. This is py.incref.
  dup     : Var → Var → Instr
  -- Give up an owning reference. This is py.decref.
  drop    : Var → Instr
  -- A non-owning read. No runtime operation, and the obligation is on lifetime
  -- rather than on the count.
  borrow  : Var → Var → Instr
  -- Field access on a boxed object, at the object's own lane.
  getField : Var → Var → FieldId → Instr
  setField : Var → FieldId → Var → Instr

------------------------------------------------------------------------
-- Terminators.
--
-- `br` and `condBr` carry OPERANDS for the successor's parameters. That is the
-- construct the SIGSEGV turns on: after the branch, the successor's parameter
-- and the operand that fed it are two names for one entity.
--
-- `invoke` is the shape a call with an unwind edge has: one successor on the
-- normal path and one pad. Without it, a release placed on an unwind edge --
-- which is where families A and B lived -- has nowhere to be placed.

data Term : Set where
  br     : BlockId → List Var → Term
  condBr : Var → BlockId → List Var → BlockId → List Var → Term
  ret    : Var → Term
  -- A call that may throw: normal successor, then the landing pad.
  invoke : Var → BlockId → List Var → BlockId → List Var → Term
  -- Rethrow out of the function.
  unwind : Term

record Block : Set where
  constructor block
  field
    label  : BlockId
    -- Block PARAMETERS. Every entry to this block binds these to the operands
    -- the branch supplied.
    params : List Var
    body   : List Instr
    term   : Term

open Block public

record Function : Set where
  constructor function
  field
    blocks : List Block
    entry  : BlockId

open Function public

lookupBlock′ : List Block → BlockId → List Block
lookupBlock′ []       _ = []
lookupBlock′ (b ∷ bs) l with label b ≟ l
... | yes _ = b ∷ []
... | no  _ = lookupBlock′ bs l

-- Total, and it reports absence rather than inventing a block: a branch to a
-- label that does not exist is a malformed program and the step relation has to
-- be able to say so.
findBlock : Function → BlockId → List Block
findBlock f l = lookupBlock′ (blocks f) l
