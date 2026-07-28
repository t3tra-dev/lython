{-# OPTIONS --safe #-}

-- The step relation. This is the module the gap analysis called the binding
-- constraint: without it an operation has no POSITION, so it cannot be in the
-- wrong one, and six of the eleven inexpressible defects were placement
-- defects.
--
-- Sequential and finite. The note is explicit that memory safety needs only the
-- reflexive-transitive closure of a finite step relation -- coinduction is for
-- liveness, and is not required here.

open import Proof.Memory.Element using (ElemSig)

module Proof.Program.Step (Sig : ElemSig) where

open ElemSig Sig

open import Data.List using (List; []; _∷_)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc)
open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Relation.Binary.PropositionalEquality using (_≡_; refl)

open import Proof.RC.Object using (ObjId; obj; Life; live; finalizing; dead;
  RuntimeCount; counted; immortal; bumpUp; bumpDown)
open import Proof.RC.OwnerSite using (OwnerSite; local; temp; SiteMap; strongAt;
  occupy; vacate; logicalRC)
open import Proof.Memory.Descriptor Sig using (Desc)
open import Proof.RC.Machine Sig
open import Proof.Program.Syntax
open import Proof.Program.Env

------------------------------------------------------------------------
-- Program state.
--
-- `current` and `pending` together are the position -- the thing whose absence
-- made placement inexpressible. `env` is the name→entity map, `mach` is the
-- heap and object table underneath.

record PState : Set where
  constructor pstate
  field
    current : BlockId
    pending : List Instr
    env     : Env
    mach    : Machine

open PState public

-- The owner site a name occupies. Names in a program ARE owner sites, and this
-- is where the program layer and the refcount layer meet: `logicalRC` counts
-- these, so a program that binds a name without occupying a site would make the
-- runtime counter disagree with the ghost count by construction.
siteOf : Var → OwnerSite
siteOf x = local 0 x

-- Decrementing one object's counter, and moving it to `finalizing` when it
-- reaches zero. Defined here rather than reused from Proof.RC.Ops because that
-- module's `release` also consults the site map; here the rule has already
-- decided which site is going, so only the counter half is wanted.
stepDownAt : ObjTable → ObjId → ObjTable
stepDownAt ts o = updateObj ts o down
  where
    down : ObjCell → ObjCell
    down c with bumpDown (count c)
    ... | counted zero = record c { count = counted zero ; life = finalizing }
    ... | n            = record c { count = n }

------------------------------------------------------------------------
-- Instruction steps.
--
-- Each constructor is a rule, and the premises are the conditions under which
-- the operation is legal. An operation whose premises cannot be met simply has
-- no step -- which is how a malformed program gets stuck rather than
-- proceeding into nonsense.

data _⊢_—→ᵢ_ (f : Function) : PState → PState → Set where

  -- `new`: fresh object, refcount 1, one owner site, one owned name.
  --
  -- The cell is CONSTRUCTED by the rule rather than taken as a parameter. The
  -- first version let the rule install any cell at all, including one already
  -- at count 0 -- and then `live-positive` was unprovable, not because the
  -- invariant was wrong but because the IR could produce a state violating it.
  -- Starting at 1 is also the right semantics: at 0 the object would be
  -- reclaimable the instant it exists.
  step-new :
    ∀ {bid rest es m x c o bk} →
    lookupObj (objects m) o ≡ nothing →
    f ⊢ pstate bid (new x c ∷ rest) es m
      —→ᵢ pstate bid rest
            (bindVar es x (bind o owned))
            (machine (heap m) ((o , cell live (counted 1) bk) ∷ objects m)
                     (occupy (sites m) (siteOf x) o))

  -- `move`: the entity changes hands. NO change to the count and none to the
  -- object table -- the number of owner sites is the same, one vacated and one
  -- occupied. A rule that touched the count here would make a move a dup.
  step-move :
    ∀ {bid rest es m src dst o} →
    lookupVar es src ≡ just (bind o owned) →
    f ⊢ pstate bid (move dst src ∷ rest) es m
      —→ᵢ pstate bid rest
            (bindVar (unbindVar es src) dst (bind o owned))
            (machine (heap m) (objects m)
                     (occupy (vacate (sites m) (siteOf src)) (siteOf dst) o))

  -- `dup` = py.incref: counter up, one more owner site, one more owned name.
  -- All three move together or the invariant is broken by the rule itself.
  step-dup :
    ∀ {bid rest es m src dst o} →
    lookupVar es src ≡ just (bind o owned) →
    f ⊢ pstate bid (dup dst src ∷ rest) es m
      —→ᵢ pstate bid rest
            (bindVar es dst (bind o owned))
            (machine (heap m)
                     (updateObj (objects m) o
                                (λ c → record c { count = bumpUp (count c) }))
                     (occupy (sites m) (siteOf dst) o))

  -- `drop` = py.decref. The name goes, the site goes, the counter goes down.
  -- Reaching zero moves the object to `finalizing` and does NOT free it: the
  -- storage handoff is a separate step with its own precondition.
  -- The object must be LIVE. Without this premise a drop on a dead object would
  -- move it back to `finalizing` -- a resurrection the model would then have to
  -- explain, and `dead-unowned` would be false at the state it produced.
  step-drop :
    ∀ {bid rest es m x o c} →
    lookupVar es x ≡ just (bind o owned) →
    lookupObj (objects m) o ≡ just c →
    life c ≡ live →
    f ⊢ pstate bid (drop x ∷ rest) es m
      —→ᵢ pstate bid rest
            (unbindVar es x)
            (machine (heap m) (stepDownAt (objects m) o)
                     (vacate (sites m) (siteOf x)))

  -- `borrow`: a second NAME, no second owner site, no runtime operation. This
  -- is the rule that makes eliding a retain correct rather than merely cheaper.
  step-borrow :
    ∀ {bid rest es m src dst o md} →
    lookupVar es src ≡ just (bind o md) →
    f ⊢ pstate bid (borrow dst src ∷ rest) es m
      —→ᵢ pstate bid rest (bindVar es dst (bind o (borrowed src))) m

------------------------------------------------------------------------
-- Terminator steps.
--
-- `step-br` is the one the SIGSEGV turns on: it binds the successor's
-- PARAMETERS to the operands' entities, which creates a second name for each.

-- Every rule below looks up the CURRENT block and requires its terminator to
-- be the one it is about. The first version did not, and the relation then
-- allowed a branch to any label whatsoever -- a step relation that lets control
-- go anywhere makes every reachability theorem vacuous, which is the opposite of
-- what this layer is for.
data _⊢_—→ₜ_ (f : Function) : PState → PState → Set where

  step-br :
    ∀ {bid es m l args cur nxt es'} →
    findBlock f bid ≡ cur ∷ [] →
    term cur ≡ br l args →
    findBlock f l ≡ nxt ∷ [] →
    bindParams es (params nxt) args ≡ just es' →
    f ⊢ pstate bid [] es m —→ₜ pstate l (body nxt) es' m

  -- Both branches of a conditional are steps, and the model does not decide
  -- which. That is deliberate: safety has to hold for EVERY path, so a rule
  -- that picked one would prove safety of the path it happened to pick.
  step-cond-then :
    ∀ {bid es m x l₁ a₁ l₂ a₂ cur nxt es'} →
    findBlock f bid ≡ cur ∷ [] →
    term cur ≡ condBr x l₁ a₁ l₂ a₂ →
    findBlock f l₁ ≡ nxt ∷ [] →
    bindParams es (params nxt) a₁ ≡ just es' →
    f ⊢ pstate bid [] es m —→ₜ pstate l₁ (body nxt) es' m

  step-cond-else :
    ∀ {bid es m x l₁ a₁ l₂ a₂ cur nxt es'} →
    findBlock f bid ≡ cur ∷ [] →
    term cur ≡ condBr x l₁ a₁ l₂ a₂ →
    findBlock f l₂ ≡ nxt ∷ [] →
    bindParams es (params nxt) a₂ ≡ just es' →
    f ⊢ pstate bid [] es m —→ₜ pstate l₂ (body nxt) es' m

  -- A call that returns normally.
  step-invoke-normal :
    ∀ {bid es m x l a pad pa cur nxt es'} →
    findBlock f bid ≡ cur ∷ [] →
    term cur ≡ invoke x l a pad pa →
    findBlock f l ≡ nxt ∷ [] →
    bindParams es (params nxt) a ≡ just es' →
    f ⊢ pstate bid [] es m —→ₜ pstate l (body nxt) es' m

  -- THE UNWIND EDGE. Families A and B were releases placed on an edge no input
  -- reaches; without this constructor there is no such edge in the model
  -- either, and the defect would be as inexpressible here as it was before.
  step-invoke-throw :
    ∀ {bid es m x l a pad pa cur nxt es'} →
    findBlock f bid ≡ cur ∷ [] →
    term cur ≡ invoke x l a pad pa →
    findBlock f pad ≡ nxt ∷ [] →
    bindParams es (params nxt) pa ≡ just es' →
    f ⊢ pstate bid [] es m —→ₜ pstate pad (body nxt) es' m

-- `by-term`, not `term`: Syntax already exports a `term` field, and a clash
-- here would be resolved by whichever import came last rather than by intent.
data _⊢_—→_ (f : Function) : PState → PState → Set where
  by-instr : ∀ {s t} → f ⊢ s —→ᵢ t → f ⊢ s —→ t
  by-term  : ∀ {s t} → f ⊢ s —→ₜ t → f ⊢ s —→ t

------------------------------------------------------------------------
-- Reachability: the reflexive-transitive closure.

data _⊢_—→*_ (f : Function) : PState → PState → Set where
  done : ∀ {s} → f ⊢ s —→* s
  more : ∀ {s t u} → f ⊢ s —→ t → f ⊢ t —→* u → f ⊢ s —→* u

Reachable : Function → PState → PState → Set
Reachable f s t = f ⊢ s —→* t

—→*-trans : ∀ {f s t u} → f ⊢ s —→* t → f ⊢ t —→* u → f ⊢ s —→* u
—→*-trans done        q = q
—→*-trans (more p ps) q = more p (—→*-trans ps q)
