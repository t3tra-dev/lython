{-# OPTIONS --safe #-}

-- The step relation. This is the module the gap analysis called the binding
-- constraint: without it an operation has no POSITION, so it cannot be in the
-- wrong one, and six of the eleven inexpressible defects were placement
-- defects.
--
-- Sequential and finite. The note is explicit that memory safety needs only the
-- reflexive-transitive closure of a finite step relation -- coinduction is for
-- liveness, and is not required here.
--
-- The rules carry more premises than they first did, and every one of them was
-- added because WITHOUT it the rule takes a well-formed machine to an
-- ill-formed one. They are legality conditions, not proof scaffolding: a `dup`
-- of a dead object is a resurrection, a `new` onto storage that is not there is
-- a dangling reference at birth, and a `drop` whose name is shadowed leaves the
-- environment claiming ownership of a reference that no longer exists.
--
-- The guard against making the rules unsatisfiable by piling on premises is
-- Proof.Program.Run, which derives all five on a concrete block. A rule nobody
-- can fire preserves everything.

open import Proof.Memory.Element using (ElemSig)

module Proof.Program.Step (Sig : ElemSig) where

open ElemSig Sig

open import Data.Bool using (Bool; true; false; if_then_else_)
open import Data.List using (List; []; _∷_)
open import Data.Maybe using (Maybe; just; nothing; maybe′)
open import Data.Nat using (ℕ; zero; suc)
open import Data.Product using (_×_; _,_; Σ; proj₁; proj₂)
open import Relation.Binary.PropositionalEquality using (_≡_; refl)

open import Proof.RC.Object using (ObjId; obj; objAllocation; objGeneration;
  Life; live; finalizing; dead;
  RuntimeCount; counted; immortal; bumpUp; bumpDown)
open import Proof.RC.OwnerSite using (OwnerSite; local; temp; ThreadId; SiteMap;
  strongAt; occupy; vacate; logicalRC; Holds)
open import Proof.Memory.Heap using (Heap; Block; lookupBlock; generation;
  liveness)
  renaming (live to blockLive)
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
--
-- `onThread` is not decoration. Without it `siteOf` had to name a fixed thread,
-- and the sequential relation could not describe anything the second thread
-- did -- so the concurrent layer could only ever schedule instructions that
-- leave the site map alone. `OwnerSite.local` has been thread-indexed from the
-- start precisely so this would be expressible; the index is now used.
record PState : Set where
  constructor pstate
  field
    onThread : ThreadId
    current  : BlockId
    pending  : List Instr
    env      : Env
    mach     : Machine

open PState public

-- The owner site a name occupies. Names in a program ARE owner sites, and this
-- is where the program layer and the refcount layer meet: `logicalRC` counts
-- these, so a program that binds a name without occupying a site would make the
-- runtime counter disagree with the ghost count by construction.
siteOf : ThreadId → Var → OwnerSite
siteOf t x = local t x

-- The pair a terminator transforms. Named so the rules read as one operation
-- on (names, sites) rather than as two coincidences.
env-and-sites : Env → Machine → Env × SiteMap
env-and-sites es m = es , sites m

------------------------------------------------------------------------
-- Stepping a counter down.
--
-- Written as two total functions rather than with a `with`, because a `with`
-- becomes an auxiliary function and then no preservation proof can see what the
-- new cell's life and count are -- which is the whole content of the `drop`
-- case. Same reason `sameObj`, `entityOf` and `ownedCount` are written the way
-- they are.

lifeAfterDown : RuntimeCount → Life → Life
lifeAfterDown (counted zero)    _ = finalizing
lifeAfterDown (counted (suc _)) l = l
lifeAfterDown immortal          l = l

stepDownCell : ObjCell → ObjCell
stepDownCell c = cell (lifeAfterDown (bumpDown (count c)) (life c))
                      (bumpDown (count c))
                      (backing c)
                      (arity c)

-- Decrementing one object's counter, and moving it to `finalizing` when it
-- reaches zero. Defined here rather than reused from Proof.RC.Ops because that
-- module's `release` also consults the site map; here the rule has already
-- decided which site is going, so only the counter half is wanted.
stepDownAt : ObjTable → ObjId → ObjTable
stepDownAt ts o = updateObj ts o stepDownCell

------------------------------------------------------------------------
-- Instruction steps.

data _⊢_—→ᵢ_ (f : Function) : PState → PState → Set where

  -- `new`: fresh object, refcount 1, one owner site, one owned name.
  --
  -- The cell is CONSTRUCTED by the rule rather than taken as a parameter. The
  -- first version let the rule install any cell at all, including one already
  -- at count 0 -- and then `live-positive` was unprovable, not because the
  -- invariant was wrong but because the IR could produce a state violating it.
  --
  -- Freshness is required in the object table AND in the ghost state: an id
  -- that some site already holds is not fresh however empty the table is, and
  -- occupying a second site for it would put the counter at 1 while the ghost
  -- count went to 2.
  --
  -- The three heap premises are what makes the reference valid at birth.
  step-new :
    ∀ {t bid rest es m x c o bk b} →
    lookupObj (objects m) o ≡ nothing →
    logicalRC (sites m) o ≡ 0 →
    lookupBlock (heap m) (objAllocation o) ≡ just b →
    generation b ≡ objGeneration o →
    liveness b ≡ blockLive →
    f ⊢ pstate t bid (new x c ∷ rest) es m
      —→ᵢ pstate t bid rest
            (bindVar es x (bind o owned))
            (machine (heap m) ((o , cell live (counted 1) bk 0) ∷ objects m)
                     (occupy (sites m) (siteOf t x) o))

  -- `move`: the entity changes hands. NO change to the count and none to the
  -- object table -- the number of owner sites is the same, one vacated and one
  -- occupied. A rule that touched the count here would make a move a dup.
  --
  -- The second premise is the SSA condition, and it is not decoration.
  -- `bindVar` shadows and `unbindVar` removes only the first binding, so
  -- without it a move can vacate the source's site while a shadowed binding of
  -- the same name survives -- an environment that owns a reference the site map
  -- no longer records, which is precisely the state an over-release comes from.
  step-move :
    ∀ {t bid rest es m src dst o} →
    lookupVar es src ≡ just (bind o owned) →
    lookupVar (unbindVar es src) src ≡ nothing →
    f ⊢ pstate t bid (move dst src ∷ rest) es m
      —→ᵢ pstate t bid rest
            (bindVar (unbindVar es src) dst (bind o owned))
            (machine (heap m) (objects m)
                     (occupy (vacate (sites m) (siteOf t src)) (siteOf t dst) o))

  -- `dup` = py.incref: counter up, one more owner site, one more owned name.
  -- All three move together or the invariant is broken by the rule itself.
  --
  -- The object must be in the table and LIVE. Dup of a dead object is a
  -- resurrection: the site count would go up while `dead-unowned` says a dead
  -- object has none. Exactly the premise `drop` already carried, for exactly
  -- the same reason, and its absence here was an asymmetry rather than a
  -- decision.
  step-dup :
    ∀ {t bid rest es m src dst o c} →
    lookupVar es src ≡ just (bind o owned) →
    lookupObj (objects m) o ≡ just c →
    life c ≡ live →
    f ⊢ pstate t bid (dup dst src ∷ rest) es m
      —→ᵢ pstate t bid rest
            (bindVar es dst (bind o owned))
            (machine (heap m)
                     (updateObj (objects m) o
                                (λ x → record x { count = bumpUp (count x) }))
                     (occupy (sites m) (siteOf t dst) o))

  -- `drop` = py.decref. The name goes, the site goes, the counter goes down.
  -- Reaching zero moves the object to `finalizing` and does NOT free it: the
  -- storage handoff is a separate step with its own precondition.
  step-drop :
    ∀ {t bid rest es m x o c} →
    lookupVar es x ≡ just (bind o owned) →
    lookupObj (objects m) o ≡ just c →
    life c ≡ live →
    lookupVar (unbindVar es x) x ≡ nothing →
    f ⊢ pstate t bid (drop x ∷ rest) es m
      —→ᵢ pstate t bid rest
            (unbindVar es x)
            (machine (heap m) (stepDownAt (objects m) o)
                     (vacate (sites m) (siteOf t x)))

  -- `borrow`: a second NAME, no second owner site, no runtime operation. This
  -- is the rule that makes eliding a retain correct rather than merely cheaper.
  step-borrow :
    ∀ {t bid rest es m src dst o md} →
    lookupVar es src ≡ just (bind o md) →
    f ⊢ pstate t bid (borrow dst src ∷ rest) es m
      —→ᵢ pstate t bid rest (bindVar es dst (bind o (borrowed src))) m

------------------------------------------------------------------------
-- ⭐ A block argument is a MOVE.
--
-- This is the decision the development localised and did not take for a long
-- time. `br` binds the successor's parameter to the operand's binding; the open
-- question was whether the operand's NAME survives.
--
-- If it does, one entity has two owned names and one owner site. That is the
-- disagreement `Proof.Program.Coherence` used to exhibit and the shape of the
-- shipped SIGSEGV: a pass reading the owned-name count as "references to
-- release" emits two drops for one reference.
--
-- It does not. The operand's name dies at the branch and its owner site moves
-- to the parameter -- exactly what `step-move` does, and for the same reason:
-- the number of owner sites is unchanged, one vacated and one occupied, so the
-- correct number of runtime operations is ZERO.
--
-- The alternative -- a DUP, occupying a second site -- would also be sound and
-- would cost a retain per block argument on every loop iteration. A loop-
-- carried value would pay a retain/release pair per turn for a reference that
-- never went anywhere. Move is free and is what the IR now means.

-- Borrowed bindings hold no site, so there is nothing to relocate for them.
-- Making that an `if` on `isOwned` rather than a `with` on the mode is the
-- usual reason: a `with` becomes an auxiliary function and the preservation
-- proof cannot see which branch was taken.
relocate : ThreadId → SiteMap → Var → Var → Binding → SiteMap
relocate t ss a p b =
  if isOwned (mode b)
    then occupy (vacate ss (siteOf t a)) (siteOf t p) (entity b)
    else ss

-- One argument moved into one parameter, at both levels at once.
--
-- The second lookup is the SSA condition, built into the operation rather than
-- carried as a premise: `bindVar` shadows and `unbindVar` removes only the
-- first binding, so a shadowed operand would survive its own move and the
-- environment would own a reference the site map no longer records. A program
-- that shadows simply has no step here, which is what "never silently
-- mis-execute" means at this layer.
moveOne : ThreadId → Env × SiteMap → Var → Var → Maybe (Env × SiteMap)
moveOne t (es , ss) p a =
  maybe′ (λ b →
           maybe′ (λ _ → nothing)
                  (just (bindVar (unbindVar es a) p b , relocate t ss a p b))
                  (lookupVar (unbindVar es a) a))
         nothing
         (lookupVar es a)

-- Length mismatch yields `nothing`: a branch supplying the wrong number of
-- operands is a malformed program, and silently zipping the shorter list would
-- make the model accept it.
moveArgs : ThreadId → Env × SiteMap → List Var → List Var → Maybe (Env × SiteMap)
moveArgs t st []       []       = just st
moveArgs t st (p ∷ ps) (a ∷ as) =
  maybe′ (λ st' → moveArgs t st' ps as) nothing (moveOne t st p a)
moveArgs t st []       (_ ∷ _)  = nothing
moveArgs t st (_ ∷ _)  []       = nothing

-- The machine a terminator hands on: same heap, same object table, relocated
-- sites. No terminator touches a counter -- that is the content of "a block
-- argument is a move".
afterArgs : Machine → SiteMap → Machine
afterArgs m ss = machine (heap m) (objects m) ss

------------------------------------------------------------------------
-- Terminator steps.
--
-- Every rule below looks up the CURRENT block and requires its terminator to
-- be the one it is about. The first version did not, and the relation then
-- allowed a branch to any label whatsoever -- a step relation that lets control
-- go anywhere makes every reachability theorem vacuous, which is the opposite
-- of what this layer is for.

data _⊢_—→ₜ_ (f : Function) : PState → PState → Set where

  step-br :
    ∀ {t bid es m l args cur nxt es' ss'} →
    findBlock f bid ≡ cur ∷ [] →
    term cur ≡ br l args →
    findBlock f l ≡ nxt ∷ [] →
    moveArgs t (env-and-sites es m) (params nxt) args ≡ just (es' , ss') →
    f ⊢ pstate t bid [] es m —→ₜ pstate t l (body nxt) es' (afterArgs m ss')

  -- Both branches of a conditional are steps, and the model does not decide
  -- which. That is deliberate: safety has to hold for EVERY path, so a rule
  -- that picked one would prove safety of the path it happened to pick.
  step-cond-then :
    ∀ {t bid es m x l₁ a₁ l₂ a₂ cur nxt es' ss'} →
    findBlock f bid ≡ cur ∷ [] →
    term cur ≡ condBr x l₁ a₁ l₂ a₂ →
    findBlock f l₁ ≡ nxt ∷ [] →
    moveArgs t (env-and-sites es m) (params nxt) a₁ ≡ just (es' , ss') →
    f ⊢ pstate t bid [] es m —→ₜ pstate t l₁ (body nxt) es' (afterArgs m ss')

  step-cond-else :
    ∀ {t bid es m x l₁ a₁ l₂ a₂ cur nxt es' ss'} →
    findBlock f bid ≡ cur ∷ [] →
    term cur ≡ condBr x l₁ a₁ l₂ a₂ →
    findBlock f l₂ ≡ nxt ∷ [] →
    moveArgs t (env-and-sites es m) (params nxt) a₂ ≡ just (es' , ss') →
    f ⊢ pstate t bid [] es m —→ₜ pstate t l₂ (body nxt) es' (afterArgs m ss')

  -- A call that returns normally.
  step-invoke-normal :
    ∀ {t bid es m x l a pad pa cur nxt es' ss'} →
    findBlock f bid ≡ cur ∷ [] →
    term cur ≡ invoke x l a pad pa →
    findBlock f l ≡ nxt ∷ [] →
    moveArgs t (env-and-sites es m) (params nxt) a ≡ just (es' , ss') →
    f ⊢ pstate t bid [] es m —→ₜ pstate t l (body nxt) es' (afterArgs m ss')

  -- THE UNWIND EDGE. Families A and B were releases placed on an edge no input
  -- reaches; without this constructor there is no such edge in the model
  -- either, and the defect would be as inexpressible here as it was before.
  step-invoke-throw :
    ∀ {t bid es m x l a pad pa cur nxt es' ss'} →
    findBlock f bid ≡ cur ∷ [] →
    term cur ≡ invoke x l a pad pa →
    findBlock f pad ≡ nxt ∷ [] →
    moveArgs t (env-and-sites es m) (params nxt) pa ≡ just (es' , ss') →
    f ⊢ pstate t bid [] es m —→ₜ pstate t pad (body nxt) es' (afterArgs m ss')

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
