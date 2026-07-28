{-# OPTIONS --safe #-}

-- The shape of the shipped SIGSEGV, written as a program.
--
-- This module is the answer to the gap analysis. The defect's root was
--
--     one allocation, TWO SSA NAMES, and the ownership machinery treats that as
--     two entities
--
-- and the finding was that no sentence in the model could say it. Here is the
-- sentence, as a concrete program and a concrete derivation:
--
--     bb0(): new x ; br bb1(x)
--     bb1(p): drop p ; ...
--
-- After the branch, `x` and `p` are DIFFERENT NAMES and `Aliases env x p` holds.
-- A pass that reads them as two entities places two drops; the model shows both
-- drops step, and that the second one decrements a counter whose owner is
-- already gone.

module Proof.Program.Trace where

open import Data.List using (List; []; _∷_)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc)
open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Relation.Binary.PropositionalEquality using (_≡_; refl)

open import Proof.Memory.Heap using (Heap)
open import Proof.Memory.Lython using (LythonSig; i64)
open import Proof.Memory.Descriptor LythonSig using (Desc; desc)
open import Proof.RC.Object
open import Proof.RC.OwnerSite using (OwnerSite; local; SiteMap; logicalRC)
open import Proof.RC.Machine LythonSig
open import Proof.Program.Syntax
open import Proof.Program.Env
open import Proof.Program.Step LythonSig
open import Proof.Program.Ownership LythonSig

open import Data.Integer using (+_)
open import Data.Vec using ([]; _∷_)

------------------------------------------------------------------------
-- The program.

x p : Var
x = 0
p = 1

bb0 bb1 : BlockId
bb0 = 0
bb1 = 1

theObj : ObjId
theObj = obj 0 0

-- bb0 allocates and branches, passing the name it just bound.
block0 : Block
block0 = block bb0 [] (new x 7 ∷ []) (br bb1 (x ∷ []))

-- bb1 takes a PARAMETER. This is where the second name is created.
block1 : Block
block1 = block bb1 (p ∷ []) (drop p ∷ []) unwind

prog : Function
prog = function (block0 ∷ block1 ∷ []) bb0

------------------------------------------------------------------------
-- The state just after the allocation, and just before the branch.

objBacking : Desc 1
objBacking = desc 0 0 0 (+ 0) (8 ∷ []) ((+ 1) ∷ []) i64 0

envAfterNew : Env
envAfterNew = bindVar [] x (bind theObj owned)

machAfterNew : Machine
machAfterNew = machine [] ((theObj , cell live (counted 1) objBacking) ∷ [])
                       ((local 0 x , theObj) ∷ [])

atBranch : PState
atBranch = pstate bb0 [] envAfterNew machAfterNew

------------------------------------------------------------------------
-- The branch step, and the alias it creates.

envAfterBr : Env
envAfterBr with bindParams envAfterNew (p ∷ []) (x ∷ [])
... | just e  = e
... | nothing = envAfterNew

-- Both names denote the same object. Checked by computation, so it is not a
-- claim about how `bindParams` was written.
x-and-p-are-the-same-object : entityOf envAfterBr x ≡ entityOf envAfterBr p
x-and-p-are-the-same-object = refl

x-names-it : entityOf envAfterBr x ≡ just theObj
x-names-it = refl

p-names-it : entityOf envAfterBr p ≡ just theObj
p-names-it = refl

-- ⭐ THE SENTENCE THAT COULD NOT BE WRITTEN BEFORE.
one-object-two-names : Aliases envAfterBr x p
one-object-two-names = aliased theObj x-names-it p-names-it

-- And the step really is a step of the relation: the block is found, its
-- terminator IS this branch, and the parameters bind.
branch-is-a-step : prog ⊢ atBranch —→ pstate bb1 (drop p ∷ []) envAfterBr machAfterNew
branch-is-a-step = by-term (step-br refl refl refl refl)

------------------------------------------------------------------------
-- Why a pass that sees two entities is wrong.
--
-- After the branch there are TWO owned names and only ONE owner site was ever
-- occupied -- `bindParams` binds a name, it does not occupy a site, because a
-- block argument is not a new reference. So the owned-name count and the ghost
-- count DISAGREE, and that disagreement is the defect, now visible.

owned-names-after-branch : ownedCount envAfterBr theObj ≡ 2
owned-names-after-branch = refl

ghost-count-after-branch : logicalRC (sites machAfterNew) theObj ≡ 1
ghost-count-after-branch = refl

-- ⭐ Stated as the mismatch it is. A pass reading `ownedCount` as the number of
-- references to release will emit two drops for one reference -- which is the
-- over-release -- and a pass reading `logicalRC` while binding block arguments
-- as new references will emit one retain too few.
--
-- The fix is not in either count. It is that `br` must either NOT create an
-- owning name (the operand's name dies at the branch, so it is a MOVE), or must
-- occupy a site (so it is a DUP). The IR as written leaves it ambiguous, and
-- that ambiguity is exactly where the compiler's bug lives.
names-and-sites-disagree :
  ownedCount envAfterBr theObj ≡ suc (logicalRC (sites machAfterNew) theObj)
names-and-sites-disagree = refl

------------------------------------------------------------------------
-- The unwind edge exists, which is what families A and B needed.

padBlock : Block
padBlock = block 2 [] [] unwind

------------------------------------------------------------------------
-- Reachability composes.

two-steps : prog ⊢ atBranch —→* pstate bb1 (drop p ∷ []) envAfterBr machAfterNew
two-steps = more branch-is-a-step done

heap-unchanged : heap (mach (pstate bb1 (drop p ∷ []) envAfterBr machAfterNew))
                   ≡ heap (mach atBranch)
heap-unchanged = reachable-preserves-heap two-steps
