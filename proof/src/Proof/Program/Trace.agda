{-# OPTIONS --safe #-}

-- The shipped SIGSEGV's shape, and the IR decision that removes it.
--
-- This module used to be the answer to the gap analysis. The defect's root was
--
--     one allocation, TWO SSA NAMES, and the ownership machinery treats that as
--     two entities
--
-- and the finding was that no sentence in the model could say it. The sentence
-- was written here, on this program:
--
--     bb0(): alloc x ; init x ; br bb1(x)
--     bb1(p): drop p ; ...
--
-- and it said: after the branch `x` and `p` are two owned names of one entity
-- while only one owner site was ever occupied, so the owned-name count and the
-- ghost count DISAGREE. A pass reading the first as "references to release"
-- emits two drops for one reference.
--
-- ⭐ A BLOCK ARGUMENT IS NOW A MOVE. The operand's name dies at the branch and
-- its owner site moves to the parameter. The disagreement is not patched over,
-- it is unreachable: `moveArgs` unbinds `x`, so no state has both names owned.
--
-- What this module shows now is the two halves of that: the branch really does
-- move, and the counts really do agree afterwards.

module Proof.Program.Trace where

open import Data.List using (List; []; _∷_)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc)
open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Relation.Binary.PropositionalEquality using (_≡_; refl)
open import Relation.Nullary using (¬_)

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
open import Proof.Program.Preservation LythonSig using (block-arguments-are-free)

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
block0 = block bb0 [] (alloc x 7 ∷ init x ∷ []) (br bb1 (x ∷ []))

-- bb1 takes a PARAMETER. This is where the second name used to be created.
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
machAfterNew = machine [] ((theObj , cell live (counted 1) objBacking 0) ∷ [])
                       ((local 0 x , theObj) ∷ [])

atBranch : PState
atBranch = pstate 0 bb0 [] envAfterNew machAfterNew

------------------------------------------------------------------------
-- The branch, as a move.

moved : Maybe (Env × SiteMap)
moved = moveArgs 0 (env-and-sites envAfterNew machAfterNew) (p ∷ []) (x ∷ [])

envAfterBr : Env
envAfterBr with moved
... | just r  = proj₁ r
... | nothing = envAfterNew

sitesAfterBr : SiteMap
sitesAfterBr with moved
... | just r  = proj₂ r
... | nothing = sites machAfterNew

machAfterBr : Machine
machAfterBr = afterArgs machAfterNew sitesAfterBr

-- ⭐ The parameter denotes the object ...
p-names-it : entityOf envAfterBr p ≡ just theObj
p-names-it = refl

-- ... and the operand does NOT. This is the equation the whole decision is
-- about: a pass that still sees `x` after the branch places a release for it,
-- and that release is the over-release the compiler shipped.
x-is-gone : entityOf envAfterBr x ≡ nothing
x-is-gone = refl

-- and they are therefore NOT aliases. The sentence the model could not write
-- has become one the model can refute.
no-longer-aliased : ¬ Aliases envAfterBr x p
no-longer-aliased al with x-holds-it al
... | ()

------------------------------------------------------------------------
-- The step.

branch-term : prog ⊢ atBranch —→ₜ pstate 0 bb1 (drop p ∷ []) envAfterBr machAfterBr
branch-term = step-br refl refl refl refl

branch-is-a-step : prog ⊢ atBranch —→ pstate 0 bb1 (drop p ∷ []) envAfterBr machAfterBr
branch-is-a-step = by-term branch-term

------------------------------------------------------------------------
-- ⭐ The counts agree.
--
-- One owned name, one owner site, and a counter nobody touched. All three are
-- checked separately, because the defect was two of them disagreeing and a
-- single number cannot show that.

owned-names-after-branch : ownedCount envAfterBr theObj ≡ 1
owned-names-after-branch = refl

ghost-count-after-branch : logicalRC (sites machAfterBr) theObj ≡ 1
ghost-count-after-branch = refl

counts-agree : ownedCount envAfterBr theObj ≡ logicalRC (sites machAfterBr) theObj
counts-agree = refl

-- The site MOVED rather than being added: it is the parameter's now, and the
-- operand's is gone.
site-relocated : sites machAfterBr ≡ (local 0 p , theObj) ∷ []
site-relocated = refl

-- ⭐ And the counter did not move. A block argument costs nothing -- which is
-- the reason to make it a move rather than a dup: a loop-carried value would
-- otherwise pay a retain/release pair per turn for a reference that never went
-- anywhere.
counter-untouched : countOf machAfterBr theObj ≡ countOf machAfterNew theObj
counter-untouched = refl

objects-untouched : objects machAfterBr ≡ objects machAfterNew
objects-untouched = block-arguments-are-free branch-term

------------------------------------------------------------------------
-- The unwind edge exists, which is what families A and B needed.

padBlock : Block
padBlock = block 2 [] [] unwind

------------------------------------------------------------------------
-- Reachability composes, and the heap is untouched.

two-steps : prog ⊢ atBranch —→* pstate 0 bb1 (drop p ∷ []) envAfterBr machAfterBr
two-steps = more branch-is-a-step done

heap-unchanged : heap (mach (pstate 0 bb1 (drop p ∷ []) envAfterBr machAfterBr))
                   ≡ heap (mach atBranch)
heap-unchanged = reachable-preserves-heap two-steps
