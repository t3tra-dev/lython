{-# OPTIONS --safe #-}

-- Owned NAMES against owner SITES, as a preserved property and a broken one.
--
-- `Proof.Program.Trace` showed the two counts disagreeing after a branch and
-- left it there: a `refl` about one state. That is a fact, not a localisation.
-- What makes it a localisation is the other half -- that the INSTRUCTIONS keep
-- them in step -- because then the disagreement is attributable to `br` rather
-- than being something the model does everywhere.
--
--   `new` / `dup`    bind an owned name AND occupy a site
--   `drop`           unbind one AND vacate one
--   `move`           swaps both
--   `borrow`         binds a name and occupies NOTHING -- and stays coherent,
--                    because `ownedCount` does not count borrows either
--   `br`             binds a name and occupies nothing -- and does NOT stay
--                    coherent, because the name it binds IS owned
--
-- That last line is the defect. A block argument is currently neither a move
-- (which would unbind the operand) nor a dup (which would occupy a site), and
-- the model now says which of the two it has to become.

module Proof.Program.Coherence where

open import Data.Bool using (Bool; true; false)
open import Data.Empty using (⊥; ⊥-elim)
open import Data.List using (List; []; _∷_)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc; _<_; s≤s; z≤n)
open import Data.Product using (_×_; _,_)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; sym; trans; subst)
open import Relation.Nullary using (¬_)

open import Proof.Memory.Lython using (LythonSig)
open import Proof.RC.Object using (ObjId; obj; sameObj)
open import Proof.RC.OwnerSite using (logicalRC)
open import Proof.RC.Machine LythonSig using (Machine; ghostRC; sites)
open import Proof.Program.Env using (Env; ownedCount)
open import Proof.Program.Step LythonSig using (PState; pstate; env; mach; _⊢_—→*_)
open import Proof.Lython.Invalid LythonSig
  using (Leaked; leaked; still-owned; unnamed; Invalidity; leak)
open import Proof.Lython.Detect LythonSig using (leaked?; Every; every-nil; every-cons)
open import Proof.Lython.Invalid LythonSig using (Valid)
open import Proof.Lython.Decide LythonSig
  using (AllChecksSilent; checks-silent; silence-means-valid)
-- The property and the leak theorem live in the parameterised module, so that
-- the general preservation proof and these concrete witnesses are about the same
-- predicate rather than two copies of it.
open import Proof.Program.Leak LythonSig
  using (NameSiteCoherent; Coherent; coherent-has-no-leaks;
         no-reachable-state-leaks)

import Proof.Program.Run  as R
import Proof.Program.Trace as T

------------------------------------------------------------------------
-- ⭐ Every state of the six-instruction run is coherent.
--
-- Each is a case split on `sameObj R.theObj o` and two `refl`s. That both
-- branches close is the content: the `true` branch says the counts agree for
-- the object in play, and the `false` branch says they agree at 0 for every
-- other object -- which is where a leak would show.

coherent₀ : Coherent R.s₀
coherent₀ o = refl

coherent₁ : Coherent R.s₁
coherent₁ o with sameObj R.theObj o
... | true  = refl
... | false = refl

coherent₂ : Coherent R.s₂
coherent₂ o with sameObj R.theObj o
... | true  = refl
... | false = refl

-- The borrow state. `z` is a third name for the object and the counts still
-- agree, because neither side counts a borrow. This is "a borrow costs nothing"
-- as a preservation statement rather than as an observation about one number.
coherent₃ : Coherent R.s₃
coherent₃ o with sameObj R.theObj o
... | true  = refl
... | false = refl

coherent₄ : Coherent R.s₄
coherent₄ o with sameObj R.theObj o
... | true  = refl
... | false = refl

coherent₅ : Coherent R.s₅
coherent₅ o with sameObj R.theObj o
... | true  = refl
... | false = refl

-- At the end: no owned name, no site. The borrowed `z` is still bound, and
-- coherence holds anyway -- which is exactly why coherence does NOT license the
-- reclaim that `Proof.Program.Run.reclaim-is-licensed` shows is accepted there.
-- Two different properties, and only one of them is about safety.
coherent₆ : Coherent R.s₆
coherent₆ o with sameObj R.theObj o
... | true  = refl
... | false = refl

------------------------------------------------------------------------
-- ⭐ And the branch keeps it.
--
-- This section used to say the opposite, and the change is the whole content of
-- the decision. With `br` binding the successor's parameter and LEAVING the
-- operand bound, `Proof.Program.Trace` reached a state with two owned names and
-- one site, and `branch-breaks-coherence` was a theorem.
--
-- A block argument is now a MOVE. The operand's name dies at the branch and its
-- site travels with it, so the two counts move together and the mismatch is not
-- reachable -- there is no state in this model where both names are owned.

branch-keeps-coherence : NameSiteCoherent T.envAfterBr T.machAfterBr
branch-keeps-coherence o with sameObj T.theObj o
... | true  = refl
... | false = refl

-- Read as the equation it is: one owned name, one owner site.
--
-- The two readings used to send a compiler writer to different places --
-- `ownedCount` is what a release-placement pass consults and `ghostRC` is what
-- the counter implements -- and the defect was that the branch made them
-- disagree. They agree now, so both readings give the same answer.
the-counts-match : ownedCount T.envAfterBr T.theObj
                     ≡ ghostRC T.machAfterBr T.theObj
the-counts-match = refl

------------------------------------------------------------------------
-- What these witnesses are FOR, now that the general theorem exists.
--
-- This section used to say "these are seven witnesses, not a preservation
-- theorem", and that was the honest reading: nothing proved
-- `Coherent s → f ⊢ s —→ t → Coherent t`, so leak-freedom held of the states
-- written down here and no others.
--
-- `Proof.Program.Leak.step-preserves-coherence` closes it, and these witnesses
-- keep a different job. They rule out the property being unsatisfiable -- a
-- preservation theorem for an empty predicate is free -- and they attribute a
-- break to ONE rule, which a theorem quantified over all steps cannot do.
--
-- ⭐ And `coherent₀` is what the general theorem needs as its hypothesis: the
-- run starts coherent, so `no-reachable-state-leaks` applies to every state it
-- reaches rather than to the seven listed above.

------------------------------------------------------------------------
-- ⭐ Coherence is exactly leak-freedom.
--
-- `coherent-has-no-leaks` now lives in `Proof.Program.Leak`, next to the
-- preservation theorem that makes it reach further than a list of states.

-- Every state of the run is leak-free -- and not because each was checked. This
-- goes through reachability: one coherent start, and the theorem covers the
-- whole run.
run-is-leak-free : ∀ o → ¬ Leaked (env R.s₆) (mach R.s₆) o
run-is-leak-free =
  no-reachable-state-leaks R.whole-block R.start-is-well-formed coherent₀

------------------------------------------------------------------------
-- ⭐ And a leak, exhibited.
--
-- The state is `Run.s₅`'s machine -- one owner site, counter 1 -- under
-- `Run.s₆`'s environment, in which the only surviving name is a BORROW. That is
-- precisely what an elided `drop x` produces: the name goes out of scope and
-- the site stays.
--
-- It is not reachable by the step relation, and that is the point: `step-drop`
-- removes the name and the site together, so the model cannot leak. The leak is
-- what a COMPILER produces when it emits the scope exit and not the release,
-- and this is the state to check for.
--
-- ⭐ "Not reachable" used to be a claim in this comment. `leak-is-unreachable`
-- below is the claim as a proof.

leakyEnv : Env
leakyEnv = env R.s₆

leakyMach : Machine
leakyMach = mach R.s₅

the-leak : Leaked leakyEnv leakyMach R.theObj
the-leak = leaked (s≤s z≤n) refl

-- The checker finds it.
leak-checker-fires : leaked? leakyEnv leakyMach R.theObj ≡ true
leak-checker-fires = refl

-- and is silent on the coherent state one step later, where the site went too.
leak-checker-silent : leaked? (env R.s₆) (mach R.s₆) R.theObj ≡ false
leak-checker-silent = refl

-- ⭐ So the leaky state is NOT coherent -- which is the same fact as the
-- theorem above, read the other way, and is what makes `NameSiteCoherent` worth
-- checking rather than merely stating.
leaky-state-is-incoherent : ¬ NameSiteCoherent leakyEnv leakyMach
leaky-state-is-incoherent coh = coherent-has-no-leaks leakyEnv leakyMach coh R.theObj the-leak

-- `Invalidity` gains its fourth constructor here.
leak-is-an-invalidity : Invalidity leakyEnv leakyMach
leak-is-an-invalidity = leak the-leak

------------------------------------------------------------------------
-- ⭐ AND THE LEAK IS UNREACHABLE.
--
-- Not "no path was found" and not "the rules look fine": if a path existed,
-- `no-reachable-state-leaks` applied to it would refute `the-leak`. This is the
-- statement the suite's leak stage was built to check, and it is now about the
-- step relation rather than about a list of states.
--
-- What it does NOT say, and the distinction is the useful part: it does not say
-- this compiler is leak-free. It says a leak cannot be produced by running
-- instructions, so a leak needs a BOUNDARY the relation does not have -- function
-- entry or scope exit. A leak is not a misplaced release, which is the opposite
-- of families A-E.

leakyState : PState
leakyState = pstate 0 R.bb0 [] leakyEnv leakyMach

leak-is-unreachable : ¬ (R.prog ⊢ R.s₀ —→* leakyState)
leak-is-unreachable r =
  no-reachable-state-leaks r R.start-is-well-formed coherent₀ R.theObj the-leak

------------------------------------------------------------------------
-- ⭐ `Valid`, established on a state the step relation reaches.
--
-- The four checks, run over the lists the state itself provides, and every one
-- silent. This is the first state in the development shown to satisfy `Valid` --
-- and it is `s₅`, in the middle of the run, rather than a state written to suit.

-- `s₂`: two owned names, two owner sites on ONE thread, a live counted object
-- and no borrows. Every check is silent.
validState : PState
validState = R.s₂

private
  quiet : AllChecksSilent (env validState) (mach validState)
  quiet = checks-silent
    (every-cons refl (every-cons refl every-nil))
    (every-cons refl (every-cons refl (every-cons refl (every-cons refl every-nil))))
    (every-cons refl (every-cons refl (every-cons refl (every-cons refl every-nil))))
    -- Both sites belong to thread 0, so nothing is shared and a plain refcount
    -- update is safe here. That is `needsAtomic?` earning its keep: the answer
    -- comes from the site map, with no escape analysis.
    (every-cons refl (every-cons refl (every-cons refl (every-cons refl every-nil))))

state-is-valid : Valid (env validState) (mach validState)
state-is-valid = silence-means-valid (env validState) (mach validState) quiet

-- ⭐ And the state one step later is NOT, because the move stranded the borrow.
-- The same four checks, and one of them fires.
next-state-is-invalid : Invalidity (env R.s₆) (mach R.s₆)
next-state-is-invalid = R.invalid-by-dangling-borrow

-- so `Valid` is a real constraint here rather than something every state meets.
next-state-is-not-valid : ¬ Valid (env R.s₆) (mach R.s₆)
next-state-is-not-valid v = v next-state-is-invalid
