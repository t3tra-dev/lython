{-# OPTIONS --safe #-}

-- Where an owning reference can live, and the count that follows from it.
--
-- The central move of the whole refcount layer:
--
--   the reference count is not a number the runtime maintains and the proof
--   trusts. It is the NUMBER OF OWNER SITES holding the object, computed from
--   ghost state -- and `counted-exact` in Proof.RC.Invariant is the obligation
--   that the runtime's counter implements it.
--
-- Stated the other way round -- "the count is whatever the counter says" --
-- there is nothing to violate, and no incref/decref placement can be wrong.

module Proof.RC.OwnerSite where

open import Data.Bool using (Bool; true; false; if_then_else_; _∧_)
open import Data.Empty using (⊥; ⊥-elim)
open import Data.List using (List; []; _∷_; length; filter)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc; _≟_; _+_)
open import Data.Nat.Base using (_≡ᵇ_)
open import Data.Product using (_×_; _,_)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; cong)
open import Relation.Nullary using (Dec; yes; no; ¬_)

open import Proof.RC.Object using (ObjId; obj; _≟-obj_; sameObj; sameObj-refl;
  sameObj-sound; ≡ᵇ-refl)

ThreadId LocalSlot FieldId GlobalId QueueId Ticket TempId : Set
ThreadId  = ℕ
LocalSlot = ℕ
FieldId   = ℕ
GlobalId  = ℕ
QueueId   = ℕ
Ticket    = ℕ
TempId    = ℕ

-- Every place an owning reference can be. The list is not decoration: an owner
-- site the model forgets is a reference the count does not include, and the
-- invariant would then be provable about a machine that leaks.
--
-- `temp` is the one that is easy to leave out and is exactly where this
-- compiler's defects have lived: the transient reference held while a field is
-- being updated, or while a value is in flight into a container literal.
data OwnerSite : Set where
  local  : ThreadId → LocalSlot → OwnerSite
  field′ : ObjId → FieldId → OwnerSite
  global : GlobalId → OwnerSite
  queue  : QueueId → Ticket → OwnerSite
  temp   : ThreadId → TempId → OwnerSite

_≟-site_ : (s t : OwnerSite) → Dec (s ≡ t)
local t s   ≟-site local t' s'  with t ≟ t' | s ≟ s'
... | yes refl | yes refl = yes refl
... | no ¬p    | _        = no λ { refl → ¬p refl }
... | _        | no ¬q    = no λ { refl → ¬q refl }
field′ o f  ≟-site field′ o' f' with o ≟-obj o' | f ≟ f'
... | yes refl | yes refl = yes refl
... | no ¬p    | _        = no λ { refl → ¬p refl }
... | _        | no ¬q    = no λ { refl → ¬q refl }
global g    ≟-site global g'    with g ≟ g'
... | yes refl = yes refl
... | no ¬p    = no λ { refl → ¬p refl }
queue q k   ≟-site queue q' k'  with q ≟ q' | k ≟ k'
... | yes refl | yes refl = yes refl
... | no ¬p    | _        = no λ { refl → ¬p refl }
... | _        | no ¬q    = no λ { refl → ¬q refl }
temp t i    ≟-site temp t' i'   with t ≟ t' | i ≟ i'
... | yes refl | yes refl = yes refl
... | no ¬p    | _        = no λ { refl → ¬p refl }
... | _        | no ¬q    = no λ { refl → ¬q refl }
local _ _   ≟-site field′ _ _ = no λ ()
local _ _   ≟-site global _   = no λ ()
local _ _   ≟-site queue _ _  = no λ ()
local _ _   ≟-site temp _ _   = no λ ()
field′ _ _  ≟-site local _ _  = no λ ()
field′ _ _  ≟-site global _   = no λ ()
field′ _ _  ≟-site queue _ _  = no λ ()
field′ _ _  ≟-site temp _ _   = no λ ()
global _    ≟-site local _ _  = no λ ()
global _    ≟-site field′ _ _ = no λ ()
global _    ≟-site queue _ _  = no λ ()
global _    ≟-site temp _ _   = no λ ()
queue _ _   ≟-site local _ _  = no λ ()
queue _ _   ≟-site field′ _ _ = no λ ()
queue _ _   ≟-site global _   = no λ ()
queue _ _   ≟-site temp _ _   = no λ ()
temp _ _    ≟-site local _ _  = no λ ()
temp _ _    ≟-site field′ _ _ = no λ ()
temp _ _    ≟-site global _   = no λ ()
temp _ _    ≟-site queue _ _  = no λ ()

------------------------------------------------------------------------
-- The site map: which object, if any, each occupied site holds.
--
-- An association list rather than a function, so that `logicalRC` can COUNT it.
-- As a function `OwnerSite → Maybe ObjId` the count would need the domain to be
-- finite and enumerable, which is extra structure carried for no gain.

SiteMap : Set
SiteMap = List (OwnerSite × ObjId)

-- Boolean, for the same reason the object table is: `with s ≟-site t` compiles
-- to an auxiliary function, and then `vacate` and `logicalRC` branch on terms a
-- goal cannot abstract together. The counting lemmas below are unprovable
-- otherwise, and they are what WFRC preservation rests on.
sameSite : OwnerSite → OwnerSite → Bool
sameSite (local t s)  (local t' s')  = (t ≡ᵇ t') ∧ (s ≡ᵇ s')
sameSite (field′ o f) (field′ o' f') = sameObj o o' ∧ (f ≡ᵇ f')
sameSite (global g)   (global g')    = g ≡ᵇ g'
sameSite (queue q k)  (queue q' k')  = (q ≡ᵇ q') ∧ (k ≡ᵇ k')
sameSite (temp t i)   (temp t' i')   = (t ≡ᵇ t') ∧ (i ≡ᵇ i')
sameSite _            _              = false

sameSite-refl : ∀ s → sameSite s s ≡ true
sameSite-refl (local t s)  rewrite ≡ᵇ-refl t | ≡ᵇ-refl s = refl
sameSite-refl (field′ o f) rewrite sameObj-refl o | ≡ᵇ-refl f = refl
sameSite-refl (global g)   rewrite ≡ᵇ-refl g = refl
sameSite-refl (queue q k)  rewrite ≡ᵇ-refl q | ≡ᵇ-refl k = refl
sameSite-refl (temp t i)   rewrite ≡ᵇ-refl t | ≡ᵇ-refl i = refl

strongAt : SiteMap → OwnerSite → Maybe ObjId
strongAt []             _ = nothing
strongAt ((s , o) ∷ ss) t = if sameSite s t then just o else strongAt ss t

-- THE definition. The reference count of an object is how many owner sites hold
-- it -- an ordinary count over ghost state, with no runtime counter involved.
logicalRC : SiteMap → ObjId → ℕ
logicalRC []             _ = 0
logicalRC ((_ , p) ∷ ss) o = if sameObj p o then suc (logicalRC ss o) else logicalRC ss o

------------------------------------------------------------------------
-- The three site operations, and what each does to the count.

occupy : SiteMap → OwnerSite → ObjId → SiteMap
occupy ss s o = (s , o) ∷ ss

vacate : SiteMap → OwnerSite → SiteMap
vacate []             _ = []
vacate ((s , o) ∷ ss) t = if sameSite s t then ss else (s , o) ∷ vacate ss t

------------------------------------------------------------------------
-- Counting lemmas.

occupy-same : ∀ (ss : SiteMap) (s : OwnerSite) (o : ObjId) →
              logicalRC (occupy ss s o) o ≡ suc (logicalRC ss o)
occupy-same ss s o rewrite sameObj-refl o = refl

occupy-other : ∀ (ss : SiteMap) (s : OwnerSite) (o p : ObjId) → sameObj o p ≡ false →
               logicalRC (occupy ss s o) p ≡ logicalRC ss p
-- Stated over the BOOLEAN rather than over the proposition, because that is
-- what `logicalRC` branches on. Callers turn a disequality into it with
-- `sameObj-sound` contraposed.
occupy-other ss s o p ne rewrite ne = refl

-- ⭐ Vacating a site that HOLDS the object drops the count by exactly one.
--
-- This is the lemma WFRC preservation for `drop` needs, and the reason the site
-- map had to move to a boolean test: `strongAt` and `vacate` branch on the same
-- decision, and with `_≟-site_` they branch on two different auxiliary
-- functions that no single `with` abstracts.
vacate-holder : ∀ (ss : SiteMap) (s : OwnerSite) (o : ObjId) →
                strongAt ss s ≡ just o →
                logicalRC ss o ≡ suc (logicalRC (vacate ss s) o)
vacate-holder [] s o ()
vacate-holder ((t , p) ∷ ss) s o held with sameSite t s
... | true  = helper (just-inj held)
  where
    just-inj : ∀ {A : Set} {x y : A} → just x ≡ just y → x ≡ y
    just-inj refl = refl
    helper : p ≡ o → logicalRC ((t , p) ∷ ss) o ≡ suc (logicalRC ss o)
    helper refl rewrite sameObj-refl p = refl
... | false with sameObj p o
...   | true  = cong suc (vacate-holder ss s o held)
...   | false = vacate-holder ss s o held

-- And vacating a site that does NOT hold the object leaves its count alone.
-- Without this, `move` -- which vacates one site and occupies another -- could
-- not be shown to preserve anything.
vacate-other : ∀ (ss : SiteMap) (s : OwnerSite) (o : ObjId) →
               strongAt ss s ≡ nothing →
               logicalRC (vacate ss s) o ≡ logicalRC ss o
vacate-other [] s o _ = refl
vacate-other ((t , p) ∷ ss) s o miss with sameSite t s
... | true  = ⊥-elim (bad miss)
  where
    open import Data.Empty using (⊥-elim)
    bad : just p ≡ nothing → ⊥
    bad ()
    open import Data.Empty using (⊥)
... | false with sameObj p o
...   | true  = cong suc (vacate-other ss s o miss)
...   | false = vacate-other ss s o miss
