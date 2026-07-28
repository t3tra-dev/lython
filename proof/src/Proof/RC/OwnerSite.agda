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

open import Data.List using (List; []; _∷_; length; filter)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc; _≟_; _+_)
open import Data.Product using (_×_; _,_)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; cong)
open import Relation.Nullary using (Dec; yes; no; ¬_)

open import Proof.RC.Object using (ObjId; obj; _≟-obj_)

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

strongAt : SiteMap → OwnerSite → Maybe ObjId
strongAt []             _ = nothing
strongAt ((s , o) ∷ ss) t with s ≟-site t
... | yes _ = just o
... | no  _ = strongAt ss t

-- THE definition. The reference count of an object is how many owner sites hold
-- it -- an ordinary count over ghost state, with no runtime counter involved.
logicalRC : SiteMap → ObjId → ℕ
logicalRC []             _ = 0
logicalRC ((_ , p) ∷ ss) o with p ≟-obj o
... | yes _ = suc (logicalRC ss o)
... | no  _ = logicalRC ss o

------------------------------------------------------------------------
-- The three site operations, and what each does to the count.
--
-- These are the ONLY ways the map changes, which is what makes the counting
-- lemmas below exhaustive rather than merely true of the cases considered.

-- Taking a site that was free. Corresponds to py.incref's destination, or to
-- the destination of a move.
occupy : SiteMap → OwnerSite → ObjId → SiteMap
occupy ss s o = (s , o) ∷ ss

-- Releasing a site. Removes the FIRST binding for it, which is the right
-- semantics because `occupy` shadows: a site is occupied once at a time, and
-- the model's `strongAt` already reads the most recent.
vacate : SiteMap → OwnerSite → SiteMap
vacate []             _ = []
vacate ((s , o) ∷ ss) t with s ≟-site t
... | yes _ = ss
... | no  _ = (s , o) ∷ vacate ss t

------------------------------------------------------------------------
-- Counting lemmas.

occupy-same : ∀ (ss : SiteMap) (s : OwnerSite) (o : ObjId) →
              logicalRC (occupy ss s o) o ≡ suc (logicalRC ss o)
occupy-same ss s o with o ≟-obj o
... | yes _  = refl
... | no ¬p  = ⊥-elim (¬p refl)
  where open import Data.Empty using (⊥-elim)

occupy-other : ∀ (ss : SiteMap) (s : OwnerSite) (o p : ObjId) → ¬ (o ≡ p) →
               logicalRC (occupy ss s o) p ≡ logicalRC ss p
occupy-other ss s o p ne with o ≟-obj p
... | yes q = ⊥-elim (ne q)
  where open import Data.Empty using (⊥-elim)
... | no  _ = refl
