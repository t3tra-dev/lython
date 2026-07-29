{-# OPTIONS --safe #-}

-- The build target. `make` typechecks this and therefore everything below it.
--
-- A module that is not reachable from here is not checked by `make`, and a
-- proof nobody checks is a comment. This file is the analogue of the compiler
-- suite's layer manifest: membership is the thing that makes a check run at all.

module Everything where

-- Shared vocabulary.
import Proof.Prelude

-- The memory model: identity, storage, descriptors.
import Proof.Memory.Fault
import Proof.Memory.Byte
import Proof.Memory.Element
import Proof.Memory.Index
import Proof.Memory.Heap
import Proof.Memory.Descriptor
import Proof.Memory.Resolve
import Proof.Memory.Properties

-- The memref dialect, transcribed op by op, and realloc.
import Proof.MemRef.Dialect
import Proof.MemRef.Realloc

-- Quantitative type theory's multiplicities, and the reference modes they do
-- not determine.
import Proof.QTT.Quantity
import Proof.QTT.Trace

-- Reference counting: owner sites, the ghost count the runtime counter has to
-- implement, py.incref / py.decref, and the invariant tying them together.
import Proof.RC.Object
import Proof.RC.OwnerSite
import Proof.RC.Machine
import Proof.RC.Ops
import Proof.RC.Invariant
import Proof.RC.Properties
import Proof.RC.Trace
-- Machines that SATISFY the invariant. Without these `WFRC` is a record nobody
-- has built, and every theorem conditional on it is vacuous.
import Proof.RC.WellFormed
-- Aggregates: field paths as a judgment, and the multiplicity an omitted
-- aggregate release leaks.
import Proof.RC.Aggregate
import Proof.Object.Trace
import Proof.Program.Trace
import Proof.Lython.Trace

-- The one-lane object: a redesign, not a transcription. An object reference is
-- ONE descriptor and every field is an index into it, with the mutable part in
-- a separate buffer the box names by identity.
import Proof.Object.Word
import Proof.Object.WordSig
import Proof.Object.Layout
import Proof.Object.Box
import Proof.Object.Ops
import Proof.Object.Coherence
-- The shape witness bundled with its descriptor, so it cannot be mis-paired.
import Proof.Object.Shaped

-- The program layer: names distinct from allocations, control flow, unwind
-- edges, a step relation and reachability. This is where placement becomes
-- expressible -- six of the eleven defects the gap analysis found inexpressible
-- were placement defects.
import Proof.Program.Syntax
import Proof.Program.Env
import Proof.Program.Step
import Proof.Program.Ownership
import Proof.Program.Preservation
-- What the IR RECORDS against what is TRUE. A pass reads attributes, not the
-- semantics, so "ownership taken and not recorded" is a defect no care inside
-- the pass can reach -- and it was inexpressible while `mode` was the only
-- notion of ownership in the model.
import Proof.Program.Recorded
-- A leak, tied to reachability. Coherence -- owned names equal owner sites -- is
-- preserved by every rule, so no sequence of instructions can produce a leak;
-- every leak is therefore a MISSING operation at scope exit rather than a
-- misplaced one.
import Proof.Program.Leak
-- Every instruction rule, taken as an actual step. Proof.Program.Trace derives
-- only terminator steps, so without this the five rules the refcount story
-- rests on had never been applied.
import Proof.Program.Run
-- Owned names against owner sites: preserved by every instruction, broken by
-- `br`. The half that turns "the counts disagree" into an attribution.
import Proof.Program.Coherence

-- Concurrency: threads, a nondeterministic scheduler, happens-before and the
-- race predicate. No permission algebra yet, so no race-freedom theorem.
import Proof.Concurrent.Event
import Proof.Concurrent.Machine
import Proof.Concurrent.Trace
-- Race freedom for the races this IR can have: no permission algebra needed,
-- because the only conflicting traffic is the refcount word.
import Proof.Concurrent.RaceFree

-- Lython-specific invalidity: the handful of things THIS language forbids, with
-- decision procedures and soundness. Not a permission algebra.
import Proof.Lython.Invalid
import Proof.Lython.Detect
-- `Valid`, decided: all four invalidities have sound and complete procedures,
-- and silence on all of them means no `Invalidity` is derivable.
import Proof.Lython.Decide

-- The instantiation, and a trace that exercises it. Both are load-bearing: the
-- modules above are parameterised over an element signature, and every theorem
-- in them holds vacuously for a signature that is never supplied.
import Proof.Memory.Lython
import Proof.Memory.Trace
