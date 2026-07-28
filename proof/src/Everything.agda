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

-- Reference counting: owner sites, the ghost count the runtime counter has to
-- implement, py.incref / py.decref, and the invariant tying them together.
import Proof.RC.Object
import Proof.RC.OwnerSite
import Proof.RC.Machine
import Proof.RC.Ops
import Proof.RC.Invariant
import Proof.RC.Properties
import Proof.RC.Trace
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

-- The program layer: names distinct from allocations, control flow, unwind
-- edges, a step relation and reachability. This is where placement becomes
-- expressible -- six of the eleven defects the gap analysis found inexpressible
-- were placement defects.
import Proof.Program.Syntax
import Proof.Program.Env
import Proof.Program.Step
import Proof.Program.Ownership
import Proof.Program.Preservation

-- Concurrency: threads, a nondeterministic scheduler, happens-before and the
-- race predicate. No permission algebra yet, so no race-freedom theorem.
import Proof.Concurrent.Event
import Proof.Concurrent.Machine

-- Lython-specific invalidity: the handful of things THIS language forbids, with
-- decision procedures and soundness. Not a permission algebra.
import Proof.Lython.Invalid
import Proof.Lython.Detect

-- The instantiation, and a trace that exercises it. Both are load-bearing: the
-- modules above are parameterised over an element signature, and every theorem
-- in them holds vacuously for a signature that is never supplied.
import Proof.Memory.Lython
import Proof.Memory.Trace
