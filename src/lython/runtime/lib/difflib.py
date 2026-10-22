"""
Module difflib -- helpers for computing deltas between objects.

Lython port of CPython 3.14's Lib/difflib.py, restricted to the well-typed
statically compilable surface. It ships as SOURCE inside the compiler and
compiles with the program. Signatures follow typeshed stdlib/difflib.pyi,
the bodies are CPython's.

Function get_close_matches(word, possibilities, n=3, cutoff=0.6):
    Use SequenceMatcher to return list of the best "good enough" matches.

Function context_diff(a, b):
    For two lists of strings, return a delta in context diff format.

Function ndiff(a, b):
    Return a delta: the difference between `a` and `b` (lists of strings).

Function restore(delta, which):
    Return one of the two sequences that generated an ndiff delta.

Function unified_diff(a, b):
    For two lists of strings, return a delta in unified diff format.

Class SequenceMatcher:
    A flexible class for comparing pairs of sequences of any type.

Class Differ:
    For producing human-readable deltas from sequences of lines of text.

Deviations from CPython, pending language surface:
  - SequenceMatcher[T] keeps its sequences as `list[T]`: a field typed
    `Sequence[T]` (typeshed) has no concrete method evidence for the
    subscript. The constructor and set_seq1/set_seq2 take any `Sequence[T]`,
    a str included, and copy it, so `.a` / `.b` read back as lists and
    set_seq2 recomputes even when handed the same object again.
  - Differ.compare, ndiff, unified_diff, context_diff and restore take
    `list[str]` / `Iterable[str]` rather than `Sequence[str]`: a generator
    with a protocol-typed parameter is refused at lowering.
  - restore reads the whole delta when it is called, and raises ValueError
    for a bad `which` there rather than at the first next().
  - unified_diff and context_diff do not check their argument types at run
    time (CPython's _check_types): the annotations are enforced statically.
  - IS_LINE_JUNK has no `pat` argument (it is undocumented in CPython).
  - Differ's helpers (_dump, _plain_replace, _fancy_replace, _fancy_helper,
    _qformat) return lists where CPython's are generators, and compare()
    yields one opcode's lines at a time from them. The output is the same;
    a chain of generators delegating to generators is what the generator
    lowering does not handle here (docs/unittest-port.md, C-2).
  - HtmlDiff and diff_bytes are not provided.
  - get_close_matches orders with sorted() where CPython uses heapq.nlargest;
    the two are documented to agree.
"""

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import NamedTuple

__all__ = ['get_close_matches', 'ndiff', 'restore', 'SequenceMatcher',
           'Differ','IS_CHARACTER_JUNK', 'IS_LINE_JUNK', 'context_diff',
           'unified_diff', 'Match']

class Match(NamedTuple):
    a: int
    b: int
    size: int

_Opcode = tuple[str, int, int, int, int]

def _calculate_ratio(matches: int, length: int) -> float:
    if length != 0:
        return 2.0 * matches / length
    return 1.0

class SequenceMatcher[T]:

    """
    SequenceMatcher is a flexible class for comparing pairs of sequences of
    any type, so long as the sequence elements are hashable.  The basic
    algorithm predates, and is a little fancier than, an algorithm
    published in the late 1980's by Ratcliff and Obershelp under the
    hyperbolic name "gestalt pattern matching".  The basic idea is to find
    the longest contiguous matching subsequence that contains no "junk"
    elements (R-O doesn't address junk).  The same idea is then applied
    recursively to the pieces of the sequences to the left and to the right
    of the matching subsequence.  This does not yield minimal edit
    sequences, but does tend to yield matches that "look right" to people.
    """

    def __init__(self, isjunk: Callable[[T], bool] | None = None,
                 a: Sequence[T] = '', b: Sequence[T] = '',
                 autojunk: bool = True) -> None:
        """Construct a SequenceMatcher.

        Optional arg isjunk is None (the default), or a one-argument
        function that takes a sequence element and returns true iff the
        element is junk.  None is equivalent to passing "lambda x: 0", i.e.
        no elements are considered to be junk.

        Optional arg a is the first of two sequences to be compared.  By
        default, an empty string.  The elements of a must be hashable.  See
        also .set_seqs() and .set_seq1().

        Optional arg b is the second of two sequences to be compared.  By
        default, an empty string.  The elements of b must be hashable. See
        also .set_seqs() and .set_seq2().

        Optional arg autojunk should be set to False to disable the
        "automatic junk heuristic" that treats popular elements as junk
        (see module documentation for more information).
        """
        self.isjunk = isjunk
        self.a: list[T] = []
        self.b: list[T] = []
        self.autojunk = autojunk
        self.matching_blocks: list[Match] | None = None
        self.opcodes: list[_Opcode] | None = None
        self.fullbcount: dict[T, int] | None = None
        self.b2j: dict[T, list[int]] = {}
        self.bjunk: set[T] = set()
        self.bpopular: set[T] = set()
        self.set_seqs(a, b)

    def set_seqs(self, a: Sequence[T], b: Sequence[T]) -> None:
        """Set the two sequences to be compared."""
        self.set_seq1(a)
        self.set_seq2(b)

    def set_seq1(self, a: Sequence[T]) -> None:
        """Set the first sequence to be compared.

        The second sequence to be compared is not changed.
        """
        self.a = list(a)
        self.matching_blocks = None
        self.opcodes = None

    def set_seq2(self, b: Sequence[T]) -> None:
        """Set the second sequence to be compared.

        The first sequence to be compared is not changed.
        """
        self.b = list(b)
        self.matching_blocks = None
        self.opcodes = None
        self.fullbcount = None
        self.__chain_b()

    def __chain_b(self) -> None:
        # The first trick is to build b2j ignoring the possibility
        # of junk.  I.e., we don't call isjunk at all yet.  Throwing
        # out the junk later is much cheaper than building b2j "right"
        # from the start.
        b = self.b
        b2j: dict[T, list[int]] = {}
        self.b2j = b2j
        for i, elt in enumerate(b):
            indices = b2j.setdefault(elt, [])
            indices.append(i)

        # Purge junk elements
        junk: set[T] = set()
        self.bjunk = junk
        isjunk = self.isjunk
        if isjunk:
            for elt in b2j.keys():
                if isjunk(elt):
                    junk.add(elt)
            for elt in junk: # separate loop avoids separate list of keys
                del b2j[elt]

        # Purge popular elements that are not junk
        popular: set[T] = set()
        self.bpopular = popular
        n = len(b)
        if self.autojunk and n >= 200:
            ntest = n // 100 + 1
            for elt, idxs in b2j.items():
                if len(idxs) > ntest:
                    popular.add(elt)
            for elt in popular: # ditto; as fast for 1% deletion
                del b2j[elt]

    def find_longest_match(self, alo: int = 0, ahi: int | None = None,
                           blo: int = 0, bhi: int | None = None) -> Match:
        """Find longest matching block in a[alo:ahi] and b[blo:bhi].

        By default it will find the longest match in the entirety of a and b.

        If isjunk is not defined:

        Return (i,j,k) such that a[i:i+k] is equal to b[j:j+k], where
            alo <= i <= i+k <= ahi
            blo <= j <= j+k <= bhi
        and for all (i',j',k') meeting those conditions,
            k >= k'
            i <= i'
            and if i == i', j <= j'

        If isjunk is defined, first the longest matching block is
        determined as above, but with the additional restriction that no
        junk element appears in the block.  Then that block is extended as
        far as possible by matching (only) junk elements on both sides.

        If no blocks match, return (alo, blo, 0).
        """
        a, b, b2j, bjunk = self.a, self.b, self.b2j, self.bjunk
        if ahi is None:
            ahi = len(a)
        if bhi is None:
            bhi = len(b)
        besti, bestj, bestsize = alo, blo, 0
        # find longest junk-free match
        # during an iteration of the loop, j2len[j] = length of longest
        # junk-free match ending with a[i-1] and b[j]
        j2len: dict[int, int] = {}
        nothing: list[int] = []
        for i in range(alo, ahi):
            # look at all instances of a[i] in b; note that because
            # b2j has no junk keys, the loop is skipped if a[i] is junk
            newj2len: dict[int, int] = {}
            for j in b2j.get(a[i], nothing):
                # a[i] matches b[j]
                if j < blo:
                    continue
                if j >= bhi:
                    break
                k = j2len.get(j-1, 0) + 1
                newj2len[j] = k
                if k > bestsize:
                    besti, bestj, bestsize = i-k+1, j-k+1, k
            j2len = newj2len

        # Extend the best by non-junk elements on each end.  In particular,
        # "popular" non-junk elements aren't in b2j, which greatly speeds
        # the inner loop above, but also means "the best" match so far
        # doesn't contain any junk *or* popular non-junk elements.
        while besti > alo and bestj > blo and \
              not b[bestj-1] in bjunk and \
              a[besti-1] == b[bestj-1]:
            besti, bestj, bestsize = besti-1, bestj-1, bestsize+1
        while besti+bestsize < ahi and bestj+bestsize < bhi and \
              not b[bestj+bestsize] in bjunk and \
              a[besti+bestsize] == b[bestj+bestsize]:
            bestsize += 1

        # Now that we have a wholly interesting match (albeit possibly
        # empty!), we may as well suck up the matching junk on each
        # side of it too.
        while besti > alo and bestj > blo and \
              b[bestj-1] in bjunk and \
              a[besti-1] == b[bestj-1]:
            besti, bestj, bestsize = besti-1, bestj-1, bestsize+1
        while besti+bestsize < ahi and bestj+bestsize < bhi and \
              b[bestj+bestsize] in bjunk and \
              a[besti+bestsize] == b[bestj+bestsize]:
            bestsize = bestsize + 1

        return Match(besti, bestj, bestsize)

    def get_matching_blocks(self) -> list[Match]:
        """Return list of triples describing matching subsequences.

        Each triple is of the form (i, j, n), and means that
        a[i:i+n] == b[j:j+n].  The triples are monotonically increasing in
        i and in j.  It's also guaranteed that if (i, j, n) and (i', j', n')
        are adjacent triples in the list, and the second is not the last
        triple in the list, then i+n != i' or j+n != j'.  IOW, adjacent
        triples never describe adjacent equal blocks.

        The last triple is a dummy, (len(a), len(b), 0), and is the only
        triple with n==0.
        """
        cached = self.matching_blocks
        if cached is not None:
            return cached
        la, lb = len(self.a), len(self.b)

        # This is most naturally expressed as a recursive algorithm, but
        # at least one user bumped into extreme use cases that exceeded
        # the recursion limit on their box.  So, now we maintain a list
        # ('queue`) of blocks we still need to look at, and append partial
        # results to `matching_blocks` in a loop; the matches are sorted
        # at the end.
        queue = [(0, la, 0, lb)]
        matching_blocks: list[Match] = []
        while queue:
            alo, ahi, blo, bhi = queue.pop()
            x = self.find_longest_match(alo, ahi, blo, bhi)
            i, j, k = x
            # a[alo:i] vs b[blo:j] unknown
            # a[i:i+k] same as b[j:j+k]
            # a[i+k:ahi] vs b[j+k:bhi] unknown
            if k != 0: # if k is 0, there was no matching block
                matching_blocks.append(x)
                if alo < i and blo < j:
                    queue.append((alo, i, blo, j))
                if i+k < ahi and j+k < bhi:
                    queue.append((i+k, ahi, j+k, bhi))
        matching_blocks.sort()

        # It's possible that we have adjacent equal blocks in the
        # matching_blocks list now.  This code collapses them.
        i1 = j1 = k1 = 0
        non_adjacent: list[Match] = []
        for i2, j2, k2 in matching_blocks:
            # Is this block adjacent to i1, j1, k1?
            if i1 + k1 == i2 and j1 + k1 == j2:
                # Yes, so collapse them -- this just increases the length of
                # the first block by the length of the second, and the first
                # block so lengthened remains the block to compare against.
                k1 += k2
            else:
                # Not adjacent.  Remember the first block (k1==0 means it's
                # the dummy we started with), and make the second block the
                # new block to compare against.
                if k1 != 0:
                    non_adjacent.append(Match(i1, j1, k1))
                i1, j1, k1 = i2, j2, k2
        if k1 != 0:
            non_adjacent.append(Match(i1, j1, k1))

        non_adjacent.append(Match(la, lb, 0))
        self.matching_blocks = non_adjacent
        return non_adjacent

    def get_opcodes(self) -> list[_Opcode]:
        """Return list of 5-tuples describing how to turn a into b.

        Each tuple is of the form (tag, i1, i2, j1, j2).  The first tuple
        has i1 == j1 == 0, and remaining tuples have i1 == the i2 from the
        tuple preceding it, and likewise for j1 == the previous j2.

        The tags are strings, with these meanings:

        'replace':  a[i1:i2] should be replaced by b[j1:j2]
        'delete':   a[i1:i2] should be deleted.
                    Note that j1==j2 in this case.
        'insert':   b[j1:j2] should be inserted at a[i1:i1].
                    Note that i1==i2 in this case.
        'equal':    a[i1:i2] == b[j1:j2]
        """
        cached = self.opcodes
        if cached is not None:
            return cached
        i = j = 0
        answer: list[_Opcode] = []
        self.opcodes = answer
        for ai, bj, size in self.get_matching_blocks():
            # invariant:  we've pumped out correct diffs to change
            # a[:i] into b[:j], and the next matching block is
            # a[ai:ai+size] == b[bj:bj+size].  So we need to pump
            # out a diff to change a[i:ai] into b[j:bj], pump out
            # the matching block, and move (i,j) beyond the match
            tag = ''
            if i < ai and j < bj:
                tag = 'replace'
            elif i < ai:
                tag = 'delete'
            elif j < bj:
                tag = 'insert'
            if tag:
                answer.append( (tag, i, ai, j, bj) )
            i, j = ai+size, bj+size
            # the list of matching blocks is terminated by a
            # sentinel with size 0
            if size != 0:
                answer.append( ('equal', ai, i, bj, j) )
        return answer

    def get_grouped_opcodes(self, n: int = 3) -> Iterator[list[_Opcode]]:
        """ Isolate change clusters by eliminating ranges with no changes.

        Return a generator of groups with up to n lines of context.
        Each group is in the same format as returned by get_opcodes().
        """

        codes = list(self.get_opcodes())
        if not codes:
            codes = [("equal", 0, 1, 0, 1)]
        # Fixup leading and trailing groups if they show no changes.
        if codes[0][0] == 'equal':
            tag, i1, i2, j1, j2 = codes[0]
            codes[0] = tag, max(i1, i2-n), i2, max(j1, j2-n), j2
        if codes[-1][0] == 'equal':
            tag, i1, i2, j1, j2 = codes[-1]
            codes[-1] = tag, i1, min(i2, i1+n), j1, min(j2, j1+n)

        nn = n + n
        group: list[_Opcode] = []
        for tag, i1, i2, j1, j2 in codes:
            # End the current group and start a new one whenever
            # there is a large range with no changes.
            if tag == 'equal' and i2-i1 > nn:
                group.append((tag, i1, min(i2, i1+n), j1, min(j2, j1+n)))
                yield group
                group = []
                i1, j1 = max(i1, i2-n), max(j1, j2-n)
            group.append((tag, i1, i2, j1 ,j2))
        if group and not (len(group)==1 and group[0][0] == 'equal'):
            yield group

    def ratio(self) -> float:
        """Return a measure of the sequences' similarity (float in [0,1]).

        Where T is the total number of elements in both sequences, and
        M is the number of matches, this is 2.0*M / T.
        Note that this is 1 if the sequences are identical, and 0 if
        they have nothing in common.
        """

        matches = 0
        for triple in self.get_matching_blocks():
            matches += triple[-1]
        return _calculate_ratio(matches, len(self.a) + len(self.b))

    def quick_ratio(self) -> float:
        """Return an upper bound on ratio() relatively quickly.

        This isn't defined beyond that it is an upper bound on .ratio(), and
        is faster to compute.
        """

        # viewing a and b as multisets, set matches to the cardinality
        # of their intersection; this counts the number of matches
        # without regard to order, so is clearly an upper bound
        fullbcount = self.fullbcount
        if fullbcount is None:
            fullbcount = {}
            self.fullbcount = fullbcount
            for elt in self.b:
                fullbcount[elt] = fullbcount.get(elt, 0) + 1
        # avail[x] is the number of times x appears in 'b' less the
        # number of times we've seen it in 'a' so far ... kinda
        avail: dict[T, int] = {}
        matches = 0
        for elt in self.a:
            if elt in avail:
                numb = avail[elt]
            else:
                numb = fullbcount.get(elt, 0)
            avail[elt] = numb - 1
            if numb > 0:
                matches = matches + 1
        return _calculate_ratio(matches, len(self.a) + len(self.b))

    def real_quick_ratio(self) -> float:
        """Return an upper bound on ratio() very quickly.

        This isn't defined beyond that it is an upper bound on .ratio(), and
        is faster to compute than either .ratio() or .quick_ratio().
        """

        la, lb = len(self.a), len(self.b)
        # can't have more matches than the number of elements in the
        # shorter sequence
        return _calculate_ratio(min(la, lb), la + lb)


def get_close_matches(word: str, possibilities: Iterable[str], n: int = 3,
                      cutoff: float = 0.6) -> list[str]:
    """Use SequenceMatcher to return list of the best "good enough" matches.

    word is a sequence for which close matches are desired (typically a
    string).

    possibilities is a list of sequences against which to match word
    (typically a list of strings).

    Optional arg n (default 3) is the maximum number of close matches to
    return.  n must be > 0.

    Optional arg cutoff (default 0.6) is a float in [0, 1].  Possibilities
    that don't score at least that similar to word are ignored.

    The best (no more than n) matches among the possibilities are returned
    in a list, sorted by similarity score, most similar first.
    """

    if not n >  0:
        raise ValueError("n must be > 0: %r" % (n,))
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("cutoff must be in [0.0, 1.0]: %r" % (cutoff,))
    result: list[tuple[float, str]] = []
    s: SequenceMatcher[str] = SequenceMatcher()
    s.set_seq2(word)
    for x in possibilities:
        s.set_seq1(x)
        if s.real_quick_ratio() >= cutoff and \
           s.quick_ratio() >= cutoff and \
           s.ratio() >= cutoff:
            result.append((s.ratio(), x))

    # Move the best scorers to head of list
    result = sorted(result, reverse=True)[:n]
    # Strip scores for the best n matches
    return [x for score, x in result]


def _keep_original_ws(s: str, tag_s: str) -> str:
    """Replace whitespace with the original whitespace characters in `s`"""
    return ''.join([
        c if tag_c == " " and c.isspace() else tag_c
        for c, tag_c in zip(s, tag_s)
    ])


class Differ:
    r"""
    Differ is a class for comparing sequences of lines of text, and
    producing human-readable differences or deltas.  Differ uses
    SequenceMatcher both to compare sequences of lines, and to compare
    sequences of characters within similar (near-matching) lines.

    Each line of a Differ delta begins with a two-letter code:

        '- '    line unique to sequence 1
        '+ '    line unique to sequence 2
        '  '    line common to both sequences
        '? '    line not present in either input sequence

    Lines beginning with '? ' attempt to guide the eye to intraline
    differences, and were not present in either input sequence.  These lines
    can be confusing if the sequences contain tab characters.
    """

    def __init__(self, linejunk: Callable[[str], bool] | None = None,
                 charjunk: Callable[[str], bool] | None = None) -> None:
        """
        Construct a text differencer, with optional filters.

        The two optional keyword parameters are for filter functions:

        - `linejunk`: A function that should accept a single string argument,
          and return true iff the string is junk. The module-level function
          `IS_LINE_JUNK` may be used to filter out lines without visible
          characters, except for at most one splat ('#').  It is recommended
          to leave linejunk None; the underlying SequenceMatcher class has
          an adaptive notion of "noise" lines that's better than any static
          definition the author has ever been able to craft.

        - `charjunk`: A function that should accept a string of length 1. The
          module-level function `IS_CHARACTER_JUNK` may be used to filter out
          whitespace characters (a blank or tab; **note**: bad idea to include
          newline in this!).  Use of IS_CHARACTER_JUNK is recommended.
        """

        self.linejunk = linejunk
        self.charjunk = charjunk

    def compare(self, a: list[str], b: list[str]) -> Iterator[str]:
        r"""
        Compare two sequences of lines; generate the resulting delta.

        Each sequence must contain individual single-line strings ending with
        newlines. Such sequences can be obtained from the `readlines()` method
        of file-like objects.  The delta generated also consists of newline-
        terminated strings, ready to be printed as-is via the writelines()
        method of a file-like object.
        """

        cruncher = SequenceMatcher(self.linejunk, a, b)
        for tag, alo, ahi, blo, bhi in cruncher.get_opcodes():
            if tag == 'replace':
                g = self._fancy_replace(a, alo, ahi, b, blo, bhi)
            elif tag == 'delete':
                g = self._dump('-', a, alo, ahi)
            elif tag == 'insert':
                g = self._dump('+', b, blo, bhi)
            elif tag == 'equal':
                g = self._dump(' ', a, alo, ahi)
            else:
                raise ValueError('unknown tag %r' % (tag,))

            yield from g

    def _dump(self, tag: str, x: list[str], lo: int, hi: int) -> list[str]:
        """Generate comparison results for a same-tagged range."""
        return ['%s %s' % (tag, x[i]) for i in range(lo, hi)]

    def _plain_replace(self, a: list[str], alo: int, ahi: int,
                       b: list[str], blo: int, bhi: int) -> list[str]:
        assert alo < ahi and blo < bhi
        # dump the shorter block first -- reduces the burden on short-term
        # memory if the blocks are of very different sizes
        if bhi - blo < ahi - alo:
            first  = self._dump('+', b, blo, bhi)
            second = self._dump('-', a, alo, ahi)
        else:
            first  = self._dump('-', a, alo, ahi)
            second = self._dump('+', b, blo, bhi)

        return first + second

    def _fancy_replace(self, a: list[str], alo: int, ahi: int,
                       b: list[str], blo: int, bhi: int) -> list[str]:
        r"""
        When replacing one block of lines with another, search the blocks
        for *similar* lines; the best-matching pair (if any) is used as a
        synch point, and intraline difference marking is done on the
        similar pair. Lots of work, but often worth it.
        """
        # Don't synch up unless the lines have a similarity score above
        # cutoff. Previously only the smallest pair was handled here,
        # and if there are many pairs with the best ratio, recursion
        # could grow very deep, and runtime cubic. See:
        # https://github.com/python/cpython/issues/119105
        #
        # Later, more pathological cases prompted removing recursion
        # entirely.
        result: list[str] = []
        cutoff = 0.74999
        cruncher: SequenceMatcher[str] = SequenceMatcher(self.charjunk)
        WINDOW = 10
        best_i: int | None = None
        best_j: int | None = None
        # smallest indices not yet resolved
        # `blo + 0` rather than `blo`: the one int, carried into the loop both
        # as range()'s start and as this local, is refused by the ownership
        # verifier (docs/unittest-port.md, C-2).
        dump_i, dump_j = alo, blo + 0
        for j in range(blo, bhi):
            cruncher.set_seq2(b[j])
            # Search the corresponding i's within WINDOW for rhe highest
            # ratio greater than `cutoff`.
            aequiv = alo + (j - blo)
            arange = range(max(aequiv - WINDOW, dump_i),
                           min(aequiv + WINDOW + 1, ahi))
            if not arange: # likely exit if `a` is shorter than `b`
                break
            best_ratio = cutoff
            for i in arange:
                cruncher.set_seq1(a[i])
                # Ordering by cheapest to most expensive ratio is very
                # valuable, most often getting out early.
                if (cruncher.real_quick_ratio() > best_ratio
                      and cruncher.quick_ratio() > best_ratio
                      and cruncher.ratio() > best_ratio):
                    best_i, best_j, best_ratio = i, j, cruncher.ratio()

            if best_i is None or best_j is None:
                # found nothing to synch on yet - move to next j
                continue

            # pump out straight replace from before this synch pair
            result += self._fancy_helper(a, dump_i, best_i,
                                         b, dump_j, best_j)
            # do intraline marking on the synch pair
            aelt, belt = a[best_i], b[best_j]
            if aelt != belt:
                # pump out a '-', '?', '+', '?' quad for the synched lines
                atags = btags = ""
                cruncher.set_seqs(aelt, belt)
                for tag, ai1, ai2, bj1, bj2 in cruncher.get_opcodes():
                    la, lb = ai2 - ai1, bj2 - bj1
                    if tag == 'replace':
                        atags += '^' * la
                        btags += '^' * lb
                    elif tag == 'delete':
                        atags += '-' * la
                    elif tag == 'insert':
                        btags += '+' * lb
                    elif tag == 'equal':
                        atags += ' ' * la
                        btags += ' ' * lb
                    else:
                        raise ValueError('unknown tag %r' % (tag,))
                result += self._qformat(aelt, belt, atags, btags)
            else:
                # the synch pair is identical
                result.append('  ' + aelt)
            dump_i, dump_j = best_i + 1, best_j + 1
            best_i = best_j = None

        # pump out straight replace from after the last synch pair
        result += self._fancy_helper(a, dump_i, ahi,
                                     b, dump_j, bhi)
        return result

    def _fancy_helper(self, a: list[str], alo: int, ahi: int,
                      b: list[str], blo: int, bhi: int) -> list[str]:
        g: list[str] = []
        if alo < ahi:
            if blo < bhi:
                g = self._plain_replace(a, alo, ahi, b, blo, bhi)
            else:
                g = self._dump('-', a, alo, ahi)
        elif blo < bhi:
            g = self._dump('+', b, blo, bhi)
        return g

    def _qformat(self, aline: str, bline: str, atags: str,
                 btags: str) -> list[str]:
        r"""
        Format "?" output and deal with tabs.
        """
        atags = _keep_original_ws(aline, atags).rstrip()
        btags = _keep_original_ws(bline, btags).rstrip()

        lines = ["- " + aline]
        if atags:
            lines.append(f"? {atags}\n")

        lines.append("+ " + bline)
        if btags:
            lines.append(f"? {btags}\n")
        return lines

def IS_LINE_JUNK(line: str) -> bool:
    r"""
    Return True for ignorable line: if `line` is blank or contains a single '#'.
    """

    return line.strip() in '#'

def IS_CHARACTER_JUNK(ch: str, ws: str = " \t") -> bool:
    r"""
    Return True for ignorable character: iff `ch` is a space or tab.
    """

    return ch in ws


########################################################################
###  Unified Diff
########################################################################

def _format_range_unified(start: int, stop: int) -> str:
    'Convert range to the "ed" format'
    # Per the diff spec at http://www.unix.org/single_unix_specification/
    beginning = start + 1     # lines start numbering with one
    length = stop - start
    if length == 1:
        return '{}'.format(beginning)
    if length == 0:
        beginning -= 1        # empty ranges begin at line just before the range
    return '{},{}'.format(beginning, length)

def unified_diff(a: list[str], b: list[str], fromfile: str = '',
                 tofile: str = '', fromfiledate: str = '',
                 tofiledate: str = '', n: int = 3,
                 lineterm: str = '\n') -> Iterator[str]:
    r"""
    Compare two sequences of lines; generate the delta as a unified diff.

    Unified diffs are a compact way of showing line changes and a few
    lines of context.  The number of context lines is set by 'n' which
    defaults to three.

    By default, the diff control lines (those with ---, +++, or @@) are
    created with a trailing newline.

    For inputs that do not have trailing newlines, set the lineterm
    argument to "" so that the output will be uniformly newline free.

    The unidiff format normally has a header for filenames and modification
    times.  Any or all of these may be specified using strings for
    'fromfile', 'tofile', 'fromfiledate', and 'tofiledate'.
    """

    started = False
    matcher: SequenceMatcher[str] = SequenceMatcher(None, a, b)
    for group in matcher.get_grouped_opcodes(n):
        if not started:
            started = True
            fromdate = '\t{}'.format(fromfiledate) if fromfiledate else ''
            todate = '\t{}'.format(tofiledate) if tofiledate else ''
            yield '--- {}{}{}'.format(fromfile, fromdate, lineterm)
            yield '+++ {}{}{}'.format(tofile, todate, lineterm)

        first, last = group[0], group[-1]
        file1_range = _format_range_unified(first[1], last[2])
        file2_range = _format_range_unified(first[3], last[4])
        yield '@@ -{} +{} @@{}'.format(file1_range, file2_range, lineterm)

        for tag, i1, i2, j1, j2 in group:
            if tag == 'equal':
                for line in a[i1:i2]:
                    yield ' ' + line
                continue
            if tag in {'replace', 'delete'}:
                for line in a[i1:i2]:
                    yield '-' + line
            if tag in {'replace', 'insert'}:
                for line in b[j1:j2]:
                    yield '+' + line


########################################################################
###  Context Diff
########################################################################

def _format_range_context(start: int, stop: int) -> str:
    'Convert range to the "ed" format'
    # Per the diff spec at http://www.unix.org/single_unix_specification/
    beginning = start + 1     # lines start numbering with one
    length = stop - start
    if length == 0:
        beginning -= 1        # empty ranges begin at line just before the range
    if length <= 1:
        return '{}'.format(beginning)
    return '{},{}'.format(beginning, beginning + length - 1)

# See http://www.unix.org/single_unix_specification/
def context_diff(a: list[str], b: list[str], fromfile: str = '',
                 tofile: str = '', fromfiledate: str = '',
                 tofiledate: str = '', n: int = 3,
                 lineterm: str = '\n') -> Iterator[str]:
    r"""
    Compare two sequences of lines; generate the delta as a context diff.

    Context diffs are a compact way of showing line changes and a few
    lines of context.  The number of context lines is set by 'n' which
    defaults to three.

    By default, the diff control lines (those with *** or ---) are
    created with a trailing newline.

    For inputs that do not have trailing newlines, set the lineterm
    argument to "" so that the output will be uniformly newline free.

    The context diff format normally has a header for filenames and
    modification times.  Any or all of these may be specified using
    strings for 'fromfile', 'tofile', 'fromfiledate', and 'tofiledate'.
    If not specified, the strings default to blanks.
    """

    prefix = dict(insert='+ ', delete='- ', replace='! ', equal='  ')
    started = False
    matcher: SequenceMatcher[str] = SequenceMatcher(None, a, b)
    for group in matcher.get_grouped_opcodes(n):
        if not started:
            started = True
            fromdate = '\t{}'.format(fromfiledate) if fromfiledate else ''
            todate = '\t{}'.format(tofiledate) if tofiledate else ''
            yield '*** {}{}{}'.format(fromfile, fromdate, lineterm)
            yield '--- {}{}{}'.format(tofile, todate, lineterm)

        first, last = group[0], group[-1]
        yield '***************' + lineterm

        file1_range = _format_range_context(first[1], last[2])
        yield '*** {} ****{}'.format(file1_range, lineterm)

        if any([tag in {'replace', 'delete'} for tag, _, _, _, _ in group]):
            for tag, i1, i2, _, _ in group:
                if tag != 'insert':
                    for line in a[i1:i2]:
                        yield prefix[tag] + line

        file2_range = _format_range_context(first[3], last[4])
        yield '--- {} ----{}'.format(file2_range, lineterm)

        if any([tag in {'replace', 'insert'} for tag, _, _, _, _ in group]):
            for tag, _, _, j1, j2 in group:
                if tag != 'delete':
                    for line in b[j1:j2]:
                        yield prefix[tag] + line


def ndiff(a: list[str], b: list[str],
          linejunk: Callable[[str], bool] | None = None,
          charjunk: Callable[[str], bool] | None = IS_CHARACTER_JUNK) -> Iterator[str]:
    r"""
    Compare `a` and `b` (lists of strings); return a `Differ`-style delta.

    Optional keyword parameters `linejunk` and `charjunk` are for filter
    functions, or can be None:

    - linejunk: A function that should accept a single string argument and
      return true iff the string is junk.  The default is None, and is
      recommended; the underlying SequenceMatcher class has an adaptive
      notion of "noise" lines.

    - charjunk: A function that accepts a character (string of length
      1), and returns true iff the character is junk. The default is
      the module-level function IS_CHARACTER_JUNK, which filters out
      whitespace characters (a blank or tab; note: it's a bad idea to
      include newline in this!).
    """
    return Differ(linejunk, charjunk).compare(a, b)


def restore(delta: Iterable[str], which: int) -> Iterator[str]:
    r"""
    Generate one of the two sequences that generated a delta.

    Given a `delta` produced by `Differ.compare()` or `ndiff()`, extract
    lines originating from file 1 or 2 (parameter `which`), stripping off line
    prefixes.
    """
    if which == 1:
        tag = "- "
    elif which == 2:
        tag = "+ "
    else:
        raise ValueError('unknown delta choice (must be 1 or 2): %r'
                           % which) from None
    return _restore(list(delta), tag)


def _restore(delta: list[str], tag: str) -> Iterator[str]:
    # restore()'s generator, over the delta read into a list: a generator
    # with an `Iterable` parameter has no frame lane for it.
    prefixes = ("  ", tag)
    for line in delta:
        if line[:2] in prefixes:
            yield line[2:]
