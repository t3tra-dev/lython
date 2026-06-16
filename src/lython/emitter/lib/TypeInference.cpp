#include "TypeInference.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <deque>
#include <utility>

namespace lython::emitter::typing {
namespace {

std::string describeReason(const std::optional<Reason> &reason) {
  if (!reason || reason->message.empty())
    return {};
  return ": " + reason->message;
}

std::set<std::uint64_t> setDifference(const std::set<std::uint64_t> &lhs,
                                      const std::set<std::uint64_t> &rhs) {
  std::set<std::uint64_t> result;
  for (std::uint64_t value : lhs)
    if (!rhs.count(value))
      result.insert(value);
  return result;
}

void mergeInto(std::set<std::uint64_t> &target,
               const std::set<std::uint64_t> &source) {
  target.insert(source.begin(), source.end());
}

std::optional<Term> varargElementType(const Term &vararg) {
  if (vararg.kind != Term::Kind::Con || vararg.name != "tuple" ||
      vararg.args.size() != 1)
    return std::nullopt;
  return vararg.args.front();
}

bool isObjectTop(const Term &term) {
  return term.kind == Term::Kind::Con &&
         (term.name == "object" || term.name == "!py.object");
}

} // namespace

Term Term::var(std::uint64_t id, llvm::StringRef name) {
  Term term;
  term.kind = Kind::Var;
  term.id = id;
  term.name = name.str();
  return term;
}

Term Term::con(llvm::StringRef name, llvm::ArrayRef<Term> args) {
  Term term;
  term.kind = Kind::Con;
  term.name = name.str();
  term.args.assign(args.begin(), args.end());
  return term;
}

Term Term::func(llvm::ArrayRef<Term> args, llvm::ArrayRef<Term> results,
                llvm::ArrayRef<Term> kwonly,
                std::optional<unsigned> positionalOnlyCount) {
  return func(args, results, kwonly, llvm::ArrayRef<Term>{},
              llvm::ArrayRef<Term>{}, positionalOnlyCount);
}

Term Term::func(llvm::ArrayRef<Term> args, llvm::ArrayRef<Term> results,
                llvm::ArrayRef<Term> kwonly, llvm::ArrayRef<Term> vararg,
                llvm::ArrayRef<Term> kwarg,
                std::optional<unsigned> positionalOnlyCount) {
  Term term;
  term.kind = Kind::Func;
  term.args.assign(args.begin(), args.end());
  term.results.assign(results.begin(), results.end());
  term.kwonly.assign(kwonly.begin(), kwonly.end());
  term.vararg.assign(vararg.begin(), vararg.end());
  term.kwarg.assign(kwarg.begin(), kwarg.end());
  term.positionalOnlyCount = positionalOnlyCount;
  return term;
}

std::string Term::display() const {
  if (kind == Kind::Var) {
    if (!name.empty())
      return name;
    return "t" + std::to_string(id);
  }

  auto joinTerms = [](llvm::ArrayRef<Term> terms) {
    std::string storage;
    llvm::raw_string_ostream os(storage);
    llvm::interleaveComma(terms, os,
                          [&](const Term &term) { os << term.display(); });
    return storage;
  };

  if (kind == Kind::Con) {
    if (args.empty())
      return name;
    return name + "<" + joinTerms(args) + ">";
  }

  std::string result = "func<[" + joinTerms(args) + "]";
  if (positionalOnlyCount && *positionalOnlyCount != 0)
    result += ", posonly = " + std::to_string(*positionalOnlyCount);
  if (!kwonly.empty())
    result += ", kwonly = [" + joinTerms(kwonly) + "]";
  if (!vararg.empty())
    result += ", vararg = " + vararg.front().display();
  if (!kwarg.empty())
    result += ", kwarg = " + kwarg.front().display();
  result += " -> [" + joinTerms(results) + "]>";
  return result;
}

bool operator==(const Term &lhs, const Term &rhs) {
  return lhs.kind == rhs.kind && lhs.id == rhs.id && lhs.name == rhs.name &&
         lhs.args == rhs.args && lhs.kwonly == rhs.kwonly &&
         lhs.vararg == rhs.vararg && lhs.kwarg == rhs.kwarg &&
         lhs.results == rhs.results &&
         lhs.positionalOnlyCount == rhs.positionalOnlyCount;
}

bool operator!=(const Term &lhs, const Term &rhs) { return !(lhs == rhs); }

std::optional<Term> Oracle::attribute(const Term &, llvm::StringRef) const {
  return std::nullopt;
}

std::optional<Term> Oracle::call(const Term &callee,
                                 llvm::ArrayRef<Term> args) const {
  if (callee.kind != Term::Kind::Func || callee.results.size() != 1 ||
      !callee.kwonly.empty())
    return std::nullopt;
  if (callee.vararg.empty() && callee.args.size() != args.size())
    return std::nullopt;
  if (!callee.vararg.empty() && callee.args.size() > args.size())
    return std::nullopt;
  return callee.results.front();
}

std::optional<Term> Oracle::subscript(const Term &container,
                                      const Term &) const {
  if (container.kind != Term::Kind::Con)
    return std::nullopt;
  if (container.name == "list" && container.args.size() == 1)
    return container.args.front();
  if (container.name == "dict" && container.args.size() == 2)
    return container.args[1];
  if (container.name == "tuple" && container.args.size() == 1)
    return container.args.front();
  return std::nullopt;
}

std::optional<Term> Oracle::awaitable(const Term &awaitable) const {
  if (awaitable.kind != Term::Kind::Con)
    return std::nullopt;
  if (awaitable.args.size() == 1 && (awaitable.name == "async.value" ||
                                     awaitable.name == "protocol:Awaitable" ||
                                     awaitable.name == "protocol:Future" ||
                                     awaitable.name == "protocol:Task"))
    return awaitable.args.front();
  if (awaitable.name == "protocol:Coroutine" && awaitable.args.size() == 3)
    return awaitable.args[2];
  return std::nullopt;
}

void Environment::push() { scopes.emplace_back(); }

void Environment::pop() {
  if (scopes.size() == 1)
    throw Error("cannot pop root type environment scope");
  scopes.pop_back();
}

void Environment::bind(llvm::StringRef name, Scheme scheme) {
  scopes.back()[name.str()] = std::move(scheme);
}

const Scheme &Environment::lookup(llvm::StringRef name) const {
  std::string key = name.str();
  for (const auto &scope : llvm::reverse(scopes)) {
    auto found = scope.find(key);
    if (found != scope.end())
      return found->second;
  }
  throw Error("unresolved type binding '" + key + "'");
}

std::set<std::uint64_t> Environment::freeVars() const {
  std::set<std::uint64_t> result;
  for (const auto &scope : scopes) {
    for (const auto &[name, scheme] : scope) {
      (void)name;
      std::set<std::uint64_t> bodyFree = typing::freeVars(scheme.body);
      for (std::uint64_t variable : scheme.variables)
        bodyFree.erase(variable);
      mergeInto(result, bodyFree);
    }
  }
  return result;
}

AlgorithmM::AlgorithmM(const Oracle &oracle) : oracle(oracle) {}

Term AlgorithmM::fresh(llvm::StringRef name) {
  return Term::var(nextTypeVar++, name);
}

Scheme AlgorithmM::scheme(Term term, std::set<std::uint64_t> variables) const {
  return Scheme{std::move(variables), std::move(term)};
}

void AlgorithmM::bind(llvm::StringRef name, Term term, bool generalized) {
  if (generalized) {
    env.bind(name, generalize(term));
    return;
  }
  env.bind(name, scheme(std::move(term)));
}

Term AlgorithmM::lookup(llvm::StringRef name) {
  return instantiate(env.lookup(name));
}

Scheme AlgorithmM::generalize(const Term &term) const {
  Term applied = apply(term);
  return Scheme{setDifference(freeVars(applied), env.freeVars()), applied};
}

Term AlgorithmM::instantiate(const Scheme &scheme) {
  if (scheme.variables.empty())
    return scheme.body;

  std::map<std::uint64_t, Term> replacements;
  for (std::uint64_t variable : scheme.variables)
    replacements[variable] = fresh();
  return applySubstitution(scheme.body, replacements);
}

void AlgorithmM::requireEqual(Term lhs, Term rhs, Reason reason) {
  constraints.push_back(Constraint{Constraint::Kind::Equal,
                                   std::move(lhs),
                                   std::move(rhs),
                                   {},
                                   {},
                                   std::move(reason)});
}

void AlgorithmM::requireAttribute(Term owner, llvm::StringRef name, Term result,
                                  Reason reason) {
  constraints.push_back(Constraint{Constraint::Kind::Attribute,
                                   std::move(owner),
                                   std::move(result),
                                   {},
                                   name.str(),
                                   std::move(reason)});
}

void AlgorithmM::requireCall(Term callee, llvm::ArrayRef<Term> args,
                             Term result, Reason reason) {
  Constraint constraint{
      Constraint::Kind::Call, std::move(callee), std::move(result), {}, {},
      std::move(reason)};
  constraint.args.assign(args.begin(), args.end());
  constraints.push_back(std::move(constraint));
}

void AlgorithmM::requireSubscript(Term container, Term index, Term result,
                                  Reason reason) {
  constraints.push_back(Constraint{Constraint::Kind::Subscript,
                                   std::move(container),
                                   std::move(result),
                                   {std::move(index)},
                                   {},
                                   std::move(reason)});
}

void AlgorithmM::requireAwait(Term awaitable, Term result, Reason reason) {
  constraints.push_back(Constraint{Constraint::Kind::Await,
                                   std::move(awaitable),
                                   std::move(result),
                                   {},
                                   {},
                                   std::move(reason)});
}

void AlgorithmM::solve() {
  std::deque<Constraint> pending;
  for (const Constraint &constraint : constraints)
    pending.push_back(apply(constraint));
  constraints.clear();

  while (!pending.empty()) {
    Constraint constraint = apply(pending.front());
    pending.pop_front();

    if (constraint.kind == Constraint::Kind::Equal) {
      unify(std::move(constraint.lhs), std::move(constraint.rhs),
            constraint.reason);
      continue;
    }

    if (constraint.kind == Constraint::Kind::Call &&
        constraint.lhs.kind == Term::Kind::Func) {
      if (!constraint.lhs.kwonly.empty() || constraint.lhs.results.size() != 1)
        throw Error("cannot resolve call of " + constraint.lhs.display() +
                    ": " + constraint.reason.message);
      if (constraint.lhs.vararg.empty() &&
          constraint.lhs.args.size() != constraint.args.size())
        throw Error("cannot resolve call of " + constraint.lhs.display() +
                    ": " + constraint.reason.message);
      if (!constraint.lhs.vararg.empty() &&
          constraint.lhs.args.size() > constraint.args.size())
        throw Error("cannot resolve call of " + constraint.lhs.display() +
                    ": " + constraint.reason.message);
      pending.push_front(Constraint{Constraint::Kind::Equal,
                                    constraint.rhs,
                                    constraint.lhs.results.front(),
                                    {},
                                    {},
                                    constraint.reason});
      for (auto [expected, actual] :
           llvm::zip(constraint.lhs.args, constraint.args)) {
        if (isObjectTop(expected))
          continue;
        pending.push_front(Constraint{Constraint::Kind::Equal,
                                      expected,
                                      actual,
                                      {},
                                      {},
                                      constraint.reason});
      }
      if (!constraint.lhs.vararg.empty() &&
          constraint.args.size() > constraint.lhs.args.size()) {
        std::optional<Term> element =
            varargElementType(constraint.lhs.vararg.front());
        if (!element)
          throw Error("cannot resolve call of " + constraint.lhs.display() +
                      ": " + constraint.reason.message);
        llvm::ArrayRef<Term> extraArgs(constraint.args);
        extraArgs = extraArgs.drop_front(constraint.lhs.args.size());
        for (const Term &actual : extraArgs) {
          if (isObjectTop(*element))
            continue;
          pending.push_front(Constraint{Constraint::Kind::Equal,
                                        *element,
                                        actual,
                                        {},
                                        {},
                                        constraint.reason});
        }
      }
      continue;
    }

    std::optional<Term> resolved;
    if (constraint.kind == Constraint::Kind::Attribute)
      resolved = oracle.attribute(constraint.lhs, constraint.name);
    else if (constraint.kind == Constraint::Kind::Call)
      resolved = oracle.call(constraint.lhs, constraint.args);
    else if (constraint.kind == Constraint::Kind::Subscript)
      resolved = oracle.subscript(constraint.lhs, constraint.args.front());
    else
      resolved = oracle.awaitable(constraint.lhs);

    if (!resolved) {
      std::string op;
      if (constraint.kind == Constraint::Kind::Attribute)
        op = "attribute '" + constraint.name + "' on ";
      else if (constraint.kind == Constraint::Kind::Call)
        op = "call of ";
      else if (constraint.kind == Constraint::Kind::Subscript)
        op = "subscript of ";
      else
        op = "await of ";
      throw Error("cannot resolve " + op + constraint.lhs.display() + ": " +
                  constraint.reason.message);
    }

    pending.push_front(Constraint{Constraint::Kind::Equal,
                                  constraint.rhs,
                                  *resolved,
                                  {},
                                  {},
                                  constraint.reason});
  }
}

Term AlgorithmM::resolve(const Term &term) {
  solve();
  Term result = apply(term);
  if (!freeVars(result).empty())
    throw Error("unresolved type variable in " + result.display());
  return result;
}

Term AlgorithmM::apply(const Term &term) const {
  return applySubstitution(term, substitution);
}

void AlgorithmM::unify(Term lhs, Term rhs, std::optional<Reason> reason) {
  lhs = apply(lhs);
  rhs = apply(rhs);
  if (lhs == rhs)
    return;
  if (lhs.kind == Term::Kind::Var) {
    bindVar(lhs, rhs, std::move(reason));
    return;
  }
  if (rhs.kind == Term::Kind::Var) {
    bindVar(rhs, lhs, std::move(reason));
    return;
  }
  if (lhs.kind == Term::Kind::Con && rhs.kind == Term::Kind::Con) {
    if (lhs.name != rhs.name || lhs.args.size() != rhs.args.size())
      failUnify(lhs, rhs, reason);
    for (auto [lhsArg, rhsArg] : llvm::zip(lhs.args, rhs.args))
      unify(lhsArg, rhsArg, reason);
    return;
  }
  if (lhs.kind == Term::Kind::Func && rhs.kind == Term::Kind::Func) {
    if (lhs.args.size() != rhs.args.size() ||
        lhs.kwonly.size() != rhs.kwonly.size() ||
        lhs.vararg.size() != rhs.vararg.size() ||
        lhs.kwarg.size() != rhs.kwarg.size() ||
        lhs.results.size() != rhs.results.size())
      failUnify(lhs, rhs, reason);
    if (lhs.positionalOnlyCount && rhs.positionalOnlyCount &&
        *lhs.positionalOnlyCount != *rhs.positionalOnlyCount)
      failUnify(lhs, rhs, reason);
    for (auto [lhsArg, rhsArg] : llvm::zip(lhs.args, rhs.args))
      unify(lhsArg, rhsArg, reason);
    for (auto [lhsArg, rhsArg] : llvm::zip(lhs.kwonly, rhs.kwonly))
      unify(lhsArg, rhsArg, reason);
    for (auto [lhsArg, rhsArg] : llvm::zip(lhs.vararg, rhs.vararg))
      unify(lhsArg, rhsArg, reason);
    for (auto [lhsArg, rhsArg] : llvm::zip(lhs.kwarg, rhs.kwarg))
      unify(lhsArg, rhsArg, reason);
    for (auto [lhsResult, rhsResult] : llvm::zip(lhs.results, rhs.results))
      unify(lhsResult, rhsResult, reason);
    return;
  }
  failUnify(lhs, rhs, reason);
}

AlgorithmM::Constraint AlgorithmM::apply(const Constraint &constraint) const {
  Constraint result = constraint;
  result.lhs = apply(result.lhs);
  result.rhs = apply(result.rhs);
  for (Term &arg : result.args)
    arg = apply(arg);
  return result;
}

void AlgorithmM::bindVar(const Term &var, const Term &term,
                         std::optional<Reason> reason) {
  if (term.kind == Term::Kind::Var && term.id == var.id)
    return;
  if (freeVars(term).count(var.id))
    throw Error("recursive type " + var.display() + " occurs in " +
                term.display() + describeReason(reason));
  substitution[var.id] = term;
}

void AlgorithmM::failUnify(const Term &lhs, const Term &rhs,
                           std::optional<Reason> reason) const {
  throw Error("cannot unify " + lhs.display() + " with " + rhs.display() +
              describeReason(reason));
}

std::set<std::uint64_t> freeVars(const Term &term) {
  if (term.kind == Term::Kind::Var)
    return {term.id};

  std::set<std::uint64_t> result;
  for (const Term &arg : term.args)
    mergeInto(result, freeVars(arg));
  for (const Term &arg : term.kwonly)
    mergeInto(result, freeVars(arg));
  for (const Term &arg : term.vararg)
    mergeInto(result, freeVars(arg));
  for (const Term &arg : term.kwarg)
    mergeInto(result, freeVars(arg));
  for (const Term &resultType : term.results)
    mergeInto(result, freeVars(resultType));
  return result;
}

Term applySubstitution(const Term &term,
                       const std::map<std::uint64_t, Term> &substitution) {
  if (term.kind == Term::Kind::Var) {
    auto found = substitution.find(term.id);
    if (found == substitution.end())
      return term;
    return applySubstitution(found->second, substitution);
  }

  Term result = term;
  for (Term &arg : result.args)
    arg = applySubstitution(arg, substitution);
  for (Term &arg : result.kwonly)
    arg = applySubstitution(arg, substitution);
  for (Term &arg : result.vararg)
    arg = applySubstitution(arg, substitution);
  for (Term &arg : result.kwarg)
    arg = applySubstitution(arg, substitution);
  for (Term &resultType : result.results)
    resultType = applySubstitution(resultType, substitution);
  return result;
}

} // namespace lython::emitter::typing
