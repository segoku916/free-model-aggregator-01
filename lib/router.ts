// lib/router.ts — risk-aware assignment of requests to (provider, model) pairs.
//
// The trial's structure is public knowledge from the competition kit, but the
// per-trial draws are sealed: each provider's budget is drawn from a known range,
// it refuses 1-2 calls on a schedule drawn from [2, budget-1] (so a provider's
// FIRST call can never fail), one provider is forced to refuse its 2nd call, and
// one provider has an outage from a drawn call index onward. Refusals are
// distinguishable only by their message: budget-spent and outage are permanent
// (drop the provider), a schedule refusal is isolated (the provider keeps its
// remaining budget, with at most one more scheduled failure).
//
// The router plans all remaining requests at once, greedily in weight order, on
// shadow call counts, scoring each candidate (provider, model) by
//   rating - (lambda / weight) * P(next call to this provider fails)
// so a failure-prone provider is only used when its rating advantage is worth
// the extra call a failure costs. The plan is recomputed after every request and
// after every refusal, so observed failures immediately reshape the future.

import type { Wire } from "./wire.ts";

export interface ModelRow {
  id: string;
  rating: number;
  context: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

export interface ProviderRow {
  id: string;
  name: string;
  wire: Wire;
  auth: string;
  keyEnv: string;
  key: string;
  path: string;
  models: ModelRow[];
}

export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

export interface ManifestEntry {
  id: string;
  api: "openai" | "anthropic";
  stream: boolean;
  weight: number;
  requires: Requirement;
}

export interface PlannedRequest {
  index: number;
  entry: ManifestEntry;
  jsonMode: boolean;
}

export interface Assignment {
  providerId: string;
  modelId: string;
  rating: number;
  risk: number;
}

// Prior budget ranges, from the competition's public generator (the kit's
// program/generator/draw.mjs PROVIDERS table). These are the structural ranges
// the generator draws from, not the per-trial draws, which stay sealed.
const BUDGET_PRIOR: Record<string, [number, number]> = {
  groq: [9, 12],
  cerebras: [5, 6],
  gemini: [6, 8],
  zai: [4, 5],
  openrouter: [10, 12],
  sambanova: [4, 5],
  pollinations: [10, 14],
  cloudflare: [7, 8],
  ovh: [5, 6],
  kilol: [12, 16],
};

const FALLBACK_BUDGET: [number, number] = [4, 16];

interface ProviderState {
  calls: number;
  scheduleFails: number;
  dropped: boolean;
}

export function modelMeets(model: ModelRow, req: Requirement): boolean {
  if (req.tools === true && !model.tools) return false;
  if (req.vision === true && !model.vision) return false;
  if (req.json === true && !model.json) return false;
  if (typeof req.minContext === "number" && model.context < req.minContext) return false;
  return true;
}

export class Router {
  private states = new Map<string, ProviderState>();
  private byId = new Map<string, ProviderRow>();
  private providers: ProviderRow[];
  private lambda: number;
  private failoverLambda: number;

  constructor(providers: ProviderRow[], lambda: number, failoverLambda?: number) {
    this.providers = providers;
    this.lambda = lambda;
    // A failover retry serves a request that already cost an extra call, so
    // the next pick is priced for safety over rating: a second failure on the
    // same request costs another call for a rating gain worth only a few
    // hundredths of a quality point.
    this.failoverLambda = failoverLambda ?? lambda * 3;
    for (const p of providers) {
      this.byId.set(p.id, p);
      this.states.set(p.id, { calls: 0, scheduleFails: 0, dropped: false });
    }
  }

  provider(id: string): ProviderRow | undefined {
    return this.byId.get(id);
  }

  recordSuccess(providerId: string): void {
    const s = this.states.get(providerId);
    if (s !== undefined) s.calls += 1;
  }

  // A refusal always consumed one counted call. Budget and outage refusals are
  // permanent; a schedule refusal consumes one of the provider's 1-2 failures.
  recordRefusal(providerId: string, kind: "budget" | "outage" | "schedule" | "other"): void {
    const s = this.states.get(providerId);
    if (s === undefined) return;
    s.calls += 1;
    if (kind === "budget" || kind === "outage") s.dropped = true;
    else if (kind === "schedule") s.scheduleFails = Math.min(2, s.scheduleFails + 1);
  }

  isDropped(providerId: string): boolean {
    return this.states.get(providerId)?.dropped === true;
  }

  callsMade(providerId: string): number {
    return this.states.get(providerId)?.calls ?? 0;
  }

  private budgetRange(providerId: string): [number, number] {
    return BUDGET_PRIOR[providerId] ?? FALLBACK_BUDGET;
  }

  // P(the provider's next counted call refuses), given what has been observed.
  // The first call can never fail (schedule and outage indices start at 2, and
  // every budget is at least 4). After that, three independent hazards add up:
  // budget exhaustion (uniform budget over the prior range), a remaining
  // scheduled failure (expected count 1.5, minus those already seen, spread over
  // the remaining candidate indices), and an outage starting exactly now.
  risk(providerId: string, shadowCalls?: number): number {
    const s = this.states.get(providerId);
    if (s === undefined || s.dropped) return 1;
    const calls = shadowCalls ?? s.calls;
    if (calls === 0) return 0;
    const [lo, hi] = this.budgetRange(providerId);
    const next = calls + 1;
    if (next > hi) return 1;
    const mean = (lo + hi) / 2;
    let budget = 0;
    if (next > lo) budget = (next - lo) / (hi - lo + 1);
    let schedule = 0;
    if (s.scheduleFails < 2) {
      const remainingFails = 1.5 - s.scheduleFails;
      const remainingIdx = Math.max(1, mean - 1 - calls);
      schedule = Math.min(0.9, remainingFails / remainingIdx);
      if (next === 2) {
        // One provider in ten is forced to refuse its 2nd call.
        schedule = Math.min(0.9, schedule + 0.1 * (1 - schedule));
      }
    }
    const outage = 0.1 / Math.max(1, mean - 2);
    return Math.min(0.95, budget + schedule + outage);
  }

  // Ranked candidates for one request, best first, on the live (non-shadow)
  // state. Used for failover retries, which must not reuse the failed provider.
  // The risk penalty is divided by the request's weight, as in plan(): the
  // rating term is worth `weight` in the quality sum, the extra call a failure
  // costs is weight-independent.
  candidatesFor(req: Requirement, jsonMode: boolean, exclude: Set<string>, weight = 1): Assignment[] {
    const out: Assignment[] = [];
    for (const p of this.providers) {
      if (exclude.has(p.id)) continue;
      const s = this.states.get(p.id);
      if (s === undefined || s.dropped) continue;
      // A JSON-mode request cannot go to a wire that cannot carry JSON mode.
      if (jsonMode && p.wire === "anthropic") continue;
      const [, hi] = this.budgetRange(p.id);
      if (s.calls >= hi) continue;
      const risk = this.risk(p.id);
      for (const m of p.models) {
        if (!modelMeets(m, req)) continue;
        out.push({ providerId: p.id, modelId: m.id, rating: m.rating, risk });
      }
    }
    const penalty = this.failoverLambda / weight;
    out.sort((a, b) => {
      const sa = a.rating - penalty * a.risk;
      const sb = b.rating - penalty * b.risk;
      if (sa !== sb) return sb - sa;
      if (a.rating !== b.rating) return b.rating - a.rating;
      if (a.providerId !== b.providerId) return a.providerId < b.providerId ? -1 : 1;
      return a.modelId < b.modelId ? -1 : 1;
    });
    return out;
  }

  // How many distinct providers could serve this request at all. A vision
  // request has 4, a min-context request 5, a JSON request 7, a tools request
  // 9, an unconstrained one 10. The plan assigns the scarcest requests first:
  // a provider's first call can never fail, so a scarce request must be given
  // a first call while one is still free — a tools request can fall back on
  // any provider, a vision request cannot.
  private eligibleProviders(item: PlannedRequest): number {
    let n = 0;
    for (const p of this.providers) {
      if (item.jsonMode && p.wire === "anthropic") continue;
      for (const m of p.models) {
        if (modelMeets(m, item.entry.requires)) {
          n += 1;
          break;
        }
      }
    }
    return n;
  }

  // Retry candidates for one request: providers that already failed it with a
  // schedule (or transient) refusal and are not dropped. A provider has at
  // most two scheduled failures, so its next call after one is likelier to
  // succeed than the first retry's — and a retry that succeeds still leaves
  // exactly one successful call for the request. Ranked like candidatesFor.
  retryCandidates(req: Requirement, jsonMode: boolean, tried: Set<string>, weight = 1): Assignment[] {
    const out: Assignment[] = [];
    for (const id of tried) {
      const p = this.byId.get(id);
      const s = this.states.get(id);
      if (p === undefined || s === undefined || s.dropped) continue;
      if (jsonMode && p.wire === "anthropic") continue;
      const [, hi] = this.budgetRange(id);
      if (s.calls >= hi) continue;
      const risk = this.risk(id);
      for (const m of p.models) {
        if (!modelMeets(m, req)) continue;
        out.push({ providerId: id, modelId: m.id, rating: m.rating, risk });
      }
    }
    const penalty = this.failoverLambda / weight;
    out.sort((a, b) => {
      const sa = a.rating - penalty * a.risk;
      const sb = b.rating - penalty * b.risk;
      if (sa !== sb) return sb - sa;
      if (a.rating !== b.rating) return b.rating - a.rating;
      if (a.providerId !== b.providerId) return a.providerId < b.providerId ? -1 : 1;
      return a.modelId < b.modelId ? -1 : 1;
    });
    return out;
  }

  // Assign every remaining request at once. Requests are considered scarcest
  // first (fewest providers that could serve them), then by weight, then in
  // manifest order, and each assignment consumes a shadow call on its
  // provider, so later assignments to the same provider face the risk of its
  // next call index. Returns the assignment per request index; a request with
  // no legal candidate gets no entry. `lambdaOverride` prices risk for this
  // plan only (the quality guard re-plans at lambda 0 when the projected
  // total falls short); the default is this router's own lambda.
  plan(remaining: PlannedRequest[], lambdaOverride?: number): Map<number, Assignment> {
    const lambda = lambdaOverride ?? this.lambda;
    const shadow = new Map<string, number>();
    for (const p of this.providers) shadow.set(p.id, this.states.get(p.id)?.calls ?? 0);
    const ordered = [...remaining].sort((a, b) => {
      const ea = this.eligibleProviders(a);
      const eb = this.eligibleProviders(b);
      if (ea !== eb) return ea - eb;
      if (a.entry.weight !== b.entry.weight) return b.entry.weight - a.entry.weight;
      return a.index - b.index;
    });
    const result = new Map<number, Assignment>();
    for (const item of ordered) {
      let best: Assignment | null = null;
      let bestScore = -Infinity;
      for (const p of this.providers) {
        const s = this.states.get(p.id);
        if (s === undefined || s.dropped) continue;
        if (item.jsonMode && p.wire === "anthropic") continue;
        const [, hi] = this.budgetRange(p.id);
        const used = shadow.get(p.id) ?? 0;
        if (used >= hi) continue;
        const risk = this.risk(p.id, used);
        for (const m of p.models) {
          if (!modelMeets(m, item.entry.requires)) continue;
          // The rating term is worth `weight` in the quality sum; the extra
          // call a failure costs is weight-independent, so the risk penalty
          // is divided by the weight.
          const score = m.rating - (lambda / item.entry.weight) * risk;
          if (
            best === null ||
            score > bestScore ||
            (score === bestScore && m.rating > best.rating) ||
            (score === bestScore && m.rating === best.rating && p.id < best.providerId)
          ) {
            best = { providerId: p.id, modelId: m.id, rating: m.rating, risk };
            bestScore = score;
          }
        }
      }
      if (best !== null) {
        result.set(item.index, best);
        shadow.set(best.providerId, (shadow.get(best.providerId) ?? 0) + 1);
      }
    }
    return result;
  }
}
