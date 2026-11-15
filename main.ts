// main.ts — a risk-aware aggregator over ten simulated free-tier providers.
//
// One long-running program: it reads the trial brief, probes every provider's
// model list, registers its catalog, serves the client's OpenAI and Anthropic
// requests, and exits 0 when the client is done. Every answer comes from one
// provider call; when a provider refuses, the refusal's message says whether
// the provider is spent (budget/outage: drop it) or merely hit its failure
// schedule (keep it, its remaining budget intact), and the request is answered
// by another provider. Routing is a greedy plan over the remaining requests
// (see lib/router.ts), recomputed after every request and every refusal.
//
// Deterministic: no wall clock, no unseeded randomness anywhere in the request
// path. Run with --list, it prints the catalog it would register and exits 0.

import { createServer } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import {
  authHeaders,
  buildCatalog,
  buildProviderRows,
  freeModelIds,
  loadBriefFromFile,
  loadBriefFromInstance,
  probeCatalogs,
} from "./lib/brief.ts";
import type { Brief, Catalog } from "./lib/brief.ts";
import {
  Router,
  type ManifestEntry,
  type PlannedRequest,
} from "./lib/router.ts";
import {
  anthropicJsonBody,
  anthropicStreamBody,
  anthropicModelList,
  buildAnthropicRequest,
  buildGeminiRequest,
  buildOpenAIRequest,
  canonFromAnthropic,
  canonFromOpenAI,
  openAIJsonBody,
  openAIStreamBody,
  openAIModelList,
  parseErrorBody,
  parseProviderAnswer,
  type CanonRequest,
  type ProviderAnswer,
  type ProviderRefusal,
  type Wire,
} from "./lib/wire.ts";

// The risk penalty in rating-milli per unit failure probability (see
// lib/router.ts). Tunable via AGG_LAMBDA / AGG_FAILOVER_LAMBDA for local
// grading; the final env never sets them, so the constants below are what
// ships. Tuned on 48 synthetic instances drawn with the kit's own generator:
// lambda 400 with the quality guard below scores 99 on all 48 (earned min
// 37804, 321 points above the 97 floor) at a mean 22.69 upstream calls.
const LAMBDA = (() => {
  const v = Number(process.env.AGG_LAMBDA ?? "400");
  return Number.isFinite(v) && v >= 0 ? v : 400;
})();
const FAILOVER_LAMBDA = (() => {
  const v = Number(process.env.AGG_FAILOVER_LAMBDA ?? String(LAMBDA * 3));
  return Number.isFinite(v) && v >= 0 ? v : LAMBDA * 3;
})();

// The quality guard: the projected weighted rating total (earned so far plus
// the plan's remaining ratings) below which the plan is re-run at lambda 0,
// pricing rating over call-saving for the rest of the trial. 37800 is exact
// quality 98.83 on the 38248 bound — comfortably above the 37483 that floors
// to quality 97, with room for the downgrades failures still cause.
const GUARD_THRESHOLD = (() => {
  const v = Number(process.env.AGG_GUARD ?? "37800");
  return Number.isFinite(v) && v > 0 ? v : 37800;
})();

const MAX_BODY_BYTES = 2 * 1024 * 1024;
const PROVIDER_TIMEOUT_MS = 10000;
const WORK_POLL_MS = 100;

function log(...args: unknown[]): void {
  process.stderr.write(`${args.map(String).join(" ")}\n`);
}

async function loadBrief(envUrl: string | null): Promise<Brief | null> {
  const promptFile = process.env.MAKEFUN_PROMPT_FILE;
  if (promptFile !== undefined && promptFile !== "") {
    const brief = loadBriefFromFile(promptFile);
    if (brief !== null) return brief;
  }
  if (envUrl !== null) return loadBriefFromInstance(envUrl);
  return null;
}

// A provider call that succeeded upstream but whose answer cannot be parsed:
// failing over would make a second successful call for the same request, which
// invalidates it anyway, so this aborts the request instead of retrying.
class FatalAnswerError extends Error {}

type ProviderCall =
  | { ok: true; answer: ProviderAnswer }
  | { ok: false; refusal: ProviderRefusal };

async function callProvider(
  envUrl: string,
  provider: { id: string; wire: Wire; auth: string; key: string; path: string },
  modelId: string,
  canon: CanonRequest,
): Promise<ProviderCall> {
  let url: string;
  let body: Record<string, unknown>;
  if (provider.wire === "openai") {
    url = `${envUrl}${provider.path}/v1/chat/completions`;
    body = buildOpenAIRequest(canon, modelId);
  } else if (provider.wire === "anthropic") {
    url = `${envUrl}${provider.path}/v1/messages`;
    body = buildAnthropicRequest(canon, modelId);
  } else {
    const verb = canon.stream ? "streamGenerateContent" : "generateContent";
    url = `${envUrl}${provider.path}/v1beta/models/${encodeURIComponent(modelId)}:${verb}`;
    body = buildGeminiRequest(canon);
  }
  const headers: Record<string, string> = {
    "content-type": "application/json",
    ...authHeaders({ id: provider.id, name: provider.id, wire: provider.wire, auth: provider.auth, keyEnv: "", key: provider.key, path: provider.path }),
  };
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
    });
  } catch (err) {
    return {
      ok: false,
      refusal: {
        status: 0,
        code: "network_error",
        message: err instanceof Error ? err.message : "network error",
        kind: "other",
      },
    };
  }
  const text = await res.text();
  if (!res.ok) return { ok: false, refusal: parseErrorBody(provider.wire, res.status, text) };
  try {
    return { ok: true, answer: parseProviderAnswer(provider.wire, text, canon) };
  } catch {
    throw new FatalAnswerError(`unparsable ${provider.wire} answer from ${provider.id}`);
  }
}

class Aggregator {
  private router: Router;
  private manifestIndex = 0;
  private servedModels: string[];
  private envUrl: string;
  private brief: Brief;
  // Weighted rating sum earned so far (rating x weight per served request).
  private earned = 0;

  constructor(envUrl: string, brief: Brief, catalog: Catalog) {
    this.envUrl = envUrl;
    this.brief = brief;
    this.router = new Router(buildProviderRows(brief), LAMBDA, FAILOVER_LAMBDA);
    this.servedModels = freeModelIds(catalog);
  }

  // Whether a future request (whose body is not seen yet) carries JSON mode:
  // only the OpenAI wire puts response_format in the body, so only an
  // OpenAI-wire request with a json requirement does.
  private futureJsonMode(entry: ManifestEntry): boolean {
    return entry.api === "openai" && entry.requires.json === true;
  }

  private remainingRequests(from: number, currentJsonMode: boolean): PlannedRequest[] {
    const out: PlannedRequest[] = [];
    for (let i = from; i < this.brief.manifest.length; i++) {
      const entry = this.brief.manifest[i];
      out.push({
        index: i,
        entry,
        jsonMode: i === from ? currentJsonMode : this.futureJsonMode(entry),
      });
    }
    return out;
  }

  // Serve one client request. Returns the provider answer and the model id
  // that answered it, or null when no provider could serve it.
  async serve(
    api: "openai" | "anthropic",
    body: Record<string, unknown>,
  ): Promise<{ answer: ProviderAnswer; modelId: string; attempts: number } | null> {
    const index = this.manifestIndex;
    this.manifestIndex += 1;
    const entry = this.brief.manifest[index] ?? {
      id: `r${index + 1}`,
      api,
      stream: false,
      weight: 1,
      requires: {},
    };
    const canon = api === "openai" ? canonFromOpenAI(body) : canonFromAnthropic(body);
    const exclude = new Set<string>();
    // Providers that failed this request with a schedule (or transient)
    // refusal: still alive, with budget left, so retryable as a last resort.
    const retryable = new Set<string>();
    const remaining = this.remainingRequests(index, canon.jsonMode);
    let plan = this.router.plan(remaining);
    // Quality guard: if the plan's projected total (earned plus the plan's
    // remaining ratings) falls below the guard threshold, re-plan at lambda
    // 0 — rating over call-saving — so a run of bad luck cannot drag the
    // trial's quality toward the floor that would cost a whole score point.
    let projected = this.earned;
    for (const [i, a] of plan) projected += a.rating * this.brief.manifest[i].weight;
    if (projected < GUARD_THRESHOLD) {
      plan = this.router.plan(remaining, 0);
      let guarded = this.earned;
      for (const [i, a] of plan) guarded += a.rating * this.brief.manifest[i].weight;
      log(`${entry.id}: quality guard: projected ${projected} < ${GUARD_THRESHOLD}, re-planned at lambda 0 (now ${guarded})`);
    }
    let assignment = plan.get(index);
    let attempts = 0;
    while (assignment !== undefined && attempts < 12) {
      const provider = this.router.provider(assignment.providerId);
      if (provider === undefined) break;
      attempts += 1;
      let result: ProviderCall;
      try {
        result = await callProvider(this.envUrl, provider, assignment.modelId, canon);
      } catch (err) {
        if (err instanceof FatalAnswerError) {
          log(`${entry.id}: fatal: ${err.message}`);
          return null;
        }
        throw err;
      }
      if (result.ok) {
        this.router.recordSuccess(provider.id);
        this.earned += assignment.rating * entry.weight;
        log(`${entry.id}: ${provider.id}/${assignment.modelId} ok (attempt ${attempts})`);
        return { answer: result.answer, modelId: assignment.modelId, attempts };
      }
      const kind = result.refusal.kind;
      this.router.recordRefusal(
        provider.id,
        kind === "budget" || kind === "outage" || kind === "schedule" ? kind : "other",
      );
      log(
        `${entry.id}: ${provider.id}/${assignment.modelId} refused ${result.refusal.status} ${result.refusal.code} (${kind}) — ${result.refusal.message}`,
      );
      exclude.add(provider.id);
      if (kind === "schedule" || kind === "other") retryable.add(provider.id);
      // Fresh providers first; when none is left, retry a provider whose
      // refusal was only a scheduled one — its next call may well succeed,
      // and a request answered on a retry is still exactly one success.
      const fresh = this.router.candidatesFor(entry.requires, canon.jsonMode, exclude, entry.weight);
      if (fresh.length > 0) {
        assignment = fresh[0];
        continue;
      }
      const retries = this.router.retryCandidates(entry.requires, canon.jsonMode, retryable, entry.weight);
      assignment = retries[0];
      if (assignment !== undefined) {
        log(`${entry.id}: retrying ${assignment.providerId}/${assignment.modelId} after its schedule refusal`);
      }
    }
    log(`${entry.id}: no provider could serve it`);
    return null;
  }

  modelList(req: IncomingMessage): Record<string, unknown> {
    // The wire is the auth header: x-api-key asks for the Anthropic shape,
    // anything else the OpenAI shape.
    return req.headers["x-api-key"] !== undefined
      ? anthropicModelList(this.servedModels)
      : openAIModelList(this.servedModels);
  }
}

async function readBody(req: IncomingMessage): Promise<string | null> {
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of req) {
    const buf = chunk as Buffer;
    total += buf.length;
    if (total > MAX_BODY_BYTES) return null;
    chunks.push(buf);
  }
  return Buffer.concat(chunks).toString("utf8");
}

function sendJson(res: ServerResponse, status: number, value: unknown): void {
  const body = `${JSON.stringify(value)}\n`;
  res.writeHead(status, { "content-type": "application/json", "content-length": Buffer.byteLength(body) });
  res.end(body);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, {
    "content-type": "text/event-stream; charset=utf-8",
    "cache-control": "no-cache",
    connection: "keep-alive",
    "content-length": Buffer.byteLength(text),
  });
  res.end(text);
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const listOnly = args.includes("--list");
  const envUrl = process.env.MAKEFUN_ENV_URL ?? null;
  const brief = await loadBrief(envUrl);
  if (brief === null) {
    if (listOnly) {
      process.stdout.write(`${JSON.stringify({ providers: [] })}\n`);
      process.exit(0);
    }
    log("no brief: set MAKEFUN_PROMPT_FILE or MAKEFUN_ENV_URL");
    process.exit(1);
  }
  const probed = envUrl !== null ? await probeCatalogs(envUrl, brief.directory) : new Map();
  const catalog = buildCatalog(brief, probed);
  if (listOnly) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    process.exit(0);
  }
  if (envUrl === null) {
    log("no env: set MAKEFUN_ENV_URL");
    process.exit(1);
  }
  const aggregator = new Aggregator(envUrl, brief, catalog);
  const server = createServer((req, res) => {
    void (async () => {
      try {
        const url = new URL(req.url ?? "/", "http://localhost");
        if (req.method === "GET" && url.pathname === "/v1/models") {
          sendJson(res, 200, aggregator.modelList(req));
          return;
        }
        if (req.method === "POST" && (url.pathname === "/v1/chat/completions" || url.pathname === "/v1/messages")) {
          const api = url.pathname === "/v1/messages" ? "anthropic" : "openai";
          const raw = await readBody(req);
          if (raw === null) {
            sendJson(res, 413, { error: { message: "request body too large", type: "invalid_request_error" } });
            return;
          }
          let body: unknown;
          try {
            body = JSON.parse(raw);
          } catch {
            sendJson(res, 400, { error: { message: "request body is not JSON", type: "invalid_request_error" } });
            return;
          }
          if (body === null || typeof body !== "object" || Array.isArray(body)) {
            sendJson(res, 400, { error: { message: "request body is not one JSON object", type: "invalid_request_error" } });
            return;
          }
          const stream = (body as Record<string, unknown>).stream === true;
          const served = await aggregator.serve(api, body as Record<string, unknown>);
          if (served === null) {
            sendJson(res, 502, { error: { message: "no provider could serve this request", type: "api_error" } });
            return;
          }
          if (api === "openai") {
            if (stream) sendSse(res, openAIStreamBody(served.answer, served.modelId));
            else sendJson(res, 200, openAIJsonBody(served.answer, served.modelId));
          } else {
            if (stream) sendSse(res, anthropicStreamBody(served.answer, served.modelId));
            else sendJson(res, 200, anthropicJsonBody(served.answer, served.modelId));
          }
          return;
        }
        sendJson(res, 404, { error: { message: "not found", type: "invalid_request_error" } });
      } catch (err) {
        log(`handler error: ${err instanceof Error ? err.message : String(err)}`);
        if (!res.headersSent) {
          sendJson(res, 500, { error: { message: "internal error", type: "api_error" } });
        }
      }
    })();
  });
  await new Promise<void>((resolve) => server.listen(0, "0.0.0.0", resolve));
  const address = server.address();
  const port = typeof address === "object" && address !== null ? address.port : 0;
  if (port === 0) {
    log("could not listen");
    process.exit(1);
  }
  let registerRes: Response;
  try {
    registerRes = await fetch(`${envUrl}/register`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ protocol_version: 1, port, catalog }),
      signal: AbortSignal.timeout(5000),
    });
  } catch (err) {
    log(`register failed: ${err instanceof Error ? err.message : String(err)}`);
    process.exit(1);
  }
  if (!registerRes.ok) {
    log(`register refused: HTTP ${registerRes.status}`);
    process.exit(1);
  }
  log(`registered on port ${port}, ${catalog.providers.length} providers`);
  const timer = setInterval(() => {
    void (async () => {
      let res: Response;
      try {
        res = await fetch(`${envUrl}/work`, { signal: AbortSignal.timeout(5000) });
      } catch {
        return;
      }
      if (!res.ok) return;
      let body: unknown;
      try {
        body = await res.json();
      } catch {
        return;
      }
      if (body !== null && typeof body === "object" && (body as Record<string, unknown>).done === true) {
        clearInterval(timer);
        server.close(() => process.exit(0));
      }
    })();
  }, WORK_POLL_MS);
}

void main();
