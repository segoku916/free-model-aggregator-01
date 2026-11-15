// lib/brief.ts — the trial brief (provider directory, quality table, workload
// manifest), the catalog probes, and the catalog the entry registers.
//
// The brief lives in the prompt file as three single-line JSON documents, one
// per schema. The directory carries the sim keys in plaintext. The catalog is
// built from live probes of each provider's model list: a model's free flag
// comes from the quality table when the table names it (the table is free-only),
// else from the listing itself — that is what keeps the one paid model
// (openrouter's claude-opus-5.5, absent from the table, free:false in the
// listing) out of the free set. A probe that fails falls back to the quality
// table's rows for that provider, so the free set stays exact either way.

import { readFileSync } from "node:fs";
import type { ModelRow, ProviderRow, ManifestEntry } from "./router.ts";
import type { Wire } from "./wire.ts";

export interface DirectoryProvider {
  id: string;
  name: string;
  wire: Wire;
  auth: string;
  keyEnv: string;
  key: string;
  path: string;
}

export interface RatingRow {
  model: string;
  provider: string;
  rating: number;
  context: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

export interface Brief {
  directory: DirectoryProvider[];
  ratings: RatingRow[];
  manifest: ManifestEntry[];
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function asWire(v: unknown): Wire | null {
  return v === "openai" || v === "anthropic" || v === "gemini" ? v : null;
}

function parseBriefText(text: string): Brief | null {
  let directory: DirectoryProvider[] | null = null;
  let ratings: RatingRow[] | null = null;
  let manifest: ManifestEntry[] | null = null;
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("{")) continue;
    let doc: unknown;
    try {
      doc = JSON.parse(trimmed);
    } catch {
      continue;
    }
    if (!isRecord(doc) || typeof doc.schema !== "string") continue;
    if (doc.schema === "free-model-aggregator-01/providers/v1" && Array.isArray(doc.providers)) {
      const rows: DirectoryProvider[] = [];
      for (const p of doc.providers) {
        if (!isRecord(p)) continue;
        const wire = asWire(p.wire);
        if (typeof p.id !== "string" || wire === null) continue;
        rows.push({
          id: p.id,
          name: typeof p.name === "string" ? p.name : p.id,
          wire,
          auth: typeof p.auth === "string" ? p.auth : "none",
          keyEnv: typeof p.key_env === "string" ? p.key_env : "",
          key: typeof p.key === "string" ? p.key : "",
          path: typeof p.path === "string" ? p.path : `/p/${p.id}`,
        });
      }
      directory = rows;
    } else if (doc.schema === "free-model-aggregator-01/ratings/v1" && Array.isArray(doc.ratings)) {
      const rows: RatingRow[] = [];
      for (const r of doc.ratings) {
        if (!isRecord(r)) continue;
        if (typeof r.model !== "string" || typeof r.provider !== "string") continue;
        if (typeof r.rating_milli !== "number") continue;
        rows.push({
          model: r.model,
          provider: r.provider,
          rating: r.rating_milli,
          context: typeof r.context_length === "number" ? r.context_length : 0,
          tools: r.tools === true,
          vision: r.vision === true,
          json: r.json === true,
        });
      }
      ratings = rows;
    } else if (doc.schema === "free-model-aggregator-01/workload/v1" && Array.isArray(doc.manifest)) {
      const rows: ManifestEntry[] = [];
      for (const m of doc.manifest) {
        if (!isRecord(m)) continue;
        if (typeof m.id !== "string") continue;
        const req = isRecord(m.requires) ? m.requires : {};
        rows.push({
          id: m.id,
          api: m.api === "anthropic" ? "anthropic" : "openai",
          stream: m.stream === true,
          weight: typeof m.weight === "number" ? m.weight : 1,
          requires: {
            tools: req.tools === true ? true : undefined,
            vision: req.vision === true ? true : undefined,
            json: req.json === true ? true : undefined,
            minContext: typeof req.min_context === "number" ? req.min_context : undefined,
          },
        });
      }
      manifest = rows;
    }
  }
  if (directory === null || ratings === null || manifest === null) return null;
  return { directory, ratings, manifest };
}

export function loadBriefFromFile(path: string): Brief | null {
  try {
    return parseBriefText(readFileSync(path, "utf8"));
  } catch {
    return null;
  }
}

// Fallback when there is no prompt file: the env's /instance route serves the
// directory (with absolute base_urls) and the manifest, but no quality table,
// so routing falls back to context length alone for model choice.
export async function loadBriefFromInstance(envUrl: string): Promise<Brief | null> {
  let res: Response;
  try {
    res = await fetch(`${envUrl}/instance`, { signal: AbortSignal.timeout(5000) });
  } catch {
    return null;
  }
  if (!res.ok) return null;
  let doc: unknown;
  try {
    doc = await res.json();
  } catch {
    return null;
  }
  if (!isRecord(doc) || !Array.isArray(doc.providers) || !isRecord(doc.workload)) return null;
  const directory: DirectoryProvider[] = [];
  for (const p of doc.providers) {
    if (!isRecord(p)) continue;
    const wire = asWire(p.wire);
    if (typeof p.id !== "string" || wire === null) continue;
    let path = `/p/${p.id}`;
    if (typeof p.base_url === "string") {
      try {
        path = new URL(p.base_url).pathname;
      } catch {
        path = p.base_url;
      }
    }
    directory.push({
      id: p.id,
      name: typeof p.name === "string" ? p.name : p.id,
      wire,
      auth: typeof p.auth === "string" ? p.auth : "none",
      keyEnv: typeof p.key_env === "string" ? p.key_env : "",
      key: typeof p.key === "string" ? p.key : "",
      path,
    });
  }
  const workload = doc.workload;
  if (!Array.isArray(workload.manifest)) return null;
  const manifest: ManifestEntry[] = [];
  for (const m of workload.manifest) {
    if (!isRecord(m) || typeof m.id !== "string") continue;
    const req = isRecord(m.requires) ? m.requires : {};
    manifest.push({
      id: m.id,
      api: m.api === "anthropic" ? "anthropic" : "openai",
      stream: m.stream === true,
      weight: typeof m.weight === "number" ? m.weight : 1,
      requires: {
        tools: req.tools === true ? true : undefined,
        vision: req.vision === true ? true : undefined,
        json: req.json === true ? true : undefined,
        minContext: typeof req.min_context === "number" ? req.min_context : undefined,
      },
    });
  }
  return { directory, ratings: [], manifest };
}

// Auth headers for a provider, from its auth scheme. The anthropic wire also
// carries the anthropic-version header the client sends.
export function authHeaders(p: DirectoryProvider): Record<string, string> {
  const headers: Record<string, string> = {};
  if (p.auth === "bearer" && p.key !== "") headers["authorization"] = `Bearer ${p.key}`;
  else if (p.auth === "x-api-key" && p.key !== "") headers["x-api-key"] = p.key;
  else if (p.auth === "x-goog-api-key" && p.key !== "") headers["x-goog-api-key"] = p.key;
  if (p.wire === "anthropic") headers["anthropic-version"] = "2023-06-01";
  return headers;
}

function catalogPath(p: DirectoryProvider): string {
  return p.wire === "gemini" ? "/v1beta/models" : "/v1/models";
}

interface ProbedModel {
  id: string;
  free: boolean | null;
  context: number;
}

// One provider's model listing, in its own wire's shape. Returns null on any
// failure (timeout, non-2xx, unparsable body); the caller falls back to the
// quality table.
async function probeOne(envUrl: string, p: DirectoryProvider): Promise<ProbedModel[] | null> {
  let res: Response;
  try {
    res = await fetch(`${envUrl}${p.path}${catalogPath(p)}`, {
      headers: authHeaders(p),
      signal: AbortSignal.timeout(5000),
    });
  } catch {
    return null;
  }
  if (!res.ok) return null;
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    return null;
  }
  const rows: ProbedModel[] = [];
  const push = (id: unknown, free: unknown, context: unknown): void => {
    if (typeof id !== "string" || id === "") return;
    rows.push({
      id,
      free: typeof free === "boolean" ? free : null,
      context: typeof context === "number" ? context : 0,
    });
  };
  if (p.wire === "gemini") {
    if (!isRecord(body) || !Array.isArray(body.models)) return null;
    for (const m of body.models) {
      if (!isRecord(m)) continue;
      const name = typeof m.name === "string" ? m.name : "";
      const id = name.startsWith("models/") ? name.slice("models/".length) : name;
      push(id, m.free, m.context_length);
    }
  } else {
    if (!isRecord(body) || !Array.isArray(body.data)) return null;
    for (const m of body.data) {
      if (!isRecord(m)) continue;
      push(m.id, m.free, m.context_length);
    }
  }
  return rows;
}

export async function probeCatalogs(
  envUrl: string,
  directory: DirectoryProvider[],
): Promise<Map<string, ProbedModel[] | null>> {
  const out = new Map<string, ProbedModel[] | null>();
  await Promise.all(
    directory.map(async (p) => {
      out.set(p.id, await probeOne(envUrl, p));
    }),
  );
  return out;
}

// The catalog to register (and to print for --list): one entry per provider in
// directory order, base_url the provider's own path, models deduplicated by id.
// A model the quality table names is free; otherwise its listing's word stands.
// With no probe at all, the quality table's rows are the whole catalog.
export function buildCatalog(
  brief: Brief,
  probed: Map<string, ProbedModel[] | null>,
): Catalog {
  const rated = new Map<string, RatingRow>();
  for (const r of brief.ratings) rated.set(`${r.provider}/${r.model}`, r);
  const providers: CatalogProvider[] = [];
  for (const p of brief.directory) {
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    const rows = probed.get(p.id);
    if (rows !== null && rows !== undefined) {
      for (const m of rows) {
        if (seen.has(m.id)) continue;
        seen.add(m.id);
        const row = rated.get(`${p.id}/${m.id}`);
        const free = row !== undefined ? true : m.free === true;
        models.push({
          id: m.id,
          free,
          context_length: row?.context ?? m.context,
        });
      }
    } else {
      for (const r of brief.ratings) {
        if (r.provider !== p.id) continue;
        if (seen.has(r.model)) continue;
        seen.add(r.model);
        models.push({ id: r.model, free: true, context_length: r.context });
      }
    }
    providers.push({
      id: p.id,
      base_url: p.path,
      key_env: p.keyEnv,
      wire: p.wire,
      models,
    });
  }
  return { providers };
}

// The router's view: free models only, with their ratings and capabilities,
// from the quality table. A provider with no rated models contributes none.
export function buildProviderRows(brief: Brief): ProviderRow[] {
  const rows: ProviderRow[] = [];
  for (const p of brief.directory) {
    const models: ModelRow[] = [];
    for (const r of brief.ratings) {
      if (r.provider !== p.id) continue;
      models.push({
        id: r.model,
        rating: r.rating,
        context: r.context,
        tools: r.tools,
        vision: r.vision,
        json: r.json,
      });
    }
    rows.push({
      id: p.id,
      name: p.name,
      wire: p.wire,
      auth: p.auth,
      keyEnv: p.keyEnv,
      key: p.key,
      path: p.path,
      models,
    });
  }
  return rows;
}

// The union of free model ids in the registered catalog, first-seen order,
// for the entry's own GET /v1/models.
export function freeModelIds(catalog: Catalog): string[] {
  const ids: string[] = [];
  const seen = new Set<string>();
  for (const p of catalog.providers) {
    for (const m of p.models) {
      if (m.free && !seen.has(m.id)) {
        seen.add(m.id);
        ids.push(m.id);
      }
    }
  }
  return ids;
}
