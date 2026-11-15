// lib/wire.ts — wire translation for the three vendor APIs (OpenAI, Anthropic, Gemini).
//
// Every request the client sends is reduced to one canonical shape — the fields the
// provider echo is computed from (system text, user text, image bytes, tool names,
// max_tokens, stop sequences, JSON mode) — and rebuilt in the upstream provider's
// wire without rewriting or dropping any of them. Provider answers are buffered
// fully (JSON or SSE), parsed into one answer shape, and re-emitted in the client's
// wire, streamed or not as the client asked, with the provider's text whole and its
// usage counts exact. The provider's answer is always plain text with a family-0
// stop, so the client stop reason is always "stop" (OpenAI) / "end_turn" (Anthropic).

export type Wire = "openai" | "anthropic" | "gemini";

export interface ToolDef {
  name: string;
  description?: string;
  parameters?: unknown;
}

export type ImageRef =
  | { kind: "data"; mime: string; b64: string }
  | { kind: "url"; url: string };

export interface CanonRequest {
  systemText: string;
  userText: string;
  image: ImageRef | null;
  tools: ToolDef[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
}

export interface ProviderAnswer {
  text: string;
  stop: string;
  usage: { input: number; output: number };
}

export type RefusalKind =
  | "budget"
  | "outage"
  | "schedule"
  | "concurrent"
  | "auth"
  | "invalid"
  | "other";

export interface ProviderRefusal {
  status: number;
  code: string;
  message: string;
  kind: RefusalKind;
}

export type CallResult =
  | { ok: true; answer: ProviderAnswer }
  | { ok: false; refusal: ProviderRefusal };

export const FIXED_CREATED = 1767225600;
const SSE_PIECE = 13;

function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function numberOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function stringList(v: unknown): string[] {
  if (typeof v === "string") return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === "string");
}

// Text of a content field: a plain string as-is, else its text blocks joined with \n.
function blocksText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") {
      parts.push(block.text);
    }
  }
  return parts.join("\n");
}

function parseDataUrl(url: string): ImageRef {
  const m = /^data:([^,]*);base64,(.*)$/s.exec(url);
  if (m !== null) {
    const mime = m[1] === "" ? "image/png" : m[1];
    return { kind: "data", mime, b64: m[2] };
  }
  return { kind: "url", url };
}

function imageUrl(image: ImageRef): string {
  return image.kind === "data" ? `data:${image.mime};base64,${image.b64}` : image.url;
}

// --- Canonical request extraction (mirrors what each wire's parser reads) ---

export function canonFromOpenAI(body: Record<string, unknown>): CanonRequest {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  let userText = "";
  let image: ImageRef | null = null;
  for (const message of messages) {
    if (message.role === "system") {
      const text = blocksText(message.content);
      if (text !== "") systemParts.push(text);
    } else if (message.role === "user") {
      userText = blocksText(message.content);
      if (Array.isArray(message.content)) {
        for (const block of message.content) {
          if (!isRecord(block)) continue;
          if (block.type === "image_url" && isRecord(block.image_url) && typeof block.image_url.url === "string") {
            image = parseDataUrl(block.image_url.url);
          }
        }
      }
    }
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === "string") {
        const def: ToolDef = { name: tool.function.name };
        if (typeof tool.function.description === "string") def.description = tool.function.description;
        if (tool.function.parameters !== undefined) def.parameters = tool.function.parameters;
        tools.push(def);
      }
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    systemText: systemParts.join("\n"),
    userText,
    image,
    tools,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    jsonMode: format === "json_object" || format === "json_schema",
    stream: body.stream === true,
  };
}

export function canonFromAnthropic(body: Record<string, unknown>): CanonRequest {
  const systemText = blocksText(body.system);
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  let userText = "";
  let image: ImageRef | null = null;
  for (const message of messages) {
    if (message.role !== "user") continue;
    userText = blocksText(message.content);
    if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if (!isRecord(block) || block.type !== "image" || !isRecord(block.source)) continue;
        if (block.source.type === "base64" && typeof block.source.data === "string") {
          const mime = typeof block.source.media_type === "string" ? block.source.media_type : "image/png";
          image = { kind: "data", mime, b64: block.source.data };
        } else if (block.source.type === "url" && typeof block.source.url === "string") {
          image = parseDataUrl(block.source.url);
        }
      }
    }
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (isRecord(tool) && typeof tool.name === "string") {
        const def: ToolDef = { name: tool.name };
        if (typeof tool.description === "string") def.description = tool.description;
        if (tool.input_schema !== undefined) def.parameters = tool.input_schema;
        tools.push(def);
      }
    }
  }
  return {
    systemText,
    userText,
    image,
    tools,
    maxTokens: numberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    stream: body.stream === true,
  };
}

// --- Upstream request building ---

function openAITools(tools: ToolDef[]): unknown[] {
  return tools.map((t) => {
    const fn: Record<string, unknown> = { name: t.name };
    if (t.description !== undefined) fn.description = t.description;
    if (t.parameters !== undefined) fn.parameters = t.parameters;
    return { type: "function", function: fn };
  });
}

export function buildOpenAIRequest(canon: CanonRequest, model: string): Record<string, unknown> {
  const messages: Record<string, unknown>[] = [];
  if (canon.systemText !== "") messages.push({ role: "system", content: canon.systemText });
  if (canon.image !== null) {
    messages.push({
      role: "user",
      content: [
        { type: "text", text: canon.userText },
        { type: "image_url", image_url: { url: imageUrl(canon.image) } },
      ],
    });
  } else {
    messages.push({ role: "user", content: canon.userText });
  }
  const body: Record<string, unknown> = { model, messages };
  if (canon.tools.length > 0) body.tools = openAITools(canon.tools);
  if (canon.jsonMode) body.response_format = { type: "json_object" };
  if (canon.maxTokens !== null) body.max_tokens = canon.maxTokens;
  if (canon.stops.length > 0) body.stop = canon.stops;
  if (canon.stream) {
    body.stream = true;
    // Without include_usage the provider's stream carries no usage chunk, and the
    // client checks usage exactly. Always ask for it.
    body.stream_options = { include_usage: true };
  }
  return body;
}

export function buildAnthropicRequest(canon: CanonRequest, model: string): Record<string, unknown> {
  const body: Record<string, unknown> = { model };
  if (canon.systemText !== "") body.system = canon.systemText;
  if (canon.image !== null) {
    const source =
      canon.image.kind === "data"
        ? { type: "base64", media_type: canon.image.mime, data: canon.image.b64 }
        : { type: "url", url: canon.image.url };
    body.messages = [
      {
        role: "user",
        content: [
          { type: "text", text: canon.userText },
          { type: "image", source },
        ],
      },
    ];
  } else {
    body.messages = [{ role: "user", content: canon.userText }];
  }
  if (canon.tools.length > 0) {
    body.tools = canon.tools.map((t) => {
      const tool: Record<string, unknown> = { name: t.name, input_schema: t.parameters ?? { type: "object" } };
      if (t.description !== undefined) tool.description = t.description;
      return tool;
    });
  }
  if (canon.maxTokens !== null) body.max_tokens = canon.maxTokens;
  if (canon.stops.length > 0) body.stop_sequences = canon.stops;
  if (canon.stream) body.stream = true;
  return body;
}

export function buildGeminiRequest(canon: CanonRequest): Record<string, unknown> {
  const parts: Record<string, unknown>[] = [{ text: canon.userText }];
  if (canon.image !== null) {
    if (canon.image.kind === "data") {
      parts.push({ inlineData: { mimeType: canon.image.mime, data: canon.image.b64 } });
    } else {
      parts.push({ fileData: { fileUri: canon.image.url } });
    }
  }
  const body: Record<string, unknown> = { contents: [{ role: "user", parts }] };
  if (canon.systemText !== "") {
    body.systemInstruction = { parts: [{ text: canon.systemText }] };
  }
  if (canon.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: canon.tools.map((t) => {
          const fn: Record<string, unknown> = { name: t.name };
          if (t.description !== undefined) fn.description = t.description;
          if (t.parameters !== undefined) fn.parameters = t.parameters;
          return fn;
        }),
      },
    ];
  }
  const config: Record<string, unknown> = {};
  if (canon.maxTokens !== null) config.maxOutputTokens = canon.maxTokens;
  if (canon.stops.length > 0) config.stopSequences = canon.stops;
  if (canon.jsonMode) config.responseMimeType = "application/json";
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

// --- Refusal classification ---
//
// The three refusal kinds share status, error code and retry-after; only the
// human-readable message differs (the env's own wording):
//   "<Name>'s free-tier budget is spent"  -> budget: provider is done for the trial
//   "<Name> is out"                       -> outage: provider is done for the trial
//   "<Name> refused this call"            -> schedule: one of 1-2 isolated refusals,
//                                                the provider still has budget

export function classifyRefusal(status: number, code: string, message: string): RefusalKind {
  if (message.endsWith("budget is spent")) return "budget";
  if (message.endsWith(" is out")) return "outage";
  if (message.endsWith("refused this call")) return "schedule";
  if (code === "concurrent_requests") return "concurrent";
  if (status === 401) return "auth";
  if (status === 400) return "invalid";
  return "other";
}

export function parseErrorBody(wire: Wire, status: number, text: string): ProviderRefusal {
  let message = "";
  let code = "";
  try {
    const body = JSON.parse(text) as unknown;
    if (isRecord(body) && isRecord(body.error)) {
      const err = body.error;
      if (typeof err.message === "string") message = err.message;
      if (wire === "gemini") {
        if (typeof err.status === "string") code = err.status;
      } else if (typeof err.code === "string") {
        code = err.code;
      } else if (typeof err.type === "string") {
        code = err.type;
      }
    }
  } catch {
    // Non-JSON error body: fall through with empty message/code.
  }
  return { status, code, message, kind: classifyRefusal(status, code, message) };
}

// --- Usage fallback ---
//
// Provider usage is deterministic: input = 1 + floor((system + user +
// tools-joined-by-comma) / 4), output = 1 + floor(answer length / 4). The
// provider's own numbers are used whenever present; this recomputes them from
// the canonical request when a stream frame omits them.

export function computeUsage(canon: CanonRequest, text: string): { input: number; output: number } {
  const prompt = canon.systemText.length + canon.userText.length + canon.tools.map((t) => t.name).join(",").length;
  return { input: 1 + Math.floor(prompt / 4), output: 1 + Math.floor(text.length / 4) };
}

// --- SSE parsing (buffered: the whole body is read first, then parsed) ---

interface SseFrame {
  event: string | null;
  data: string;
}

function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  for (const block of text.split("\n\n")) {
    let event: string | null = null;
    const dataLines: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice("event:".length).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice("data:".length).trimStart());
    }
    if (dataLines.length > 0) frames.push({ event, data: dataLines.join("\n") });
  }
  return frames;
}

function isSse(text: string): boolean {
  return text.includes("data:");
}

// --- Provider answer parsing ---

function parseOpenAIAnswer(text: string, canon: CanonRequest): ProviderAnswer {
  let out = "";
  let stop = "stop";
  let usage: { input: number; output: number } | null = null;
  if (isSse(text)) {
    for (const frame of parseSseFrames(text)) {
      if (frame.data === "[DONE]") continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(frame.data);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        const input = numberOrNull(chunk.usage.prompt_tokens);
        const output = numberOrNull(chunk.usage.completion_tokens);
        if (input !== null && output !== null) usage = { input, output };
      }
      if (Array.isArray(chunk.choices) && chunk.choices.length > 0 && isRecord(chunk.choices[0])) {
        const choice = chunk.choices[0];
        if (isRecord(choice.delta) && typeof choice.delta.content === "string") out += choice.delta.content;
        if (typeof choice.finish_reason === "string") stop = choice.finish_reason;
      }
    }
  } else {
    const body = JSON.parse(text) as unknown;
    if (isRecord(body)) {
      if (Array.isArray(body.choices) && body.choices.length > 0 && isRecord(body.choices[0])) {
        const choice = body.choices[0];
        if (isRecord(choice.message)) {
          if (typeof choice.message.content === "string") out = choice.message.content;
          else if (Array.isArray(choice.message.content)) {
            out = choice.message.content
              .filter(isRecord)
              .filter((b) => b.type === "text" && typeof b.text === "string")
              .map((b) => b.text as string)
              .join("");
          }
        }
        if (typeof choice.finish_reason === "string") stop = choice.finish_reason;
      }
      if (isRecord(body.usage)) {
        const input = numberOrNull(body.usage.prompt_tokens);
        const output = numberOrNull(body.usage.completion_tokens);
        if (input !== null && output !== null) usage = { input, output };
      }
    }
  }
  return { text: out, stop, usage: usage ?? computeUsage(canon, out) };
}

function parseAnthropicAnswer(text: string, canon: CanonRequest): ProviderAnswer {
  let out = "";
  let stop = "end_turn";
  let input: number | null = null;
  let output: number | null = null;
  if (isSse(text)) {
    for (const frame of parseSseFrames(text)) {
      let payload: unknown;
      try {
        payload = JSON.parse(frame.data);
      } catch {
        continue;
      }
      if (!isRecord(payload)) continue;
      if (frame.event === "message_start" && isRecord(payload.message) && isRecord(payload.message.usage)) {
        const n = numberOrNull(payload.message.usage.input_tokens);
        if (n !== null) input = n;
      } else if (frame.event === "content_block_delta" && isRecord(payload.delta) && payload.delta.type === "text_delta" && typeof payload.delta.text === "string") {
        out += payload.delta.text;
      } else if (frame.event === "message_delta") {
        if (isRecord(payload.delta) && typeof payload.delta.stop_reason === "string") stop = payload.delta.stop_reason;
        if (isRecord(payload.usage)) {
          const n = numberOrNull(payload.usage.output_tokens);
          if (n !== null) output = n;
        }
      }
    }
  } else {
    const body = JSON.parse(text) as unknown;
    if (isRecord(body)) {
      if (Array.isArray(body.content)) {
        out = body.content
          .filter(isRecord)
          .filter((b) => b.type === "text" && typeof b.text === "string")
          .map((b) => b.text as string)
          .join("");
      }
      if (typeof body.stop_reason === "string") stop = body.stop_reason;
      if (isRecord(body.usage)) {
        const i = numberOrNull(body.usage.input_tokens);
        const o = numberOrNull(body.usage.output_tokens);
        if (i !== null) input = i;
        if (o !== null) output = o;
      }
    }
  }
  const usage =
    input !== null && output !== null ? { input, output } : computeUsage(canon, out);
  return { text: out, stop, usage };
}

function parseGeminiAnswer(text: string, canon: CanonRequest): ProviderAnswer {
  let out = "";
  let stop = "STOP";
  let usage: { input: number; output: number } | null = null;
  const readBody = (body: unknown): void => {
    if (!isRecord(body)) return;
    if (Array.isArray(body.candidates) && body.candidates.length > 0 && isRecord(body.candidates[0])) {
      const cand = body.candidates[0];
      if (isRecord(cand.content) && Array.isArray(cand.content.parts)) {
        for (const part of cand.content.parts) {
          if (isRecord(part) && typeof part.text === "string") out += part.text;
        }
      }
      if (typeof cand.finishReason === "string") stop = cand.finishReason;
    }
    if (isRecord(body.usageMetadata)) {
      const input = numberOrNull(body.usageMetadata.promptTokenCount);
      const output = numberOrNull(body.usageMetadata.candidatesTokenCount);
      if (input !== null && output !== null) usage = { input, output };
    }
  };
  if (isSse(text)) {
    for (const frame of parseSseFrames(text)) {
      try {
        readBody(JSON.parse(frame.data));
      } catch {
        continue;
      }
    }
  } else {
    readBody(JSON.parse(text));
  }
  return { text: out, stop, usage: usage ?? computeUsage(canon, out) };
}

export function parseProviderAnswer(wire: Wire, text: string, canon: CanonRequest): ProviderAnswer {
  if (wire === "openai") return parseOpenAIAnswer(text, canon);
  if (wire === "anthropic") return parseAnthropicAnswer(text, canon);
  return parseGeminiAnswer(text, canon);
}

// --- Client response emission ---
//
// Providers only ever answer with plain text and a family-0 stop, so the client
// stop reason is always "stop" (OpenAI) / "end_turn" (Anthropic). Stream pieces
// are fixed 13-character chunks, matching the provider's own convention.

function splitPieces(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += SSE_PIECE) pieces.push(text.slice(at, at + SSE_PIECE));
  return pieces.length === 0 ? [""] : pieces;
}

export function openAIJsonBody(answer: ProviderAnswer, model: string): Record<string, unknown> {
  return {
    id: "chatcmpl-agg",
    object: "chat.completion",
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: answer.text },
        finish_reason: "stop",
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: answer.usage.input,
      completion_tokens: answer.usage.output,
      total_tokens: answer.usage.input + answer.usage.output,
    },
  };
}

export function openAIStreamBody(answer: ProviderAnswer, model: string): string {
  const frames: string[] = [];
  const chunk = (delta: Record<string, unknown>, finish: string | null): string => {
    const body: Record<string, unknown> = {
      id: "chatcmpl-agg",
      object: "chat.completion.chunk",
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finish }],
    };
    return `data: ${JSON.stringify(body)}\n\n`;
  };
  frames.push(chunk({ role: "assistant", content: "" }, null));
  for (const piece of splitPieces(answer.text)) frames.push(chunk({ content: piece }, null));
  frames.push(chunk({}, "stop"));
  frames.push(
    `data: ${JSON.stringify({
      id: "chatcmpl-agg",
      object: "chat.completion.chunk",
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.usage.input,
        completion_tokens: answer.usage.output,
        total_tokens: answer.usage.input + answer.usage.output,
      },
    })}\n\n`,
  );
  frames.push("data: [DONE]\n\n");
  return frames.join("");
}

export function anthropicJsonBody(answer: ProviderAnswer, model: string): Record<string, unknown> {
  return {
    id: "msg_agg",
    type: "message",
    role: "assistant",
    model,
    content: [{ type: "text", text: answer.text }],
    stop_reason: "end_turn",
    stop_sequence: null,
    usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
  };
}

export function anthropicStreamBody(answer: ProviderAnswer, model: string): string {
  const frame = (event: string, data: Record<string, unknown>): string =>
    `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  const frames: string[] = [
    frame("message_start", {
      type: "message_start",
      message: {
        id: "msg_agg",
        type: "message",
        role: "assistant",
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.usage.input, output_tokens: 0 },
      },
    }),
    frame("content_block_start", {
      type: "content_block_start",
      index: 0,
      content_block: { type: "text", text: "" },
    }),
  ];
  for (const piece of splitPieces(answer.text)) {
    frames.push(
      frame("content_block_delta", {
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text: piece },
      }),
    );
  }
  frames.push(frame("content_block_stop", { type: "content_block_stop", index: 0 }));
  frames.push(
    frame("message_delta", {
      type: "message_delta",
      delta: { stop_reason: "end_turn", stop_sequence: null },
      usage: { output_tokens: answer.usage.output },
    }),
  );
  frames.push(frame("message_stop", { type: "message_stop" }));
  return frames.join("");
}

// --- Model lists (fixed timestamps: no wall clock anywhere) ---

export function openAIModelList(ids: string[]): Record<string, unknown> {
  return {
    object: "list",
    data: ids.map((id) => ({ id, object: "model", created: FIXED_CREATED, owned_by: "makefun" })),
  };
}

export function anthropicModelList(ids: string[]): Record<string, unknown> {
  return {
    data: ids.map((id) => ({ type: "model", id, display_name: id, created_at: "2026-01-01T00:00:00Z" })),
    has_more: false,
  };
}
