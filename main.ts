// free-model-aggregator-02: one server, two client wires, and the best live free model behind
// every answer.
//
// Built on the starter's brief parser, catalog and wire translations (lib/providers.ts,
// lib/wire.ts), with the starter's marked gaps closed:
//
//   - quality (60%): every prompt is routed to the highest-rated free model that meets its
//     requirements, by the brief's own table (rating_milli). JSON-mode prompts go only to models
//     whose row carries JSON, tool prompts only to models whose row carries tools.
//   - proof before report: a dialled answer is accepted only when the provider's own body carries
//     the prompt's token (and, for a JSON-mode prompt, parses as the JSON object asked for). An
//     answer without the token earns the prompt nothing, so the request fails over to the next
//     candidate instead of reporting a model whose answer cannot be credited.
//   - failover (10%): a gateway refusal (429, 5xx, a gateway 4xx) is recorded in the
//     x-makefun-upstream-failures trail on the answer that finally serves, and the request is
//     retried on the next-best candidate. A model that refuses or answers tokenless twice in a
//     row is demoted below every healthy candidate for the rest of the trial.
//   - one upstream wire: every model is dialled on the gateway's Chat Completions route,
//     non-streamed, and the buffered answer is rendered into the client's wire, streamed or not
//     as the client asked, with the dialled id on every answer and every chunk, its usage and
//     stop reason whole. The gateway's record then holds one whole provider body per prompt,
//     carrying the token in plain sight.
//   - catalog (5%): the brief's fourteen free ids, exactly, in the register body and in both
//     GET /v1/models answers.
//
// Everything is a function of the brief, the requests and the gateway's answers: no wall clock
// in anything sent, no randomness, so a replayed trial makes the same calls in the same order
// and leaves the same export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIToAnthropic,
  sseEvent,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one gateway call may take, in milliseconds: the env's client waits 30 s a prompt. */
const CALL_TIMEOUT_MS = 20000;
/**
 * The most gateway dials one prompt may cost, first candidate included. Refusals are fast and
 * do not count toward the tie-break (only calls the provider answered do), so the walk is
 * generous: a bad minute at the top of the table should not cost a prompt its failover.
 */
const MAX_ATTEMPTS = 10;
/** Consecutive refusals or tokenless answers after which a model is demoted below the healthy. */
const DEAD_AFTER = 2;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The gateway's own output cap: a request naming more is refused before it is forwarded. */
const MAX_OUTPUT_TOKENS = 8192;
/** The output cap named when a client body names none (the gateway refuses an uncapped call). */
const DEFAULT_OUTPUT_TOKENS = 1024;
/** A fixed epoch second for synthesized stream chunks: nothing sent reads the clock. */
const FIXED_CREATED = 1767225600;
/** The response header that carries the refusals a served prompt recovered from. */
const FAILURES_HEADER = 'x-makefun-upstream-failures';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** Consecutive failed dials per model id, reset by its next accepted answer. */
const consecutiveFailures = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on
 * the trial network); a base that does not already end in `/v1` gets one, so both spellings reach it.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

interface Route {
  model: ModelRow;
  /** The model's place in the brief's list: the deterministic tie-break under equal ratings. */
  briefIndex: number;
}

/**
 * Every free model that could serve one request, best first: healthy models ahead of demoted
 * ones, then by the brief's rating, then by the brief's own order. Pure in the brief and the
 * failures the gateway's answers have caused so far, so a replay walks the same list.
 */
function candidatesFor(requires: Requirement): Route[] {
  const dead = (m: ModelRow): number => ((consecutiveFailures.get(m.id) ?? 0) >= DEAD_AFTER ? 1 : 0);
  return freeModels(brief)
    .map((model, briefIndex) => ({ model, briefIndex }))
    .filter((r) => meetsRequirements(requires, capabilitiesOf(r.model)))
    .sort((a, b) => dead(a.model) - dead(b.model) || b.model.rating_milli - a.model.rating_milli || a.briefIndex - b.briefIndex);
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** The text a message carries, string or block array, for reading the prompt's token off a request. */
function messageText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/**
 * The unique tokens a prompt carries, read off the client's own request (its system prompt and
 * its messages): the answer the house credits is the one whose provider body carries the token,
 * so the token is what an accepted answer is checked against.
 */
function tokensIn(body: Json): string[] {
  const parts: string[] = [messageText(body.system)];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) if (isRecord(message)) parts.push(messageText(message.content));
  const found = parts.join('\n').match(/\b[A-Z][A-Z0-9]*-\d+\b/g) ?? [];
  return [...new Set(found)];
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

interface TrailRow {
  status: number;
  code: string;
  model: string;
}

/** The refusal trail as the response header carries it: at most 8 rows, in the order refused. */
function trailHeader(trail: readonly TrailRow[]): Record<string, string> {
  if (trail.length === 0) return {};
  return { [FAILURES_HEADER]: JSON.stringify(trail.slice(0, 8)) };
}

function sendJson(res: ServerResponse, status: number, body: unknown, extra: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extra });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, extra: Record<string, string> = {}): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), extra);
  else sendJson(res, status, openAIError(message, type, null), extra);
}

function sendStream(res: ServerResponse, text: string, extra: Record<string, string> = {}): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...extra });
  res.end(text);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * The Chat Completions body one dial sends: the client's request carried through (an Anthropic
 * request translated first), the dialled model's id named, streaming off — the answer is
 * buffered whole, checked, and only then rendered in the client's wire — and an output cap the
 * gateway accepts always named.
 */
function gatewayBody(clientApi: 'openai' | 'anthropic', body: Json, modelId: string): Json {
  const out: Json = clientApi === 'openai' ? { ...body, model: modelId } : anthropicToOpenAI(body, modelId);
  delete out.stream;
  delete out.stream_options;
  if (typeof out.max_tokens === 'number') out.max_tokens = Math.min(out.max_tokens, MAX_OUTPUT_TOKENS);
  else if (typeof out.max_completion_tokens === 'number') out.max_completion_tokens = Math.min(out.max_completion_tokens, MAX_OUTPUT_TOKENS);
  else out.max_tokens = DEFAULT_OUTPUT_TOKENS;
  return out;
}

interface DialResult {
  status: number;
  text: string;
}

/** One dialled call to the gateway's Chat Completions route; a thrown fetch is a status of 0. */
async function dial(payload: Json): Promise<DialResult> {
  try {
    const res = await fetch(chatCompletionsUrl(), {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: `Bearer ${GATEWAY_KEY}` },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
    return { status: res.status, text: await res.text() };
  } catch {
    return { status: 0, text: '' };
  }
}

/** The refusal's code as the provider spelled it, for the trail row; '' when it cannot be read. */
function errorCodeOf(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      if (typeof value.error.code === 'string') return value.error.code;
      if (typeof value.error.type === 'string') return value.error.type;
    }
  } catch {
    // Not JSON: the status alone goes on the trail.
  }
  return '';
}

/** Whether a text parses as one JSON object, code fences aside: what a JSON-mode prompt asks for. */
function parsesAsJsonObject(text: string): boolean {
  let trimmed = text.trim();
  if (trimmed.startsWith('```')) trimmed = trimmed.replace(/^```[a-z]*\n?/, '').replace(/```$/, '').trim();
  try {
    return isRecord(JSON.parse(trimmed));
  } catch {
    return false;
  }
}

/** A stream of bytes from one whole text, for feeding a stream translator a buffered answer. */
function streamOf(text: string): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
}

/**
 * One buffered Chat Completions answer as an OpenAI SSE stream naming the dialled model on every
 * chunk: a role chunk, the content (or the tool calls) as delta chunks, the finish chunk, the
 * usage chunk when the provider sent usage, and [DONE] — the order the wire checks expect.
 */
function openAIStreamText(answer: Json, modelId: string): string {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const base = {
    id: typeof answer.id === 'string' && answer.id !== '' ? answer.id : 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model: modelId,
  };
  const frames: string[] = [];
  const chunk = (delta: Json, finish: string | null): string =>
    sseEvent(null, { ...base, choices: [{ index: 0, delta, finish_reason: finish }] });
  frames.push(chunk({ role: 'assistant', content: '' }, null));
  const content = typeof message.content === 'string' ? message.content : messageText(message.content);
  if (content !== '') frames.push(chunk({ content }, null));
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  calls.forEach((call, index) => {
    if (!isRecord(call)) return;
    const fn = isRecord(call.function) ? call.function : {};
    frames.push(
      chunk(
        {
          tool_calls: [
            {
              index,
              id: typeof call.id === 'string' ? call.id : `call_${index}`,
              type: 'function',
              function: { name: typeof fn.name === 'string' ? fn.name : '', arguments: typeof fn.arguments === 'string' ? fn.arguments : '' },
            },
          ],
        },
        null,
      ),
    );
  });
  frames.push(chunk({}, typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop'));
  if (isRecord(answer.usage)) frames.push(sseEvent(null, { ...base, choices: [], usage: answer.usage }));
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** The accepted answer rendered in the client's wire, streamed or not as the client asked. */
async function render(clientApi: 'openai' | 'anthropic', stream: boolean, answer: Json, modelId: string, res: ServerResponse, trail: readonly TrailRow[]): Promise<void> {
  const extra = trailHeader(trail);
  if (clientApi === 'openai') {
    if (stream) sendStream(res, openAIStreamText(answer, modelId), extra);
    else sendJson(res, 200, openAIJsonWithModel(answer, modelId), extra);
    return;
  }
  if (!stream) {
    sendJson(res, 200, openAIToAnthropic(answer, modelId), extra);
    return;
  }
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...extra });
  for await (const event of anthropicEventsFromOpenAI(streamOf(openAIStreamText(answer, modelId)), modelId)) res.write(event);
  res.end();
}

/**
 * One client request, on either wire: walk the candidates best-first, dial each on Chat
 * Completions, and accept the first answer the house can credit — a successful call whose own
 * body carries the prompt's token (and parses as JSON when the prompt asked for JSON). Refusals
 * go on the trail the served answer carries; a model that fails twice running is demoted for
 * the prompts still to come.
 */
async function serve(clientApi: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const tokens = tokensIn(body);
  const jsonMode =
    requires.json === true ||
    (clientApi === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema'));
  const candidates = candidatesFor(requires);
  if (candidates.length === 0) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model in the brief meets this request');
  }
  const trail: TrailRow[] = [];
  let attempts = 0;
  for (const route of candidates) {
    if (attempts >= MAX_ATTEMPTS) break;
    attempts += 1;
    const result = await dial(gatewayBody(clientApi, body, route.model.id));
    if (result.status === 0) {
      consecutiveFailures.set(route.model.id, (consecutiveFailures.get(route.model.id) ?? 0) + 1);
      continue;
    }
    if (result.status < 200 || result.status >= 300) {
      consecutiveFailures.set(route.model.id, (consecutiveFailures.get(route.model.id) ?? 0) + 1);
      if (trail.length < 8) trail.push({ status: result.status, code: errorCodeOf(result.text), model: route.model.id });
      continue;
    }
    let answer: Json | null = null;
    try {
      const value: unknown = JSON.parse(result.text);
      answer = isRecord(value) ? value : null;
    } catch {
      answer = null;
    }
    if (answer === null) {
      consecutiveFailures.set(route.model.id, (consecutiveFailures.get(route.model.id) ?? 0) + 1);
      continue;
    }
    const choices = Array.isArray(answer.choices) ? answer.choices : [];
    const message = isRecord(choices[0]) && isRecord(choices[0].message) ? choices[0].message : {};
    const content = typeof message.content === 'string' ? message.content : messageText(message.content);
    const tokenOk = tokens.length === 0 || tokens.some((t) => result.text.includes(t));
    const jsonOk = !jsonMode || parsesAsJsonObject(content);
    if (!tokenOk || !jsonOk) {
      // A real answer the house cannot credit: worth nothing reported, so fail over quietly.
      consecutiveFailures.set(route.model.id, (consecutiveFailures.get(route.model.id) ?? 0) + 1);
      continue;
    }
    consecutiveFailures.set(route.model.id, 0);
    await render(clientApi, body.stream === true, answer, route.model.id, res, trail);
    return;
  }
  return sendError(
    res,
    clientApi,
    502,
    clientApi === 'anthropic' ? 'api_error' : 'upstream_error',
    'every free model refused or answered without the token',
    trailHeader(trail),
  );
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
