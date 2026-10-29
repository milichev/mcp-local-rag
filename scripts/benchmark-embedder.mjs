import { spawn, spawnSync } from 'node:child_process'
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(__dirname, '..')
const bin = resolve(repoRoot, 'dist/index.js')
const QUERY = 'order'
const DEFAULT_CASES = resolve(__dirname, 'benchmark-embedder.example.json')

function usage() {
  console.error(
    'Usage: node scripts/benchmark-embedder.mjs <docs-path> [--format md|json] [--cases path]'
  )
  process.exit(2)
}

function flagValue(args, name) {
  const i = args.indexOf(name)
  return i >= 0 ? args[i + 1] : undefined
}

const args = process.argv.slice(2)
const docsPathArg = args.find((a) => !a.startsWith('--'))
const format = flagValue(args, '--format') ?? 'md'
const casesPath = resolve(flagValue(args, '--cases') ?? DEFAULT_CASES)
if (!docsPathArg || (format !== 'md' && format !== 'json')) usage()
const docsPath = resolve(docsPathArg)
const CASES = JSON.parse(readFileSync(casesPath, 'utf8'))
if (!Array.isArray(CASES)) {
  console.error(`Cases file must be a JSON array: ${casesPath}`)
  process.exit(2)
}

const CACHE_DIR = resolve(repoRoot, 'tmp/models')

function dbPathFor(caseId) {
  return resolve(repoRoot, 'tmp/lancedb', caseId)
}

function envFor(c, dbPath) {
  const env = {
    ...process.env,
    DB_PATH: dbPath,
    CACHE_DIR,
    BASE_DIRS: JSON.stringify([docsPath]),
    MODEL_NAME: c.modelName,
  }
  if (c.dtype) env.RAG_DTYPE = c.dtype
  else delete env.RAG_DTYPE
  if (c.modelConfig !== undefined) env.RAG_MODEL_CONFIG = JSON.stringify(c.modelConfig)
  else delete env.RAG_MODEL_CONFIG
  return env
}

function prepareDirs(c) {
  const dbPath = dbPathFor(c.id)
  rmSync(dbPath, { recursive: true, force: true })
  mkdirSync(dbPath, { recursive: true })
  mkdirSync(CACHE_DIR, { recursive: true })
  return { dbPath }
}

function parseTimeL(text) {
  const real = Number(/([\d.]+)\s+real/.exec(text)?.[1])
  const maxRssBytes = Number(/(\d+)\s+maximum resident set size/.exec(text)?.[1])
  return {
    elapsedSec: real,
    peakMb: maxRssBytes / (1024 * 1024),
  }
}

function runWithTime(argv, env) {
  const result = spawnSync('/usr/bin/time', ['-l', process.execPath, ...argv], {
    cwd: repoRoot,
    env,
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024,
  })
  if (result.stdout) process.stdout.write(result.stdout)
  if (result.stderr) process.stderr.write(result.stderr)
  if (result.error) throw result.error
  if (result.status !== 0) {
    throw new Error(`command failed (${result.status}): ${argv.join(' ')}`)
  }
  return parseTimeL(result.stderr ?? '')
}

function rssMb(pid) {
  const out = spawnSync('ps', ['-o', 'rss=', '-p', String(pid)], { encoding: 'utf8' })
  return Number(out.stdout.trim()) / 1024
}

/** Start MCP, sample idle RSS after init, then kill — server does not exit on its own. */
function runMcpIdle(env) {
  return new Promise((resolveIdle, reject) => {
    const started = performance.now()
    let done = false
    const child = spawn(process.execPath, [bin], {
      cwd: repoRoot,
      env,
      stdio: ['ignore', 'ignore', 'pipe'],
    })
    let stderr = ''
    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString()
      if (!done && stderr.includes('RAGServer initialized')) {
        done = true
        const idleMb = rssMb(child.pid)
        const elapsedSec = (performance.now() - started) / 1000
        child.kill('SIGTERM')
        resolveIdle({ elapsedSec, idleMb })
      }
    })
    child.on('error', (err) => {
      if (!done) {
        done = true
        reject(err)
      }
    })
    child.on('exit', (code, signal) => {
      if (!done) {
        done = true
        reject(new Error(`MCP exited before init (code=${code}, signal=${signal})\n${stderr}`))
      }
    })
  })
}

async function runValidCase(c) {
  console.error(`\n=== ${c.id} ===`)
  const { dbPath } = prepareDirs(c)
  const env = envFor(c, dbPath)

  console.error('mcp idle...')
  const mcp = await runMcpIdle(env)
  console.error(`mcp: ${mcp.elapsedSec.toFixed(1)}s idle=${mcp.idleMb.toFixed(1)}MB`)

  console.error('sync...')
  const sync = runWithTime([bin, 'sync', docsPath], env)
  console.error(`sync: ${sync.elapsedSec.toFixed(1)}s peak=${sync.peakMb.toFixed(1)}MB`)

  console.error('query...')
  const query = runWithTime([bin, 'query', QUERY], env)
  console.error(`query: ${query.elapsedSec.toFixed(1)}s peak=${query.peakMb.toFixed(1)}MB`)

  return {
    case: c.id,
    mcp_s: mcp.elapsedSec,
    idle_mb: mcp.idleMb,
    sync_s: sync.elapsedSec,
    sync_peak_mb: sync.peakMb,
    query_s: query.elapsedSec,
    query_peak_mb: query.peakMb,
  }
}

function formatMd(rows) {
  const lines = [
    '| case | mcp | idle | sync | sync peak | query | query peak |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
  ]
  for (const r of rows) {
    lines.push(
      `| ${r.case} | ${r.mcp_s.toFixed(1)}s | ${r.idle_mb.toFixed(1)}MB | ${r.sync_s.toFixed(1)}s | ${r.sync_peak_mb.toFixed(1)}MB | ${r.query_s.toFixed(1)}s | ${r.query_peak_mb.toFixed(1)}MB |`
    )
  }
  return `${lines.join('\n')}\n`
}

function stamp() {
  const d = new Date()
  const p = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`
}

console.error('Building...')
const build = spawnSync('pnpm', ['build'], { cwd: repoRoot, stdio: 'inherit' })
if (build.status !== 0) process.exit(build.status ?? 1)

const rows = []
for (const c of CASES) {
  rows.push(await runValidCase(c))
}

const outDir = resolve(repoRoot, 'tmp')
mkdirSync(outDir, { recursive: true })
const outPath = resolve(outDir, `benchmark-embedder-${stamp()}.${format}`)
const body = format === 'json' ? `${JSON.stringify(rows, null, 2)}\n` : formatMd(rows)
writeFileSync(outPath, body)
console.error(`\nWrote ${outPath}`)
console.log(body)
