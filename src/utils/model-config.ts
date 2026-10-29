import { isRecord } from './type-guards.js'

/** Parse top-level model config overrides. Invalid input must stop model loading. */
export function parseModelConfig(value: string | undefined): Record<string, unknown> | undefined {
  if (!value?.trim()) {
    return undefined
  }
  let parsed: unknown
  try {
    parsed = JSON.parse(value)
  } catch {
    throw new Error('RAG_MODEL_CONFIG must be a JSON object')
  }
  if (!isRecord(parsed) || Array.isArray(parsed)) {
    throw new Error('RAG_MODEL_CONFIG must be a JSON object')
  }
  for (const key of ['__proto__', 'constructor', 'prototype']) {
    if (Object.hasOwn(parsed, key)) {
      throw new Error(`RAG_MODEL_CONFIG cannot override "${key}"`)
    }
  }
  return parsed
}
