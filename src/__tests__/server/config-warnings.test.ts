import { describe, expect, it } from 'vitest'
import {
  parseGroupingMode,
  parseHybridWeight,
  parseMaxDistance,
  parseMaxFiles,
} from '../../cli/options.js'
import { parseChunkMinLength, resolveServerConfig } from '../../server-main.js'

// ============================================
// Unit Tests: Parser Functions
// ============================================

describe('parseGroupingMode', () => {
  it('returns undefined with no warning for empty input', () => {
    expect(parseGroupingMode(undefined)).toEqual({ value: undefined })
    expect(parseGroupingMode('')).toEqual({ value: undefined })
  })

  it('returns valid grouping modes', () => {
    expect(parseGroupingMode('similar')).toEqual({ value: 'similar' })
    expect(parseGroupingMode('related')).toEqual({ value: 'related' })
    expect(parseGroupingMode('SIMILAR')).toEqual({ value: 'similar' })
    expect(parseGroupingMode(' Related ')).toEqual({ value: 'related' })
  })
})

describe('parseMaxDistance', () => {
  it('returns undefined with no warning for empty input', () => {
    expect(parseMaxDistance(undefined)).toEqual({ value: undefined })
    expect(parseMaxDistance('')).toEqual({ value: undefined })
  })

  it('returns valid positive numbers', () => {
    expect(parseMaxDistance('0.5')).toEqual({ value: 0.5 })
    expect(parseMaxDistance('1.0')).toEqual({ value: 1.0 })
    expect(parseMaxDistance('0.001')).toEqual({ value: 0.001 })
  })
})

describe('parseMaxFiles', () => {
  it('returns undefined with no warning for empty input', () => {
    expect(parseMaxFiles(undefined)).toEqual({ value: undefined })
    expect(parseMaxFiles('')).toEqual({ value: undefined })
  })

  it('returns valid positive integers', () => {
    expect(parseMaxFiles('1')).toEqual({ value: 1 })
    expect(parseMaxFiles('10')).toEqual({ value: 10 })
  })
})

describe('parseHybridWeight', () => {
  it('returns undefined with no warning for empty input', () => {
    expect(parseHybridWeight(undefined)).toEqual({ value: undefined })
    expect(parseHybridWeight('')).toEqual({ value: undefined })
  })

  it('returns valid values in 0.0-1.0 range', () => {
    expect(parseHybridWeight('0')).toEqual({ value: 0 })
    expect(parseHybridWeight('0.5')).toEqual({ value: 0.5 })
    expect(parseHybridWeight('1')).toEqual({ value: 1 })
    expect(parseHybridWeight('1.0')).toEqual({ value: 1.0 })
  })
})

describe('parseChunkMinLength', () => {
  it('returns undefined with no warning for empty input', () => {
    expect(parseChunkMinLength(undefined)).toEqual({ value: undefined })
    expect(parseChunkMinLength('')).toEqual({ value: undefined })
  })

  it('returns valid integer values', () => {
    expect(parseChunkMinLength('100')).toEqual({ value: 100 })
    expect(parseChunkMinLength('50')).toEqual({ value: 50 })
  })

  it('returns valid boundary values', () => {
    expect(parseChunkMinLength('1')).toEqual({ value: 1 })
    expect(parseChunkMinLength('10000')).toEqual({ value: 10000 })
  })
})

describe('invalid search and chunk settings', () => {
  it.each([
    { name: 'RAG_GROUPING', parse: parseGroupingMode, raw: 'invalid' },
    { name: 'RAG_MAX_DISTANCE', parse: parseMaxDistance, raw: '0' },
    { name: 'RAG_MAX_DISTANCE', parse: parseMaxDistance, raw: '-1' },
    { name: 'RAG_MAX_DISTANCE', parse: parseMaxDistance, raw: 'abc' },
    { name: 'RAG_MAX_DISTANCE', parse: parseMaxDistance, raw: 'Infinity' },
    { name: 'RAG_MAX_FILES', parse: parseMaxFiles, raw: '0' },
    { name: 'RAG_MAX_FILES', parse: parseMaxFiles, raw: '-1' },
    { name: 'RAG_MAX_FILES', parse: parseMaxFiles, raw: 'abc' },
    { name: 'RAG_HYBRID_WEIGHT', parse: parseHybridWeight, raw: '-0.1' },
    { name: 'RAG_HYBRID_WEIGHT', parse: parseHybridWeight, raw: '1.1' },
    { name: 'RAG_HYBRID_WEIGHT', parse: parseHybridWeight, raw: 'abc' },
    { name: 'CHUNK_MIN_LENGTH', parse: parseChunkMinLength, raw: '0' },
    { name: 'CHUNK_MIN_LENGTH', parse: parseChunkMinLength, raw: '-1' },
    { name: 'CHUNK_MIN_LENGTH', parse: parseChunkMinLength, raw: '10001' },
    { name: 'CHUNK_MIN_LENGTH', parse: parseChunkMinLength, raw: 'abc' },
    { name: 'CHUNK_MIN_LENGTH', parse: parseChunkMinLength, raw: '50.5' },
    { name: 'CHUNK_MIN_LENGTH', parse: parseChunkMinLength, raw: '50abc' },
    { name: 'CHUNK_MIN_LENGTH', parse: parseChunkMinLength, raw: '1e2' },
    { name: 'CHUNK_MIN_LENGTH', parse: parseChunkMinLength, raw: '0x10' },
    { name: 'CHUNK_MIN_LENGTH', parse: parseChunkMinLength, raw: ' 50' },
  ])('$name ignores $raw with a warning', ({ name, parse, raw }) => {
    const result = parse(raw)
    expect(result.value).toBeUndefined()
    expect(result.warning).toContain(`Invalid ${name} value: "${raw}"`)
  })
})

describe('MAX_FILE_SIZE configuration', () => {
  it.each([
    'invalid',
    '0',
    '-1',
    '524288001',
    'Infinity',
    '12oops',
    '1.5',
    '1e8',
    '0x100000',
    '100MB',
  ])('warns and retains the default for %s', async (value) => {
    const config = await resolveServerConfig({ MAX_FILE_SIZE: value }, process.cwd())
    expect(config.maxFileSize).toBe(104_857_600)
    expect(config.configWarnings?.join('\n')).toContain('Invalid MAX_FILE_SIZE')
  })

  it.each(['1', '524288000'])('accepts valid size %s', async (value) => {
    const config = await resolveServerConfig({ MAX_FILE_SIZE: value }, process.cwd())
    expect(config.maxFileSize).toBe(Number(value))
    expect(config.configWarnings?.join('\n') ?? '').not.toContain('Invalid MAX_FILE_SIZE')
  })
})

describe('model configuration overrides', () => {
  it('passes parsed overrides into the server configuration', async () => {
    const config = await resolveServerConfig(
      { RAG_MODEL_CONFIG: '{"vision_config":null}' },
      process.cwd()
    )
    expect(config.modelConfig).toEqual({ vision_config: null })
  })

  it('rejects invalid overrides before server startup', async () => {
    await expect(resolveServerConfig({ RAG_MODEL_CONFIG: '[]' }, process.cwd())).rejects.toThrow(
      'RAG_MODEL_CONFIG'
    )
  })
})
