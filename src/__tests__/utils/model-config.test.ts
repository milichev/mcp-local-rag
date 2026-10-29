import { describe, expect, it } from 'vitest'
import { parseModelConfig } from '../../utils/model-config.js'

describe('parseModelConfig', () => {
  it.each([undefined, '', '  '])('treats %s as unset', (value) => {
    expect(parseModelConfig(value)).toBeUndefined()
  })

  it('keeps nulls and nested model-specific values', () => {
    expect(parseModelConfig('{"audio_config":null,"custom":{"enabled":false}}')).toEqual({
      audio_config: null,
      custom: { enabled: false },
    })
  })

  it.each(['{', 'null', '[]', '1', 'true', '"value"'])('rejects %s', (value) => {
    expect(() => parseModelConfig(value)).toThrow('RAG_MODEL_CONFIG must be a JSON object')
  })

  it.each(['__proto__', 'constructor', 'prototype'])('rejects reserved key %s', (key) => {
    expect(() => parseModelConfig(`{"${key}":{}}`)).toThrow('cannot override')
  })
})
