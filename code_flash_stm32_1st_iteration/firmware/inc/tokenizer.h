#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

/*
 * Encodes as many characters as possible from `src` (a NUL-terminated UTF-8
 * string) into `dst`, up to `dst_cap` token ids. Returns the number of tokens
 * written. Bytes that don't map to any vocabulary entry are silently skipped.
 */
size_t tokenize(const char *src, uint8_t *dst, size_t dst_cap);

/*
 * Appends the UTF-8 encoding of one token id to `dst` (bounded by `dst_cap`
 * including the trailing NUL). Returns number of bytes appended (excluding
 * the NUL). Safe for id in [0, VOCAB_SIZE); other ids emit "?".
 */
size_t detokenize_one(uint8_t id, char *dst, size_t dst_cap);

#endif /* TOKENIZER_H */
